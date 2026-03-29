from timm.loss import SoftTargetCrossEntropy
from timm.data.mixup import Mixup
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# =============================================================
#  Configurations
# =============================================================
MODE = 'test'          # 'train' | 'test'
WEIGHTS_PATH = 'best_model.pth'  # test mode 讀取；train mode 存到這裡
RESUME = False            # True = 讀 WEIGHTS_PATH 繼續訓練

TRAIN_DIR = '../data/train'  # 子資料夾為 class label
VAL_DIR = '../data/val'    # 子資料夾為 class label
TEST_DIR = '../data/test'   # 直接放圖片，無子資料夾

NUM_CLASSES = 100
BATCH_SIZE = 32
EPOCHS = 80
LR = 1e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
DROP_RATE = 0.3
DROP_PATH_RATE = 0.2

USE_TTA = True  # 推論時是否開水平翻轉 TTA
SAVE_CSV = True   # 是否儲存預測結果 CSV

MIXUP_ALPHA = 0.8  # Mixup 的混合強度
CUTMIX_ALPHA = 1.0  # CutMix 的混合強度
MIX_PROB = 0.5  # 每個 Batch 套用 Mixup/CutMix 的機率 (1.0 代表必定觸發)
# =============================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Mode: {MODE.upper()} | Device: {device}")


# ==================== Transforms ====================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(320, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

val_transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.CenterCrop(320),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==================== Unlabeled Test Dataset ====================
class TestDataset(Dataset):
    """test 資料夾下直接放圖片（無子資料夾），回傳 (tensor, filename)"""
    EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, root, transform=None):
        self.transform = transform
        self.paths = sorted([
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.splitext(f)[1].lower() in self.EXTS
        ])
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}")
        print(f"TestDataset: {len(self.paths)} images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(path)


# ==================== Dataset & DataLoader ====================
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transform)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4)

idx_to_class = {v: k for k, v in val_dataset.class_to_idx.items()}

if MODE == 'train':
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4)
    print(
        f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
else:
    test_dataset = TestDataset(root=TEST_DIR, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4)
    print(f"Val: {len(val_dataset)} images | Test: {len(test_dataset)} images")


# ==================== Model ====================
def build_model(pretrained=True):
    drop = DROP_RATE if MODE == 'train' else 0.0
    dp = DROP_PATH_RATE if MODE == 'train' else 0.0
    return timm.create_model(
        'resnet50d',
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
        drop_rate=drop,
        drop_path_rate=dp
    )


if MODE == 'train':
    if RESUME:
        print(f"Resuming from {WEIGHTS_PATH}")
        model = build_model(pretrained=False)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    else:
        model = build_model(pretrained=True)
else:
    print(f"Loading weights from {WEIGHTS_PATH}")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

model = model.to(device)


# ==================== Helper Functions ====================
def predict(model, inputs):
    """帶 TTA 的推論，回傳 softmax 機率"""
    if USE_TTA:
        p1 = torch.softmax(model(inputs), dim=1)
        p2 = torch.softmax(model(torch.flip(inputs, dims=[3])), dim=1)
        return (p1 + p2) / 2
    return torch.softmax(model(inputs), dim=1)


def run_val(model, loader, criterion=None):
    """有 label 的 val set：回傳 (avg_loss, acc, preds, labels, probs)"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            probs = predict(model, inputs)
            preds = torch.argmax(probs, dim=1)

            if criterion is not None:
                loss = criterion(torch.log(probs + 1e-9), labels)
                total_loss += loss.item() * inputs.size(0)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total if criterion is not None else None
    acc = 100.0 * correct / total
    return avg_loss, acc, np.array(all_preds), np.array(
        all_labels), np.array(all_probs)


def run_test_inference(model, loader):
    """無 label 的 test set：回傳 (filenames, preds, probs)"""
    model.eval()
    all_filenames, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, filenames in tqdm(loader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            probs = predict(model, inputs)
            preds = torch.argmax(probs, dim=1)
            all_filenames.extend(filenames)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_filenames, np.array(all_preds), np.array(all_probs)


def save_confusion_matrix(
        all_labels, all_preds,
        save_path='confusion_matrix.png',
        title='Confusion Matrix'):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def print_val_report(all_labels, all_preds, all_probs):
    total = len(all_labels)
    correct = (all_preds == all_labels).sum()
    acc = 100.0 * correct / total

    top5_correct = sum(
        1 for i, label in enumerate(all_labels)
        if label in np.argsort(all_probs[i])[-5:]
    )
    top5_acc = 100.0 * top5_correct / total

    print(f"\n{'='*45}")
    print(f"  Accuracy      : {correct}/{total} = {acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"  TTA enabled   : {USE_TTA}")
    print(f"{'='*45}")

    per_class_c = np.zeros(NUM_CLASSES)
    per_class_t = np.zeros(NUM_CLASSES)
    for pred, label in zip(all_preds, all_labels):
        per_class_t[label] += 1
        if pred == label:
            per_class_c[label] += 1
    per_class_acc = per_class_c / (per_class_t + 1e-9) * 100

    print("\n  Worst 10 classes:")
    for cls_idx in np.argsort(per_class_acc)[:10]:
        name = idx_to_class[cls_idx]
        print(f"    [{cls_idx:3d}] {name:25s} "
              f"{per_class_acc[cls_idx]:.1f}%  "
              f"({int(per_class_c[cls_idx])}/{int(per_class_t[cls_idx])})")

    if SAVE_CSV:
        with open('val_predictions.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'true_label', 'true_class',
                             'pred_label', 'pred_class',
                             'confidence', 'correct'])
            for i, (label, pred) in enumerate(zip(all_labels, all_preds)):
                writer.writerow([
                    i,
                    label, idx_to_class[label],
                    pred, idx_to_class[pred],
                    f"{all_probs[i][pred]*100:.2f}",
                    int(label == pred)
                ])
        print("  Val predictions saved to val_predictions.csv")

    save_confusion_matrix(
        all_labels,
        all_preds,
        save_path='confusion_matrix_val.png')


def save_test_csv(filenames, all_preds, all_probs, save_path='prediction.csv'):
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'pred_label'])
        for fname, pred, probs in zip(filenames, all_preds, all_probs):
            writer.writerow([
                fname.replace(
                    '.jpg',
                    '').replace(
                    '.jpeg',
                    '').replace(
                    '.png',
                    '').replace(
                    '.bmp',
                    '').replace(
                    '.webp',
                    ''),
                idx_to_class[pred]
            ])
    print(f"Test predictions saved to {save_path}")


# =============================================================
#  TRAIN MODE
# =============================================================
if MODE == 'train':
    # Mixup Object
    mixup_fn = Mixup(
        mixup_alpha=MIXUP_ALPHA,
        cutmix_alpha=CUTMIX_ALPHA,
        prob=MIX_PROB,
        switch_prob=0.5,       # 50% 機率用 Mixup，50% 機率用 CutMix
        mode='batch',          # 以 Batch 為單位進行混合
        label_smoothing=LABEL_SMOOTH,
        num_classes=NUM_CLASSES
    )

    # Loss Function
    train_criterion = SoftTargetCrossEntropy()
    val_criterion = nn.NLLLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []}

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = mixup_fn(inputs, labels)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = train_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            _, true_labels = torch.max(labels, 1)

            train_total += labels.size(0)
            train_correct += (predicted == true_labels).sum().item()

        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100.0 * train_correct / train_total

        # --- Val ---
        epoch_val_loss, epoch_val_acc, val_preds, val_labels, _ = run_val(
            model, val_loader, val_criterion
        )

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        current_lr = scheduler.get_last_lr()[0]

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(
                f"  ★ New best! Saved → {WEIGHTS_PATH}  ({best_val_acc:.2f}%)")

        print(f"\n--- Epoch [{epoch+1}/{EPOCHS}]  lr={current_lr:.2e} ---")
        print(f"Train| Loss: {epoch_train_loss:.4f} \
              | Acc: {epoch_train_acc:.2f}%")
        print(f"Val  | Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.2f}%")
        print("-" * 40)

    # --- 訓練結束後跑 test inference ---
    print("\n====== Training done. Running test inference... ======")
    test_dataset = TestDataset(root=TEST_DIR, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4)
    filenames, test_preds, test_probs = run_test_inference(model, test_loader)
    if SAVE_CSV:
        save_test_csv(filenames, test_preds, test_probs)

    save_confusion_matrix(val_labels, val_preds,
                          save_path=f'cm_epoch_{EPOCHS}.png',
                          title=f'Confusion Matrix - Epoch {EPOCHS}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
    plt.close()
    print("Training curve saved to training_curve.png")
    print(f"Best Val Acc: {best_val_acc:.2f}%")


# =============================================================
#  TEST MODE
# =============================================================
else:
    # 1. Val set：算準確率 + confusion matrix
    print("\n[1/2] Evaluating on val set...")
    _, _, val_preds, val_labels, val_probs = run_val(model, val_loader)
    print_val_report(val_labels, val_preds, val_probs)

    # 2. Test set：純推論輸出 CSV
    print("\n[2/2] Inferencing on test set...")
    filenames, test_preds, test_probs = run_test_inference(model, test_loader)
    if SAVE_CSV:
        save_test_csv(filenames, test_preds, test_probs)
