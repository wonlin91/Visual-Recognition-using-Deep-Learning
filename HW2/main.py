import argparse
import copy
import json
import os
import time
import math
from collections import defaultdict
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.ops import box_iou, nms
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────
NUM_CLASSES = 10  # digits 0-9; category_id 1~10 in COCO labels
NUM_QUERIES = 300
NO_OBJ_IDX = 0


# ─────────────────────────────────────────────
#  EMA
# ─────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model, decay=0.9997):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, mp in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)
        for ema_b, mb in zip(self.ema.buffers(), model.buffers()):
            ema_b.data.copy_(mb.data)


# ─────────────────────────────────────────────
#  Model builder
# ─────────────────────────────────────────────
def build_model(
    num_classes=NUM_CLASSES, num_queries=NUM_QUERIES, pretrained_backbone=True
):
    config = DeformableDetrConfig(
        num_labels=num_classes,
        backbone="resnet50",
        use_pretrained_backbone=pretrained_backbone,
        ignore_mismatched_sizes=True,
        # 架構參數
        d_model=256,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        dim_feedforward=1024,
        dropout=0.1,
        num_queries=num_queries,
        two_stage=False,
        # 訓練相關
        auxiliary_loss=True,  # 中間層 loss，幫助收斂
    )
    model = DeformableDetrForObjectDetection(config)
    return model


# ─────────────────────────────────────────────
#  Warmup + Cosine LR
# ─────────────────────────────────────────────

def build_warmup_cosine_scheduler(
    optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.05
):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
#  Box utilities
# ─────────────────────────────────────────────
def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_coco_to_detr(boxes, img_w, img_h):
    """COCO [x_min, y_min, w, h] (unnorm) → normalised [cx, cy, w, h]"""
    x, y, w, h = boxes.unbind(-1)
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return torch.stack([cx, cy, nw, nh], dim=-1)


def box_detr_to_coco(boxes, img_w, img_h):
    """Normalised [cx, cy, w, h] → COCO [x_min, y_min, w, h] (unnorm)"""
    cx, cy, w, h = boxes.unbind(-1)
    x = (cx - w / 2) * img_w
    y = (cy - h / 2) * img_h
    bw = w * img_w
    bh = h * img_h
    return torch.stack([x, y, bw, bh], dim=-1)


# ─────────────────────────────────────────────
#  Postprocess
# ─────────────────────────────────────────────
def postprocess_single(
    logits: torch.Tensor,  # (Q, num_classes+1)
    boxes: torch.Tensor,  # (Q, 4) normalised cxcywh
    score_thresh: float = 0.15,
    nms_iou_thresh: float = 0.5,
    top_k: int = 20,
):
    """
    HuggingFace Deformable DETR 輸出的 logits 是 sigmoid（不是 softmax），
    每個類別獨立預測，所以取 sigmoid 後的最大值。
    """
    prob = logits.sigmoid()  # (Q, num_classes+1)
    scores, labels = prob.max(-1)
    labels = labels + 1  # 還原 1-indexed

    keep = scores > score_thresh
    scores = scores[keep]
    labels = labels[keep]
    boxes = boxes[keep]

    if len(scores) == 0:
        return scores, labels, boxes

    # per-class NMS
    xyxy = box_cxcywh_to_xyxy(boxes)
    keep_idx = nms(xyxy, scores, iou_threshold=nms_iou_thresh)
    scores = scores[keep_idx]
    labels = labels[keep_idx]
    boxes = boxes[keep_idx]

    # top-K
    if len(scores) > top_k:
        topk_idx = scores.topk(top_k).indices
        scores = scores[topk_idx]
        labels = labels[topk_idx]
        boxes = boxes[topk_idx]

    return scores, labels, boxes


# ─────────────────────────────────────────────
#  model_forward wrapper（統一介面給 eval_utils）
# ─────────────────────────────────────────────
def model_forward(model, imgs):
    """把 HuggingFace 輸出包成 dict，讓 eval_utils 不用改。"""
    out = model(pixel_values=imgs)
    return {
        "pred_logits": out.logits,  # (B, Q, num_classes+1)
        "pred_boxes": out.pred_boxes,  # (B, Q, 4) normalised cxcywh
    }


# ─────────────────────────────────────────────
#  Albumentations
# ─────────────────────────────────────────────
def get_albu_transforms(train=True):
    if train:
        return A.Compose(
            [
                A.Resize(512, 512),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.15,
                    rotate_limit=0.05,  # 數字不要旋轉，6/9 會混淆
                    p=0.5,
                    border_mode=0,
                    value=0,
                ),
                # A.RandomResizedCrop(
                #     size=(512, 512),
                #     scale=(0.7, 1.0),
                #     ratio=(0.8, 1.2),
                #     p=0.5,
                # ),
                # A.OneOf([
                #     A.GaussNoise(var_limit=(10.0, 40.0)),
                #     A.GaussianBlur(blur_limit=(3, 5)),
                # ], p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["class_labels"],
                min_visibility=0.3,
                clip=True,
            ),
        )
    else:
        return A.Compose(
            [
                A.Resize(512, 512),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["class_labels"],
            ),
        )


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────
class CocoDigitDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        orig_w, orig_h = img.size

        boxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            x_min = max(0.0, x)
            y_min = max(0.0, y)
            x_max = min(float(orig_w), x + bw)
            y_max = min(float(orig_h), y + bh)
            bw_c = x_max - x_min
            bh_c = y_max - y_min
            if bw_c > 1 and bh_c > 1:
                boxes.append([x_min, y_min, bw_c, bh_c])
                labels.append(ann["category_id"])

        img_np = np.array(img)

        if self._transforms is not None:
            t = self._transforms(image=img_np, bboxes=boxes, class_labels=labels)
            img = t["image"]  # Tensor (C, H, W)
            boxes = list(t["bboxes"])
            labels = list(t["class_labels"])
        else:
            img = T.ToTensor()(img)

        if len(boxes) > 0:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([self.ids[idx]]),
            "orig_size": torch.tensor([orig_h, orig_w]),
        }
        return img, target


class TestImageDataset(torch.utils.data.Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, img_folder, transforms=None):
        self.img_folder = img_folder
        self.transforms = transforms
        self.paths = sorted(
            p for p in Path(img_folder).iterdir() if p.suffix.lower() in self.EXTS
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        try:
            image_id = int(path.stem)
        except ValueError:
            image_id = idx

        img_np = np.array(img)
        if self.transforms:
            t = self.transforms(image=img_np, bboxes=[], class_labels=[])
            img = t["image"]
        else:
            img = T.ToTensor()(img)

        _, th, tw = img.shape
        meta = {
            "image_id": image_id,
            "orig_size": (h, w),
            "trans_size": (th, tw),
        }
        return img, meta


# ─────────────────────────────────────────────
#  collate_fn
# ─────────────────────────────────────────────
def collate_fn(batch):
    imgs, targets = zip(*batch)
    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)

    padded = []
    new_targets = []

    for img, tgt in zip(imgs, targets):
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        padded.append(F.pad(img, (0, pad_w, 0, pad_h), value=0))

        if "boxes" in tgt and len(tgt["boxes"]) > 0:
            tgt["boxes"] = box_coco_to_detr(tgt["boxes"].clone(), max_w, max_h)

        tgt = {**tgt, "padded_size": (max_h, max_w)}
        new_targets.append(tgt)

    return torch.stack(padded), new_targets


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, device, epoch, ema=None, scaler=None):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)

        # ── 轉成 HuggingFace 格式 ──────────────────
        # key: "class_labels"（不是 "labels"）, "boxes" 是歸一化 cxcywh
        hf_targets = [
            {
                "class_labels": t["labels"].to(device) - 1,
                "boxes": t["boxes"].to(device),
            }
            for t in targets
        ]

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(pixel_values=imgs, labels=hf_targets)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values=imgs, labels=hf_targets)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()

        if step % 50 == 0:
            elapsed = time.time() - t0
            # HuggingFace 把細項 loss 存在 loss_dict
            ld = outputs.loss_dict if hasattr(outputs, "loss_dict") else {}
            ce = ld.get("loss_ce", ld.get("loss_ce_0", 0.0))
            bbox = ld.get("loss_bbox", ld.get("loss_bbox_0", 0.0))
            giou = ld.get("loss_giou", ld.get("loss_giou_0", 0.0))
            print(
                f"  Epoch {epoch} | step {step}/{len(dataloader)} "
                f"| loss={loss.item():.4f} "
                f"| ce={ce:.3f} bbox={bbox:.3f} giou={giou:.3f} "
                f"| {elapsed:.1f}s"
            )
            t0 = time.time()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        hf_targets = [
            {
                "class_labels": t["labels"].to(device) - 1,
                "boxes": t["boxes"].to(device),
            }
            for t in targets
        ]
        outputs = model(pixel_values=imgs, labels=hf_targets)
        total_loss += outputs.loss.item()
    return total_loss / len(dataloader)


# ─────────────────────────────────────────────
#  evaluate_with_metrics_v2
# ─────────────────────────────────────────────
class APMeter:
    def __init__(self, num_classes=10, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.pred_records = defaultdict(list)
        self.gt_counts = defaultdict(int)

    def add(self, pred_scores, pred_labels, pred_boxes, gt_labels, gt_boxes):
        gt_matched = torch.zeros(len(gt_labels), dtype=torch.bool)
        order = pred_scores.argsort(descending=True)
        for idx in order:
            score = pred_scores[idx].item()
            label = pred_labels[idx].item()
            box = pred_boxes[idx]
            gt_cls_mask = gt_labels == label
            gt_cls_idx = gt_cls_mask.nonzero(as_tuple=True)[0]
            is_tp = False
            if len(gt_cls_idx) > 0:
                ious = box_iou(
                    box_cxcywh_to_xyxy(box.unsqueeze(0)),
                    box_cxcywh_to_xyxy(gt_boxes[gt_cls_idx]),
                )[0]
                best_val, best_local = ious.max(0)
                best_gt = gt_cls_idx[best_local]
                if best_val >= self.iou_threshold and not gt_matched[best_gt]:
                    is_tp = True
                    gt_matched[best_gt] = True
            self.pred_records[label].append((score, is_tp))
        for lbl in gt_labels.tolist():
            self.gt_counts[lbl] += 1

    def _ap(self, records, n_gt):
        if n_gt == 0:
            return float("nan")
        records.sort(key=lambda x: -x[0])
        tp, fp = 0, 0
        precs, recs = [], []
        for _, is_tp in records:
            if is_tp:
                tp += 1
            else:
                fp += 1
            precs.append(tp / (tp + fp))
            recs.append(tp / n_gt)
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            ps = [p for p, r in zip(precs, recs) if r >= thr]
            ap += max(ps) if ps else 0.0
        return ap / 11.0

    def compute(self):
        ap_dict = {
            c: self._ap(self.pred_records.get(c, []), self.gt_counts.get(c, 0))
            for c in range(1, self.num_classes + 1)
        }
        valid = [v for v in ap_dict.values() if not np.isnan(v)]
        return ap_dict, float(np.mean(valid)) if valid else 0.0


@torch.no_grad()
def run_eval_metrics(
    model,
    dataloader,
    device,
    output_dir,
    epoch,
    score_thresh=0.15,
    nms_iou_thresh=0.5,
    top_k=20,
):
    """完整評估：AP/mAP + Confusion Matrix + Score Distribution"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    iou_thresholds = np.linspace(0.5, 0.95, 10).tolist()
    meters = {t: APMeter(NUM_CLASSES, t) for t in iou_thresholds}

    all_pred_cls, all_true_cls = [], []
    all_fg_scores, score_log = [], []

    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        out = model_forward(model, imgs)  # 使用 wrapper

        prob = out["pred_logits"].sigmoid()  # Deformable DETR 用 sigmoid

        for i, tgt in enumerate(targets):
            gt_labels = tgt["labels"].cpu()
            gt_boxes = tgt["boxes"].cpu()

            fg = prob[i].max(-1).values
            all_fg_scores.extend(fg.cpu().tolist())
            if len(score_log) < 5:
                img_id = tgt.get("image_id", i)
                if hasattr(img_id, "item"):
                    img_id = img_id.item()
                top5 = fg.topk(min(5, len(fg))).values
                score_log.append((img_id, [round(s.item(), 4) for s in top5]))

            scores, labels, pred_boxes = postprocess_single(
                out["pred_logits"][i].cpu(),
                out["pred_boxes"][i].cpu(),
                score_thresh=score_thresh,
                nms_iou_thresh=nms_iou_thresh,
                top_k=top_k,
            )

            for meter in meters.values():
                meter.add(scores, labels, pred_boxes, gt_labels, gt_boxes)

            # confusion matrix matching
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                iou_mat = box_iou(
                    box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes)
                )
                matched_p, matched_g = set(), set()
                order = iou_mat.flatten().argsort(descending=True)
                for idx in order:
                    pi = idx.item() // len(gt_boxes)
                    gi = idx.item() % len(gt_boxes)
                    if iou_mat[pi, gi] < 0.5:
                        break
                    if pi in matched_p or gi in matched_g:
                        continue
                    all_pred_cls.append(labels[pi].item())
                    all_true_cls.append(gt_labels[gi].item())
                    matched_p.add(pi)
                    matched_g.add(gi)
                for gi in range(len(gt_labels)):
                    if gi not in matched_g:
                        all_pred_cls.append(NO_OBJ_IDX)
                        all_true_cls.append(gt_labels[gi].item())
            else:
                for gl in gt_labels.tolist():
                    all_pred_cls.append(NO_OBJ_IDX)
                    all_true_cls.append(gl)

    # ── AP / mAP ──────────────────────────────────────────────
    ap50_dict, mAP50 = meters[0.5].compute()
    mAP_coco = float(np.mean([meters[t].compute()[1] for t in iou_thresholds]))

    print(f"\n  ══ AP / mAP (Epoch {epoch}) ══")
    vals = "  " + " ".join(
        f"{ap50_dict.get(i+1, float('nan')):.4f}".rjust(8) for i in range(NUM_CLASSES)
    )
    head = "  " + " ".join(f"digit-{i}".rjust(8) for i in range(NUM_CLASSES))
    print(head)
    print(vals)
    print(f"  mAP@0.5      = {mAP50:.4f}")
    print(f"  mAP@[.5:.95] = {mAP_coco:.4f}")

    with open(os.path.join(output_dir, f"ap_epoch{epoch:04d}.txt"), "w") as f:
        f.write(f"mAP@0.5={mAP50:.4f}  mAP@[.5:.95]={mAP_coco:.4f}\n")
        for c in range(1, NUM_CLASSES + 1):
            f.write(f"  digit {c-1}: {ap50_dict.get(c, float('nan')):.4f}\n")

    # ── Score Distribution ────────────────────────────────────
    print(f"\n  [Score Log] (epoch {epoch}):")
    for fname, sc in score_log:
        print(f"    img {fname}: {' '.join(f'{s:.3f}' for s in sc)}")
    if len(all_fg_scores) >= 10:
        arr = np.array(all_fg_scores)
        p90, p95 = np.percentile(arr, 90), np.percentile(arr, 95)
        print(f"  p90={p90:.3f}  p95={p95:.3f}  → 建議 threshold={p90:.3f}~{p95:.3f}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(
            all_fg_scores, bins=50, color="steelblue", edgecolor="white", lw=0.3
        )
        axes[0].axvline(p90, color="green", linestyle="--", label=f"p90={p90:.2f}")
        axes[0].set_title(f"Score Dist (Epoch {epoch})")
        axes[0].legend(fontsize=8)
        high = [s for s in all_fg_scores if s > 0.1]
        if high:
            axes[1].hist(high, bins=40, color="coral", edgecolor="white", lw=0.3)
            axes[1].axvline(p90, color="green", linestyle="--", label=f"p90={p90:.2f}")
            axes[1].set_title("Scores > 0.1")
            axes[1].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"score_dist_epoch{epoch:04d}.png"), dpi=120
        )
        plt.close()

    # ── Confusion Matrix ──────────────────────────────────────
    if all_true_cls:
        labels_list = list(range(NUM_CLASSES + 1))
        tick_names = ["miss"] + [str(i) for i in range(NUM_CLASSES)]
        cm = confusion_matrix(all_true_cls, all_pred_cls, labels=labels_list)
        row_mask = cm.sum(axis=1) > 0
        cm_show = cm[row_mask][:, row_mask]
        names_show = [tick_names[i] for i, m in enumerate(row_mask) if m]
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_show, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(names_show)))
        ax.set_xticklabels(names_show)
        ax.set_yticks(range(len(names_show)))
        ax.set_yticklabels(names_show)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix (Epoch {epoch})")
        thr = cm_show.max() / 2
        for r in range(cm_show.shape[0]):
            for c in range(cm_show.shape[1]):
                ax.text(
                    c,
                    r,
                    str(cm_show[r, c]),
                    ha="center",
                    va="center",
                    color="white" if cm_show[r, c] > thr else "black",
                    fontsize=8,
                )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"confusion_matrix_epoch{epoch:04d}.png"), dpi=120
        )
        plt.close()

        mask = [cls != NO_OBJ_IDX for cls in all_true_cls]
        t_f = [all_true_cls[j] for j in range(len(mask)) if mask[j]]
        p_f = [all_pred_cls[j] for j in range(len(mask)) if mask[j]]
        report = classification_report(
            t_f,
            p_f,
            labels=list(range(1, NUM_CLASSES + 1)),
            target_names=[str(i) for i in range(NUM_CLASSES)],
            zero_division=0,
        )
        print(f"\n  Per-class Report:\n{report}")
        with open(os.path.join(output_dir, f"report_epoch{epoch:04d}.txt"), "w") as f:
            f.write(report)

    return mAP50, mAP_coco, ap50_dict


# ─────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_dataset = CocoDigitDetection(
        img_folder=os.path.join(args.data_dir, "train"),
        ann_file=os.path.join(args.data_dir, "train.json"),
        transforms=get_albu_transforms(train=True),
    )
    val_dataset = CocoDigitDetection(
        img_folder=os.path.join(args.data_dir, "valid"),
        ann_file=os.path.join(args.data_dir, "valid.json"),
        transforms=get_albu_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── 建立模型 ──────────────────────────────────────
    model = build_model(
        num_classes=NUM_CLASSES,
        num_queries=NUM_QUERIES,
        pretrained_backbone=True,
    ).to(device)

    # ── EMA ───────────────────────────────────────────
    ema = ModelEMA(model, decay=0.9997)

    # ── Optimizer（backbone 小 LR）────────────────────
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = build_warmup_cosine_scheduler(
        optimizer, warmup_epochs=5, total_epochs=args.epochs
    )

    # ── AMP ───────────────────────────────────────────
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.output_dir, exist_ok=True)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            ema=ema,
            scaler=scaler,
        )
        # 用 EMA 模型評估
        val_loss = evaluate(ema.ema, val_loader, device)
        scheduler.step()

        if epoch % args.eval_interval == 0 or epoch == args.epochs or epoch == 5:
            mAP50, mAP_coco, _ = run_eval_metrics(
                ema.ema,
                val_loader,
                device,
                args.output_dir,
                epoch,
                score_thresh=0.15,
                nms_iou_thresh=0.5,
                top_k=20,
            )
            print(f"  mAP@0.5={mAP50:.4f}  mAP@[.5:.95]={mAP_coco:.4f}")

        print(f"  → Train loss: {train_loss:.4f}  |  Val loss: {val_loss:.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_loss,
            "num_queries": NUM_QUERIES,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "latest.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            torch.save(ckpt, os.path.join(args.output_dir, f"epoch_{epoch:04d}.pth"))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curve.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
#  Test / Inference
# ─────────────────────────────────────────────
@torch.no_grad()
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference on: {device}")

    test_dataset = TestImageDataset(
        img_folder=os.path.join(args.data_dir, "test"),
        transforms=get_albu_transforms(train=False),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # ── 載入模型 ──────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)

    # 優先用儲存的 num_queries，避免架構不符
    nq = ckpt.get("num_queries", NUM_QUERIES)
    model = build_model(
        num_classes=NUM_CLASSES, num_queries=nq, pretrained_backbone=False
    ).to(device)

    # 嘗試載入 EMA 權重（更穩定），沒有才用一般權重
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("Loaded EMA weights.")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights.")

    model.eval()
    print(f"Checkpoint: epoch {ckpt.get('epoch', '?')}, num_queries={nq}")

    predictions = []

    for imgs, targets in test_loader:
        imgs = imgs.to(device)
        out = model_forward(model, imgs)

        for i, meta in enumerate(targets):
            image_id = meta["image_id"]
            orig_h, orig_w = meta["orig_size"]
            trans_h, trans_w = meta["trans_size"]
            pad_h, pad_w = meta["padded_size"]

            kept_scores, kept_labels, kept_boxes = postprocess_single(
                out["pred_logits"][i].cpu(),
                out["pred_boxes"][i].cpu(),
                score_thresh=args.threshold,
                nms_iou_thresh=0.5,
                top_k=20,
            )

            # padded canvas → absolute coords → scale back to orig
            coco_boxes = box_detr_to_coco(kept_boxes, img_w=pad_w, img_h=pad_h)
            scale_x = orig_w / trans_w
            scale_y = orig_h / trans_h
            coco_boxes[:, 0] *= scale_x
            coco_boxes[:, 1] *= scale_y
            coco_boxes[:, 2] *= scale_x
            coco_boxes[:, 3] *= scale_y

            for score, label, bbox in zip(
                kept_scores.tolist(), kept_labels.tolist(), coco_boxes.tolist()
            ):
                predictions.append(
                    {
                        "image_id": image_id,
                        "bbox": bbox,
                        "score": score,
                        "category_id": label,
                    }
                )

    out_path = os.path.join(args.output_dir, "pred.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f)
    print(f"Saved {len(predictions)} predictions → {out_path}")


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Deformable DETR Digit Detection")
    parser.add_argument("--mode", choices=["train", "test"], required=True)

    parser.add_argument("--data_dir", type=str, default="../nycu-hw2-data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--checkpoint", type=str, default="./outputs/best.pth")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=3)
    parser.add_argument("--save_interval", type=int, default=0)

    parser.add_argument("--threshold", type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        test(args)
