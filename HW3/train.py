"""
Cell Instance Segmentation — Mask R-CNN Training (with ablation hooks)

Used directly:
    python train.py --gpu 1                        # baseline run
    python train.py --gpu 1 --tag baseline

Used by run_ablation.py via train_one_run(...).
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from shared import cfg, setup_device, set_seed, load_image_tif, build_model

matplotlib.use("Agg")

# ─────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data_root", type=str, default="../data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Try 4 first; A6000 can handle 8 if VRAM allows.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Linear scaling rule: 0.005 for batch=2 -> 0.01 for batch=4.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader workers; bump to 12 if CPU is idle.",
    )
    p.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable mixed precision (default: AMP on).",
    )
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--tag", type=str, default="baseline")
    return p.parse_args()


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class CellCocoDataset(Dataset):
    def __init__(self, json_path, img_dir, transforms=None):
        self.coco = COCO(json_path)
        self.img_dir = img_dir
        self.transforms = transforms
        self.img_ids = sorted(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.imgs[img_id]
        fname = info["file_name"]
        folder = os.path.splitext(fname)[0]
        img_path = os.path.join(self.img_dir, folder, "image.tif")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, fname)

        image = load_image_tif(img_path)
        h, w = info["height"], info["width"]

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        all_masks, all_labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                mask = maskUtils.decode(seg).astype(np.uint8)
            else:
                rle = maskUtils.frPyObjects(seg, h, w)
                mask = maskUtils.decode(maskUtils.merge(rle)).astype(np.uint8)
            if mask.sum() == 0:
                continue
            all_masks.append(mask)
            all_labels.append(int(ann["category_id"]))

        if self.transforms:
            out = self.transforms(image=image, masks=all_masks if all_masks else [])
            image = out["image"]
            aug_masks = out["masks"]
            sm, sl = [], []
            for m, lbl in zip(aug_masks, all_labels):
                m_arr = np.asarray(m, dtype=np.uint8)
                if m_arr.sum() > 0:
                    sm.append(m_arr)
                    sl.append(lbl)
            all_masks, all_labels = sm, sl
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        ih, iw = image.shape[1], image.shape[2]
        if not all_masks:
            return image, _empty_target(img_id, ih, iw)

        vm, vb, vl = [], [], []
        for m, lbl in zip(all_masks, all_labels):
            ys, xs = np.where(m > 0)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            if x2 <= x1 or y2 <= y1:
                continue
            vb.append([x1, y1, x2, y2])
            vm.append(m)
            vl.append(lbl)

        if not vb:
            return image, _empty_target(img_id, ih, iw)

        boxes = torch.tensor(vb, dtype=torch.float32)
        labels = torch.tensor(vl, dtype=torch.int64)
        masks = torch.tensor(np.stack(vm), dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return image, {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": area,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }


def _empty_target(img_id, h, w):
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros(0, dtype=torch.int64),
        "masks": torch.zeros((0, h, w), dtype=torch.uint8),
        "image_id": torch.tensor([img_id]),
        "area": torch.zeros(0, dtype=torch.float32),
        "iscrowd": torch.zeros(0, dtype=torch.int64),
    }


# ─────────────────────────────────────────────
# AUG
# ─────────────────────────────────────────────
def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.3),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(5, 30), p=0.2),
            A.ElasticTransform(alpha=50, sigma=5, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transforms():
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def collate_fn(batch):
    return tuple(zip(*batch))


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def plot_curves(history, save_path):
    """
    Plot training loss and validation AP50 from history list.
    history: list of {"epoch": int, "loss": float, "ap50": float}
    """
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    ap50s = [h["ap50"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax = axes[0]
    ax.plot(epochs, losses, marker="o", markersize=4, label="train_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # AP50 curve
    ax = axes[1]
    ax.plot(
        epochs, ap50s, marker="s", markersize=4, color="tab:green", label="val_AP50"
    )
    if ap50s:
        best = max(ap50s)
        best_ep = epochs[ap50s.index(best)]
        ax.axhline(
            best,
            ls="--",
            color="tab:red",
            alpha=0.6,
            label=f"best={best:.4f} @ ep{best_ep}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP50 (segm)")
    ax.set_title("Validation AP50")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, save_path, title="Confusion Matrix"):
    """
    Plot a 5x5 confusion matrix.
    cm: np.ndarray shape (5, 5), row=GT, col=Pred, index 0=background (FP/FN)
    """
    class_names = ["background", "class1", "class2", "class3", "class4"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    vmax = max(cm.max(), 1)
    for i in range(5):
        for j in range(5):
            v = int(cm[i, j])
            color = "white" if v > 0.5 * vmax else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────
# TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────
def train_one_epoch(model, optimizer, loader, device, epoch, scaler=None):
    """
    Mixed-precision training when scaler is provided.
    On A6000 / 30/40-series GPUs, AMP gives ~1.5-2x speedup
    by using Tensor Cores (BF16 matmul).
    """
    model.train()
    total = 0.0
    use_amp = scaler is not None and device.type == "cuda"
    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}")

    for images, targets in pbar:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [
            {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
        ]

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total / len(loader)


def _mask_iou(a, b):
    """Compute IoU between two binary masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device, coco_gt_api, val_ds, score_thresh=0.05):
    """
    Run validation, compute AP50 and confusion matrix.

    Returns:
        ap50 (float)
        cm   (np.ndarray, shape 5x5)
            row = GT class (0=background/FN), col = Pred class (0=background/FP)
    """
    model.eval()
    coco_dt = []
    # cm[gt_label][pred_label] — index 0 is background (FP / FN slot)
    cm = np.zeros((5, 5), dtype=np.int64)

    # Build image_id → dataset index map for GT mask lookup
    id_to_idx = {int(info["id"]): i for i, info in enumerate(val_ds.coco.imgs.values())}

    for images, targets in tqdm(loader, desc="  Val", leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for pred, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())

            # ── Collect predictions for AP50 ──────────────
            pred_entries = []  # each: {"mask", "label", "score"}
            for mask, label, score, box in zip(
                pred["masks"].cpu().numpy(),
                pred["labels"].cpu().numpy(),
                pred["scores"].cpu().numpy(),
                pred["boxes"].cpu().numpy(),
            ):
                if score < score_thresh:
                    continue
                binary = (mask[0] > 0.5).astype(np.uint8)
                if binary.sum() == 0:
                    continue

                rle = maskUtils.encode(np.asfortranarray(binary))
                rle["counts"] = rle["counts"].decode("utf-8")
                x1, y1, x2, y2 = box
                coco_dt.append(
                    {
                        "image_id": img_id,
                        "category_id": int(label),
                        "segmentation": rle,
                        "score": float(score),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    }
                )
                pred_entries.append(
                    {
                        "mask": binary,
                        "label": int(label),
                        "score": float(score),
                    }
                )

            # ── Update confusion matrix ────────────────────
            if img_id in id_to_idx:
                _, gt_masks, gt_labels, _ = val_ds.load_raw_by_id(img_id)
                gts = [
                    {"mask": m, "label": int(l)} for m, l in zip(gt_masks, gt_labels)
                ]
                _update_cm(cm, pred_entries, gts, iou_thresh=0.5)

    # ── AP50 via pycocotools ───────────────────────────────
    if not coco_dt:
        return 0.0, cm

    dt = coco_gt_api.loadRes(coco_dt)
    ev = COCOeval(coco_gt_api, dt, "segm")
    ev.params.iouThrs = np.array([0.5])
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0]), cm


def _update_cm(cm, preds, gts, iou_thresh=0.5):
    """
    Greedy match predictions to GT (by score descending).
    Matched pair → cm[gt_label][pred_label] += 1
    Unmatched pred → cm[0][pred_label] += 1   (FP)
    Unmatched GT   → cm[gt_label][0]   += 1   (FN)
    """
    preds_sorted = sorted(preds, key=lambda x: -x["score"])
    gt_used = [False] * len(gts)

    for p in preds_sorted:
        best_iou, best_idx = 0.0, -1
        for i, g in enumerate(gts):
            if gt_used[i]:
                continue
            iou = _mask_iou(p["mask"], g["mask"])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx >= 0 and best_iou >= iou_thresh:
            gt_used[best_idx] = True
            cm[gts[best_idx]["label"], p["label"]] += 1
        else:
            cm[0, p["label"]] += 1  # FP

    for i, used in enumerate(gt_used):
        if not used:
            cm[gts[i]["label"], 0] += 1  # FN


# ─────────────────────────────────────────────
# CORE TRAIN FUNCTION  (callable from run_ablation.py)
# ─────────────────────────────────────────────
def train_one_run(
    *,
    tag: str,
    device,
    data_root: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    model_kwargs: dict,
    num_workers: int = 8,
    use_amp: bool = True,
):
    set_seed(cfg.SEED)

    train_json = os.path.join(data_root, "annotations", "train.json")
    val_json = os.path.join(data_root, "annotations", "val.json")
    img_dir = os.path.join(data_root, "train")

    run_dir = os.path.join(output_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train_log.txt")
    ckpt_path = os.path.join(run_dir, "best_model.pth")
    log_f = open(log_path, "w")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*70}\n=== RUN: {tag} ===\n{'='*70}")
    log(f"model_kwargs: {model_kwargs}")
    log(f"epochs={epochs}, batch_size={batch_size}, lr={lr}")

    train_ds = CellCocoDataset(train_json, img_dir, get_train_transforms())
    val_ds = CellCocoDataset(val_json, img_dir, get_val_transforms())
    log(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    coco_gt = COCO(val_json)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(2, num_workers // 2),
        collate_fn=collate_fn,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    model = build_model(cfg.NUM_CLASSES, **model_kwargs)
    model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Params — Total: {n_total/1e6:.1f}M | Trainable: {n_trainable/1e6:.1f}M")
    if n_trainable / 1e6 >= 200:
        log("  [WARN] Exceeds 200M parameter limit!")

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=cfg.MOMENTUM,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA
    )

    scaler = "bf16" if (use_amp and device.type == "cuda") else None
    if scaler:
        log("AMP enabled (bfloat16 autocast)")

    best_ap50 = 0.0
    history = []  # records every epoch (loss) + eval epochs (ap50)
    best_cm = None

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, scaler=scaler
        )
        scheduler.step()

        if epoch == epochs:
            ap50, cm = evaluate(model, val_loader, device, coco_gt, val_ds)
            log(f"[{tag}] Epoch {epoch:03d}  Loss={loss:.4f}  AP50={ap50:.4f}")
            history.append({"epoch": epoch, "loss": loss, "ap50": ap50})

            # ── Save training curve (updated every eval) ──
            plot_curves(history, os.path.join(run_dir, "training_curves.png"))

            # ── Save latest confusion matrix ──────────────
            plot_confusion(
                cm,
                os.path.join(run_dir, "confusion_matrix_latest.png"),
                title=f"Confusion Matrix ({tag}, ep{epoch}, AP50={ap50:.4f})",
            )

            if ap50 > best_ap50:
                best_ap50 = ap50
                best_cm = cm.copy()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "ap50": ap50,
                        "model_kwargs": model_kwargs,
                    },
                    ckpt_path,
                )
                log(f"  ✓ Saved best (AP50={ap50:.4f})")

                # ── Save best confusion matrix ─────────────
                plot_confusion(
                    best_cm,
                    os.path.join(run_dir, "confusion_matrix_best.png"),
                    title=f"Confusion Matrix ({tag}, best ep{epoch}, AP50={ap50:.4f})",
                )
        else:
            history.append({"epoch": epoch, "loss": loss, "ap50": None})
            log(f"[{tag}] Epoch {epoch:03d}  Loss={loss:.4f}")

        # ── Save loss-only curve every epoch ──────────────
        plot_curves(
            [h for h in history if h["ap50"] is not None],
            os.path.join(run_dir, "training_curves.png"),
        )

    summary = {
        "tag": tag,
        "best_ap50": best_ap50,
        "n_trainable": n_trainable,
        "model_kwargs": model_kwargs,
        "history": history,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"\n=== {tag} done. Best AP50: {best_ap50:.4f} ===\n")
    log(f"Curves : {run_dir}/training_curves.png")
    log(f"CM best: {run_dir}/confusion_matrix_best.png")
    log_f.close()

    del model, optimizer, scheduler, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_ap50, summary


# ─────────────────────────────────────────────
# STANDALONE ENTRY POINT
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    device = setup_device(args.gpu)
    cfg.DEVICE = device

    train_one_run(
        tag=args.tag,
        device=device,
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_kwargs=dict(),
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
