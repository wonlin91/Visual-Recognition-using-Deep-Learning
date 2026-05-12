"""
Generate COCO-format submission JSON.
Reads model_kwargs from the checkpoint so any ablation model loads correctly.

Usage:
    python inference.py --gpu 1 --model outputs/baseline/best_model.pth
    python inference.py --gpu 1 --model outputs/fpn_dcn/best_model.pth
"""

import os
import json
import glob
import argparse
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as maskUtils
from tqdm import tqdm

from shared import cfg, setup_device, load_image_tif, build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data_root", type=str, default="../data")
    p.add_argument("--model", type=str, default="./outputs/baseline/best_model.pth")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Default: same dir as model / submission.json",
    )
    p.add_argument("--score_thresh", type=float, default=0.3)
    return p.parse_args()


def get_test_transforms():
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


@torch.no_grad()
def generate_submission(
    model_path, test_dir, test_json_path, output_path, device, score_thresh=0.3
):

    ckpt = torch.load(model_path, map_location=device)
    model_kwargs = ckpt.get("model_kwargs", {})
    print(f"Building model with kwargs: {model_kwargs}")

    model = build_model(cfg.NUM_CLASSES, **model_kwargs)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    model.roi_heads.score_thresh = score_thresh

    print(f"Loaded epoch={ckpt.get('epoch', '?')}  AP50={ckpt.get('ap50', '?')}")

    with open(test_json_path, "r") as f:
        test_id_map = json.load(f)
    fname_to_id = {entry["file_name"]: entry["id"] for entry in test_id_map}

    transform = get_test_transforms()
    results = []
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.tif")))
    print(f"Inference on {len(test_files)} images (score_thresh={score_thresh})\n")

    for img_path in tqdm(test_files):
        fname = os.path.basename(img_path)
        image_id = fname_to_id.get(fname)
        if image_id is None:
            print(f"  [Skip] {fname} not in JSON")
            continue

        image = load_image_tif(img_path)
        out = transform(image=image, masks=[])
        img_t = out["image"].unsqueeze(0).to(device)
        pred = model(img_t)[0]

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
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "segmentation": rle,
                    "score": float(score),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                }
            )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"\nSaved → {output_path}  ({len(results)} predictions)")


def main():
    args = parse_args()
    device = setup_device(args.gpu)
    cfg.DEVICE = device

    if args.output is None:
        run_dir = os.path.dirname(args.model)
        args.output = os.path.join(run_dir, "submission.json")

    generate_submission(
        model_path=args.model,
        test_dir=os.path.join(args.data_root, "test"),
        test_json_path=os.path.join(args.data_root, "test_image_name_to_ids.json"),
        output_path=args.output,
        device=device,
        score_thresh=args.score_thresh,
    )


if __name__ == "__main__":
    main()
