"""
shared.py — config, device setup, image loader, parameterized model builder.
Imported by train.py, inference.py, and run_ablation.py.
"""

import random
import numpy as np
import tifffile as tiff

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import project_masks_on_boxes


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
class Config:
    NUM_CLASSES = 5  # background + class1~4

    SEED = 42
    NUM_EPOCHS = 50
    BATCH_SIZE = 2
    LR = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    LR_STEP_SIZE = 15
    LR_GAMMA = 0.1

    # Default anchors (small for cells)
    ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,))
    ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * 5

    # RPN / Detection
    RPN_PRE_NMS_TOP_N_TRAIN = 2000
    RPN_POST_NMS_TOP_N_TRAIN = 1000
    RPN_PRE_NMS_TOP_N_TEST = 1000
    RPN_POST_NMS_TOP_N_TEST = 500
    BOX_DETECTIONS_PER_IMG = 300
    RPN_NMS_THRESH = 0.7
    BOX_NMS_THRESH = 0.5
    BOX_SCORE_THRESH = 0.05

    SUBMISSION_SCORE_THRESH = 0.3

    DATA_ROOT = "../data"
    TRAIN_JSON = "../data/annotations/train.json"
    VAL_JSON = "../data/annotations/val.json"
    TRAIN_DIR = "../data/train"
    TEST_DIR = "../data/test"
    TEST_JSON = "../data/test_image_name_to_ids.json"
    OUTPUT_DIR = "./outputs"
    DEVICE = None


cfg = Config()


# ─────────────────────────────────────────────
# DEVICE & SEED
# ─────────────────────────────────────────────
def setup_device(gpu_index: int) -> torch.device:
    if gpu_index < 0 or not torch.cuda.is_available():
        print("[Device] CPU")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if gpu_index >= n:
        print(
            f"[Device] GPU {gpu_index} unavailable ({n} found). Using cuda:0.")
        gpu_index = 0
    device = torch.device(f"cuda:{gpu_index}")
    p = torch.cuda.get_device_properties(device)
    print(
        f"[Device] cuda:{gpu_index} — {p.name}({p.total_memory // 1024**2} MB)")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return device


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# IMAGE LOADER
# ─────────────────────────────────────────────
def load_image_tif(img_path: str) -> np.ndarray:
    img = tiff.imread(img_path)
    if (
        img.ndim == 3
        and img.shape[0] in (1, 3, 4)
        and img.shape[0] < img.shape[1]
    ):
        img = img.transpose(1, 2, 0)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        lo, hi = img.min(), img.max()
        img = ((img.astype(np.float32) - lo) /
               (hi - lo + 1e-8) * 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────
# CUSTOM MASK HEAD (for ablation: more conv layers)
# ─────────────────────────────────────────────
class DeepMaskHead(nn.Module):
    """
    Mask head with configurable number of conv layers.
    Default Mask R-CNN uses 4 conv layers; we make this tunable.
    Replaces both conv5_mask + mask_predictor logits in one module.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_convs: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(num_convs):
            layers.append(nn.Conv2d(ch, hidden_dim, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch = hidden_dim
        self.conv_block = nn.Sequential(*layers)
        # Upsample 2x then 1x1 to logits
        self.deconv = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, 2, 0)
        self.relu = nn.ReLU(inplace=True)
        self.mask_logits = nn.Conv2d(hidden_dim, num_classes, 1)

        # Init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.relu(self.deconv(x))
        return self.mask_logits(x)


# ─────────────────────────────────────────────
# DICE + FOCAL LOSS  (for ablation)
# ─────────────────────────────────────────────
def dice_focal_mask_loss(
    mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs
):
    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs]
              for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, idxs, discretization_size)
        for m, p, idxs in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    # Pick the logit channel for each instance's GT class
    idx_range = torch.arange(labels.shape[0], device=labels.device)
    pred = mask_logits[idx_range, labels]  # N,H,W
    gt = mask_targets.float()  # N,H,W

    # Focal loss component (alpha=0.25, gamma=2.0)
    bce = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
    p_t = torch.exp(-bce)
    alpha = 0.25
    gamma = 2.0
    alpha_t = alpha * gt + (1 - alpha) * (1 - gt)
    focal = alpha_t * (1 - p_t) ** gamma * bce
    focal = focal.mean()

    # Dice loss component
    pred_sig = torch.sigmoid(pred)
    smooth = 1.0
    inter = (pred_sig * gt).sum(dim=(1, 2))
    union = pred_sig.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
    dice = 1 - (2 * inter + smooth) / (union + smooth)
    dice = dice.mean()

    return focal + dice


def patch_maskrcnn_loss(model):
    import torchvision.models.detection.roi_heads as roi_heads_module

    roi_heads_module.maskrcnn_loss = dice_focal_mask_loss


# ─────────────────────────────────────────────
# DEFORMABLE CONV WRAPPER (for ablation)
# ─────────────────────────────────────────────
class DeformConv2dBlock(nn.Module):
    """
    DCN v1 block: standard offset prediction + deformable conv + ReLU.
    Wraps torchvision.ops.DeformConv2d.
    """

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3):
        super().__init__()
        from torchvision.ops import DeformConv2d

        padding = kernel_size // 2
        # 2 offsets per kernel position (dx, dy)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.deform = DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.relu = nn.ReLU(inplace=True)
        # init offsets to zero so DCN starts ≈ regular conv
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.relu(self.deform(x, offset))


def add_dcn_to_fpn(model):
    """
    Append a DCN block after each FPN level output.
    The FPN module (model.backbone.fpn) returns an OrderedDict of P2..P5+pool.
    We wrap its forward to apply DCN after the standard FPN output.
    """
    fpn = model.backbone.fpn
    # 256 is the standard FPN out channels in torchvision
    dcn_blocks = nn.ModuleDict(
        {name: DeformConv2dBlock(256, 256, 3)
         for name in ["0", "1", "2", "3", "pool"]}
    )
    fpn.dcn_blocks = dcn_blocks

    orig_forward = fpn.forward

    def new_forward(x):
        out = orig_forward(x)  # OrderedDict
        for k in list(out.keys()):
            if k in fpn.dcn_blocks:
                out[k] = fpn.dcn_blocks[k](out[k])
        return out

    fpn.forward = new_forward


# ─────────────────────────────────────────────
# MAIN MODEL BUILDER  (parameterized for ablation)
# ─────────────────────────────────────────────
def build_model(
    num_classes: int,
    *,
    anchor_sizes=None,
    aspect_ratios=None,
    mask_num_convs: int = 4,  # default Mask R-CNN: 4
    use_dcn_in_fpn: bool = False,
    use_dice_focal: bool = False,
    roi_sampling_ratio: int = 2,  # default torchvision: 2
) -> nn.Module:
    """
    Parameterized Mask R-CNN builder for ablation studies.

    Args:
        anchor_sizes      : tuple of anchor size tuples per FPN level
        aspect_ratios     : tuple of aspect ratio tuples per FPN level
        mask_num_convs    : depth of mask head (default 4)
        use_dcn_in_fpn    : insert DCN after each FPN output level
        use_dice_focal    : replace BCE mask loss with Dice + Focal
        roi_sampling_ratio: sampling ratio for both box and mask RoIAlign
    """
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.rpn import RPNHead

    if anchor_sizes is None:
        anchor_sizes = cfg.ANCHOR_SIZES
    if aspect_ratios is None:
        aspect_ratios = cfg.ASPECT_RATIOS

    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # ── Box predictor ──
    in_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_box, num_classes)

    # ── Mask predictor ──
    if mask_num_convs == 4:
        in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256,
                                                           num_classes)
    else:
        in_mask = 256
        model.roi_heads.mask_head = nn.Identity()
        model.roi_heads.mask_predictor = DeepMaskHead(
            in_channels=in_mask, num_classes=num_classes,
            num_convs=mask_num_convs
        )

    # ── Anchors ──
    anchor_gen = AnchorGenerator(sizes=anchor_sizes,
                                 aspect_ratios=aspect_ratios)
    model.rpn.anchor_generator = anchor_gen
    model.rpn.head = RPNHead(256, anchor_gen.num_anchors_per_location()[0])

    # ── RoIAlign sampling ratio ──
    if roi_sampling_ratio != 2:
        model.roi_heads.box_roi_pool.sampling_ratio = roi_sampling_ratio
        model.roi_heads.mask_roi_pool.sampling_ratio = roi_sampling_ratio

    # ── RPN settings ──
    model.rpn.pre_nms_top_n_train = cfg.RPN_PRE_NMS_TOP_N_TRAIN
    model.rpn.post_nms_top_n_train = cfg.RPN_POST_NMS_TOP_N_TRAIN
    model.rpn.pre_nms_top_n_test = cfg.RPN_PRE_NMS_TOP_N_TEST
    model.rpn.post_nms_top_n_test = cfg.RPN_POST_NMS_TOP_N_TEST
    model.rpn.nms_thresh = cfg.RPN_NMS_THRESH

    # ── Detection head ──
    model.roi_heads.detections_per_img = cfg.BOX_DETECTIONS_PER_IMG
    model.roi_heads.nms_thresh = cfg.BOX_NMS_THRESH
    model.roi_heads.score_thresh = cfg.BOX_SCORE_THRESH

    # ── Optional: DCN in FPN ──
    if use_dcn_in_fpn:
        add_dcn_to_fpn(model)

    # ── Optional: Dice + Focal loss for mask head ──
    if use_dice_focal:
        patch_maskrcnn_loss(model)

    return model
