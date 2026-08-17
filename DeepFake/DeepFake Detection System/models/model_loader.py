
from __future__ import annotations
import logging, os, sys
from typing import Tuple
import torch, torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
sys.path.insert(0, "/home/claude/DeepSentinel")
from config import FACE_CROP_SIZE, IMAGE_MEAN, IMAGE_STD, DROPOUT_PROB, PRIMARY_WEIGHTS

logger = logging.getLogger(__name__)


class SentinelNet(nn.Module):
    """EfficientNet-B0 backbone + custom 2-layer GELU classification head."""

    def __init__(self, dropout=DROPOUT_PROB, pretrained=True):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tv_models.efficientnet_b0(weights=weights)
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 256),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, 1),
        )
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        logger.info("SentinelNet built (EfficientNet-B0).")

    def forward(self, x):
        return self.head(self.pool(self.feature_extractor(x)).flatten(1))

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self(x))

    def freeze_backbone(self):
        for p in self.feature_extractor.parameters(): p.requires_grad = False
        logger.info("Backbone frozen.")

    def unfreeze_top_blocks(self, num_blocks=3):
        for p in self.feature_extractor.parameters(): p.requires_grad = False
        total = len(self.feature_extractor)
        for block in list(self.feature_extractor.children())[max(0, total-num_blocks):]:
            for p in block.parameters(): p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("Top %d blocks unfrozen. Trainable params: %s", num_blocks, f"{trainable:,}")


def persist_checkpoint(model, path=PRIMARY_WEIGHTS, extras=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model_state": model.state_dict()}
    if extras: payload.update(extras)
    torch.save(payload, path)
    logger.info("Saved checkpoint -> %s", path)


def load_from_checkpoint(path=PRIMARY_WEIGHTS, device="cpu", strict=True):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run training first.")
    payload = torch.load(path, map_location=device)
    model   = SentinelNet(pretrained=False)
    model.load_state_dict(payload.pop("model_state"), strict=strict)
    model.to(device).eval()
    logger.info("Loaded checkpoint from %s", path)
    return model, payload


def build_inference_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((FACE_CROP_SIZE, FACE_CROP_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def build_training_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((FACE_CROP_SIZE + 16, FACE_CROP_SIZE + 16)),
        T.RandomCrop(FACE_CROP_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])
