"""
Chest X-Ray Model Service — DenseNet-121 & ResNet-50 Model Management.

Loads pretrained models from torchvision with modified classifier heads
for 14-class multi-label chest X-ray classification (ChestX-ray14).
Models are cached in memory for fast repeated inference.
"""

import os
import logging
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights, ResNet50_Weights

from ..core.config import get_settings

logger = logging.getLogger(__name__)

WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "weights",
)


class ChestXRayDenseNet(nn.Module):
    """DenseNet-121 adapted for 14-class multi-label chest X-ray classification."""

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.densenet = models.densenet121(weights=weights)

        # Replace classifier head for multi-label output
        num_ftrs = self.densenet.classifier.in_features  # 1024
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x)


class ChestXRayResNet(nn.Module):
    """ResNet-50 baseline adapted for 14-class multi-label chest X-ray classification."""

    def __init__(self, num_classes: int = 14, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)

        # Replace FC head for multi-label output
        num_ftrs = self.resnet.fc.in_features  # 2048
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class ChestModelService:
    """Service for loading and managing chest X-ray classification models."""

    def __init__(self):
        self.settings = get_settings()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._densenet: Optional[ChestXRayDenseNet] = None
        self._resnet: Optional[ChestXRayResNet] = None
        logger.info(f"ChestModelService initialized. Device: {self.device}")

    def get_densenet(self):
        """Load and return cached DenseNet-121 model trained on 6 chest X-ray datasets."""
        if self._densenet is None:
            logger.info("Loading DenseNet-121 model...")
            try:
                import torchxrayvision as xrv
                logger.info("Loading multi-dataset DenseNet-121 (densenet121-res224-all) from torchxrayvision...")
                # Use the 'all' weights trained on NIH+CheXpert+PadChest+RSNA+MIMIC-CXR+OpenI
                # These weights produce logits (NOT probabilities) regardless of apply_sigmoid
                model = xrv.models.DenseNet(weights="densenet121-res224-all", apply_sigmoid=False)
                # Disable piecewise operating threshold warping — we apply sigmoid ourselves
                model.op_threshs = None
                model.to(self.device)
                model.eval()
                self._densenet = model
                logger.info(
                    f"Multi-dataset DenseNet-121 loaded. "
                    f"Pathologies: {list(model.pathologies)}"
                )
                return self._densenet
            except Exception as e:
                logger.warning(f"Failed to load torchxrayvision DenseNet: {e}, falling back to custom weights.")

            self._densenet = ChestXRayDenseNet(
                num_classes=self.settings.num_classes, pretrained=True
            )

            # Load fine-tuned weights if available
            weights_path = self.settings.densenet_weights_path
            if weights_path and os.path.exists(weights_path):
                logger.info(f"Loading fine-tuned DenseNet weights from {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
                if isinstance(state_dict, nn.Module):
                    self._densenet = state_dict
                else:
                    self._densenet.load_state_dict(state_dict)
            else:
                default_path = os.path.join(WEIGHTS_DIR, "densenet121_chestxray.pth")
                if os.path.exists(default_path):
                    logger.info(f"Loading fine-tuned DenseNet weights from {default_path}")
                    loaded = torch.load(default_path, map_location=self.device, weights_only=False)
                    if isinstance(loaded, nn.Module):
                        self._densenet = loaded
                    else:
                        self._densenet.load_state_dict(loaded)

            self._densenet.to(self.device)
            self._densenet.eval()
            logger.info("DenseNet-121 loaded and ready.")

        return self._densenet

    def get_resnet(self):
        """Load and return cached ResNet-50 baseline model."""
        if self._resnet is None:
            logger.info("Loading ResNet-50 model...")
            default_path = os.path.join(WEIGHTS_DIR, "resnet50_chestxray.pth")
            weights_path = self.settings.resnet_weights_path or default_path

            # 1. Load local fine-tuned weights if available (instant, no network download)
            if os.path.exists(weights_path):
                try:
                    logger.info(f"Loading fine-tuned ResNet weights from {weights_path}")
                    loaded = torch.load(weights_path, map_location=self.device, weights_only=False)
                    if isinstance(loaded, nn.Module):
                        self._resnet = loaded
                    else:
                        model = ChestXRayResNet(num_classes=self.settings.num_classes, pretrained=False)
                        model.load_state_dict(loaded)
                        self._resnet = model
                    self._resnet.to(self.device)
                    self._resnet.eval()
                    logger.info("Fine-tuned ResNet-50 loaded successfully from local weights.")
                    return self._resnet
                except Exception as e:
                    logger.warning(f"Failed to load local ResNet weights: {e}")

            # 2. Try torchxrayvision pretrained ResNet-50
            try:
                import torchxrayvision as xrv
                logger.info("Loading pretrained chest ResNet-50 from torchxrayvision...")
                model = xrv.models.ResNet(weights="resnet50-res512-all", apply_sigmoid=False)
                model.op_threshs = None
                model.to(self.device)
                model.eval()
                self._resnet = model
                logger.info("Pretrained chest ResNet-50 loaded successfully.")
                return self._resnet
            except Exception as e:
                logger.warning(f"Failed to load torchxrayvision ResNet: {e}")

            # 3. Fallback to ImageNet pretrained architecture
            self._resnet = ChestXRayResNet(
                num_classes=self.settings.num_classes, pretrained=True
            )
            self._resnet.to(self.device)
            self._resnet.eval()
            logger.info("ResNet-50 loaded with default weights.")

        return self._resnet

    def get_device(self) -> torch.device:
        return self.device

    def get_model_info(self) -> list:
        """Return architecture metadata for both models."""
        densenet = self.get_densenet()
        resnet = self.get_resnet()

        densenet_params = sum(p.numel() for p in densenet.parameters())
        resnet_params = sum(p.numel() for p in resnet.parameters())

        return [
            {
                "name": "DenseNet-121",
                "architecture": "Dense Convolutional Network (121 layers)",
                "parameters": densenet_params,
                "parameters_human": f"{densenet_params / 1e6:.2f}M",
                "pretrained_on": "ImageNet → Fine-tuned for ChestX-ray14",
                "input_size": "224 × 224 × 3",
                "classifier_head": "Linear(1024 → 512) → ReLU → Dropout(0.3) → Linear(512 → 14)",
                "key_feature": "Dense connectivity pattern reduces vanishing gradients",
            },
            {
                "name": "ResNet-50",
                "architecture": "Residual Network (50 layers)",
                "parameters": resnet_params,
                "parameters_human": f"{resnet_params / 1e6:.2f}M",
                "pretrained_on": "ImageNet → Fine-tuned for ChestX-ray14",
                "input_size": "224 × 224 × 3",
                "classifier_head": "Linear(2048 → 512) → ReLU → Dropout(0.3) → Linear(512 → 14)",
                "key_feature": "Skip connections (baseline comparison model)",
            },
        ]


# ─── Singleton ───────────────────────────────────────────────
_model_service: Optional[ChestModelService] = None


def get_chest_model_service() -> ChestModelService:
    global _model_service
    if _model_service is None:
        _model_service = ChestModelService()
    return _model_service
