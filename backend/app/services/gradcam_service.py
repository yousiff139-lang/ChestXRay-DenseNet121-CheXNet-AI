"""
Grad-CAM Service — Disease-Relevant Region Visualization.

Uses the pytorch-grad-cam library to generate class activation
heatmaps showing which regions of the chest X-ray influenced the
model's prediction for a given disease class.

Target layers:
  - DenseNet-121: model.features.denseblock4 / features[-2]  (last DenseBlock conv block)
  - ResNet-50:    model.resnet.layer4[-1] / layer4[-1]        (last bottleneck block)
"""

import base64
import io
import logging
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..models.classification import CHEST_XRAY_CLASSES
from .chest_model_service import ChestModelService, get_chest_model_service

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


class GradCAMService:
    """Service for generating Grad-CAM heatmap visualizations."""

    def __init__(self):
        self.model_service: ChestModelService = get_chest_model_service()
        self.transform = transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _get_target_layer(self, model_name: str):
        """Return the appropriate target convolutional layer for Grad-CAM."""
        if model_name == "densenet":
            model = self.model_service.get_densenet()
            if hasattr(model, "features"):
                if hasattr(model.features, "norm5"):
                    return [model.features.norm5]
                if hasattr(model.features, "denseblock4"):
                    return [model.features.denseblock4]
                return [model.features[-2]]
            if hasattr(model, "densenet") and hasattr(model.densenet, "features"):
                if hasattr(model.densenet.features, "norm5"):
                    return [model.densenet.features.norm5]
                if hasattr(model.densenet.features, "denseblock4"):
                    return [model.densenet.features.denseblock4]
                return [model.densenet.features[-2]]
            return [model.features[-2]]
        else:
            model = self.model_service.get_resnet()
            if hasattr(model, "model") and hasattr(model.model, "layer4"):
                return [model.model.layer4[-1]]
            if hasattr(model, "resnet") and hasattr(model.resnet, "layer4"):
                return [model.resnet.layer4[-1]]
            return [model.layer4[-1]]

    def generate_gradcam(
        self,
        image_path: str,
        model_name: str = "densenet",
        target_class_idx: Optional[int] = None,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.5,
    ) -> Optional[str]:
        """
        Generate Grad-CAM heatmap overlay for a chest X-ray.

        Args:
            image_path: Path to the chest X-ray image
            model_name: "densenet" or "resnet"
            target_class_idx: Index of the target disease class (0-13).
                              If None, uses the highest-probability prediction.
            colormap: OpenCV colormap for the heatmap
            alpha: Blending weight for the overlay (0=only heatmap, 1=only image)

        Returns:
            Base64-encoded PNG image of the Grad-CAM overlay, or None on failure
        """
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        except ImportError:
            logger.warning(
                "pytorch-grad-cam not installed. Run: pip install grad-cam"
            )
            return None

        try:
            # Load and preprocess image
            original_image = Image.open(image_path)
            original_resized = original_image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
            rgb_array = np.float32(np.array(original_resized)) / 255.0

            device = self.model_service.get_device()
            if model_name == "densenet":
                model = self.model_service.get_densenet()
            else:
                model = self.model_service.get_resnet()

            # Handle 1-channel medical models (torchxrayvision)
            if hasattr(model, "pathologies"):
                import torchxrayvision as xrv

                res = 224 if model_name == "densenet" else 512
                img_l = original_image.convert("L").resize((res, res), Image.Resampling.BILINEAR)
                img_np = np.array(img_l, dtype=np.float32)
                img_norm = xrv.datasets.normalize(img_np, 255.0)
                img_norm = img_norm[None, ...]
                input_tensor = torch.from_numpy(img_norm).unsqueeze(0).float().to(device)
            else:
                input_tensor = self.transform(original_image.convert("RGB")).unsqueeze(0).to(device)

            model.eval()

            # Resolve actual model output index for the target pathology
            actual_target_idx = target_class_idx
            if hasattr(model, "pathologies"):
                if target_class_idx is not None and 0 <= target_class_idx < len(CHEST_XRAY_CLASSES):
                    disease_name = CHEST_XRAY_CLASSES[target_class_idx]
                    variants = [
                        disease_name,
                        disease_name.replace("_", " "),
                        disease_name.replace(" ", "_"),
                        disease_name.lower(),
                        disease_name.replace("_", " ").lower(),
                    ]
                    for var in variants:
                        if var in model.pathologies:
                            actual_target_idx = model.pathologies.index(var)
                            break
                    else:
                        for i, p in enumerate(model.pathologies):
                            if p.strip() and p.strip().lower() == disease_name.lower().replace("_", " "):
                                actual_target_idx = i
                                break
                elif target_class_idx is None:
                    with torch.no_grad():
                        logits = model(input_tensor)
                        actual_target_idx = int(logits.argmax(dim=1).item())
            else:
                if target_class_idx is None:
                    with torch.no_grad():
                        logits = model(input_tensor)
                        actual_target_idx = int(logits.argmax(dim=1).item())

            # Generate Grad-CAM
            target_layers = self._get_target_layer(model_name)
            targets = [ClassifierOutputTarget(actual_target_idx)]

            with GradCAM(model=model, target_layers=target_layers) as cam:
                grayscale_cam = cam(
                    input_tensor=input_tensor, targets=targets
                )
                raw_cam = grayscale_cam[0, :]
                if raw_cam.shape != (INPUT_SIZE, INPUT_SIZE):
                    raw_cam = cv2.resize(raw_cam, (INPUT_SIZE, INPUT_SIZE))
                grayscale_cam = raw_cam

            # Apply colormap
            heatmap = cv2.applyColorMap(
                np.uint8(255 * grayscale_cam), colormap
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = np.float32(heatmap) / 255.0

            # Blend heatmap with original image
            overlay = (1 - alpha) * heatmap + alpha * rgb_array
            overlay = np.clip(overlay, 0, 1)
            overlay = np.uint8(255 * overlay)

            # Encode to base64 PNG
            overlay_image = Image.fromarray(overlay)
            buffer = io.BytesIO()
            overlay_image.save(buffer, format="PNG", quality=95)
            buffer.seek(0)
            base64_str = base64.b64encode(buffer.read()).decode("utf-8")

            logger.info(
                f"Grad-CAM generated for class {target_class_idx} "
                f"using {model_name}"
            )
            return base64_str

        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}", exc_info=True)
            return None


# ─── Singleton ───────────────────────────────────────────────
_gradcam_service: Optional[GradCAMService] = None


def get_gradcam_service() -> GradCAMService:
    global _gradcam_service
    if _gradcam_service is None:
        _gradcam_service = GradCAMService()
    return _gradcam_service
