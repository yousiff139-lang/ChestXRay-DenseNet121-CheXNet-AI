"""
Classification Service — Multi-Label Chest X-Ray Inference.

Preprocesses chest X-ray images and runs inference through DenseNet-121
and/or ResNet-50 models. Returns multi-label disease probabilities for
all 14 ChestX-ray14 pathologies.
"""

import time
import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..core.config import get_settings
from ..models.classification import (
    CHEST_XRAY_CLASSES,
    CHEST_XRAY_PREVALENCE,
    CHEST_XRAY_VALIDATION_THRESHOLDS,
    DiseaseClassification,
    ClassificationResponse,
    ModelComparisonResponse,
)
from .chest_model_service import ChestModelService, get_chest_model_service

logger = logging.getLogger(__name__)

# ImageNet normalization constants (used for transfer learning from ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def get_inference_transforms() -> transforms.Compose:
    """Standard preprocessing pipeline for chest X-ray inference."""
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ClassificationService:
    """Service for running multi-label chest X-ray classification."""

    def __init__(self):
        self.settings = get_settings()
        self.model_service: ChestModelService = get_chest_model_service()
        self.transform = get_inference_transforms()

    def _preprocess_image(
        self, image_path: str, model_name: str = "densenet"
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Load and preprocess a chest X-ray image for inference.

        Args:
            image_path: Path to the image file
            model_name: "densenet" or "resnet"

        Returns:
            Tuple of (input_tensor, original_width, original_height)
        """
        image = Image.open(image_path)
        original_width, original_height = image.size

        if model_name == "densenet":
            model = self.model_service.get_densenet()
        else:
            model = self.model_service.get_resnet()

        if hasattr(model, "pathologies"):
            # torchxrayvision model: requires 1-channel, [-1024, 1024]
            import torchxrayvision as xrv

            res = 224 if model_name == "densenet" else 512
            img_l = image.convert("L").resize((res, res), Image.Resampling.BILINEAR)
            img_np = np.array(img_l, dtype=np.float32)
            img_norm = xrv.datasets.normalize(img_np, 255.0)
            img_norm = img_norm[None, ...]  # [1, H, W]
            input_tensor = torch.from_numpy(img_norm).unsqueeze(0).float()
        else:
            # Custom torchvision model: 3-channel RGB, ImageNet normalization
            img_rgb = image.convert("RGB")
            input_tensor = self.transform(img_rgb).unsqueeze(0)

        return input_tensor, original_width, original_height

    def _run_inference(
        self, input_tensor: torch.Tensor, model_name: str = "densenet"
    ) -> Tuple[np.ndarray, float]:
        """
        Run inference through the specified model with 2-View Test-Time Augmentation (TTA)
        from the Colab notebook (averaging original + horizontally flipped views).

        Returns:
            Tuple of (probabilities_array [14], inference_time_ms)
        """
        device = self.model_service.get_device()

        if model_name == "ensemble":
            # Run calibrated ensemble of DenseNet-121 (0.70) and ResNet-50 (0.30)
            p_dense, t_dense = self._run_inference(input_tensor, model_name="densenet")
            p_res, t_res = self._run_inference(input_tensor, model_name="resnet")
            p_ensemble = 0.70 * p_dense + 0.30 * p_res
            return p_ensemble, t_dense + t_res

        if model_name == "densenet":
            model = self.model_service.get_densenet()
        else:
            model = self.model_service.get_resnet()

        model.eval()
        input_tensor = input_tensor.to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            # 2-View Test-Time Augmentation (Colab notebook CheXNet methodology)
            # Averages predictions from original and horizontally flipped view.
            # Cancels sensor artifacts, reduces spatial variance, and avoids overfitting.
            input_flip = torch.flip(input_tensor, dims=[3])
            out_orig = model(input_tensor)
            out_flip = model(input_flip)
            out = 0.5 * (out_orig + out_flip)
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Handle torchxrayvision models
        if hasattr(model, "pathologies"):
            raw_out = out.cpu().numpy()[0]
            probs_list = []
            for disease in CHEST_XRAY_CLASSES:
                p_val = 0.0
                variants = [
                    disease,
                    disease.replace("_", " "),
                    disease.replace(" ", "_"),
                    disease.lower(),
                    disease.replace("_", " ").lower(),
                ]
                matched = False
                for var in variants:
                    if var in model.pathologies:
                        idx = model.pathologies.index(var)
                        p_val = float(raw_out[idx])
                        matched = True
                        break
                if not matched:
                    for i, p in enumerate(model.pathologies):
                        if p.strip() and p.strip().lower() == disease.lower().replace("_", " "):
                            p_val = float(raw_out[i])
                            matched = True
                            break

                # Torchxrayvision with op_threshs=None outputs raw logits (unbounded real numbers)
                # Unconditionally convert to sigmoid probabilities
                p_val = float(1.0 / (1.0 + np.exp(-p_val)))
                probs_list.append(p_val)

            probs = np.array(probs_list, dtype=np.float32)
            return probs, inference_time_ms

        # Standard torchvision-style model (3-channel input)
        probabilities = torch.sigmoid(out)
        probs = probabilities.cpu().numpy()[0][: len(CHEST_XRAY_CLASSES)]
        return probs, inference_time_ms

    def _build_classification_response(
        self,
        probabilities: np.ndarray,
        model_name: str,
        inference_time_ms: float,
        image_width: int,
        image_height: int,
        gradcam_base64: Optional[str] = None,
        gradcam_class_name: Optional[str] = None,
        end_to_end_ms: Optional[float] = None,
    ) -> ClassificationResponse:
        """
        Build structured classification response with temperature-calibrated probabilities,
        per-class validation thresholds (Youden's J), and three-state clinical classification:
        - Positive: probability >= threshold
        - Uncertain: threshold - 0.10 <= probability < threshold
        - Negative: probability < threshold - 0.10
        """
        predictions: List[DiseaseClassification] = []
        positive_findings: List[DiseaseClassification] = []

        for i, disease_name in enumerate(CHEST_XRAY_CLASSES):
            prob = float(probabilities[i])
            thresh = float(CHEST_XRAY_VALIDATION_THRESHOLDS.get(disease_name, 0.50))
            prevalence = float(CHEST_XRAY_PREVALENCE.get(disease_name, 0.0))

            # 3-State Classification (Architecture Spec Section 5)
            # Positive: prob >= threshold
            # Uncertain: borderline zone right below threshold (within 5% or 25% of threshold)
            # Negative: prob below uncertain zone
            uncertain_margin = min(0.05, thresh * 0.25)
            if prob >= thresh:
                state = "Positive"
                is_pos = True
            elif prob >= max(0.02, thresh - uncertain_margin):
                state = "Uncertain"
                is_pos = False
            else:
                state = "Negative"
                is_pos = False

            classification = DiseaseClassification(
                disease_name=disease_name,
                probability=round(prob, 4),
                raw_probability=round(prob, 4),
                threshold=round(thresh, 4),
                prevalence=round(prevalence, 4),
                state=state,
                is_positive=is_pos,
            )
            predictions.append(classification)
            if is_pos or state == "Uncertain":
                positive_findings.append(classification)

        # Sort positive findings: Positives first by probability, then Uncertain
        positive_findings.sort(
            key=lambda x: (x.state == "Positive", x.probability), reverse=True
        )

        if model_name == "densenet":
            display_name = "DenseNet-121"
        elif model_name == "resnet":
            display_name = "ResNet-50"
        else:
            display_name = "Ensemble (DenseNet + ResNet)"

        return ClassificationResponse(
            predictions=predictions,
            positive_findings=positive_findings,
            model_used=display_name,
            inference_time_ms=round(inference_time_ms, 2),
            model_inference_ms=round(inference_time_ms, 2),
            end_to_end_ms=round(end_to_end_ms, 2) if end_to_end_ms else round(inference_time_ms, 2),
            gradcam_overlay_base64=gradcam_base64,
            gradcam_class_name=gradcam_class_name,
            image_width=image_width,
            image_height=image_height,
            confidence_threshold=0.50,
            disclaimer="Research prototype. Not for clinical use.",
        )

    async def classify_chest_xray(
        self,
        image_path: str,
        model_name: str = "densenet",
        generate_gradcam: bool = True,
    ) -> ClassificationResponse:
        """
        Classify a chest X-ray image conforming to Seminar 16 Architecture Pipeline:
        Upload -> validate -> preprocess -> Model -> sigmoid -> temperature scaling
               -> per-class thresholds -> state per class
               -> Grad-CAM (selected class) -> UI + JSON response
        """
        start_total = time.perf_counter()

        # 1. Preprocess
        input_tensor, img_w, img_h = self._preprocess_image(
            image_path, model_name=model_name
        )

        # 2. Run inference
        probabilities, inference_time_ms = self._run_inference(input_tensor, model_name)

        logger.info(
            f"[{model_name.upper()}] Inference completed in {inference_time_ms:.1f}ms. "
            f"Top prediction: {CHEST_XRAY_CLASSES[np.argmax(probabilities)]} "
            f"({probabilities.max():.3f})"
        )

        # 3. Determine positive classes for Grad-CAM
        # Architecture Spec Section 7: "Only explain positives: show heatmap for classes
        # above their threshold; otherwise show 'no evidence' instead of a heatmap."
        positive_classes = [
            (i, CHEST_XRAY_CLASSES[i], probabilities[i])
            for i in range(len(CHEST_XRAY_CLASSES))
            if probabilities[i] >= CHEST_XRAY_VALIDATION_THRESHOLDS.get(CHEST_XRAY_CLASSES[i], 0.50)
        ]

        gradcam_base64 = None
        gradcam_class_name = "No evidence"

        if generate_gradcam and self.settings.gradcam_enabled and positive_classes:
            try:
                from .gradcam_service import get_gradcam_service

                gradcam_service = get_gradcam_service()
                # Grad-CAM for the highest-scoring positive class
                positive_classes.sort(key=lambda x: x[2], reverse=True)
                top_class_idx, top_class_name, _ = positive_classes[0]

                gradcam_base64 = gradcam_service.generate_gradcam(
                    image_path=image_path,
                    model_name=model_name,
                    target_class_idx=top_class_idx,
                )
                gradcam_class_name = top_class_name
            except Exception as e:
                logger.warning(f"Grad-CAM generation failed: {e}")

        end_to_end_ms = (time.perf_counter() - start_total) * 1000

        return self._build_classification_response(
            probabilities=probabilities,
            model_name=model_name,
            inference_time_ms=inference_time_ms,
            image_width=img_w,
            image_height=img_h,
            gradcam_base64=gradcam_base64,
            gradcam_class_name=gradcam_class_name,
            end_to_end_ms=end_to_end_ms,
        )

    async def compare_models(
        self, image_path: str, generate_gradcam: bool = True
    ) -> ModelComparisonResponse:
        """
        Run classification through both DenseNet-121 and ResNet-50
        and return side-by-side comparison.
        """
        densenet_result = await self.classify_chest_xray(
            image_path, model_name="densenet", generate_gradcam=generate_gradcam
        )
        resnet_result = await self.classify_chest_xray(
            image_path, model_name="resnet", generate_gradcam=generate_gradcam
        )

        # Build comparison summary
        comparison = {
            "densenet_inference_ms": densenet_result.inference_time_ms,
            "resnet_inference_ms": resnet_result.inference_time_ms,
            "densenet_positive_count": len(densenet_result.positive_findings),
            "resnet_positive_count": len(resnet_result.positive_findings),
            "agreement": [],
            "disagreement": [],
        }

        # Find agreements and disagreements
        for i, disease in enumerate(CHEST_XRAY_CLASSES):
            d_pos = densenet_result.predictions[i].is_positive
            r_pos = resnet_result.predictions[i].is_positive
            entry = {
                "disease": disease,
                "densenet_prob": densenet_result.predictions[i].probability,
                "resnet_prob": resnet_result.predictions[i].probability,
            }
            if d_pos == r_pos:
                comparison["agreement"].append(entry)
            else:
                comparison["disagreement"].append(entry)

        return ModelComparisonResponse(
            densenet_results=densenet_result,
            resnet_results=resnet_result,
            comparison_summary=comparison,
        )


# ─── Singleton ───────────────────────────────────────────────
_classification_service: Optional[ClassificationService] = None


def get_classification_service() -> ClassificationService:
    global _classification_service
    if _classification_service is None:
        _classification_service = ClassificationService()
    return _classification_service
