"""
Unified Diagnostic Service — Orchestrating CNN Classification, YOLO Detection, and Gemini VLM.

Coordinates:
1. DenseNet-121 Multi-Label Classification & Grad-CAM Heatmap
2. ResNet-50 Comparative Baseline Classification & Grad-CAM Heatmap
3. YOLO Bounding Box Object Detection (Squaring the lesions)
4. Gemini Vision Global Audit & 500+ Word Curative Report
"""

import time
import logging
from typing import Optional, List

from .classification_service import ClassificationService, get_classification_service
from .yolo_detection_service import YoloDetectionService, get_yolo_detection_service
from .gemini_clinical_agent import GeminiClinicalAgent, get_gemini_clinical_agent
from ..models.detection import UnifiedDiagnosisResponse
from ..models.classification import DicomMetadata

logger = logging.getLogger(__name__)


class UnifiedDiagnosticService:
    """Master orchestrator for the entire multi-model chest AI engine."""

    def __init__(self):
        self.classification_service: ClassificationService = get_classification_service()
        self.yolo_service: YoloDetectionService = get_yolo_detection_service()
        self.gemini_agent: GeminiClinicalAgent = get_gemini_clinical_agent()

    async def diagnose_chest_xray(
        self,
        image_path: str,
        include_resnet: bool = True,
        generate_gradcam: bool = True,
        generate_full_report: bool = True,
        metadata: Optional[DicomMetadata] = None,
        image_base64: Optional[str] = None,
    ) -> UnifiedDiagnosisResponse:
        """
        Run the complete unified pipeline:
        1. DenseNet-121 Classification + GradCAM
        2. ResNet-50 Comparative Classification (optional)
        3. YOLO Bounding Box Detection + Overlay
        4. Gemini VLM Global Audit & Detailed Curative Report
        """
        t0 = time.perf_counter()

        # Step 1: Run DenseNet-121 Classification (and Grad-CAM)
        densenet_resp = await self.classification_service.classify_chest_xray(
            image_path=image_path,
            model_name="densenet",
            generate_gradcam=generate_gradcam,
        )

        # Step 2: Run ResNet-50 Baseline (if requested)
        resnet_predictions = None
        if include_resnet:
            try:
                resnet_resp = await self.classification_service.classify_chest_xray(
                    image_path=image_path,
                    model_name="resnet",
                    generate_gradcam=False,
                )
                resnet_predictions = resnet_resp.predictions
            except Exception as e:
                logger.warning(f"ResNet-50 baseline inference skipped: {e}")

        # Extract positive disease names from DenseNet
        positive_disease_names = [
            f.disease_name for f in densenet_resp.positive_findings
        ]

        # Step 3: Run YOLO Pathology Detection & Bounding Box Localization
        detections, annotated_overlay_b64, img_w, img_h = (
            await self.yolo_service.detect_pathologies(
                image_path=image_path,
                dense_positive_findings=positive_disease_names,
            )
        )

        # Step 4: Map Curative Protocols for all detected or classified conditions
        all_detected_conditions = list(
            set(
                [d.class_name for d in detections]
                + positive_disease_names
            )
        )
        curative_protocols = self.yolo_service.get_curative_protocols(
            all_detected_conditions
        )

        # Step 5: Gemini VLM Global Audit & Curative Report Generation
        gemini_audit = await self.gemini_agent.audit_chest_xray(image_path)
        audit_summary = gemini_audit.get("summary", "Automated diagnostic audit complete.")

        diagnostic_report = None
        if generate_full_report:
            diagnostic_report = await self.gemini_agent.generate_comprehensive_report(
                classifications=densenet_resp.predictions,
                detections=detections,
                curative_protocols=curative_protocols,
                metadata=metadata,
                image_base64=image_base64,
            )

        end_to_end_ms = (time.perf_counter() - t0) * 1000

        return UnifiedDiagnosisResponse(
            detections=detections,
            total_detections=len(detections),
            predictions=densenet_resp.predictions,
            positive_findings=densenet_resp.positive_findings,
            resnet_predictions=resnet_predictions,
            gradcam_overlay_base64=densenet_resp.gradcam_overlay_base64,
            annotated_overlay_base64=annotated_overlay_b64,
            gemini_audit_summary=audit_summary,
            diagnostic_report=diagnostic_report,
            curative_protocols=curative_protocols,
            image_width=img_w or densenet_resp.image_width or 1000,
            image_height=img_h or densenet_resp.image_height or 1000,
            inference_time_ms=round(densenet_resp.inference_time_ms, 2),
            model_inference_ms=round(densenet_resp.model_inference_ms or densenet_resp.inference_time_ms, 2),
            end_to_end_ms=round(end_to_end_ms, 2),
            metadata=metadata,
            disclaimer="Research prototype. Not for clinical diagnostic use.",
        )


# ─── Singleton ───────────────────────────────────────────────
_unified_diagnostic_service = None


def get_unified_diagnostic_service() -> UnifiedDiagnosticService:
    global _unified_diagnostic_service
    if _unified_diagnostic_service is None:
        _unified_diagnostic_service = UnifiedDiagnosticService()
    return _unified_diagnostic_service
