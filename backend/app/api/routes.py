from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Annotated
import tempfile
import os
import logging

from ..models.classification import (
    ClassificationResponse,
    ModelComparisonResponse,
    DicomClassificationResponse,
    DiagnosticReportResponse,
    DiagnosticReportRequest,
    ModelInfoResponse,
    ErrorResponse,
)
from ..models.detection import UnifiedDiagnosisResponse
from ..services.classification_service import (
    ClassificationService,
    get_classification_service,
)
from ..services.preprocessing_service import (
    PreprocessingService,
    get_preprocessing_service,
)
from ..services.diagnostic_service import (
    DiagnosticReportService,
    get_diagnostic_report_service,
)
from ..services.unified_diagnostic_service import (
    UnifiedDiagnosticService,
    get_unified_diagnostic_service,
)
from ..core.config import get_settings, Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["classification"])


# ─────────────────────────────────────────────────────────────
# POST /classify — DenseNet-121 Chest X-Ray Classification
# ─────────────────────────────────────────────────────────────
@router.post(
    "/classify",
    response_model=ClassificationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Classification failed"},
    },
)
async def classify_chest_xray(
    file: UploadFile,
    classification_service: Annotated[
        ClassificationService, Depends(get_classification_service)
    ],
    settings: Annotated[Settings, Depends(get_settings)],
    model: str = "densenet",
) -> ClassificationResponse:
    """
    Classify a chest X-ray image for 14 thoracic pathologies.

    Upload a chest X-ray (JPEG/PNG) and receive multi-label disease
    classifications with probabilities, Grad-CAM visualization, and
    inference time measurement.

    - **model**: "densenet" (DenseNet-121, default) or "resnet" (ResNet-50 baseline)
    """
    # Validate model parameter
    if model not in ("densenet", "resnet"):
        raise HTTPException(
            status_code=400,
            detail="Model must be 'densenet' or 'resnet'",
        )

    # Validate file type
    if file.content_type not in settings.allowed_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"File must be one of: {', '.join(settings.allowed_file_types)}",
        )

    # Validate file size
    if file.size and file.size > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({file.size} bytes) exceeds maximum ({settings.max_file_size} bytes)",
        )

    content = await file.read()

    if not file.size and len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({len(content)} bytes) exceeds maximum ({settings.max_file_size} bytes)",
        )

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename or "image.jpg")[1]
        ) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        result = await classification_service.classify_chest_xray(
            temp_file_path, model_name=model
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during classification"
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")


# ─────────────────────────────────────────────────────────────
# POST /classify-dicom — DICOM Chest X-Ray Classification
# ─────────────────────────────────────────────────────────────
@router.post(
    "/classify-dicom",
    response_model=DicomClassificationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid DICOM file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Classification failed"},
    },
)
async def classify_chest_xray_dicom(
    file: UploadFile,
    classification_service: Annotated[
        ClassificationService, Depends(get_classification_service)
    ],
    preprocessing_service: Annotated[
        PreprocessingService, Depends(get_preprocessing_service)
    ],
    settings: Annotated[Settings, Depends(get_settings)],
    model: str = "densenet",
) -> DicomClassificationResponse:
    """
    Classify a DICOM chest X-ray for 14 thoracic pathologies.

    Upload a DICOM file (.dcm) and receive:
    - Multi-label disease classifications with probabilities
    - DICOM metadata extraction
    - Grad-CAM visualization
    - Inference time measurement
    """
    if model not in ("densenet", "resnet"):
        raise HTTPException(status_code=400, detail="Model must be 'densenet' or 'resnet'")

    # Validate DICOM file
    allowed_extensions = settings.ALLOWED_DICOM_EXTENSIONS
    file_extension = os.path.splitext(file.filename or "")[1] if file.filename else ""

    if (
        file.content_type not in settings.allowed_dicom_file_types
        and file_extension not in allowed_extensions
    ):
        raise HTTPException(
            status_code=400,
            detail=f"File must be a DICOM file (.dcm, .dicom) or have content type: {', '.join(settings.allowed_dicom_file_types)}",
        )

    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size ({len(content)} bytes) exceeds maximum ({settings.max_file_size} bytes)",
        )

    temp_dicom_path = None
    converted_image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_file:
            temp_file.write(content)
            temp_dicom_path = temp_file.name

        # Parse DICOM metadata
        metadata = preprocessing_service.parse_dicom_metadata(temp_dicom_path)

        # Convert DICOM to standard image
        converted_image_path, image_info = preprocessing_service.convert_dicom_to_image(
            temp_dicom_path
        )

        # Run classification
        result = await classification_service.classify_chest_xray(
            converted_image_path, model_name=model
        )

        display_name = "DenseNet-121" if model == "densenet" else "ResNet-50"

        return DicomClassificationResponse(
            predictions=result.predictions,
            positive_findings=result.positive_findings,
            metadata=metadata,
            image_info=image_info,
            model_used=display_name,
            inference_time_ms=result.inference_time_ms,
            gradcam_overlay_base64=result.gradcam_overlay_base64,
            image_width=result.image_width,
            image_height=result.image_height,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DICOM classification failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during DICOM processing",
        )
    finally:
        for path in [temp_dicom_path, converted_image_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError as e:
                    logger.warning(f"Failed to delete temp file {path}: {e}")


# ─────────────────────────────────────────────────────────────
# POST /compare — DenseNet-121 vs ResNet-50 Comparison
# ─────────────────────────────────────────────────────────────
@router.post(
    "/compare",
    response_model=ModelComparisonResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format"},
        500: {"model": ErrorResponse, "description": "Comparison failed"},
    },
)
async def compare_models(
    file: UploadFile,
    classification_service: Annotated[
        ClassificationService, Depends(get_classification_service)
    ],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ModelComparisonResponse:
    """
    Run both DenseNet-121 and ResNet-50 on the same chest X-ray
    and return a side-by-side comparison with Grad-CAM for each model.
    """
    if file.content_type not in settings.allowed_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"File must be one of: {', '.join(settings.allowed_file_types)}",
        )

    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum ({settings.max_file_size} bytes)",
        )

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename or "image.jpg")[1]
        ) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        result = await classification_service.compare_models(temp_file_path)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during model comparison"
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")


# ─────────────────────────────────────────────────────────────
# POST /generate-report — Diagnostic Report from Classifications
# ─────────────────────────────────────────────────────────────
@router.post(
    "/generate-report",
    response_model=DiagnosticReportResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Report generation failed"},
    },
)
async def generate_diagnostic_report(
    request: DiagnosticReportRequest,
    diagnostic_service: Annotated[
        DiagnosticReportService, Depends(get_diagnostic_report_service)
    ],
) -> DiagnosticReportResponse:
    """
    Generate a comprehensive diagnostic report from classification results.

    This endpoint accepts classification results from a previous analysis
    and generates a professional radiology report including findings,
    clinical impression, and recommendations.
    """
    try:
        report = await diagnostic_service.generate_diagnostic_report(
            classifications=request.predictions,
            metadata=request.metadata,
            image_info=request.image_info,
            image_base64=request.image_base64,
        )

        return DiagnosticReportResponse(
            diagnostic_report=report,
            classifications_used=request.predictions,
            metadata=request.metadata,
        )

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during report generation",
        )


# ─────────────────────────────────────────────────────────────
# GET /model-info — Model Architecture Details
# ─────────────────────────────────────────────────────────────
@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(
    classification_service: Annotated[
        ClassificationService, Depends(get_classification_service)
    ],
) -> ModelInfoResponse:
    """
    Return information about available model architectures,
    dataset details, and evaluation metrics.
    """
    model_info = classification_service.model_service.get_model_info()
    return ModelInfoResponse(models=model_info)


# ─────────────────────────────────────────────────────────────
# GET /benchmark — Benchmark Comparison Metrics
# ─────────────────────────────────────────────────────────────
@router.get("/benchmark")
async def get_benchmark_comparison():
    """
    Return published and empirical benchmark metrics comparing
    DenseNet-121 and ResNet-50 across all 14 thoracic pathologies.
    """
    from ..services.evaluation_service import get_evaluation_service
    eval_service = get_evaluation_service()
    return eval_service.get_published_benchmark_comparison()


# ─────────────────────────────────────────────────────────────
# POST /detect-and-diagnose — Unified CNN + YOLO + Gemini
# ─────────────────────────────────────────────────────────────
@router.post(
    "/detect-and-diagnose",
    response_model=UnifiedDiagnosisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Diagnosis failed"},
    },
)
async def detect_and_diagnose(
    file: UploadFile,
    unified_service: Annotated[
        UnifiedDiagnosticService, Depends(get_unified_diagnostic_service)
    ],
    settings: Annotated[Settings, Depends(get_settings)],
    include_resnet: bool = True,
    generate_report: bool = True,
) -> UnifiedDiagnosisResponse:
    """
    Unified multi-model diagnostic endpoint:
    - Runs DenseNet-121 multi-label classification + Grad-CAM heatmap
    - Runs ResNet-50 comparative baseline
    - Runs YOLO object detection to square pathological lesions (Nodules, Consolidations, etc.)
    - Runs Gemini Vision global audit & 500+ word curative clinical report
    """
    if file.content_type not in settings.allowed_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"File must be one of: {', '.join(settings.allowed_file_types)}",
        )

    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum ({settings.max_file_size} bytes)",
        )

    temp_file_path = None
    try:
        import base64
        ext = os.path.splitext(file.filename or "image.jpg")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        img_b64 = f"data:image/jpeg;base64,{base64.b64encode(content).decode('ascii')}"

        result = await unified_service.diagnose_chest_xray(
            image_path=temp_file_path,
            include_resnet=include_resnet,
            generate_gradcam=True,
            generate_full_report=generate_report,
            image_base64=img_b64,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unified diagnosis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during diagnosis"
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as e:
                logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")


# ─────────────────────────────────────────────────────────────
# POST /detect-and-diagnose-dicom — DICOM Unified Pipeline
# ─────────────────────────────────────────────────────────────
@router.post(
    "/detect-and-diagnose-dicom",
    response_model=UnifiedDiagnosisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid DICOM file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Diagnosis failed"},
    },
)
async def detect_and_diagnose_dicom(
    file: UploadFile,
    unified_service: Annotated[
        UnifiedDiagnosticService, Depends(get_unified_diagnostic_service)
    ],
    preprocessing_service: Annotated[
        PreprocessingService, Depends(get_preprocessing_service)
    ],
    settings: Annotated[Settings, Depends(get_settings)],
    include_resnet: bool = True,
    generate_report: bool = True,
) -> UnifiedDiagnosisResponse:
    """
    Unified diagnostic endpoint for DICOM chest X-rays (.dcm).
    Extracts metadata, windowing, runs DenseNet-121, ResNet-50, YOLO, and Gemini.
    """
    allowed_extensions = settings.ALLOWED_DICOM_EXTENSIONS
    file_extension = os.path.splitext(file.filename or "")[1] if file.filename else ""

    if (
        file.content_type not in settings.allowed_dicom_file_types
        and file_extension not in allowed_extensions
    ):
        raise HTTPException(
            status_code=400,
            detail=f"File must be a DICOM file (.dcm, .dicom)",
        )

    content = await file.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum ({settings.max_file_size} bytes)",
        )

    temp_dicom_path = None
    converted_image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_file:
            temp_file.write(content)
            temp_dicom_path = temp_file.name

        metadata = preprocessing_service.parse_dicom_metadata(temp_dicom_path)
        converted_image_path, _ = preprocessing_service.convert_dicom_to_image(
            temp_dicom_path
        )

        with open(converted_image_path, "rb") as f:
            import base64
            conv_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('ascii')}"

        result = await unified_service.diagnose_chest_xray(
            image_path=converted_image_path,
            include_resnet=include_resnet,
            generate_gradcam=True,
            generate_full_report=generate_report,
            metadata=metadata,
            image_base64=conv_b64,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DICOM diagnosis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred during DICOM diagnosis"
        )
    finally:
        for p in [temp_dicom_path, converted_image_path]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError as e:
                    logger.warning(f"Failed to delete temp file {p}: {e}")


# ─────────────────────────────────────────────────────────────
# GET /health — Health Check
# ─────────────────────────────────────────────────────────────
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "chest-xray-classification-api"}
