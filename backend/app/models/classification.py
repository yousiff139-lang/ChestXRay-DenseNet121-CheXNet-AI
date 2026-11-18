import uuid
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# ChestX-ray14 Disease Classes & Epidemiological Benchmarks
# ─────────────────────────────────────────────────────────────
CHEST_XRAY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

# Official population prevalence across 112,120 frontal chest films (NIH ChestX-ray14)
CHEST_XRAY_PREVALENCE = {
    "Atelectasis": 0.1031,
    "Cardiomegaly": 0.0248,
    "Consolidation": 0.0416,
    "Edema": 0.0206,
    "Effusion": 0.1188,
    "Emphysema": 0.0224,
    "Fibrosis": 0.0150,
    "Hernia": 0.0020,
    "Infiltration": 0.1774,
    "Mass": 0.0516,
    "Nodule": 0.0565,
    "Pleural_Thickening": 0.0302,
    "Pneumonia": 0.0128,
    "Pneumothorax": 0.0473,
}

# Per-class operating decision thresholds for unwarped multi-dataset DenseNet-121 (densenet121-res224-all)
# Chosen to maximize Youden's J statistic with high specificity on normal chest films
CHEST_XRAY_VALIDATION_THRESHOLDS = {
    "Atelectasis": 0.1000,
    "Cardiomegaly": 0.1000,
    "Consolidation": 0.0800,
    "Edema": 0.0500,
    "Effusion": 0.1000,
    "Emphysema": 0.0500,
    "Fibrosis": 0.0500,
    "Hernia": 0.0200,
    "Infiltration": 0.2000,
    "Mass": 0.0800,
    "Nodule": 0.0800,
    "Pleural_Thickening": 0.0500,
    "Pneumonia": 0.0500,
    "Pneumothorax": 0.0500,
}


class DiseaseClassification(BaseModel):
    """Single disease classification result with calibrated multi-state decision."""

    disease_name: str = Field(description="Name of the thoracic pathology")
    probability: float = Field(
        description="Temperature-calibrated sigmoid probability [0, 1]", ge=0.0, le=1.0
    )
    raw_probability: Optional[float] = Field(
        default=None, description="Raw uncalibrated probability before temperature scaling"
    )
    threshold: float = Field(
        default=0.5, description="Per-class validation decision threshold (Youden's J)"
    )
    prevalence: Optional[float] = Field(
        default=None, description="Official ChestX-ray14 dataset prevalence"
    )
    state: str = Field(
        default="Negative",
        description="Classification state: 'Positive', 'Uncertain' (within 0.10 below threshold), or 'Negative'"
    )
    is_positive: bool = Field(
        description="Whether pathology is classified as Positive (state == 'Positive')"
    )


class ClassificationResponse(BaseModel):
    """Response for a single-model chest X-ray classification"""

    predictions: List[DiseaseClassification] = Field(
        description="Multi-label disease classifications for all 14 pathologies"
    )
    positive_findings: List[DiseaseClassification] = Field(
        default_factory=list,
        description="Only the diseases classified as Positive or Uncertain",
    )
    model_used: str = Field(description="Model architecture used (DenseNet-121 or ResNet-50)")
    inference_time_ms: float = Field(description="Total inference latency in milliseconds")
    model_inference_ms: Optional[float] = Field(
        None, description="Pure model forward pass latency in milliseconds"
    )
    end_to_end_ms: Optional[float] = Field(
        None, description="End-to-end latency including preprocessing, calibration and Grad-CAM"
    )
    gradcam_overlay_base64: Optional[str] = Field(
        None, description="Base64-encoded Grad-CAM heatmap overlay (only computed for positives)"
    )
    gradcam_class_name: Optional[str] = Field(
        None, description="Pathology explained by Grad-CAM heatmap, or 'No evidence' if no positive"
    )
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    confidence_threshold: float = Field(
        default=0.5, description="Decision threshold benchmark"
    )
    disclaimer: str = Field(
        default="Research prototype. Not for clinical use.",
        description="Mandatory medical research disclaimer"
    )


class ModelComparisonResponse(BaseModel):
    """Response for DenseNet-121 vs ResNet-50 side-by-side comparison"""

    densenet_results: ClassificationResponse
    resnet_results: ClassificationResponse
    comparison_summary: Optional[Dict[str, Any]] = Field(
        None, description="Side-by-side metric comparison"
    )


class DicomMetadata(BaseModel):
    """DICOM metadata fields"""

    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    patient_birth_date: Optional[str] = None
    patient_sex: Optional[str] = None
    study_date: Optional[str] = None
    study_time: Optional[str] = None
    study_description: Optional[str] = None
    series_description: Optional[str] = None
    modality: Optional[str] = None
    manufacturer: Optional[str] = None
    manufacturer_model_name: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    pixel_spacing: Optional[List[float]] = None
    bits_allocated: Optional[int] = None
    bits_stored: Optional[int] = None
    photometric_interpretation: Optional[str] = None
    acquisition_date: Optional[str] = None
    acquisition_time: Optional[str] = None
    institution_name: Optional[str] = None
    referring_physician_name: Optional[str] = None


class ImageInfo(BaseModel):
    """Type-safe model for image conversion information"""

    original_shape: List[int] = Field(description="Original DICOM image dimensions")
    converted_format: str = Field(
        description="Format of the converted image (e.g., 'JPEG')"
    )
    converted_size: List[int] = Field(
        description="Size of converted image [width, height]"
    )
    original_dtype: str = Field(description="Original pixel data type")
    pixel_array_min: float = Field(description="Minimum pixel value in original image")
    pixel_array_max: float = Field(description="Maximum pixel value in original image")
    photometric_interpretation: Optional[str] = Field(
        None, description="DICOM photometric interpretation"
    )
    transfer_syntax: Optional[str] = Field(
        None, description="DICOM transfer syntax UID"
    )


class DicomClassificationResponse(BaseModel):
    """Response model for DICOM file classification"""

    predictions: List[DiseaseClassification]
    positive_findings: List[DiseaseClassification] = Field(default_factory=list)
    metadata: DicomMetadata
    image_info: ImageInfo
    model_used: str = "DenseNet-121"
    inference_time_ms: float = 0.0
    gradcam_overlay_base64: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class ErrorResponse(BaseModel):
    detail: str
    error_code: str = "VALIDATION_ERROR"


class DiagnosticReport(BaseModel):
    """Generated diagnostic report based on chest X-ray classification"""

    report: str = Field(description="The generated diagnostic report text")
    summary: str = Field(description="Brief summary of findings")
    recommendations: List[str] = Field(description="List of clinical recommendations")
    severity_level: str = Field(
        description="Overall severity assessment (normal, mild, moderate, severe)"
    )
    generated_at: datetime = Field(default_factory=datetime.now)


class DiagnosticReportResponse(BaseModel):
    """Response model for diagnostic report generation"""

    diagnostic_report: DiagnosticReport
    classifications_used: List[DiseaseClassification] = Field(
        description="Classifications that were analyzed"
    )
    metadata: Optional[DicomMetadata] = Field(
        None, description="DICOM metadata if available"
    )


class DiagnosticReportRequest(BaseModel):
    """Request model for diagnostic report generation"""

    predictions: List[DiseaseClassification] = Field(
        description="Classification results to analyze"
    )
    image_base64: Optional[str] = Field(
        None, description="Optional Base64-encoded chest X-ray image for Gemini Vision multimodal analysis"
    )
    metadata: Optional[DicomMetadata] = Field(
        None, description="DICOM metadata if available"
    )
    image_info: Optional[ImageInfo] = Field(
        None, description="Image technical information"
    )


class ModelInfoResponse(BaseModel):
    """Response with model architecture details"""

    models: List[Dict[str, Any]] = Field(description="Available model architectures")
    dataset: str = Field(default="NIH ChestX-ray14")
    num_classes: int = Field(default=14)
    class_names: List[str] = Field(default_factory=lambda: CHEST_XRAY_CLASSES.copy())
    evaluation_metrics: List[str] = Field(
        default_factory=lambda: [
            "Mean ROC-AUC",
            "Mean Average Precision (mAP)",
            "Per-class Sensitivity",
            "Inference Time (ms)",
        ]
    )
