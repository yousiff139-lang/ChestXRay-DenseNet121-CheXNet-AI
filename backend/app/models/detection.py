import uuid
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from .classification import DiseaseClassification, DicomMetadata, ImageInfo, DiagnosticReport


class BoundingBox(BaseModel):
    """Bounding box in absolute and relative coordinates for HTML5 Canvas."""
    x: float = Field(description="Top-left x in pixels")
    y: float = Field(description="Top-left y in pixels")
    width: float = Field(description="Box width in pixels")
    height: float = Field(description="Box height in pixels")
    # Normalized [0, 1] for responsive canvas overlays
    rel_x: float = Field(default=0.0, description="Normalized x (0.0 - 1.0)")
    rel_y: float = Field(default=0.0, description="Normalized y (0.0 - 1.0)")
    rel_width: float = Field(default=0.0, description="Normalized width (0.0 - 1.0)")
    rel_height: float = Field(default=0.0, description="Normalized height (0.0 - 1.0)")


class ThoracicDetection(BaseModel):
    """Single pathology localized with bounding box."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    class_name: str = Field(description="Detected pathology (e.g. Nodule, Consolidation, Effusion)")
    confidence: float = Field(description="Confidence score [0, 1]", ge=0.0, le=1.0)
    bbox: BoundingBox
    verified: bool = Field(default=True, description="Whether verified by Gemini VLM gatekeeper")
    severity: str = Field(default="moderate", description="Estimated severity: mild, moderate, severe")
    color: str = Field(default="#ef4444", description="Hex color for bounding box UI rendering")
    description: Optional[str] = Field(default=None, description="Radiological description of finding")


class CurativeProtocol(BaseModel):
    """Specific curative protocol and medical treatment recommendation."""
    pathology: str = Field(description="Name of detected thoracic condition")
    clinical_action: str = Field(description="Primary clinical intervention required")
    recommended_medications: List[str] = Field(default_factory=list, description="Common medical therapies / drug classes")
    procedural_intervention: Optional[str] = Field(default=None, description="Invasive or imaging procedure (e.g., thoracentesis, chest tube, CT chest)")
    follow_up_timeline: str = Field(default="Repeat chest X-ray in 4-6 weeks", description="Recommended monitoring interval")


class UnifiedDiagnosisResponse(BaseModel):
    """Comprehensive diagnosis combining DenseNet-121, ResNet-50, YOLO Object Detection, and Gemini VLM."""
    detections: List[ThoracicDetection] = Field(default_factory=list, description="YOLO localized bounding boxes")
    total_detections: int = 0
    predictions: List[DiseaseClassification] = Field(description="DenseNet-121 multi-label classifications")
    positive_findings: List[DiseaseClassification] = Field(default_factory=list, description="Positive findings from DenseNet-121")
    resnet_predictions: Optional[List[DiseaseClassification]] = Field(default=None, description="ResNet-50 comparative predictions")
    
    # Visualizations
    gradcam_overlay_base64: Optional[str] = Field(default=None, description="DenseNet-121 Grad-CAM visual heatmap overlay")
    annotated_overlay_base64: Optional[str] = Field(default=None, description="Image with YOLO bounding boxes burned in")
    
    # Gemini VLM Insights
    gemini_audit_summary: Optional[str] = Field(default=None, description="Stage 0 global radiologist assessment")
    diagnostic_report: Optional[DiagnosticReport] = Field(default=None, description="Stage 3 structured clinical report")
    curative_protocols: List[CurativeProtocol] = Field(default_factory=list, description="Specific curative steps for detected conditions")
    
    # Metadata & Telemetry
    image_width: int = 0
    image_height: int = 0
    inference_time_ms: float = 0.0
    model_inference_ms: float = 0.0
    end_to_end_ms: float = 0.0
    metadata: Optional[DicomMetadata] = None
    disclaimer: str = "Research prototype. Not for clinical diagnostic use."
