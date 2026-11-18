export interface DiseaseClassification {
  disease_name: string;
  probability: number;
  raw_probability?: number | null;
  threshold?: number;
  prevalence?: number | null;
  state?: "Positive" | "Uncertain" | "Negative";
  is_positive: boolean;
}

export interface ClassificationResponse {
  predictions: DiseaseClassification[];
  positive_findings: DiseaseClassification[];
  model_used: string;
  inference_time_ms: number;
  model_inference_ms?: number | null;
  end_to_end_ms?: number | null;
  gradcam_overlay_base64?: string | null;
  gradcam_class_name?: string | null;
  image_width?: number;
  image_height?: number;
  confidence_threshold?: number;
  disclaimer?: string;
}

export interface ModelComparisonResponse {
  densenet_results: ClassificationResponse;
  resnet_results: ClassificationResponse;
  comparison_summary?: {
    densenet_inference_ms: number;
    resnet_inference_ms: number;
    densenet_positive_count: number;
    resnet_positive_count: number;
    agreement: Array<{
      disease: string;
      densenet_prob: number;
      resnet_prob: number;
    }>;
    disagreement: Array<{
      disease: string;
      densenet_prob: number;
      resnet_prob: number;
    }>;
  };
}

export interface DicomMetadata {
  patient_id?: string;
  patient_name?: string;
  patient_birth_date?: string;
  patient_sex?: string;
  study_date?: string;
  study_time?: string;
  study_description?: string;
  series_description?: string;
  modality?: string;
  manufacturer?: string;
  manufacturer_model_name?: string;
  rows?: number;
  columns?: number;
  pixel_spacing?: number[];
  bits_allocated?: number;
  bits_stored?: number;
  photometric_interpretation?: string;
  acquisition_date?: string;
  acquisition_time?: string;
  institution_name?: string;
  referring_physician_name?: string;
}

export interface ImageInfo {
  original_shape: number[];
  converted_format: string;
  converted_size: number[];
  original_dtype: string;
  pixel_array_min: number;
  pixel_array_max: number;
  photometric_interpretation?: string;
  transfer_syntax?: string;
}

export interface DicomClassificationResponse {
  predictions: DiseaseClassification[];
  positive_findings: DiseaseClassification[];
  metadata: DicomMetadata;
  image_info: ImageInfo;
  model_used: string;
  inference_time_ms: number;
  model_inference_ms?: number | null;
  end_to_end_ms?: number | null;
  gradcam_overlay_base64?: string | null;
  gradcam_class_name?: string | null;
  image_width?: number;
  image_height?: number;
  confidence_threshold?: number;
  disclaimer?: string;
}

export interface DiagnosticFinding {
  finding_id: string;
  disease_name: string;
  confidence: number;
  severity: "low" | "medium" | "high" | "critical";
  description: string;
  recommendations: string[];
}

export interface DiagnosticReport {
  report_id: string;
  generated_at: string;
  patient_info?: {
    patient_id?: string;
    patient_name?: string;
    age?: number;
    gender?: string;
  };
  clinical_summary: string;
  findings: DiagnosticFinding[];
  impression: string;
  recommendations: string[];
  disclaimer: string;
  urgency_level: "low" | "medium" | "high" | "urgent";
}

export interface DiagnosticReportResponse {
  diagnostic_report: DiagnosticReport;
  classifications_used: DiseaseClassification[];
  metadata?: DicomMetadata | null;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  rel_x: number;
  rel_y: number;
  rel_width: number;
  rel_height: number;
}

export interface ThoracicDetection {
  id: string;
  class_name: string;
  confidence: number;
  bbox: BoundingBox;
  verified: boolean;
  severity: "mild" | "moderate" | "severe";
  color: string;
  description?: string;
}

export interface CurativeProtocol {
  pathology: string;
  clinical_action: string;
  recommended_medications: string[];
  procedural_intervention?: string;
  follow_up_timeline: string;
}

export interface UnifiedDiagnosisResponse {
  detections: ThoracicDetection[];
  total_detections: number;
  predictions: DiseaseClassification[];
  positive_findings: DiseaseClassification[];
  resnet_predictions?: DiseaseClassification[] | null;
  gradcam_overlay_base64?: string | null;
  annotated_overlay_base64?: string | null;
  gemini_audit_summary?: string | null;
  diagnostic_report?: {
    report: string;
    summary: string;
    recommendations: string[];
    severity_level: string;
    generated_at?: string;
  } | null;
  curative_protocols: CurativeProtocol[];
  image_width: number;
  image_height: number;
  inference_time_ms: number;
  model_inference_ms: number;
  end_to_end_ms: number;
  metadata?: DicomMetadata | null;
  disclaimer: string;
}

export interface BenchmarkMetrics {
  dataset: string;
  classes: string[];
  metrics_comparison: {
    densenet_121: {
      architecture: string;
      mean_roc_auc: number;
      mean_ap: number;
      avg_sensitivity: number;
      avg_specificity: number;
      inference_speed_fps: number;
      parameters_million: number;
      key_strength: string;
    };
    resnet_50: {
      architecture: string;
      mean_roc_auc: number;
      mean_ap: number;
      avg_sensitivity: number;
      avg_specificity: number;
      inference_speed_fps: number;
      parameters_million: number;
      key_strength: string;
    };
  };
  per_class_auc: Record<string, { densenet: number; resnet: number }>;
}
