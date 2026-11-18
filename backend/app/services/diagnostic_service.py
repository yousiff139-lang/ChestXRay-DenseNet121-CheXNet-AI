"""
Diagnostic Report Service — AI-Powered Chest X-Ray Reports.

Generates structured clinical diagnostic reports from chest X-ray
classification results using the Gemini API.
"""

import logging
from typing import List, Optional

import httpx

from ..core.config import get_settings
from ..models.classification import (
    CHEST_XRAY_CLASSES,
    DiseaseClassification,
    DiagnosticReport,
    DicomMetadata,
    ImageInfo,
)

logger = logging.getLogger(__name__)


class DiagnosticReportService:
    """Service for generating AI-powered diagnostic reports from classifications."""

    def __init__(self):
        self.settings = get_settings()

    async def generate_diagnostic_report(
        self,
        classifications: List[DiseaseClassification],
        metadata: Optional[DicomMetadata] = None,
        image_info: Optional[ImageInfo] = None,
        image_base64: Optional[str] = None,
    ) -> DiagnosticReport:
        """
        Generate a comprehensive diagnostic report from classification results.

        Args:
            classifications: List of disease classification results
            metadata: Optional DICOM metadata
            image_info: Optional image technical information
            image_base64: Optional Base64-encoded image for Gemini Vision multimodal analysis

        Returns:
            DiagnosticReport with structured findings
        """
        # Build findings summary
        positive_findings = [c for c in classifications if c.is_positive]
        negative_findings = [c for c in classifications if not c.is_positive]

        # Format findings for the prompt
        findings_text = self._format_findings(positive_findings, negative_findings)
        metadata_text = self._format_metadata(metadata) if metadata else "No DICOM metadata available."

        # Try AI-generated report first
        try:
            report = await self._generate_with_gemini(
                findings_text, metadata_text, positive_findings, image_base64=image_base64
            )
            return report
        except Exception as e:
            logger.warning(f"AI report generation failed: {e}. Using template report.")
            return self._generate_template_report(positive_findings, negative_findings)

    def _format_findings(
        self,
        positive: List[DiseaseClassification],
        negative: List[DiseaseClassification],
    ) -> str:
        lines = []
        if positive:
            lines.append("POSITIVE FINDINGS:")
            for f in sorted(positive, key=lambda x: x.probability, reverse=True):
                lines.append(f"  - {f.disease_name}: {f.probability:.1%} probability")
        else:
            lines.append("NO POSITIVE FINDINGS DETECTED")

        lines.append(f"\nNEGATIVE FINDINGS ({len(negative)} pathologies screened negative):")
        for f in negative:
            lines.append(f"  - {f.disease_name}: {f.probability:.1%}")

        return "\n".join(lines)

    def _format_metadata(self, metadata: DicomMetadata) -> str:
        parts = []
        if metadata.patient_sex:
            parts.append(f"Sex: {metadata.patient_sex}")
        if metadata.modality:
            parts.append(f"Modality: {metadata.modality}")
        if metadata.study_description:
            parts.append(f"Study: {metadata.study_description}")
        if metadata.institution_name:
            parts.append(f"Institution: {metadata.institution_name}")
        return "; ".join(parts) if parts else "Limited metadata available."

    async def _generate_with_gemini(
        self,
        findings_text: str,
        metadata_text: str,
        positive_findings: List[DiseaseClassification],
        image_base64: Optional[str] = None,
    ) -> DiagnosticReport:
        """Generate diagnostic report using Gemini Vision multimodal API."""
        api_key = self.settings.gemini_api_key
        if not api_key:
            raise ValueError("Gemini API key not configured")

        prompt = f"""You are an expert thoracic radiologist reviewing a chest X-ray.
Review this case alongside the quantitative DenseNet-121 AI classification results.

PATIENT METADATA:
{metadata_text}

AI CLASSIFICATION RESULTS:
{findings_text}

Generate a professional, structured radiology report with the following sections:
1. VISUAL RADIOLOGY ASSESSMENT: Inspect the lung fields, cardiac silhouette, costophrenic angles, and mediastinum. Note any opacities, consolidations, effusions, or medical lines/tubes.
2. MODEL CONCORDANCE: State whether visual radiographic inspection supports the DenseNet-121 classification findings.
3. IMPRESSION: Summary clinical impression based on the synthesis of the image and AI model predictions.
4. RECOMMENDATIONS: Appropriate follow-up actions and clinical correlation.

Keep the tone professional, objective, and clinically clear.
"""

        model = self.settings.gemini_model or "gemini-3.5-flash-lite"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        parts = [{"text": prompt}]

        # Add image data if provided for multimodal Gemini Vision analysis
        if image_base64:
            clean_b64 = image_base64
            if "," in clean_b64:
                clean_b64 = clean_b64.split(",")[1]
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": clean_b64,
                }
            })

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json={
                    "contents": [{"parts": parts}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 1024,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        report_text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "Report generation failed.")
        )

        # Determine severity
        severity = self._assess_severity(positive_findings)

        # Build recommendations
        recommendations = self._build_recommendations(positive_findings)

        # Build summary
        if positive_findings:
            summary_parts = [f.disease_name for f in positive_findings[:3]]
            summary = f"AI analysis suggests: {', '.join(summary_parts)}. Clinical correlation recommended."
        else:
            summary = "No significant thoracic abnormalities detected by AI analysis."

        return DiagnosticReport(
            report=report_text,
            summary=summary,
            recommendations=recommendations,
            severity_level=severity,
        )

    def _generate_template_report(
        self,
        positive: List[DiseaseClassification],
        negative: List[DiseaseClassification],
    ) -> DiagnosticReport:
        """Generate a template-based report when AI is unavailable."""
        if not positive:
            report = (
                "CHEST X-RAY AI ANALYSIS REPORT\n\n"
                "FINDINGS:\n"
                "The AI classification model (DenseNet-121) did not identify any significant\n"
                "thoracic abnormalities across all 14 screened pathologies.\n\n"
                "IMPRESSION:\n"
                "No acute cardiopulmonary disease suggested by AI analysis.\n\n"
                "NOTE: This is an AI-assisted screening tool. Clinical correlation is essential."
            )
            return DiagnosticReport(
                report=report,
                summary="No significant findings detected.",
                recommendations=["Routine follow-up as clinically indicated."],
                severity_level="normal",
            )

        findings_lines = []
        for f in sorted(positive, key=lambda x: x.probability, reverse=True):
            findings_lines.append(
                f"  • {f.disease_name} — probability {f.probability:.1%}"
            )

        report = (
            "CHEST X-RAY AI ANALYSIS REPORT\n\n"
            "FINDINGS:\n"
            f"The DenseNet-121 classification model identified {len(positive)} "
            f"potential abnormalit{'y' if len(positive) == 1 else 'ies'}:\n"
            + "\n".join(findings_lines)
            + "\n\n"
            "IMPRESSION:\n"
            f"AI-assisted findings suggestive of {', '.join(f.disease_name for f in positive[:3])}. "
            "Clinical and radiological correlation recommended.\n\n"
            "NOTE: This is an AI-assisted screening tool and should not replace "
            "professional radiologist interpretation."
        )

        severity = self._assess_severity(positive)
        recommendations = self._build_recommendations(positive)
        summary = f"Potential findings: {', '.join(f.disease_name for f in positive[:3])}."

        return DiagnosticReport(
            report=report,
            summary=summary,
            recommendations=recommendations,
            severity_level=severity,
        )

    @staticmethod
    def _assess_severity(
        positive_findings: List[DiseaseClassification],
    ) -> str:
        if not positive_findings:
            return "normal"

        max_prob = max(f.probability for f in positive_findings)
        severe_conditions = {"Pneumothorax", "Hernia", "Mass", "Consolidation", "Edema"}
        has_severe = any(f.disease_name in severe_conditions for f in positive_findings)

        if has_severe and max_prob > 0.7:
            return "severe"
        elif len(positive_findings) >= 3 or max_prob > 0.6:
            return "moderate"
        else:
            return "mild"

    @staticmethod
    def _build_recommendations(
        positive_findings: List[DiseaseClassification],
    ) -> List[str]:
        recommendations = []

        if not positive_findings:
            return ["Routine follow-up as clinically indicated."]

        recommendations.append(
            "Clinical correlation with patient history and physical examination is recommended."
        )

        disease_names = {f.disease_name for f in positive_findings}

        if "Pneumothorax" in disease_names:
            recommendations.append(
                "Urgent evaluation for pneumothorax; consider CT chest if clinically indicated."
            )
        if "Mass" in disease_names or "Nodule" in disease_names:
            recommendations.append(
                "Follow-up CT chest recommended for further characterization of pulmonary lesion."
            )
        if "Effusion" in disease_names:
            recommendations.append(
                "Consider lateral decubitus view or ultrasound to assess pleural effusion volume."
            )
        if "Cardiomegaly" in disease_names:
            recommendations.append(
                "Echocardiography recommended to evaluate cardiac function."
            )
        if "Pneumonia" in disease_names or "Consolidation" in disease_names:
            recommendations.append(
                "Consider clinical assessment for infectious etiology and follow-up imaging."
            )

        recommendations.append(
            "This AI analysis should be reviewed by a qualified radiologist."
        )

        return recommendations


# ─── Singleton ───────────────────────────────────────────────
_diagnostic_service = None


def get_diagnostic_report_service() -> DiagnosticReportService:
    global _diagnostic_service
    if _diagnostic_service is None:
        _diagnostic_service = DiagnosticReportService()
    return _diagnostic_service
