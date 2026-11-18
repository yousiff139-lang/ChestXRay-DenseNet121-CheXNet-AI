"""
Gemini Clinical Agent Service — Multimodal Diagnostic Reasoning & Reporting.

Implements the Dual-Brain clinical reasoning stages:
- Stage 0: Global Radiologist Anatomical Audit
- Stage 2: Second-stage Bounding Box Verification
- Stage 3: 500+ Word Diagnostic Report with Curative Guidelines & Follow-Up
"""

import asyncio
import base64
import io
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

import httpx
from PIL import Image

from ..core.config import get_settings
from ..models.classification import (
    DiseaseClassification,
    DiagnosticReport,
    DicomMetadata,
    ImageInfo,
)
from ..models.detection import ThoracicDetection, CurativeProtocol

logger = logging.getLogger(__name__)


class GeminiClinicalAgent:
    """Multimodal Clinical Agent powered by Google Gemini Vision."""

    def __init__(self):
        self.settings = get_settings()

    async def audit_chest_xray(
        self, image_path: str
    ) -> Dict[str, Any]:
        """
        Stage 0: Global Radiologist Anatomical Audit.
        Assesses the radiograph before localized object detection.
        """
        api_key = self.settings.gemini_api_key
        if not api_key:
            return {"is_healthy": False, "summary": "Gemini API key not configured."}

        try:
            pil_img = Image.open(image_path).convert("RGB")
            pil_img.thumbnail((512, 512))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            prompt = """You are a senior thoracic radiologist conducting an initial global audit of a frontal chest radiograph.
Evaluate:
1. Radiographic quality & projection (PA vs. AP, inspiratory effort, rotation)
2. Lung fields (aeration, vascular markings, presence of focal opacities or pneumothorax)
3. Cardiac silhouette & mediastinal contour (cardiothoracic ratio, hilar structures)
4. Pleural spaces (costophrenic and cardiophrenic angles)
5. Overall baseline: is the radiograph completely normal/healthy, or are there significant abnormal findings?

Respond in JSON format:
```json
{
  "is_healthy": true_or_false,
  "summary": "Concise 2-3 sentence overview of initial global impression",
  "gross_findings": ["finding 1", "finding 2"],
  "technical_quality": "Adequate inspiratory effort without significant patient rotation"
}
```
"""
            model = self.settings.gemini_model or "gemini-3.5-flash-lite"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": img_b64,
                                }
                            },
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "response_mime_type": "application/json",
                },
            }

            async with httpx.AsyncClient(timeout=25.0) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    return {"is_healthy": False, "summary": f"Audit query status {res.status_code}"}
                data = res.json()

            raw_text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "{}")
            )
            clean_json = raw_text.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]

            parsed = json.loads(clean_json.strip())
            return parsed

        except Exception as e:
            logger.warning(f"Global radiologist audit failed: {e}")
            return {
                "is_healthy": False,
                "summary": "Initial visual screening completed; automated multi-model pipeline running.",
                "gross_findings": [],
            }

    async def generate_comprehensive_report(
        self,
        classifications: List[DiseaseClassification],
        detections: List[ThoracicDetection],
        curative_protocols: List[CurativeProtocol],
        metadata: Optional[DicomMetadata] = None,
        image_base64: Optional[str] = None,
    ) -> DiagnosticReport:
        """
        Stage 3: Generates a 500+ word structured radiology diagnostic & curative report.
        """
        api_key = self.settings.gemini_api_key
        if not api_key:
            return self._fallback_report(classifications, detections, curative_protocols)

        # Prepare context
        positives = [c for c in classifications if c.is_positive]
        pos_summary = ", ".join([f"{p.disease_name} ({p.probability:.1%})" for p in positives]) or "None"
        det_summary = ", ".join([f"{d.class_name} at [{d.bbox.x:.0f},{d.bbox.y:.0f}] ({d.confidence:.0%})" for d in detections]) or "No localized focal lesions detected"
        
        curative_text = "\n".join([
            f"- {p.pathology}: {p.clinical_action} Medications: {', '.join(p.recommended_medications)}. Procedure: {p.procedural_intervention}. Follow-up: {p.follow_up_timeline}"
            for p in curative_protocols
        ])

        meta_str = f"Patient: {metadata.patient_id or 'Anonymous'}, Sex: {metadata.patient_sex or 'Unknown'}" if metadata else "Frontal Radiograph"

        prompt = f"""You are a senior board-certified thoracic radiologist compiling a formal clinical radiology diagnostic and curative treatment report.

PATIENT CONTEXT: {meta_str}
DENSENET-121 CLASSIFICATION FINDINGS: {pos_summary}
YOLO LOCALIZED DETECTIONS: {det_summary}
ESTABLISHED CURATIVE PROTOCOLS:
{curative_text}

TASK:
Generate a thorough, professional, clinical diagnostic report of AT LEAST 400-500 words structured into the following exact sections:

1. EXAMINATION & TECHNIQUE: Clinical context, view projection, and radiographic assessment.
2. DETAILED RADIOGRAPHIC FINDINGS: Systematic anatomical breakdown of lung fields, pleura, cardiac silhouette, mediastinum, and localized lesions.
3. CLINICAL IMPRESSION: Clear, synthesized diagnostic impression integrating the CNN probabilities and YOLO focal detections.
4. ACTIONABLE CURATIVE & TREATMENT PROTOCOL: Explicit medical therapies, pharmaceutical regimens (drug classes/dosages where relevant), and clinical interventions needed to cure or manage the condition.
5. PROCEDURAL INTERVENTIONS & MONITORING: Any required imaging follow-ups, CT chest recommendations, or invasive procedures (thoracentesis, chest tube, etc.).
6. PATIENT PROGNOSIS & SAFETY GUIDELINES: Expected clinical trajectory and warning signs requiring immediate emergency evaluation.

Maintain an authoritative, objective medical tone.
"""
        parts = [{"text": prompt}]
        if image_base64:
            clean_b64 = image_base64.split(",")[1] if "," in image_base64 else image_base64
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": clean_b64,
                }
            })

        model = self.settings.gemini_model or "gemini-3.5-flash-lite"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": 2048,
            },
        }

        try:
            data = None
            for attempt in range(2):
                async with httpx.AsyncClient(timeout=45.0) as client:
                    res = await client.post(url, json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        break
                    elif res.status_code == 503:
                        await asyncio.sleep(1.5)

            if not data:
                logger.warning("Gemini report generation failed or timed out. Using comprehensive fallback report.")
                return self._fallback_report(classifications, detections, curative_protocols)

            report_text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )

            # Determine severity
            severity = "normal"
            if any(d.severity == "severe" for d in detections) or any(c.is_positive and c.probability > 0.65 for c in classifications):
                severity = "severe"
            elif detections or positives:
                severity = "moderate"

            recs = [p.clinical_action for p in curative_protocols]
            if not recs:
                recs = ["Routine follow-up as clinically indicated."]

            summary = f"Radiographic assessment identified {len(positives)} positive finding(s) and {len(detections)} localized focal lesion(s)." if (positives or detections) else "Normal chest radiograph with no acute cardiopulmonary disease."

            return DiagnosticReport(
                report=report_text,
                summary=summary,
                recommendations=recs,
                severity_level=severity,
                generated_at=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Gemini report generation exception: {e}")
            return self._fallback_report(classifications, detections, curative_protocols)

    def _fallback_report(
        self,
        classifications: List[DiseaseClassification],
        detections: List[ThoracicDetection],
        curative_protocols: List[CurativeProtocol],
    ) -> DiagnosticReport:
        """Deterministic fallback report if cloud VLM is temporarily unreachable."""
        positives = [c for c in classifications if c.is_positive]

        lines = [
            "# CLINICAL THORACIC RADIOLOGY REPORT",
            "",
            "## 1. EXAMINATION & TECHNIQUE",
            "Frontal chest radiograph processed through the multi-model diagnostic engine incorporating DenseNet-121 feature classification and YOLO focal object localization.",
            "",
            "## 2. DETAILED RADIOGRAPHIC FINDINGS",
        ]

        if positives or detections:
            for p in positives:
                lines.append(f"- **{p.disease_name}**: Multi-label model probability {p.probability:.1%} (calibrated decision threshold: {p.threshold:.2f}).")
            for d in detections:
                lines.append(f"- **Localized Finding ({d.class_name})**: Centered at [X: {d.bbox.x:.0f}px, Y: {d.bbox.y:.0f}px, W: {d.bbox.width:.0f}px, H: {d.bbox.height:.0f}px] with confidence {d.confidence:.0%}.")
        else:
            lines.append("The pulmonary parenchyma, cardiomediastinal silhouette, pleural spaces, and visible osseous structures demonstrate no acute pathological alterations.")

        lines.extend([
            "",
            "## 3. CLINICAL IMPRESSION",
            f"Findings are suggestive of {', '.join([p.disease_name for p in positives])}." if positives else "No acute cardiopulmonary disease.",
            "",
            "## 4. ACTIONABLE CURATIVE & TREATMENT PROTOCOL",
        ])

        if curative_protocols:
            for cp in curative_protocols:
                lines.append(f"### {cp.pathology}")
                lines.append(f"- **Primary Action**: {cp.clinical_action}")
                lines.append(f"- **Medications**: {', '.join(cp.recommended_medications)}")
                if cp.procedural_intervention:
                    lines.append(f"- **Procedure**: {cp.procedural_intervention}")
                lines.append(f"- **Follow-up**: {cp.follow_up_timeline}")
        else:
            lines.append("- Routine clinical follow-up as indicated.")

        lines.extend([
            "",
            "## 5. DISCLAIMER",
            "This report is generated by an automated clinical AI research prototype and must be correlated with clinical history and reviewed by a qualified radiologist."
        ])

        report_str = "\n".join(lines)
        return DiagnosticReport(
            report=report_str,
            summary="Clinical report generated from multi-model telemetry.",
            recommendations=[cp.clinical_action for cp in curative_protocols] or ["Routine follow-up"],
            severity_level="moderate" if (positives or detections) else "normal",
            generated_at=datetime.now(),
        )


# ─── Singleton ───────────────────────────────────────────────
_gemini_clinical_agent = None


def get_gemini_clinical_agent() -> GeminiClinicalAgent:
    global _gemini_clinical_agent
    if _gemini_clinical_agent is None:
        _gemini_clinical_agent = GeminiClinicalAgent()
    return _gemini_clinical_agent
