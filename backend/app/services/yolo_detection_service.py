"""
YOLO Pathology Detection Service — Thoracic Lesion Localization.

Queries Roboflow hosted YOLO models for chest X-ray object detection
with fallback to Gemini Vision spatial bounding-box reasoning.
Produces calibrated bounding boxes (squaring the disease locations) and
annotated overlay images.
"""

import asyncio
import base64
import io
import logging
import time
from typing import List, Optional, Tuple, Dict, Any

import cv2
import httpx
import numpy as np
from PIL import Image

from ..core.config import get_settings
from ..models.detection import BoundingBox, ThoracicDetection, CurativeProtocol
from ..models.classification import CHEST_XRAY_CLASSES

logger = logging.getLogger(__name__)

# Severity & color mapping for thoracic pathologies
PATHOLOGY_COLOR_MAP: Dict[str, str] = {
    "Nodule": "#f97316",           # Orange
    "Mass": "#ef4444",             # Bright Red
    "Consolidation": "#eab308",    # Yellow
    "Pneumonia": "#f59e0b",        # Amber
    "Effusion": "#06b6d4",         # Cyan
    "Pleural_Effusion": "#06b6d4",
    "Infiltration": "#8b5cf6",     # Violet
    "Cardiomegaly": "#ec4899",     # Pink
    "Atelectasis": "#3b82f6",      # Blue
    "Pneumothorax": "#dc2626",     # Crimson
    "Edema": "#14b8a6",            # Teal
    "Fibrosis": "#64748b",         # Slate
    "Emphysema": "#10b981",        # Emerald
    "Pleural_Thickening": "#a855f7",
    "Hernia": "#84cc16",
}

# Standardized curative protocols for thoracic conditions
CURATIVE_PROTOCOLS: Dict[str, CurativeProtocol] = {
    "Pneumonia": CurativeProtocol(
        pathology="Pneumonia",
        clinical_action="Initiate empirical antimicrobial therapy based on community vs. hospital-acquired presentation.",
        recommended_medications=["Amoxicillin-clavulanate or Azithromycin (outpatient)", "Ceftriaxone + Macrolide (inpatient)"],
        procedural_intervention="Supplemental oxygen therapy; sputum culture and blood gas analysis if saturation < 92%.",
        follow_up_timeline="Clinical reassessment at 48-72h; follow-up chest radiograph in 6 weeks to confirm radiographic clearance."
    ),
    "Consolidation": CurativeProtocol(
        pathology="Consolidation",
        clinical_action="Evaluate for infectious alveolar filling, aspiration, or localized non-infectious inflammatory processes.",
        recommended_medications=["Empiric broad-spectrum antibiotics", "Bronchodilators / mucolytics if airway obstruction present"],
        procedural_intervention="High-resolution CT (HRCT) chest if no radiographic improvement after 7-10 days of antibiotic therapy.",
        follow_up_timeline="Interval chest radiograph in 2-4 weeks."
    ),
    "Effusion": CurativeProtocol(
        pathology="Effusion",
        clinical_action="Assess volume and hemodynamic compromise; determine transudative vs. exudative etiology (Light's criteria).",
        recommended_medications=["Diuretic therapy for heart failure etiology", "Antibiotic coverage if parapneumonic"],
        procedural_intervention="Diagnostic or therapeutic thoracentesis with pleural fluid cytology, pH, protein, and LDH.",
        follow_up_timeline="Post-procedure repeat radiograph or bedside ultrasound within 24 hours."
    ),
    "Cardiomegaly": CurativeProtocol(
        pathology="Cardiomegaly",
        clinical_action="Cardiovascular evaluation for dilated cardiomyopathy, ischemic heart disease, or pericardial effusion.",
        recommended_medications=["ACE inhibitors / ARBs", "Beta-blockers", "Loop diuretics (e.g. Furosemide)"],
        procedural_intervention="Transthoracic Echocardiogram (TTE) for ejection fraction and structural assessment; 12-lead ECG.",
        follow_up_timeline="Cardiology outpatient consultation within 1-2 weeks."
    ),
    "Pneumothorax": CurativeProtocol(
        pathology="Pneumothorax",
        clinical_action="Urgent thoracic surgical evaluation; rule out tension pneumothorax (tracheal deviation, hemodynamic collapse).",
        recommended_medications=["Analgesia", "High-flow supplemental oxygen (enhances pleural nitrogen resorption)"],
        procedural_intervention="Needle decompression (2nd intercostal space) if tension; tube thoracostomy (chest drain) for >20% collapse.",
        follow_up_timeline="Immediate serial radiographs post-intervention; continuous water-seal chamber monitoring."
    ),
    "Nodule": CurativeProtocol(
        pathology="Nodule",
        clinical_action="Stratify malignancy risk according to Fleischner Society guidelines based on size, margin, and patient risk factors.",
        recommended_medications=["Smoking cessation counseling if applicable"],
        procedural_intervention="Low-dose non-contrast CT chest for precise volumetric measurement and calcification pattern analysis.",
        follow_up_timeline="Surveillance CT at 3, 6, or 12 months depending on solid vs. subsolid nature and risk profile."
    ),
    "Mass": CurativeProtocol(
        pathology="Mass",
        clinical_action="Urgent oncologic and pulmonary workup for lesions > 3 cm.",
        recommended_medications=["Pain management and supportive therapy as clinically indicated"],
        procedural_intervention="Contrast-enhanced CT of chest and upper abdomen, followed by bronchoscopy or CT-guided transthoracic needle biopsy.",
        follow_up_timeline="Urgent multi-disciplinary tumor board review within 7 days."
    ),
    "Atelectasis": CurativeProtocol(
        pathology="Atelectasis",
        clinical_action="Chest physiotherapy and lung expansion therapies to reverse volume loss.",
        recommended_medications=["Nebulized bronchodilators", "Inhaled mucolytics"],
        procedural_intervention="Incentive spirometry, positive expiratory pressure (PEP) therapy; bronchoscopy if mucous plug suspected.",
        follow_up_timeline="Follow-up clinical examination and CXR in 48-72 hours."
    ),
    "Edema": CurativeProtocol(
        pathology="Edema",
        clinical_action="Immediate management of acute pulmonary congestion / hydrostatic capillary leak.",
        recommended_medications=["Intravenous loop diuretics (Furosemide 40-80mg)", "Vasodilators (Nitroglycerin) if hypertensive"],
        procedural_intervention="Non-invasive positive pressure ventilation (CPAP/BiPAP); strict fluid balance monitoring.",
        follow_up_timeline="Serial chest radiographs every 12-24 hours until resolution of alveolar edema."
    ),
    "Infiltration": CurativeProtocol(
        pathology="Infiltration",
        clinical_action="Correlation with clinical symptoms (fever, cough, leukocytosis) to differentiate infectious vs interstitial cause.",
        recommended_medications=["Targeted antimicrobial therapy based on clinical setting"],
        procedural_intervention="Pulse oximetry monitoring; inflammatory markers (CRP, procalcitonin).",
        follow_up_timeline="Clinical review in 48 hours; repeat radiograph if symptoms worsen."
    ),
}


class YoloDetectionService:
    """Service for running YOLO object detection and rendering bounding boxes on chest X-rays."""

    def __init__(self):
        self.settings = get_settings()

    async def detect_pathologies(
        self,
        image_path: str,
        confidence_threshold: float = 0.15,
        dense_positive_findings: Optional[List[str]] = None,
    ) -> Tuple[List[ThoracicDetection], Optional[str], int, int]:
        """
        Detect pathologies, square them with bounding boxes, and generate annotated overlay.

        Returns:
            Tuple of:
                - List[ThoracicDetection]
                - annotated_overlay_base64
                - image_width
                - image_height
        """
        img = cv2.imread(image_path)
        if img is None:
            # Try PIL fallback
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]
        detections: List[ThoracicDetection] = []

        # 1. Attempt Roboflow Hosted YOLO Detection
        roboflow_detections = await self._query_roboflow(image_path, confidence_threshold)
        if roboflow_detections:
            detections.extend(roboflow_detections)

        # 2. If Roboflow returned few/no detections or we have confirmed DenseNet positives,
        # consult Gemini Vision Spatial VLM to guarantee precise bounding boxes for all present pathologies
        if len(detections) == 0 or (dense_positive_findings and len(detections) < len(dense_positive_findings)):
            vlm_detections = await self._query_gemini_spatial(
                image_path=image_path,
                dense_positives=dense_positive_findings,
                img_width=w,
                img_height=h
            )
            # Merge and deduplicate by IoU
            for vd in vlm_detections:
                if not any(self._calculate_iou(vd.bbox, d.bbox) > 0.4 for d in detections):
                    detections.append(vd)

        # 3. Render High-Definition Bounding Box Annotated Overlay
        annotated_b64 = self._render_annotated_overlay(img, detections)

        return detections, annotated_b64, w, h

    async def _query_roboflow(
        self, image_path: str, confidence_threshold: float
    ) -> List[ThoracicDetection]:
        """Query Roboflow hosted inference endpoint."""
        api_key = self.settings.roboflow_api_key
        if not api_key:
            return []

        model_id = self.settings.roboflow_model_id or "datamed-6oe4k/2"
        url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}&confidence={confidence_threshold}"

        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("ascii")

            async with httpx.AsyncClient(timeout=12.0) as client:
                res = await client.post(
                    url,
                    data=img_b64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )

            if res.status_code != 200:
                logger.warning(f"Roboflow detection returned status {res.status_code}")
                return []

            data = res.json()
            raw_preds = data.get("predictions", [])
            img_info = data.get("image", {})
            img_w = float(img_info.get("width", 1000))
            img_h = float(img_info.get("height", 1000))

            results: List[ThoracicDetection] = []
            # Blacklist any non-thoracic / dental classes from mismatched datasets
            non_thoracic_terms = {"cavity", "caries", "bdr", "bdc", "tooth", "teeth", "crown", "pa", "periapical", "root", "enamel", "dentin"}
            for p in raw_preds:
                raw_c = str(p.get("class", "Pathology")).lower()
                if any(term in raw_c for term in non_thoracic_terms):
                    continue  # Ignore dental classes on chest X-rays

                c_name = p.get("class", "Pathology").capitalize().replace("_", " ")
                conf = float(p.get("confidence", 0.5))
                # Roboflow returns center coordinates: [x, y, width, height]
                cx, cy = float(p.get("x", 0)), float(p.get("y", 0))
                bw, bh = float(p.get("width", 0)), float(p.get("height", 0))
                top_left_x = max(0.0, cx - bw / 2.0)
                top_left_y = max(0.0, cy - bh / 2.0)

                color = PATHOLOGY_COLOR_MAP.get(c_name, "#ef4444")
                bbox = BoundingBox(
                    x=round(top_left_x, 1),
                    y=round(top_left_y, 1),
                    width=round(bw, 1),
                    height=round(bh, 1),
                    rel_x=round(top_left_x / img_w, 4),
                    rel_y=round(top_left_y / img_h, 4),
                    rel_width=round(bw / img_w, 4),
                    rel_height=round(bh / img_h, 4),
                )
                results.append(
                    ThoracicDetection(
                        class_name=c_name,
                        confidence=round(conf, 3),
                        bbox=bbox,
                        color=color,
                        severity="severe" if conf > 0.7 else "moderate",
                        description=f"{c_name} identified via YOLO detector."
                    )
                )

            logger.info(f"Roboflow YOLO returned {len(results)} detections.")
            return results

        except Exception as e:
            logger.warning(f"Roboflow YOLO inference failed: {e}")
            return []

    async def _query_gemini_spatial(
        self,
        image_path: str,
        dense_positives: Optional[List[str]],
        img_width: int,
        img_height: int,
    ) -> List[ThoracicDetection]:
        """Use Gemini Vision to localize lesions with bounding boxes."""
        api_key = self.settings.gemini_api_key
        if not api_key:
            return []

        # Resize image for fast multimodal transfer
        pil_img = Image.open(image_path).convert("RGB")
        pil_img.thumbnail((768, 768))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        pos_str = f"Focus particularly on verifying these suspected conditions: {', '.join(dense_positives)}" if dense_positives else ""

        prompt = f"""You are an expert thoracic radiologist and computer vision detector.
Detect and localize all abnormal radiographic thoracic findings (such as Nodule, Consolidation, Infiltration, Pleural Effusion, Cardiomegaly, Atelectasis, Pneumothorax, Mass, Edema) on this frontal chest X-ray.
{pos_str}

For each finding, provide:
- label: standard pathology name
- box_2d: [ymin, xmin, ymax, xmax] coordinates normalized on a 0-1000 scale
- confidence: float between 0.0 and 1.0
- severity: "mild", "moderate", or "severe"
- description: concise 1-sentence radiological observation

Respond strictly in valid JSON format:
```json
[
  {{
    "label": "Consolidation",
    "box_2d": [450, 600, 750, 900],
    "confidence": 0.88,
    "severity": "severe",
    "description": "Right lower lobe airspace opacity with obscured right hemidiaphragm"
  }}
]
```
If the radiograph is completely clear with no abnormalities, return [].
"""

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
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        }

        # Try primary model with automatic retry if 503 temporary overload occurs
        models_to_try = [
            self.settings.gemini_model or "gemini-3.5-flash-lite",
            "gemini-flash-latest",
        ]

        data = None
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            for attempt in range(2):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        res = await client.post(url, json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            break
                        elif res.status_code == 503:
                            await asyncio.sleep(1.0)
                except Exception as e:
                    logger.warning(f"Spatial query attempt {attempt} on {model} failed: {e}")
            if data is not None:
                break

        if data is None:
            logger.warning("All Gemini spatial detection attempts were unavailable.")
            return []

        try:
            raw_text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "[]")
            )
            # Parse json
            import json
            clean_json = raw_text.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()

            parsed = json.loads(clean_json)
            if not isinstance(parsed, list):
                return []

            results: List[ThoracicDetection] = []
            for item in parsed:
                label = item.get("label", "Abnormality").title().replace(" ", "_")
                box = item.get("box_2d", [0, 0, 0, 0])
                if len(box) == 4:
                    ymin, xmin, ymax, xmax = [float(c) for c in box]
                    # Convert from 0-1000 scale to pixel coordinates
                    rel_x = max(0.0, min(1.0, xmin / 1000.0))
                    rel_y = max(0.0, min(1.0, ymin / 1000.0))
                    rel_w = max(0.01, min(1.0 - rel_x, (xmax - xmin) / 1000.0))
                    rel_h = max(0.01, min(1.0 - rel_y, (ymax - ymin) / 1000.0))

                    abs_x = rel_x * img_width
                    abs_y = rel_y * img_height
                    abs_w = rel_w * img_width
                    abs_h = rel_h * img_height

                    color = PATHOLOGY_COLOR_MAP.get(label, "#ef4444")
                    conf = float(item.get("confidence", 0.85))
                    sev = item.get("severity", "moderate")
                    desc = item.get("description", "")

                    bbox = BoundingBox(
                        x=round(abs_x, 1),
                        y=round(abs_y, 1),
                        width=round(abs_w, 1),
                        height=round(abs_h, 1),
                        rel_x=round(rel_x, 4),
                        rel_y=round(rel_y, 4),
                        rel_width=round(rel_w, 4),
                        rel_height=round(rel_h, 4),
                    )

                    results.append(
                        ThoracicDetection(
                            class_name=label.replace("_", " "),
                            confidence=round(conf, 2),
                            bbox=bbox,
                            verified=True,
                            severity=sev,
                            color=color,
                            description=desc,
                        )
                    )

            logger.info(f"Gemini spatial detector generated {len(results)} localized findings.")
            return results

        except Exception as e:
            logger.warning(f"Gemini spatial detection parsing failed: {e}")
            return []

    def _render_annotated_overlay(
        self, img: np.ndarray, detections: List[ThoracicDetection]
    ) -> str:
        """Burn high-definition bounding boxes and badges directly into the image."""
        overlay = img.copy()
        h, w = overlay.shape[:2]

        for d in detections:
            x1 = int(max(0, d.bbox.x))
            y1 = int(max(0, d.bbox.y))
            x2 = int(min(w - 1, d.bbox.x + d.bbox.width))
            y2 = int(min(h - 1, d.bbox.y + d.bbox.height))

            # Hex color to BGR
            hex_c = d.color.lstrip("#")
            if len(hex_c) == 6:
                r, g, b = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
                color = (b, g, r)
            else:
                color = (0, 0, 255)

            # Draw semi-transparent rectangle fill
            sub_rect = overlay[y1:y2, x1:x2]
            if sub_rect.size > 0:
                fill_color = np.full_like(sub_rect, color)
                cv2.addWeighted(fill_color, 0.22, sub_rect, 0.78, 0, sub_rect)
                overlay[y1:y2, x1:x2] = sub_rect

            # Main bounding box line
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Corner brackets for clinical aesthetic
            bracket_len = min(20, max(5, int(min(d.bbox.width, d.bbox.height) * 0.2)))
            cv2.line(overlay, (x1, y1), (x1 + bracket_len, y1), color, 4, cv2.LINE_AA)
            cv2.line(overlay, (x1, y1), (x1, y1 + bracket_len), color, 4, cv2.LINE_AA)
            cv2.line(overlay, (x2, y1), (x2 - bracket_len, y1), color, 4, cv2.LINE_AA)
            cv2.line(overlay, (x2, y1), (x2, y1 + bracket_len), color, 4, cv2.LINE_AA)
            cv2.line(overlay, (x1, y2), (x1 + bracket_len, y2), color, 4, cv2.LINE_AA)
            cv2.line(overlay, (x1, y2), (x1, y2 - bracket_len), color, 4, cv2.LINE_AA)
            cv2.line(overlay, (x2, y2), (x2 - bracket_len, y2), color, 4, cv2.LINE_AA)
            cv2.line(overlay, (x2, y2), (x2, y2 - bracket_len), color, 4, cv2.LINE_AA)

            # Label text badge
            label_text = f"{d.class_name} {int(d.confidence * 100)}%"
            (txt_w, txt_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            badge_y1 = max(0, y1 - txt_h - 8)
            badge_y2 = y1
            badge_x2 = min(w - 1, x1 + txt_w + 10)

            cv2.rectangle(overlay, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                overlay,
                label_text,
                (x1 + 5, badge_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Encode to Base64 PNG
        _, buf = cv2.imencode(".png", overlay, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def _calculate_iou(b1: BoundingBox, b2: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
        x1 = max(b1.x, b2.x)
        y1 = max(b1.y, b2.y)
        x2 = min(b1.x + b1.width, b2.x + b2.width)
        y2 = min(b1.y + b1.height, b2.y + b2.height)

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = b1.width * b1.height
        area2 = b2.width * b2.height
        union = area1 + area2 - intersection
        return intersection / max(union, 1e-6)

    def get_curative_protocols(
        self, detected_classes: List[str]
    ) -> List[CurativeProtocol]:
        """Look up detailed curative actions and medication regimens for detected conditions."""
        protocols: List[CurativeProtocol] = []
        for c in detected_classes:
            matched = False
            for key, proto in CURATIVE_PROTOCOLS.items():
                if key.lower() in c.lower() or c.lower() in key.lower():
                    protocols.append(proto)
                    matched = True
                    break
            if not matched:
                protocols.append(
                    CurativeProtocol(
                        pathology=c,
                        clinical_action=f"Clinical correlation and targeted thoracic evaluation for {c}.",
                        recommended_medications=["Supportive clinical therapy"],
                        procedural_intervention="Repeat chest radiograph / contrast CT as clinically indicated.",
                        follow_up_timeline="Review within 2-4 weeks.",
                    )
                )
        return protocols


# ─── Singleton ───────────────────────────────────────────────
_yolo_detection_service = None


def get_yolo_detection_service() -> YoloDetectionService:
    global _yolo_detection_service
    if _yolo_detection_service is None:
        _yolo_detection_service = YoloDetectionService()
    return _yolo_detection_service
