"""
DICOM Preprocessing Service for Chest X-Ray Classification.

Handles DICOM file parsing, metadata extraction, and image conversion
for chest X-ray classification. Simplified from the dental preprocessing
pipeline (no CLAHE/quadrant decomposition needed for classification).
"""

import logging
import os
import tempfile
from typing import Tuple

import numpy as np
import pydicom
from PIL import Image
from fastapi import HTTPException

from ..models.classification import DicomMetadata, ImageInfo

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Service for DICOM preprocessing and image preparation."""

    def parse_dicom_metadata(self, dicom_file_path: str) -> DicomMetadata:
        """
        Parse DICOM metadata from file.

        Args:
            dicom_file_path: Path to the DICOM file

        Returns:
            DicomMetadata: Parsed metadata
        """
        try:
            dicom_data = pydicom.dcmread(dicom_file_path)

            pixel_spacing = None
            if hasattr(dicom_data, "PixelSpacing"):
                pixel_spacing = [float(x) for x in dicom_data.PixelSpacing]

            metadata = DicomMetadata(
                patient_id=getattr(dicom_data, "PatientID", None),
                patient_name=(
                    str(getattr(dicom_data, "PatientName", None))
                    if hasattr(dicom_data, "PatientName")
                    else None
                ),
                patient_birth_date=getattr(dicom_data, "PatientBirthDate", None),
                patient_sex=getattr(dicom_data, "PatientSex", None),
                study_date=getattr(dicom_data, "StudyDate", None),
                study_time=getattr(dicom_data, "StudyTime", None),
                study_description=getattr(dicom_data, "StudyDescription", None),
                series_description=getattr(dicom_data, "SeriesDescription", None),
                modality=getattr(dicom_data, "Modality", None),
                manufacturer=getattr(dicom_data, "Manufacturer", None),
                manufacturer_model_name=getattr(
                    dicom_data, "ManufacturerModelName", None
                ),
                rows=getattr(dicom_data, "Rows", None),
                columns=getattr(dicom_data, "Columns", None),
                pixel_spacing=pixel_spacing,
                bits_allocated=getattr(dicom_data, "BitsAllocated", None),
                bits_stored=getattr(dicom_data, "BitsStored", None),
                photometric_interpretation=getattr(
                    dicom_data, "PhotometricInterpretation", None
                ),
                acquisition_date=getattr(dicom_data, "AcquisitionDate", None),
                acquisition_time=getattr(dicom_data, "AcquisitionTime", None),
                institution_name=getattr(dicom_data, "InstitutionName", None),
                referring_physician_name=(
                    str(getattr(dicom_data, "ReferringPhysicianName", None))
                    if hasattr(dicom_data, "ReferringPhysicianName")
                    else None
                ),
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to parse DICOM metadata: {str(e)}")
            raise HTTPException(
                status_code=400, detail=f"Failed to parse DICOM file: {str(e)}"
            )

    def convert_dicom_to_image(
        self, dicom_file_path: str
    ) -> Tuple[str, ImageInfo]:
        """
        Convert DICOM file to a standard image format for inference.

        Args:
            dicom_file_path: Path to the DICOM file

        Returns:
            Tuple of (converted_image_path, image_info)
        """
        try:
            dicom_data = pydicom.dcmread(dicom_file_path)

            # Apply modality LUT if present
            if (
                hasattr(dicom_data, "ModalityLUTSequence")
                and dicom_data.ModalityLUTSequence
            ):
                try:
                    from pydicom.pixel_data_handlers.util import apply_modality_lut
                    pixel_array = apply_modality_lut(dicom_data.pixel_array, dicom_data)
                except Exception as e:
                    logger.warning(f"Failed to apply modality LUT: {e}")
                    pixel_array = dicom_data.pixel_array
            else:
                pixel_array = dicom_data.pixel_array

            # Apply VOI LUT if present (windowing)
            if hasattr(dicom_data, "VOILUTSequence") and dicom_data.VOILUTSequence:
                try:
                    from pydicom.pixel_data_handlers.util import apply_voi_lut
                    pixel_array = apply_voi_lut(pixel_array, dicom_data)
                except Exception as e:
                    logger.warning(f"Failed to apply VOI LUT: {e}")
            elif hasattr(dicom_data, "WindowCenter") and hasattr(
                dicom_data, "WindowWidth"
            ):
                try:
                    window_center = float(
                        dicom_data.WindowCenter[0]
                        if isinstance(dicom_data.WindowCenter, list)
                        else dicom_data.WindowCenter
                    )
                    window_width = float(
                        dicom_data.WindowWidth[0]
                        if isinstance(dicom_data.WindowWidth, list)
                        else dicom_data.WindowWidth
                    )
                    pixel_array = np.clip(
                        (pixel_array - (window_center - window_width / 2))
                        / window_width
                        * 255,
                        0,
                        255,
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply windowing: {e}")

            # Handle photometric interpretation
            if hasattr(dicom_data, "PhotometricInterpretation"):
                if dicom_data.PhotometricInterpretation == "MONOCHROME1":
                    pixel_array = pixel_array.max() - pixel_array

            # Normalize pixel values to 0-255
            if pixel_array.dtype != np.uint8:
                min_val = pixel_array.min()
                max_val = pixel_array.max()
                if max_val > min_val:
                    pixel_array = (
                        (pixel_array - min_val) / (max_val - min_val) * 255
                    ).astype(np.uint8)
                else:
                    pixel_array = np.full(pixel_array.shape, 128, dtype=np.uint8)

            # Convert to RGB
            if len(pixel_array.shape) == 2:
                rgb_array = np.stack([pixel_array] * 3, axis=-1)
            elif len(pixel_array.shape) == 3:
                if pixel_array.shape[2] == 1:
                    rgb_array = np.repeat(pixel_array, 3, axis=2)
                elif pixel_array.shape[2] == 3:
                    rgb_array = pixel_array
                else:
                    if pixel_array.shape[2] >= 3:
                        rgb_array = pixel_array[:, :, :3]
                    else:
                        rgb_array = np.stack([pixel_array[:, :, 0]] * 3, axis=-1)
            else:
                raise ValueError(f"Unsupported pixel array shape: {pixel_array.shape}")

            # Save as temporary JPEG
            image = Image.fromarray(rgb_array.astype(np.uint8))
            temp_image_fd, temp_image_path = tempfile.mkstemp(suffix=".jpg")
            os.close(temp_image_fd)
            image.save(temp_image_path, "JPEG", quality=95)

            image_info = ImageInfo(
                original_shape=list(dicom_data.pixel_array.shape),
                converted_format="JPEG",
                converted_size=list(image.size),
                original_dtype=str(dicom_data.pixel_array.dtype),
                pixel_array_min=float(dicom_data.pixel_array.min()),
                pixel_array_max=float(dicom_data.pixel_array.max()),
                photometric_interpretation=getattr(
                    dicom_data, "PhotometricInterpretation", None
                ),
                transfer_syntax=(
                    str(dicom_data.file_meta.TransferSyntaxUID)
                    if hasattr(dicom_data, "file_meta")
                    and hasattr(dicom_data.file_meta, "TransferSyntaxUID")
                    else None
                ),
            )

            return temp_image_path, image_info

        except Exception as e:
            logger.error(f"Failed to convert DICOM to image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to convert DICOM file to image: {str(e)}",
            )


# ─── Singleton ───────────────────────────────────────────────
_preprocessing_service = None


def get_preprocessing_service() -> PreprocessingService:
    global _preprocessing_service
    if _preprocessing_service is None:
        _preprocessing_service = PreprocessingService()
    return _preprocessing_service
