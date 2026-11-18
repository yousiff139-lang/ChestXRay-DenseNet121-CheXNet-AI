from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "ChestXRay AI Backend"
    debug: bool = False

    # Gemini configuration for diagnostic reports
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"

    # Roboflow configuration for YOLO pathology detection
    roboflow_api_key: str = ""
    roboflow_private_key: str = ""
    roboflow_workspace: str = "dccai"
    roboflow_model_id: str = "datamed-6oe4k/2"

    # Model configuration
    num_classes: int = 14
    confidence_threshold: float = 0.5
    gradcam_enabled: bool = True

    # Pretrained weight paths (relative to backend/models/weights/)
    densenet_weights_path: str = ""  # Empty = use torchvision ImageNet pretrained
    resnet_weights_path: str = ""    # Empty = use torchvision ImageNet pretrained

    # File upload settings
    MAX_FILE_SIZE_MB: int = 15
    max_file_size: int = MAX_FILE_SIZE_MB * 1024 * 1024
    allowed_file_types: list = ["image/jpeg", "image/png", "image/jpg"]
    allowed_dicom_file_types: list = ["application/dicom", "application/octet-stream"]

    # File type extensions for validation
    ALLOWED_IMAGE_EXTENSIONS: list = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    ALLOWED_DICOM_EXTENSIONS: list = [".dcm", ".dicom", ".DCM", ".DICOM"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
