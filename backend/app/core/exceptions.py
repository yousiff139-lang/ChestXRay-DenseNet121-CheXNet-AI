from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class ChestXRayException(Exception):
    """Base exception for chest X-ray classification errors"""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ModelLoadError(ChestXRayException):
    """Raised when a model fails to load"""

    def __init__(self, message: str):
        super().__init__(message, status_code=503)


class ClassificationError(ChestXRayException):
    """Raised when classification inference fails"""

    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class GradCAMError(ChestXRayException):
    """Raised when Grad-CAM visualization fails"""

    def __init__(self, message: str):
        super().__init__(message, status_code=500)


async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions globally"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred", "error_code": "INTERNAL_ERROR"},
    )


async def chest_xray_exception_handler(request: Request, exc: ChestXRayException):
    """Handle chest X-ray specific exceptions"""
    logger.error(f"ChestXRay exception: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_code": "CHEST_XRAY_ERROR"},
    )
