import sys
import os

# Ensure local .venv site-packages is in sys.path even when spawned via multiprocessing
venv_site_packages = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".venv",
    "Lib",
    "site-packages",
)
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

from .api.routes import router as api_router
from .core.config import get_settings
from .core.exceptions import (
    ChestXRayException,
    global_exception_handler,
    chest_xray_exception_handler,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    settings = get_settings()

    app = FastAPI(
        title="Chest X-Ray Disease Classification API",
        description=(
            "AI-powered chest X-ray classification system using DenseNet-121 and ResNet-50. "
            "Identifies 14 thoracic pathologies from the NIH ChestX-ray14 dataset using "
            "multi-label classification with transfer learning. Includes Grad-CAM visualization "
            "for disease-relevant region highlighting."
        ),
        version="1.0.0",
        debug=settings.debug,
    )

    # Exception handlers
    app.add_exception_handler(Exception, global_exception_handler)
    app.add_exception_handler(
        ChestXRayException, chest_xray_exception_handler
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(api_router)

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Welcome to Chest X-Ray Disease Classification API",
            "version": "1.0.0",
            "docs": "/docs",
            "models": ["DenseNet-121", "ResNet-50"],
            "dataset": "NIH ChestX-ray14 (14 pathologies)",
        }

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
