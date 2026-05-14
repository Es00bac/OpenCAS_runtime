"""Desktop screenshot context service for body-double style collaboration."""

from .capture import (
    DesktopCapture,
    ScreenshotBackend,
    capture_desktop_image,
    choose_screenshot_backend,
    run_tesseract_ocr,
)
from .service import DesktopContextConfig, DesktopContextService
from .media import MprisMediaController

__all__ = [
    "DesktopCapture",
    "DesktopContextConfig",
    "DesktopContextService",
    "MprisMediaController",
    "ScreenshotBackend",
    "capture_desktop_image",
    "choose_screenshot_backend",
    "run_tesseract_ocr",
]
