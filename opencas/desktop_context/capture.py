"""Desktop screenshot and OCR helpers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


@dataclass(frozen=True)
class ScreenshotBackend:
    """Resolved command backend for taking one desktop screenshot."""

    name: str
    executable: str

    def command(self, output_path: Path) -> list[str]:
        target = str(output_path)
        if self.name == "spectacle":
            return [self.executable, "-b", "-n", "-o", target]
        if self.name == "grim":
            return [self.executable, target]
        if self.name == "gnome-screenshot":
            return [self.executable, "-f", target]
        if self.name == "scrot":
            return [self.executable, target]
        if self.name == "import":
            return [self.executable, "-window", "root", target]
        return [self.executable, target]


@dataclass
class DesktopCapture:
    """Result of one screenshot capture attempt."""

    success: bool
    path: Path
    backend: str
    media_type: str = "image/png"
    width: Optional[int] = None
    height: Optional[int] = None
    error: Optional[str] = None


_BACKEND_ORDER = ("spectacle", "grim", "gnome-screenshot", "scrot", "import")


def _candidate_backend_names(preferred: str = "auto") -> Iterable[str]:
    cleaned = (preferred or "auto").strip()
    if cleaned and cleaned != "auto":
        return (cleaned,)
    return _BACKEND_ORDER


def choose_screenshot_backend(
    preferred: str = "auto",
    *,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> Optional[ScreenshotBackend]:
    """Resolve the first available screenshot backend."""

    for name in _candidate_backend_names(preferred):
        executable = which(name)
        if executable:
            return ScreenshotBackend(name=name, executable=executable)
    return None


def _available_backends(
    preferred: str,
    *,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> list[ScreenshotBackend]:
    backends: list[ScreenshotBackend] = []
    for name in _candidate_backend_names(preferred):
        executable = which(name)
        if executable:
            backends.append(ScreenshotBackend(name=name, executable=executable))
    return backends


def capture_desktop_image(
    output_path: Path,
    *,
    backend: str = "auto",
    timeout_seconds: float = 20.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> DesktopCapture:
    """Capture the current desktop to *output_path* using the first usable backend."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for candidate in _available_backends(backend, which=which):
        cmd = candidate.command(output_path)
        try:
            result = runner(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except Exception as exc:
            errors.append(f"{candidate.name}: {type(exc).__name__}: {exc}")
            continue
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return DesktopCapture(success=True, path=output_path, backend=candidate.name)
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        errors.append(f"{candidate.name}: exit {result.returncode}: {stderr.strip()}")

    return DesktopCapture(
        success=False,
        path=output_path,
        backend=backend,
        error="; ".join(error for error in errors if error) or "no screenshot backend available",
    )


def run_tesseract_ocr(
    image_path: Path,
    *,
    timeout_seconds: float = 20.0,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    which: Callable[[str], Optional[str]] = shutil.which,
) -> str:
    """Return OCR text for *image_path* when tesseract is installed."""

    executable = which("tesseract")
    if not executable:
        return ""
    try:
        result = runner(
            [executable, str(image_path), "stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.decode("utf-8", errors="replace").strip()
