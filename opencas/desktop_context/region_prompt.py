"""Hotkeyable desktop region prompt helper for KDE/OpenCAS."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib import error, request


class RegionPromptError(RuntimeError):
    """Raised when the region prompt flow cannot continue."""


_QT_APP: Any = None


def default_output_dir() -> Path:
    """Return the directory used for captured region prompt images."""

    raw = os.environ.get("OPENCAS_REGION_PROMPT_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "opencas" / "region_prompts"


def timestamped_capture_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return output_dir / f"region-prompt-{stamp}-{uuid.uuid4().hex[:8]}.png"


def capture_region_with_spectacle(output_path: Path, *, timeout_seconds: int = 120) -> Path:
    """Ask KDE Spectacle to capture a rectangular region into *output_path*."""

    spectacle = shutil.which("spectacle")
    if not spectacle:
        raise RegionPromptError("KDE Spectacle is required for region capture but was not found in PATH.")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [spectacle, "-r", "-b", "-n", "-o", str(output_path)]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RegionPromptError("Region selection timed out before a screenshot was captured.") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RegionPromptError(stderr or f"Spectacle failed with exit code {result.returncode}.")
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RegionPromptError("Spectacle finished but did not produce a region screenshot.")
    return output_path


def _qt_app() -> Any:
    from PyQt6 import QtWidgets

    global _QT_APP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    _QT_APP = app
    return app


def prompt_for_text(*, title: str = "Ask OpenCAS", placeholder: str = "") -> Optional[str]:
    """Show a compact Qt prompt dialog and return the entered text."""

    from PyQt6 import QtCore, QtGui, QtWidgets

    app = _qt_app()
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.setMinimumWidth(560)
    dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
    dialog.setStyleSheet(
        """
        QDialog {
            background: #161a1d;
            color: #edf6f9;
            border: 2px solid #00d4ff;
        }
        QLabel {
            color: #a8dadc;
            font-size: 13px;
        }
        QLineEdit {
            background: #0b0f12;
            color: #f8f9fa;
            border: 1px solid #00d4ff;
            border-radius: 4px;
            padding: 9px 10px;
            font-size: 15px;
        }
        QPushButton {
            background: #00a6c8;
            color: #061014;
            border: none;
            border-radius: 4px;
            padding: 7px 14px;
            font-weight: 600;
        }
        QPushButton#cancelButton {
            background: #30363d;
            color: #dce3e8;
        }
        """
    )
    layout = QtWidgets.QVBoxLayout(dialog)
    label = QtWidgets.QLabel("Ask OpenCAS about the selected desktop region.")
    field = QtWidgets.QLineEdit()
    field.setPlaceholderText(placeholder or "Question or instruction...")
    field.returnPressed.connect(dialog.accept)
    buttons = QtWidgets.QHBoxLayout()
    buttons.addStretch(1)
    cancel = QtWidgets.QPushButton("Cancel")
    cancel.setObjectName("cancelButton")
    submit = QtWidgets.QPushButton("Ask")
    cancel.clicked.connect(dialog.reject)
    submit.clicked.connect(dialog.accept)
    buttons.addWidget(cancel)
    buttons.addWidget(submit)
    layout.addWidget(label)
    layout.addWidget(field)
    layout.addLayout(buttons)
    field.setFocus(QtCore.Qt.FocusReason.ActiveWindowFocusReason)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    text = field.text().strip()
    app.processEvents()
    return text or None


def show_text_dialog(title: str, text: str) -> None:
    """Display a response or error without requiring a terminal."""

    try:
        from PyQt6 import QtCore, QtWidgets

        app = _qt_app()
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(620, 360)
        dialog.setWindowFlags(dialog.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        layout = QtWidgets.QVBoxLayout(dialog)
        box = QtWidgets.QTextEdit()
        box.setReadOnly(True)
        box.setPlainText(text)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(box)
        layout.addWidget(close)
        dialog.exec()
        app.processEvents()
    except Exception:
        kdialog = shutil.which("kdialog")
        if kdialog:
            subprocess.run([kdialog, "--msgbox", text[:6000], "--title", title], check=False)


def _health_ok(base_url: str, *, timeout: float = 1.0) -> bool:
    try:
        with request.urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def discover_base_url(explicit: Optional[str] = None) -> str:
    """Resolve the OpenCAS API base URL for local hotkey use."""

    candidates = [
        explicit,
        os.environ.get("OPENCAS_API_BASE_URL"),
        os.environ.get("OPENCAS_BASE_URL"),
        "http://127.0.0.1:32147",
        "http://localhost:32147",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        base = str(candidate).rstrip("/")
        if _health_ok(base):
            return base
    if explicit:
        return explicit.rstrip("/")
    raise RegionPromptError(
        "OpenCAS API is not reachable. Start OpenCAS with --with-server or set OPENCAS_API_BASE_URL."
    )


def _multipart_body(fields: dict[str, Any], file_path: Path) -> tuple[bytes, str]:
    boundary = f"----OpenCASRegionPrompt{uuid.uuid4().hex}"
    body = bytearray()

    def add(value: bytes) -> None:
        body.extend(value)

    for name, value in fields.items():
        if value is None:
            continue
        add(f"--{boundary}\r\n".encode("utf-8"))
        add(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        add(str(value).encode("utf-8"))
        add(b"\r\n")

    media_type = mimetypes.guess_type(file_path.name)[0] or "image/png"
    add(f"--{boundary}\r\n".encode("utf-8"))
    add(
        (
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {media_type}\r\n\r\n"
        ).encode("utf-8")
    )
    add(file_path.read_bytes())
    add(b"\r\n")
    add(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), boundary


def submit_region_prompt(
    *,
    base_url: str,
    image_path: Path,
    prompt: str,
    session_id: Optional[str] = None,
    speak_response: bool = False,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Submit the selected image plus prompt to OpenCAS."""

    fields: dict[str, Any] = {
        "prompt": prompt,
        "speak_response": "true" if speak_response else "false",
        "voice_prefer_local": "true",
        "voice_expressive": "false",
    }
    if session_id:
        fields["session_id"] = session_id
    body, boundary = _multipart_body(fields, image_path)
    url = f"{base_url.rstrip('/')}/api/chat/region-prompt"
    req = request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RegionPromptError(f"OpenCAS returned HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RegionPromptError(f"Could not reach OpenCAS: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RegionPromptError(f"OpenCAS returned non-JSON response: {raw[:500]}") from exc
    if not isinstance(payload, dict):
        raise RegionPromptError("OpenCAS returned an invalid response payload.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a desktop region, type a prompt, and send both to OpenCAS."
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenCAS API base URL; auto-detects localhost:32147 and localhost:8080.",
    )
    parser.add_argument("--session-id", default=None, help="Optional OpenCAS chat session id.")
    parser.add_argument("--output-dir", default=None, help="Directory for captured region screenshots.")
    parser.add_argument("--image", default=None, help="Use an existing image instead of opening Spectacle.")
    parser.add_argument("--prompt", default=None, help="Prompt text; skips the Qt prompt dialog when set.")
    parser.add_argument("--speak-response", action="store_true", help="Ask OpenCAS to synthesize voice output.")
    parser.add_argument("--no-response-dialog", action="store_true", help="Do not show a Qt response dialog.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        image_path = Path(args.image).expanduser() if args.image else timestamped_capture_path(
            Path(args.output_dir).expanduser() if args.output_dir else default_output_dir()
        )
        if not args.image:
            image_path = capture_region_with_spectacle(image_path)
        elif not image_path.exists():
            raise RegionPromptError(f"Image does not exist: {image_path}")
        prompt = str(args.prompt or "").strip() or prompt_for_text()
        if not prompt:
            return 130
        base_url = discover_base_url(args.base_url)
        payload = submit_region_prompt(
            base_url=base_url,
            image_path=image_path,
            prompt=prompt,
            session_id=args.session_id,
            speak_response=bool(args.speak_response),
        )
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        response_text = str(payload.get("response") or "").strip()
        if response_text and not args.no_response_dialog:
            show_text_dialog("OpenCAS response", response_text)
        return 0
    except RegionPromptError as exc:
        message = str(exc)
        print(f"opencas-region-prompt: {message}", file=sys.stderr)
        show_text_dialog("OpenCAS region prompt failed", message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
