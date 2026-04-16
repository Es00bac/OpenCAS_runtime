"""Shared chat transport helpers for API routes and realtime surfaces."""

from __future__ import annotations

import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Optional

from fastapi import HTTPException

from .routes.identity import SomaticStateResponse

_TEXT_ATTACHMENT_SUFFIXES = {
    ".md",
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".sh",
    ".bash",
    ".zsh",
    ".toml",
    ".ini",
    ".csv",
}
_MAX_ATTACHMENT_TEXT_CHARS = 24_000


@dataclass
class ChatTurnResult:
    session_id: str
    response: str
    somatic: Optional[SomaticStateResponse]


def chat_upload_dir(runtime: Any) -> Path:
    """Return the canonical chat upload directory for *runtime*."""
    upload_dir = Path(runtime.ctx.config.state_dir).parent / "chat_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def resolve_chat_session_id(runtime: Any, session_id: Optional[str]) -> str:
    """Resolve the session id used for a chat turn."""
    return session_id or runtime.ctx.config.session_id or "default"


def serialize_somatic_state(runtime: Any) -> Optional[SomaticStateResponse]:
    """Return the current somatic state as an API payload, if available."""
    somatic_mgr = getattr(runtime.ctx, "somatic", None)
    if somatic_mgr is None:
        return None
    try:
        ss = somatic_mgr.state
        return SomaticStateResponse(
            state_id=str(ss.state_id),
            updated_at=ss.updated_at.isoformat(),
            arousal=ss.arousal,
            fatigue=ss.fatigue,
            tension=ss.tension,
            valence=ss.valence,
            focus=ss.focus,
            energy=ss.energy,
            certainty=ss.certainty,
            somatic_tag=ss.somatic_tag,
        )
    except Exception:
        return None


def guess_attachment_media_type(filename: str, hinted: Optional[str] = None) -> str:
    """Guess a stable media type for an uploaded attachment."""
    if hinted:
        return hinted
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def store_uploaded_file(
    upload_dir: Path,
    *,
    filename: Optional[str],
    content_type: Optional[str],
    fileobj: BinaryIO,
) -> Dict[str, Any]:
    """Persist an uploaded file and return the stored attachment payload."""
    dest = upload_dir / (filename or "upload")
    counter = 1
    original_dest = dest
    while dest.exists():
        stem = original_dest.stem
        suffix = original_dest.suffix
        dest = upload_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    with dest.open("wb") as handle:
        shutil.copyfileobj(fileobj, handle)
    media_type = guess_attachment_media_type(dest.name, content_type)
    return {
        "filename": dest.name,
        "path": str(dest),
        "url": f"/api/chat/uploads/{dest.name}",
        "media_type": media_type,
        "size_bytes": dest.stat().st_size,
    }


def _attachment_field(attachment: Any, field: str) -> Any:
    if isinstance(attachment, dict):
        return attachment.get(field)
    return getattr(attachment, field, None)


def _attachment_is_text(media_type: str, filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return media_type.startswith("text/") or suffix in _TEXT_ATTACHMENT_SUFFIXES


def _attachment_language_hint(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def resolve_uploaded_attachment(upload_dir: Path, attachment: Any) -> Dict[str, Any]:
    """Resolve a stored attachment reference into the materialized prompt payload."""
    raw_name = _attachment_field(attachment, "filename") or Path(
        _attachment_field(attachment, "url") or _attachment_field(attachment, "path") or ""
    ).name
    if not raw_name:
        raise HTTPException(status_code=400, detail="Attachment filename is required")
    target = upload_dir / Path(raw_name).name
    try:
        target.resolve().relative_to(upload_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid attachment path") from exc
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Attachment not found: {target.name}")

    media_type = guess_attachment_media_type(target.name, _attachment_field(attachment, "media_type"))
    payload: Dict[str, Any] = {
        "filename": target.name,
        "path": str(target),
        "url": f"/api/chat/uploads/{target.name}",
        "media_type": media_type,
        "size_bytes": target.stat().st_size,
    }
    if _attachment_is_text(media_type, target.name):
        text = target.read_text(encoding="utf-8", errors="ignore")
        truncated = len(text) > _MAX_ATTACHMENT_TEXT_CHARS
        payload["text_content"] = text[:_MAX_ATTACHMENT_TEXT_CHARS] if truncated else text
        payload["text_truncated"] = truncated
        payload["language_hint"] = _attachment_language_hint(target.name)
    return payload


async def perform_chat_turn(
    runtime: Any,
    *,
    session_id: Optional[str],
    message: str,
    attachments: Optional[Iterable[Any]] = None,
) -> ChatTurnResult:
    """Execute a chat turn with optional attachments and return the API payload."""
    sid = resolve_chat_session_id(runtime, session_id)
    upload_dir = chat_upload_dir(runtime)
    resolved_attachments = [
        resolve_uploaded_attachment(upload_dir, attachment)
        for attachment in (attachments or [])
    ]
    user_message = message.strip()
    if not user_message and resolved_attachments:
        user_message = "Please review the attached files."
    if not user_message and not resolved_attachments:
        raise HTTPException(status_code=400, detail="message or attachments required")

    user_meta = {"attachments": resolved_attachments} if resolved_attachments else None
    response_text = await runtime.converse(
        user_message,
        session_id=sid,
        user_meta=user_meta,
    )
    return ChatTurnResult(
        session_id=sid,
        response=response_text,
        somatic=serialize_somatic_state(runtime),
    )
