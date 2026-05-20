"""Shared chat transport helpers for API routes and realtime surfaces."""

from __future__ import annotations

import asyncio
import mimetypes
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, Optional
from uuid import UUID

from fastapi import HTTPException

from opencas.context.models import MessageRole
from opencas.spoken_response import prepare_spoken_response_text as _prepare_spoken_response_text

from .routes.identity import SomaticStateResponse
from .voice_service import synthesize_speech

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
_DEFAULT_CHAT_TURN_TIMEOUT_SECONDS = 240.0


@dataclass
class ChatTurnResult:
    session_id: str
    response: str
    somatic: Optional[SomaticStateResponse]
    voice_output: Optional[Dict[str, Any]] = None


def conversation_actor_meta(
    *,
    actor_type: Optional[str] = None,
    actor_label: Optional[str] = None,
    actor_note: Optional[str] = None,
    source: str = "api_chat",
) -> Dict[str, Any]:
    """Return structural metadata describing who is talking to the agent."""
    normalized_type = re.sub(r"[^a-z0-9_]+", "_", str(actor_type or "operator").strip().lower()).strip("_")
    if not normalized_type:
        normalized_type = "operator"
    label = str(actor_label or "").strip()
    if not label:
        label = "operator" if normalized_type == "operator" else normalized_type.replace("_", " ")
    is_operator = normalized_type in {"operator", "owner", "primary_user"}
    payload: Dict[str, Any] = {
        "type": normalized_type,
        "label": label,
        "source": source,
        "is_operator": is_operator,
        "memory_scope": "operator_turn" if is_operator else "collaborator_agent_turn",
        "remember_as": "owner/operator" if is_operator else f"{label} ({normalized_type})",
    }
    note = str(actor_note or "").strip()
    if note:
        payload["note"] = note[:500]
    return payload


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _message_preview(message: str, limit: int = 160) -> str:
    preview = " ".join(str(message or "").split())
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3].rstrip() + "..."


def _chat_turn_timeout_seconds(runtime: Any) -> Optional[float]:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    raw = getattr(config, "api_chat_turn_timeout_seconds", _DEFAULT_CHAT_TURN_TIMEOUT_SECONDS)
    if raw is None:
        return _DEFAULT_CHAT_TURN_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_CHAT_TURN_TIMEOUT_SECONDS
    if value <= 0:
        return None
    return max(0.001, value)


def _coerce_inflight_map(runtime: Any) -> Dict[str, Dict[str, Any]]:
    existing = getattr(runtime, "_chat_sessions_inflight", None)
    if isinstance(existing, dict):
        return existing
    if isinstance(existing, set):
        converted = {
            str(session_id): {
                "session_id": str(session_id),
                "started_at": None,
                "message_preview": "",
            }
            for session_id in existing
        }
        setattr(runtime, "_chat_sessions_inflight", converted)
        return converted
    converted: Dict[str, Dict[str, Any]] = {}
    setattr(runtime, "_chat_sessions_inflight", converted)
    return converted


def _inflight_elapsed_seconds(record: Dict[str, Any]) -> Optional[float]:
    started_at = record.get("started_at")
    if not started_at:
        return None
    try:
        started = datetime.fromisoformat(str(started_at))
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0.0, (_utc_now() - started).total_seconds())


def _public_inflight_record(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "session_id": record.get("session_id"),
        "started_at": record.get("started_at"),
        "message_preview": record.get("message_preview") or "",
    }
    elapsed = _inflight_elapsed_seconds(record)
    if elapsed is not None:
        payload["elapsed_seconds"] = round(elapsed, 3)
    return payload


def chat_inflight_snapshot(runtime: Any, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Return the current chat-turn lock state without mutating active turns."""
    raw = getattr(runtime, "_chat_sessions_inflight", None)
    if isinstance(raw, set):
        records = {
            str(sid): {
                "session_id": str(sid),
                "started_at": None,
                "message_preview": "",
            }
            for sid in raw
        }
    elif isinstance(raw, dict):
        records = raw
    else:
        records = {}

    if session_id:
        record = records.get(session_id)
        return {
            "session_id": session_id,
            "in_progress": record is not None,
            "turn": _public_inflight_record(record) if record else None,
        }
    turns = [_public_inflight_record(record) for record in records.values()]
    return {
        "in_progress": bool(turns),
        "count": len(turns),
        "turns": turns,
    }


async def _claim_session_chat_turn(runtime: Any, session_id: str, message: str) -> None:
    guard = getattr(runtime, "_chat_session_inflight_guard", None)
    if guard is None:
        guard = asyncio.Lock()
        setattr(runtime, "_chat_session_inflight_guard", guard)
    async with guard:
        in_flight = _coerce_inflight_map(runtime)
        existing = in_flight.get(session_id)
        if existing is not None:
            detail = {
                "code": "chat_turn_in_progress",
                "message": "A response is already in progress for this chat session.",
                "session_id": session_id,
                "turn": _public_inflight_record(existing),
                "operator_guidance": "Wait for the current response to finish, or open a new chat session for unrelated work.",
            }
            raise HTTPException(
                status_code=409,
                detail=detail,
            )
        now = _utc_now()
        in_flight[session_id] = {
            "session_id": session_id,
            "started_at": now.isoformat(),
            "message_preview": _message_preview(message),
        }


async def _release_session_chat_turn(runtime: Any, session_id: str) -> None:
    guard = getattr(runtime, "_chat_session_inflight_guard", None)
    in_flight = getattr(runtime, "_chat_sessions_inflight", None)
    if guard is None or in_flight is None:
        return
    async with guard:
        if isinstance(in_flight, dict):
            in_flight.pop(session_id, None)
        elif isinstance(in_flight, set):
            in_flight.discard(session_id)


def chat_upload_dir(runtime: Any) -> Path:
    """Return the canonical chat upload directory for *runtime*."""
    config = runtime.ctx.config
    agent_workspace_root = getattr(config, "agent_workspace_root", None)
    if callable(agent_workspace_root):
        workspace_root = Path(agent_workspace_root())
    else:
        managed_root = getattr(config, "managed_workspace_root", None)
        workspace_root = (
            Path(managed_root)
            if managed_root is not None
            else Path(config.state_dir).parent / "workspace"
        )
    upload_dir = workspace_root / "chat_uploads"
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


def prepare_spoken_response_text(response_text: str) -> str:
    """Adapt a chat response for voice without reading visual-only payloads aloud."""
    return _prepare_spoken_response_text(response_text)


def _desktop_context_plugin_disabled(runtime: Any) -> bool:
    ctx = getattr(runtime, "ctx", None)
    lifecycle = getattr(ctx, "plugin_lifecycle", None)
    is_tool_disabled = getattr(lifecycle, "is_tool_disabled", None)
    if not callable(is_tool_disabled):
        return False
    for tool_name in (
        "desktop_context_speak",
        "desktop_context_configure",
        "desktop_context_observe",
    ):
        try:
            if is_tool_disabled(tool_name):
                return True
        except Exception:
            return False
    return False


def _body_double_system_voice_active(runtime: Any, actor_meta: Dict[str, Any]) -> bool:
    """Return true when operator chat should use the live Body Double TTS bridge."""
    if actor_meta.get("is_operator") is False:
        return False
    service = getattr(runtime, "desktop_context", None)
    config = getattr(service, "config", None)
    if config is None or _desktop_context_plugin_disabled(runtime):
        return False
    if not bool(getattr(config, "enabled", False)):
        return False
    if not bool(getattr(config, "tts_enabled", True)):
        return False
    if not bool(getattr(config, "play_audio", True)):
        return False
    return bool(
        getattr(config, "media_commentary_mode_enabled", False)
        or str(getattr(config, "media_commentary_request", "") or "").strip()
    )


def _consume_body_double_voice_result(runtime: Any, session_id: str) -> Optional[Dict[str, Any]]:
    records = getattr(runtime, "_body_double_voice_results", None)
    if not isinstance(records, dict):
        return None
    record = records.pop(str(session_id), None)
    return record if isinstance(record, dict) else None


def _body_double_voice_output_meta(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    status = str((result or {}).get("status") or "delegated")
    reason = str((result or {}).get("reason") or "body_double_system_voice_active")
    playback = (result or {}).get("playback")
    return {
        "provider": "body-double",
        "mode": "system",
        "model": "desktop-context",
        "expressive": False,
        "voice_id": None,
        "voice_name": "Body Double system TTS",
        "warning": None,
        "status": status,
        "reason": reason,
        "playback": playback if isinstance(playback, dict) else None,
    }


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
    voice_input: Optional[Dict[str, Any]] = None,
    speak_response: bool = False,
    voice_prefer_local: bool = True,
    voice_expressive: bool = False,
    actor_type: Optional[str] = None,
    actor_label: Optional[str] = None,
    actor_note: Optional[str] = None,
    extra_user_meta: Optional[Dict[str, Any]] = None,
    suppress_body_double_voice: bool = True,
    body_double_voice_suppression_explicit: bool = False,
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

    actor_meta = conversation_actor_meta(
        actor_type=actor_type,
        actor_label=actor_label,
        actor_note=actor_note,
        source="api_chat",
    )
    explicit_body_double_voice_suppression = (
        bool(suppress_body_double_voice) and bool(body_double_voice_suppression_explicit)
    )
    delegate_voice_to_body_double = (
        _body_double_system_voice_active(runtime, actor_meta)
        and not explicit_body_double_voice_suppression
    )
    effective_suppress_body_double_voice = bool(suppress_body_double_voice)
    if delegate_voice_to_body_double:
        effective_suppress_body_double_voice = False
    user_meta: Dict[str, Any] = {"conversation_actor": actor_meta}
    if effective_suppress_body_double_voice:
        user_meta["suppress_body_double_voice"] = True
    if extra_user_meta:
        for key, value in extra_user_meta.items():
            if value is not None:
                user_meta[key] = value
    if resolved_attachments:
        user_meta["attachments"] = resolved_attachments
    if voice_input:
        user_meta["voice_input"] = voice_input
    user_meta_payload = user_meta or None
    await _claim_session_chat_turn(runtime, sid, user_message)
    try:
        turn_started = _utc_now()
        turn = runtime.converse(
            user_message,
            session_id=sid,
            user_meta=user_meta_payload,
        )
        timeout_seconds = _chat_turn_timeout_seconds(runtime)
        try:
            response_text = (
                await asyncio.wait_for(turn, timeout=timeout_seconds)
                if timeout_seconds is not None
                else await turn
            )
        except asyncio.TimeoutError as exc:
            elapsed = max(0.0, (_utc_now() - turn_started).total_seconds())
            raise HTTPException(
                status_code=504,
                detail={
                    "code": "chat_turn_timeout",
                    "message": "The chat response did not finish within the configured dashboard timeout.",
                    "session_id": sid,
                    "elapsed_seconds": round(elapsed, 3),
                    "timeout_seconds": timeout_seconds,
                    "operator_guidance": (
                        "The turn was cancelled and the chat lock was released. Retry the message, "
                        "or move long-running work into a scheduled/background task."
                    ),
                },
            ) from exc
        voice_output_meta: Optional[Dict[str, Any]] = None
        if delegate_voice_to_body_double:
            body_double_result = _consume_body_double_voice_result(runtime, sid)
            voice_output_meta = _body_double_voice_output_meta(body_double_result)
            await annotate_latest_assistant_voice_output(runtime, sid, voice_output_meta)
        elif speak_response:
            spoken_text = prepare_spoken_response_text(response_text)
            voice_result = await synthesize_speech(
                upload_dir,
                text=spoken_text,
                prefer_local=voice_prefer_local,
                expressive=voice_expressive,
            )
            voice_output_meta = voice_result.to_meta()
            await annotate_latest_assistant_voice_output(runtime, sid, voice_output_meta)
        return ChatTurnResult(
            session_id=sid,
            response=response_text,
            somatic=serialize_somatic_state(runtime),
            voice_output=voice_output_meta,
        )
    finally:
        await _release_session_chat_turn(runtime, sid)


async def annotate_latest_assistant_voice_output(
    runtime: Any,
    session_id: str,
    voice_output: Dict[str, Any],
) -> Optional[UUID]:
    """Attach *voice_output* metadata to the most recent assistant turn."""
    store = getattr(runtime.ctx, "context_store", None)
    if store is None or not hasattr(store, "list_recent") or not hasattr(store, "merge_message_meta"):
        return None
    entries = await store.list_recent(session_id, limit=8, include_hidden=True)
    for entry in reversed(entries):
        if entry.role == MessageRole.ASSISTANT:
            await store.merge_message_meta(session_id, entry.message_id, {"voice_output": voice_output})
            return entry.message_id
    return None
