"""Durable awareness records for automated system-visible messages."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from opencas.context.models import MessageRole
from opencas.memory import Episode, EpisodeKind

DEFAULT_SYSTEM_MESSAGE_SESSION_ID = "system:automated"


async def record_agent_visible_system_message(
    runtime: Any,
    *,
    content: str,
    event_kind: str,
    status: str = "",
    reason: str = "",
    source: str = "",
    source_id: str = "",
    channel: str = "",
    urgency: str = "",
    event_id: Optional[str] = None,
    session_id: Optional[str] = None,
    salience: float = 6.0,
    payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Record an automated message as both prompt context and episodic evidence."""
    text = str(content or "").strip()
    if not text:
        return {"context_recorded": False, "memory_recorded": False, "event_id": event_id}

    resolved_event_id = str(event_id or uuid4())
    resolved_session_id = _resolve_session_id(runtime, session_id)
    meta = {
        "agent_visible_system_message": True,
        "event_kind": str(event_kind or "system_message"),
        "event_id": resolved_event_id,
        "status": str(status or ""),
        "reason": str(reason or ""),
        "source": str(source or ""),
        "source_id": str(source_id or ""),
        "channel": str(channel or ""),
        "urgency": str(urgency or ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload or {},
    }
    result = {"context_recorded": False, "memory_recorded": False, "event_id": resolved_event_id}

    ctx = getattr(runtime, "ctx", None)
    context_store = getattr(ctx, "context_store", None)
    append = getattr(context_store, "append", None)
    if callable(append):
        try:
            maybe = append(resolved_session_id, MessageRole.SYSTEM, text, meta=meta)
            if hasattr(maybe, "__await__"):
                await maybe
            result["context_recorded"] = True
        except Exception as exc:
            _trace(runtime, "agent_visible_system_message_context_error", resolved_event_id, exc)

    record_episode = getattr(runtime, "_record_episode", None)
    memory_recorded = False
    if callable(record_episode):
        try:
            maybe = record_episode(
                text,
                EpisodeKind.OBSERVATION,
                session_id=resolved_session_id,
                role="system",
                payload=meta,
                salience=salience,
            )
            if hasattr(maybe, "__await__"):
                await maybe
            result["memory_recorded"] = True
            memory_recorded = True
        except TypeError:
            try:
                maybe = record_episode(
                    text,
                    EpisodeKind.OBSERVATION,
                    session_id=resolved_session_id,
                    role="system",
                )
                if hasattr(maybe, "__await__"):
                    await maybe
                result["memory_recorded"] = True
                memory_recorded = True
            except Exception as exc:
                _trace(runtime, "agent_visible_system_message_episode_error", resolved_event_id, exc)
        except Exception as exc:
            _trace(runtime, "agent_visible_system_message_episode_error", resolved_event_id, exc)

    memory = getattr(runtime, "memory", None) or getattr(ctx, "memory", None)
    save_episode = getattr(memory, "save_episode", None)
    if not memory_recorded and callable(save_episode):
        try:
            maybe = save_episode(
                Episode(
                    kind=EpisodeKind.OBSERVATION,
                    session_id=resolved_session_id,
                    content=text,
                    somatic_tag=str(event_kind or "system_message")[:80],
                    salience=max(0.0, min(10.0, float(salience))),
                    payload=meta,
                )
            )
            if hasattr(maybe, "__await__"):
                await maybe
            result["memory_recorded"] = True
        except Exception as exc:
            _trace(runtime, "agent_visible_system_message_memory_error", resolved_event_id, exc)

    return result


def _resolve_session_id(runtime: Any, session_id: Optional[str]) -> str:
    explicit = str(session_id or "").strip()
    if explicit:
        return explicit
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    configured = str(getattr(config, "session_id", "") or "").strip()
    return configured or DEFAULT_SYSTEM_MESSAGE_SESSION_ID


def _trace(runtime: Any, event: str, event_id: str, exc: Exception) -> None:
    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        try:
            trace(event, {"event_id": event_id, "error": str(exc)})
        except Exception:
            pass
