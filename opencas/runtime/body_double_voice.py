"""Body-double voice response bridge for conversation channels."""

from __future__ import annotations

from typing import Any, Optional

from opencas.spoken_response import DEFAULT_MAX_SPOKEN_RESPONSE_CHARS, prepare_spoken_response_text


async def maybe_speak_body_double_response(
    runtime: Any,
    *,
    session_id: str,
    user_input: str,
    user_meta: Optional[dict[str, Any]],
    response_text: str,
) -> Optional[dict[str, Any]]:
    """Speak a completed conversation response when body-double mode is active."""

    source = _conversation_source(user_meta)
    service = getattr(runtime, "desktop_context", None)
    if service is None:
        return None
    config = getattr(service, "config", None)
    if not bool(getattr(config, "enabled", False)):
        return _finish(
            runtime,
            session_id=session_id,
            user_input=user_input,
            result={"status": "skipped", "reason": "desktop_context_disabled"},
            source=source,
        )
    if _desktop_context_plugin_disabled(runtime):
        return _finish(
            runtime,
            session_id=session_id,
            user_input=user_input,
            result={"status": "skipped", "reason": "desktop_context_plugin_disabled"},
            source=source,
        )
    if not bool(getattr(config, "tts_enabled", True)):
        return _finish(
            runtime,
            session_id=session_id,
            user_input=user_input,
            result={"status": "skipped", "reason": "tts_disabled"},
            source=source,
        )
    if _body_double_voice_suppressed(user_meta):
        return _finish(
            runtime,
            session_id=session_id,
            user_input=user_input,
            result={"status": "skipped", "reason": "body_double_voice_suppressed"},
            source=source,
        )
    if _explicit_non_operator_actor(user_meta):
        return _finish(
            runtime,
            session_id=session_id,
            user_input=user_input,
            result={"status": "skipped", "reason": "non_operator_actor"},
            source=source,
        )
    text = _spoken_body_double_text(response_text)
    if not text:
        return _finish(
            runtime,
            session_id=session_id,
            user_input=user_input,
            result={"status": "skipped", "reason": "empty_response"},
            source=source,
        )
    reason = f"body_double_conversation_response:{source}"
    try:
        result = await service.speak_text(
            text,
            reason=reason,
            force=True,
            max_chars=DEFAULT_MAX_SPOKEN_RESPONSE_CHARS,
            allow_note_redirect=False,
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "reason": f"body_double_voice_failed:{type(exc).__name__}",
            "error": str(exc),
        }
    return _finish(
        runtime,
        session_id=session_id,
        user_input=user_input,
        result=result,
        source=source,
    )


def _spoken_body_double_text(response_text: str) -> str:
    """Convert a dashboard/chat response into something suitable to say aloud."""

    return prepare_spoken_response_text(response_text, max_chars=DEFAULT_MAX_SPOKEN_RESPONSE_CHARS)


def _explicit_non_operator_actor(user_meta: Optional[dict[str, Any]]) -> bool:
    if not isinstance(user_meta, dict):
        return False
    actor = user_meta.get("conversation_actor")
    if not isinstance(actor, dict):
        return False
    return actor.get("is_operator") is False


def _body_double_voice_suppressed(user_meta: Optional[dict[str, Any]]) -> bool:
    if not isinstance(user_meta, dict):
        return False
    return bool(user_meta.get("suppress_body_double_voice"))


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


def _conversation_source(user_meta: Optional[dict[str, Any]]) -> str:
    if not isinstance(user_meta, dict):
        return "conversation"
    actor = user_meta.get("conversation_actor")
    if isinstance(actor, dict):
        source = str(actor.get("source") or "").strip()
        if source:
            return _safe_source(source)
    if isinstance(user_meta.get("telegram_context"), dict):
        return "telegram"
    if isinstance(user_meta.get("voice_input"), dict):
        return "voice"
    return "conversation"


def _safe_source(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value.lower())
    return cleaned.strip("_") or "conversation"


def _finish(
    runtime: Any,
    *,
    session_id: str,
    user_input: str,
    result: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    _remember_result(runtime, session_id=session_id, result=result, source=source)
    _trace(runtime, session_id=session_id, user_input=user_input, result=result, source=source)
    return result


def _remember_result(
    runtime: Any,
    *,
    session_id: str,
    result: dict[str, Any],
    source: str,
) -> None:
    try:
        records = getattr(runtime, "_body_double_voice_results", None)
        if not isinstance(records, dict):
            records = {}
            setattr(runtime, "_body_double_voice_results", records)
        records[str(session_id)] = {
            "session_id": str(session_id),
            "source": source,
            "status": result.get("status"),
            "reason": result.get("reason"),
            "playback": result.get("playback"),
            "voice": result.get("voice"),
        }
    except Exception:
        pass


def _trace(
    runtime: Any,
    *,
    session_id: str,
    user_input: str,
    result: dict[str, Any],
    source: str,
) -> None:
    tracer = getattr(runtime, "_trace", None)
    if not callable(tracer):
        return
    try:
        tracer(
            "body_double_voice_response",
            {
                "session_id": session_id,
                "source": source,
                "status": result.get("status"),
                "reason": result.get("reason"),
                "input_preview": " ".join(str(user_input or "").split())[:160],
            },
        )
    except Exception:
        pass
