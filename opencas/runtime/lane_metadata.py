"""Helpers for attaching model-lane metadata to persisted runtime messages."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_runtime_lane_meta(runtime: Any) -> Dict[str, Any]:
    """Resolve the active model lane for the current runtime."""
    lane: Dict[str, Any] = {}
    llm_client = getattr(getattr(runtime, "ctx", None), "llm", None)
    default_model = getattr(llm_client, "default_model", None)
    if default_model:
        lane["model"] = default_model

    manager = getattr(llm_client, "manager", None)
    if manager is None or not default_model:
        return lane

    try:
        resolved = manager.resolve(default_model)
    except Exception:
        return lane

    lane.update(
        {
            "provider": getattr(resolved, "provider_id", None),
            "resolved_model": (
                f"{resolved.provider_id}/{resolved.model_id}"
                if getattr(resolved, "provider_id", None)
                and getattr(resolved, "model_id", None)
                else None
            ),
            "profile_id": getattr(resolved, "profile_id", None),
            "auth_source": getattr(resolved, "auth_source", None),
        }
    )
    return {key: value for key, value in lane.items() if value is not None}


def build_assistant_message_meta(
    runtime: Any,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach lane metadata to assistant messages while preserving extra fields."""
    meta: Dict[str, Any] = dict(extra or {})
    lane = build_runtime_lane_meta(runtime)
    if lane:
        meta["lane"] = lane
    return meta
