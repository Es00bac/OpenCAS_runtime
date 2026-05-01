"""Helpers for attaching model-lane metadata to persisted runtime messages."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _resolve_reasoning_supported(resolved: Any) -> bool:
    provider = getattr(resolved, "provider", None)
    checker = getattr(provider, "supports_reasoning_effort", None)
    if callable(checker):
        try:
            return bool(checker(model=getattr(resolved, "model_id", None)))
        except TypeError:
            return bool(checker())
        except Exception:
            return False
    return False


def build_default_runtime_lane_meta(runtime: Any) -> Dict[str, Any]:
    """Resolve the configured/default lane without mixing in stale last-use state."""
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
            "complexity": "standard",
        }
    )

    if _resolve_reasoning_supported(resolved):
        lane["reasoning_supported"] = True

    if lane.get("reasoning_supported") and "reasoning_effort" not in lane:
        resolver = getattr(llm_client, "resolve_reasoning_effort_for_complexity", None)
        if callable(resolver):
            try:
                resolved_effort = resolver(complexity=lane.get("complexity") or "standard")
            except Exception:
                resolved_effort = None
            if resolved_effort:
                lane["reasoning_effort"] = resolved_effort
    return {key: value for key, value in lane.items() if value is not None}


def build_last_runtime_lane_meta(runtime: Any) -> Dict[str, Any]:
    """Return the most recently used lane metadata, if any."""
    llm_client = getattr(getattr(runtime, "ctx", None), "llm", None)
    current_lane = getattr(llm_client, "current_lane_meta", None)
    if not callable(current_lane):
        return {}
    try:
        lane = current_lane() or {}
    except Exception:
        return {}
    return {key: value for key, value in dict(lane).items() if value is not None}


def build_runtime_lane_meta(
    runtime: Any,
    *,
    prefer_current: bool = True,
) -> Dict[str, Any]:
    """Resolve lane metadata while keeping configured and last-used lanes distinct."""
    if prefer_current:
        current_lane = build_last_runtime_lane_meta(runtime)
        if current_lane:
            return current_lane
    return build_default_runtime_lane_meta(runtime)


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
