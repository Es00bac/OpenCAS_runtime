"""Helpers for attaching model-lane metadata to persisted runtime messages."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_runtime_lane_meta(runtime: Any) -> Dict[str, Any]:
    """Resolve the active model lane for the current runtime."""
    lane: Dict[str, Any] = {}
    llm_client = getattr(getattr(runtime, "ctx", None), "llm", None)
    current_lane = getattr(llm_client, "current_lane_meta", None)
    if callable(current_lane):
        try:
            lane.update(current_lane() or {})
        except Exception:
            pass
    default_model = getattr(llm_client, "default_model", None)
    if default_model and "model" not in lane:
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
            "provider": lane.get("provider") or getattr(resolved, "provider_id", None),
            "resolved_model": lane.get("resolved_model") or (
                f"{resolved.provider_id}/{resolved.model_id}"
                if getattr(resolved, "provider_id", None)
                and getattr(resolved, "model_id", None)
                else None
            ),
            "profile_id": lane.get("profile_id") or getattr(resolved, "profile_id", None),
            "auth_source": lane.get("auth_source") or getattr(resolved, "auth_source", None),
        }
    )

    if "complexity" not in lane:
        lane["complexity"] = "standard"

    if "reasoning_supported" not in lane:
        provider = getattr(resolved, "provider", None)
        checker = getattr(provider, "supports_reasoning_effort", None)
        supports_reasoning = False
        if callable(checker):
            try:
                supports_reasoning = bool(checker(model=getattr(resolved, "model_id", None)))
            except TypeError:
                supports_reasoning = bool(checker())
            except Exception:
                supports_reasoning = False
        if supports_reasoning:
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
