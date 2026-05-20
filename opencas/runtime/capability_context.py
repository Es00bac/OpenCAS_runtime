"""Ground model-facing reviews in the runtime's actual capability surface."""

from __future__ import annotations

from typing import Any

from .capability_snapshot import render_capability_drift_warning


def build_runtime_capability_context(runtime: Any, *, max_items: int = 100) -> str:
    """Render concise runtime capability evidence for refusal/review prompts."""

    lines: list[str] = ["Runtime capability evidence:"]
    drift_warning = render_capability_drift_warning(
        getattr(runtime, "capability_drift_report", None)
    )
    if drift_warning:
        lines.append(drift_warning)
    feature_lines = _render_feature_inventory(runtime)
    capability_lines = _render_capabilities(runtime, max_items=max_items)
    tool_lines = _render_tools(runtime, max_items=max_items)
    inspectability_line = _render_opencas_inspectability(runtime)
    somatic_line = _render_somatic_snapshot(runtime)
    continuity_line = _render_continuity_clock(runtime)
    desktop_line = _render_desktop_context(runtime)

    if feature_lines:
        lines.extend(feature_lines)
    if capability_lines:
        lines.extend(capability_lines)
    if tool_lines:
        lines.extend(tool_lines)
    if inspectability_line:
        lines.append(inspectability_line)
    if somatic_line:
        lines.append(somatic_line)
    if continuity_line:
        lines.append(continuity_line)
    if desktop_line:
        lines.append(desktop_line)
    if (
        not feature_lines
        and not capability_lines
        and not tool_lines
        and not inspectability_line
        and not somatic_line
        and not continuity_line
        and not desktop_line
    ):
        lines.append("- No runtime capability inventory was available to this review.")

    lines.extend(
        [
            "",
            "Capability truth rules:",
            "- Do not claim listed capabilities do not exist.",
            (
                "- Distinguish capability from permission: a tool may exist while a "
                "specific request is refused by privacy, consent, safety, or disabled "
                "configuration."
            ),
            (
                "- For desktop screenshots, say whether desktop context is disabled or "
                "requires consent instead of claiming shell access is impossible."
            ),
            (
                "- For OpenCAS maintenance, listed runtime, workflow, self-inspection, "
                "and wellbeing surfaces are operator-facing diagnostics; inspect them "
                "before denying that relevant operational evidence exists."
            ),
        ]
    )
    return "\n".join(lines)


def _render_continuity_clock(runtime: Any) -> str:
    continuity = getattr(getattr(getattr(runtime, "ctx", None), "identity", None), "continuity", None)
    if continuity is None:
        return ""
    offline_seconds = getattr(continuity, "last_offline_duration_seconds", None)
    if offline_seconds is None and getattr(continuity, "last_persisted_at", None) is None:
        return (
            "- Continuity clock: prior_persistence=unavailable; "
            f"continuous_present_score={float(getattr(continuity, 'continuous_present_score', 1.0) or 0.0):.2f}"
        )
    parts = []
    if offline_seconds is not None:
        try:
            parts.append(
                "elapsed_since_last_shutdown_or_persistence="
                + _format_duration(float(offline_seconds))
            )
        except (TypeError, ValueError):
            parts.append("elapsed_since_last_shutdown_or_persistence=unavailable")
    else:
        parts.append("elapsed_since_last_shutdown_or_persistence=unavailable")
    for label in ("last_offline_started_at", "last_boot_time", "last_persisted_at"):
        value = getattr(continuity, label, None)
        if value is not None:
            parts.append(f"{label}={_iso_value(value)}")
    try:
        parts.append(
            f"continuous_present_score={float(getattr(continuity, 'continuous_present_score', 1.0) or 0.0):.2f}"
        )
    except (TypeError, ValueError):
        parts.append("continuous_present_score=unknown")
    breadcrumb = _latest_prior_continuity_breadcrumb(continuity)
    if breadcrumb:
        breadcrumb = breadcrumb.replace(";", ",")
        if len(breadcrumb) > 180:
            breadcrumb = f"{breadcrumb[:177].rstrip()}..."
        parts.append(f"latest_breadcrumb={breadcrumb}")
    return "- Continuity clock: " + "; ".join(parts)


def _latest_prior_continuity_breadcrumb(continuity: Any) -> str:
    candidates = list(getattr(continuity, "continuity_breadcrumbs", []) or [])
    fallback = str(getattr(continuity, "continuity_breadcrumb", "") or "")
    if fallback and fallback not in candidates:
        candidates.append(fallback)
    for candidate in reversed(candidates):
        text = " ".join(str(candidate or "").split())
        if text and not _is_current_turn_continuity_breadcrumb(text):
            return text
    return ""


def _is_current_turn_continuity_breadcrumb(text: str) -> bool:
    lowered = text.lower()
    return (
        "intent: start burst: conversation burst for" in lowered
        or "intent: tool loop persisted intermediate messages" in lowered
        or "intent: start burst: cycle burst for cycle" in lowered
        or "intent: resume context after" in lowered
    )


def _iso_value(value: Any) -> str:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    if total < 90:
        return f"{total} seconds"
    minutes = total // 60
    if minutes < 90:
        return f"{minutes} minutes"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    if hours < 36:
        suffix = f", {remaining_minutes} minutes" if remaining_minutes else ""
        return f"{hours} hours{suffix}"
    days = hours // 24
    remaining_hours = hours % 24
    suffix = f", {remaining_hours} hours" if remaining_hours else ""
    return f"{days} days{suffix}"


def _render_feature_inventory(runtime: Any) -> list[str]:
    """Summarize high-level capabilities from live registries and tools."""

    tool_names = _tool_names(runtime)
    capabilities = _capability_descriptors(runtime)
    capability_ids = {
        str(getattr(descriptor, "capability_id", "") or "")
        for descriptor in capabilities
    }
    capability_sources = {
        _value(getattr(descriptor, "source", ""))
        for descriptor in capabilities
    }
    plugin_owner_ids = {
        str(getattr(descriptor, "owner_id", "") or "")
        for descriptor in capabilities
        if _value(getattr(descriptor, "source", "")) == "plugin"
    }

    lines: list[str] = []
    if "plugin" in capability_sources or plugin_owner_ids:
        owner_count = len(plugin_owner_ids)
        owner_note = f"; enabled_plugin_owners={owner_count}" if owner_count else ""
        lines.append(
            "- feature plugin/extension system: available; "
            f"evidence=platform capability registry source=plugin{owner_note}"
        )

    skill_count = _skill_count(runtime)
    if skill_count and skill_count > 0:
        lines.append(
            f"- feature skills system: available; evidence=SkillRegistry has {skill_count} loaded skills"
        )
    elif any(capability_id.startswith("plugin:") for capability_id in capability_ids):
        lines.append(
            "- feature skills system: partial; evidence=plugins can register SkillEntry-backed capabilities"
        )

    if {"browser_start", "browser_navigate", "browser_snapshot"} <= tool_names and (
        "browser_click" in tool_names or "browser_type" in tool_names
    ):
        lines.append(
            "- feature browser automation: available; evidence=browser_start, browser_navigate, "
            "browser_snapshot, browser_click/browser_type"
        )

    if "mcp" in capability_sources or {"mcp_list_servers", "mcp_register_server_tools"} & tool_names:
        lines.append(
            "- feature MCP integration: available; evidence=mcp_list_servers, "
            "mcp_register_server_tools, source=mcp capability registration"
        )

    voice_tools = [
        name
        for name in (
            "desktop_context_speak",
            "desktop_context_capture",
            "phone_get_status",
            "phone_call_owner",
        )
        if name in tool_names
    ]
    if voice_tools:
        lines.append(
            "- feature voice/contact channels: available; evidence="
            + ", ".join(voice_tools)
        )

    if {"workflow_create_schedule", "workflow_list_schedules"} <= tool_names:
        schedule_actions = ["workflow_create_schedule", "workflow_list_schedules"]
        if "workflow_get_schedule" in tool_names:
            schedule_actions.append("workflow_get_schedule(detail/run history)")
        if "workflow_cancel_schedule" in tool_names:
            schedule_actions.append("workflow_cancel_schedule(unschedules/cancels by schedule_id)")
        if "workflow_update_schedule" in tool_names:
            schedule_actions.append("workflow_update_schedule(status=cancelled pauses/unschedules)")
        lines.append(
            "- feature scheduled follow-up: available; evidence="
            + ", ".join(schedule_actions)
        )

    if {"workflow_create_commitment", "workflow_list_commitments"} <= tool_names:
        commitment_actions = ["workflow_create_commitment", "workflow_list_commitments"]
        if "workflow_get_commitment" in tool_names:
            commitment_actions.append("workflow_get_commitment(detail)")
        lines.append(
            "- feature commitment tracking: available; evidence="
            + ", ".join(commitment_actions)
        )

    terminal_tools = [
        name
        for name in (
            "bash_run_command",
            "cli_discover_command",
            "pty_start",
            "pty_interact",
            "process_start",
        )
        if name in tool_names
    ]
    if terminal_tools:
        lines.append(
            "- feature Linux terminal/process use: available; evidence="
            + ", ".join(terminal_tools)
        )

    if "tui_playwright_open" in tool_names:
        lines.append(
            "- feature browser-driven terminal UI use: available; evidence="
            "tui_playwright_open, browser_start, browser_navigate, browser_snapshot"
        )

    if any(name.startswith("google_workspace_") for name in tool_names):
        lines.append(
            "- feature Google Workspace access: available; evidence=google_workspace_* tools"
        )

    if any(name.startswith("himalaya_email_") for name in tool_names):
        lines.append(
            "- feature local email access: available; "
            "evidence=himalaya_email_* tools via configured Himalaya CLI"
        )

    if {"self_inspection_query", "wellbeing_query"} & tool_names:
        lines.append(
            "- feature self-inspection/wellbeing diagnostics: available; "
            "evidence=self_inspection_query/wellbeing_query"
        )

    cancellation_evidence = []
    if "workflow_cancel_project" in tool_names:
        cancellation_evidence.append("workflow_cancel_project(project compost/cancel)")
    if "workflow_cancel_task" in tool_names:
        cancellation_evidence.append("workflow_cancel_task(BAA task cancel only)")
    if "workflow_list_tasks" in tool_names:
        cancellation_evidence.append("workflow_list_tasks(find BAA task IDs)")
    if "workflow_get_task" in tool_names:
        cancellation_evidence.append("workflow_get_task(BAA task detail)")
    if "workflow_cancel_schedule" in tool_names:
        cancellation_evidence.append("workflow_cancel_schedule(OpenCAS schedule cancel)")
    if "workflow_update_schedule" in tool_names:
        cancellation_evidence.append("workflow_update_schedule(status=cancelled for schedules)")
    if cancellation_evidence:
        lines.append(
            "- feature project/task composting and cancellation: available; "
            "evidence=" + ", ".join(cancellation_evidence)
        )

    return lines


def _render_capabilities(runtime: Any, *, max_items: int) -> list[str]:
    capabilities = _capability_descriptors(runtime)

    rendered: list[str] = []
    for descriptor in capabilities[:max_items]:
        metadata = getattr(descriptor, "metadata", {}) or {}
        risk_tier = metadata.get("risk_tier")
        status = _value(getattr(descriptor, "status", "unknown")) or "unknown"
        tools = ", ".join(getattr(descriptor, "tool_names", []) or [])
        description = " ".join(str(getattr(descriptor, "description", "") or "").split())
        if len(description) > 160:
            description = f"{description[:157].rstrip()}..."

        parts = [
            f"- capability {getattr(descriptor, 'capability_id', 'unknown')}",
            f"status={status}",
        ]
        if risk_tier:
            parts.append(f"risk={risk_tier}")
        if tools:
            parts.append(f"tools={tools}")
        if description:
            parts.append(f"description={description}")
        rendered.append("; ".join(parts))
    return rendered


def _capability_descriptors(runtime: Any) -> list[Any]:
    registry = getattr(runtime, "capability_registry", None)
    if registry is None:
        registry = getattr(getattr(runtime, "ctx", None), "capability_registry", None)
    if registry is None or not hasattr(registry, "list_capabilities"):
        return []

    try:
        return list(registry.list_capabilities())
    except Exception:
        return []


def _skill_count(runtime: Any) -> int | None:
    registry = getattr(runtime, "skill_registry", None)
    if registry is None:
        registry = getattr(getattr(runtime, "ctx", None), "skill_registry", None)
    if registry is None or not hasattr(registry, "list_skills"):
        return None

    try:
        return len(list(registry.list_skills()))
    except Exception:
        return None


def _render_tools(runtime: Any, *, max_items: int) -> list[str]:
    tools = getattr(runtime, "tools", None)
    if tools is None or not hasattr(tools, "list_tools"):
        return []

    try:
        entries = _enabled_tool_entries(runtime, list(tools.list_tools()))
    except Exception:
        return []

    rendered: list[str] = []
    for entry in sorted(entries, key=lambda item: str(getattr(item, "name", "")))[:max_items]:
        name = str(getattr(entry, "name", "") or "").strip()
        if not name:
            continue
        risk_tier = _value(getattr(entry, "risk_tier", "unknown")) or "unknown"
        description = " ".join(str(getattr(entry, "description", "") or "").split())
        if len(description) > 220:
            description = f"{description[:217].rstrip()}..."
        parts = [f"- tool {name}", f"risk={risk_tier}"]
        if description:
            parts.append(f"description={description}")
        rendered.append("; ".join(parts))
    return rendered


def _render_opencas_inspectability(runtime: Any) -> str | None:
    available = sorted(_tool_names(runtime) & _OPENCAS_INSPECTABILITY_TOOLS)
    if not available:
        return None
    listed = ", ".join(available)
    return (
        "- OpenCAS operational inspectability surfaces: "
        f"{listed}, daydream records, schedule records, and receipts are "
        "operator-facing diagnostic records, not Secure Core secrets. "
        "/api/inner-life/runtime-truth is the compact current-state packet for "
        "fresh-agent self-grounding."
    )


def _tool_names(runtime: Any) -> set[str]:
    tools = getattr(runtime, "tools", None)
    if tools is None or not hasattr(tools, "list_tools"):
        return set()

    try:
        entries = _enabled_tool_entries(runtime, list(tools.list_tools()))
    except Exception:
        return set()
    names: set[str] = set()
    for entry in entries:
        name = str(getattr(entry, "name", "") or "").strip()
        if name:
            names.add(name)
    return names


def _enabled_tool_entries(runtime: Any, entries: list[Any]) -> list[Any]:
    lifecycle = getattr(getattr(runtime, "ctx", None), "plugin_lifecycle", None)
    is_tool_disabled = getattr(lifecycle, "is_tool_disabled", None)
    if not callable(is_tool_disabled):
        return entries
    enabled: list[Any] = []
    for entry in entries:
        name = str(getattr(entry, "name", "") or "")
        try:
            disabled = bool(is_tool_disabled(name))
        except Exception:
            disabled = False
        if not disabled:
            enabled.append(entry)
    return enabled


def _render_desktop_context(runtime: Any) -> str | None:
    service = getattr(runtime, "desktop_context", None)
    if service is None or not hasattr(service, "status"):
        return None
    try:
        status = service.status()
    except Exception:
        return None
    if not isinstance(status, dict):
        return None

    config = status.get("config") if isinstance(status.get("config"), dict) else {}
    enabled = config.get("enabled")
    backend_available = status.get("capture_backend_available")
    return (
        "- desktop_context status; "
        f"enabled={enabled}; "
        f"capture_backend_available={backend_available}; "
        "desktop capture is consent/config gated even when capture tools are registered"
    )


def _render_somatic_snapshot(runtime: Any) -> str | None:
    somatic = getattr(getattr(runtime, "ctx", None), "somatic", None)
    state = getattr(somatic, "state", None)
    if state is None:
        return None
    pieces: list[str] = []
    tag = str(getattr(state, "somatic_tag", "") or "").strip()
    if tag:
        pieces.append(f"tag={tag}")
    for name in (
        "valence",
        "arousal",
        "energy",
        "focus",
        "fatigue",
        "tension",
        "certainty",
    ):
        value = getattr(state, name, None)
        if value is None:
            continue
        try:
            pieces.append(f"{name}={float(value):.2f}")
        except (TypeError, ValueError):
            pieces.append(f"{name}={value}")
    if not pieces:
        return None
    return "- Current somatic snapshot: " + ", ".join(pieces)


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


_OPENCAS_INSPECTABILITY_TOOLS = {
    "runtime_status",
    "workflow_status",
    "self_inspection_query",
    "wellbeing_query",
}
