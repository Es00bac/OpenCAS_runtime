"""Ground model-facing reviews in the runtime's actual capability surface."""

from __future__ import annotations

from typing import Any


def build_runtime_capability_context(runtime: Any, *, max_items: int = 50) -> str:
    """Render concise runtime capability evidence for refusal/review prompts."""

    lines: list[str] = ["Runtime capability evidence:"]
    capability_lines = _render_capabilities(runtime, max_items=max_items)
    tool_lines = _render_tools(runtime, max_items=max_items)
    desktop_line = _render_desktop_context(runtime)

    if capability_lines:
        lines.extend(capability_lines)
    if tool_lines:
        lines.extend(tool_lines)
    if desktop_line:
        lines.append(desktop_line)
    if not capability_lines and not tool_lines and not desktop_line:
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
        ]
    )
    return "\n".join(lines)


def _render_capabilities(runtime: Any, *, max_items: int) -> list[str]:
    registry = getattr(runtime, "capability_registry", None)
    if registry is None:
        registry = getattr(getattr(runtime, "ctx", None), "capability_registry", None)
    if registry is None or not hasattr(registry, "list_capabilities"):
        return []

    try:
        capabilities = list(registry.list_capabilities())
    except Exception:
        return []

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


def _render_tools(runtime: Any, *, max_items: int) -> list[str]:
    tools = getattr(runtime, "tools", None)
    if tools is None or not hasattr(tools, "list_tools"):
        return []

    try:
        entries = list(tools.list_tools())
    except Exception:
        return []

    rendered: list[str] = []
    for entry in sorted(entries, key=lambda item: str(getattr(item, "name", "")))[:max_items]:
        name = str(getattr(entry, "name", "") or "").strip()
        if not name:
            continue
        risk_tier = _value(getattr(entry, "risk_tier", "unknown")) or "unknown"
        rendered.append(f"- tool {name}; risk={risk_tier}")
    return rendered


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


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")
