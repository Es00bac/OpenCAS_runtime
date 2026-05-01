"""Tools for policy-limited owner contact initiated by OpenCAS."""

from __future__ import annotations

from typing import Any, Dict

from opencas.tools.models import ToolResult


class InitiativeContactToolAdapter:
    """Expose proactive owner-contact policy through the normal tool loop."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        service = getattr(self.runtime, "initiative_contact", None)
        if service is None:
            return ToolResult(False, "Initiative contact service is not available.", {})

        if name == "initiative_contact_status":
            status = service.status()
            return ToolResult(True, str(status), {"status": status})

        if name == "initiative_contact_owner":
            message = str(args.get("message") or "").strip()
            reason = str(args.get("reason") or "").strip()
            urgency = str(args.get("urgency") or "normal").strip() or "normal"
            channel = str(args.get("channel") or "auto").strip() or "auto"
            result = await service.request_contact(
                message=message,
                reason=reason,
                urgency=urgency,  # type: ignore[arg-type]
                source="tool",
                channel=channel,  # type: ignore[arg-type]
            )
            ok = result.get("status") == "sent"
            return ToolResult(ok, str(result), result)

        return ToolResult(False, f"Unknown initiative contact tool: {name}", {})
