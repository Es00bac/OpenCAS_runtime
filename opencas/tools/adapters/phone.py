"""Phone bridge tools for OpenCAS."""

from __future__ import annotations

from typing import Any, Dict

from opencas.tools.models import ToolResult


class PhoneToolAdapter:
    """Expose the runtime phone bridge through the normal tool surface."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "phone_get_status":
                getter = getattr(self.runtime, "phone_status", None)
                if not callable(getter):
                    return ToolResult(False, "Phone bridge is not available", {})
                status = await getter()
                return ToolResult(True, str(status), {"status": status})

            if name == "phone_call_owner":
                caller = getattr(self.runtime, "call_owner_via_phone", None)
                if not callable(caller):
                    return ToolResult(False, "Phone bridge is not available", {})
                message = str(args.get("message") or "").strip()
                reason = str(args.get("reason") or "").strip()
                result = await caller(message=message, reason=reason)
                return ToolResult(
                    True,
                    f"Placed owner phone call ({result.get('call_sid') or 'pending'}).",
                    result,
                )

            return ToolResult(False, f"Unknown phone tool: {name}", {})
        except Exception as exc:
            return ToolResult(False, str(exc), {"error_type": type(exc).__name__})
