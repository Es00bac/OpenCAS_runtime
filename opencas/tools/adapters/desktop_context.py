"""Tool adapter for the desktop-context body-double skill."""

from __future__ import annotations

from typing import Any, Dict

from opencas.tools.models import ToolResult


class DesktopContextToolAdapter:
    """Expose desktop observation through normal tool execution."""

    def __init__(self, runtime: Any = None, tools: Any = None) -> None:
        self.runtime = runtime
        self.tools = tools

    def _runtime(self) -> Any:
        if self.runtime is not None:
            return self.runtime
        if self.tools is not None:
            return getattr(self.tools, "runtime", None)
        return None

    def _service(self) -> Any:
        runtime = self._runtime()
        if runtime is None:
            return None
        return getattr(runtime, "desktop_context", None)

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        service = self._service()
        if service is None:
            return ToolResult(False, "Desktop context service is not available.", {})

        if name == "desktop_context_status":
            status = service.status()
            return ToolResult(True, str(status), {"status": status})

        if name == "desktop_context_configure":
            result = service.configure(**dict(args or {}))
            return ToolResult(True, str(result), result)

        if name == "desktop_context_capture":
            result = await service.capture_once(force=bool((args or {}).get("force", False)))
            return ToolResult(result.get("status") != "failed", str(result), result)

        if name == "desktop_context_observe":
            has_speak = isinstance(args, dict) and "speak" in args
            result = await service.observe_once(
                force=bool((args or {}).get("force", False)),
                reason=str((args or {}).get("reason") or "tool"),
                speak=bool(args.get("speak")) if has_speak else None,
            )
            return ToolResult(result.get("status") != "failed", str(result), result)

        if name == "desktop_context_speak":
            text = str((args or {}).get("text") or "").strip()
            if not text:
                return ToolResult(False, "text is required", {})
            result = await service.speak_text(
                text,
                reason=str((args or {}).get("reason") or "tool"),
            )
            return ToolResult(result.get("status") != "failed", str(result), result)

        return ToolResult(False, f"Unknown desktop context tool: {name}", {})
