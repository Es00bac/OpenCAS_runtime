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

    def _config_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        config = result.get("config") if isinstance(result, dict) else None
        return config if isinstance(config, dict) else {}

    def _status_config(self, status: Dict[str, Any]) -> Dict[str, Any]:
        config = status.get("config") if isinstance(status, dict) else None
        return config if isinstance(config, dict) else {}

    def _enabled_label(self, enabled: Any) -> str:
        return "enabled" if bool(enabled) else "disabled"

    def _task_label(self, task: Any) -> str:
        text = str(task or "").strip()
        return text if text else "no declared task"

    def _status_output(self, status: Dict[str, Any]) -> str:
        config = self._status_config(status)
        enabled = self._enabled_label(config.get("enabled"))
        media = "on" if config.get("media_commentary_mode_enabled") else "off"
        live = "on" if config.get("live_transcription_enabled") else "off"
        pause = "on" if config.get("pause_media_while_speaking", True) else "off"
        task = self._task_label(config.get("declared_task"))
        return (
            f"Body Double is {enabled}. Media commentary is {media}. Live transcription is {live}. "
            f"Pause media while speaking is {pause}. Task: {task}."
        )

    def _configure_output(self, result: Dict[str, Any]) -> str:
        config = self._config_from_result(result)
        enabled = self._enabled_label(config.get("enabled"))
        media = "on" if config.get("media_commentary_mode_enabled") else "off"
        live = "on" if config.get("live_transcription_enabled") else "off"
        pause = "on" if config.get("pause_media_while_speaking", True) else "off"
        task = self._task_label(config.get("declared_task"))
        return (
            f"Body Double is now {enabled}. Media commentary is {media}. Live transcription is {live}. "
            f"Pause media while speaking is {pause}. Task: {task}."
        )

    def _task_output(self, result: Dict[str, Any]) -> str:
        config = self._config_from_result(result)
        task = self._task_label(config.get("declared_task"))
        enabled = bool(config.get("enabled"))
        if task == "no declared task":
            prefix = "Body Double task cleared"
        else:
            prefix = f"Body Double task set: {task}"
        if enabled:
            return f"{prefix}. Body Double is enabled."
        return f"{prefix}. Body Double is currently disabled; enable it before observing the desktop."

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        service = self._service()
        if service is None:
            return ToolResult(False, "Desktop context service is not available.", {})

        if name == "desktop_context_status":
            status = service.status()
            return ToolResult(True, self._status_output(status), {"status": status})

        if name == "desktop_context_configure":
            result = service.configure(**dict(args or {}))
            return ToolResult(True, self._configure_output(result), result)

        if name == "desktop_context_set_task":
            task = str((args or {}).get("task") or "").strip()
            result = service.configure(
                declared_task=task or None,
                declared_task_source=str((args or {}).get("source") or "operator").strip() or "operator",
            )
            return ToolResult(True, self._task_output(result), result)

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
