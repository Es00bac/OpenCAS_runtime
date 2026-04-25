"""Tool registry for OpenCAS."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from opencas.autonomy.models import ActionRiskTier
from opencas.infra import PRE_COMMAND_EXECUTE, PRE_FILE_WRITE, PRE_TOOL_EXECUTE
from opencas.telemetry import EventKind, Tracer

from .models import ToolEntry, ToolResult
from .validation import ToolValidationPipeline


class ToolRegistry:
    """Capability-driven tool registry with policy-aware routing."""

    def __init__(
        self,
        tracer: Optional[Tracer] = None,
        validation_pipeline: Optional[ToolValidationPipeline] = None,
        hook_bus: Optional[Any] = None,
    ) -> None:
        self._tools: Dict[str, ToolEntry] = {}
        self.tracer = tracer
        self.validation_pipeline = validation_pipeline
        self.hook_bus = hook_bus
        self._plugin_tools: Dict[str, str] = {}

    def register(
        self,
        name: str,
        description: str,
        adapter,
        risk_tier: ActionRiskTier,
        parameters: Optional[Dict[str, Any]] = None,
        plugin_id: Optional[str] = None,
    ) -> None:
        """Register a tool by name."""
        self._tools[name] = ToolEntry(
            name=name,
            description=description,
            adapter=adapter,
            risk_tier=risk_tier,
            parameters=parameters or {"type": "object", "properties": {}},
        )
        if plugin_id is not None:
            self._plugin_tools[name] = plugin_id
        self._trace("tool_registered", {"name": name, "tier": risk_tier.value, "plugin_id": plugin_id})

    def unregister(self, name: str) -> None:
        """Remove a single tool by name."""

        self._tools.pop(name, None)
        self._plugin_tools.pop(name, None)

    def unregister_owner(self, owner_id: str) -> list[str]:
        """Remove all tools owned by *owner_id* and return their names."""

        removed: list[str] = []
        for tool_name, plugin_owner in list(self._plugin_tools.items()):
            if plugin_owner == owner_id:
                removed.append(tool_name)
                self.unregister(tool_name)
        return removed

    def get(self, name: str) -> Optional[ToolEntry]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolEntry]:
        return list(self._tools.values())

    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments (synchronous)."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # If we're already in an async context but execute() was called,
            # run the async version in a thread-safe way. For simplicity,
            # delegate to asyncio.run_coroutine_threadsafe when possible,
            # but since we're likely on the same thread, just run it directly.
            try:
                return asyncio.run_coroutine_threadsafe(
                    self.execute_async(name, args), loop
                ).result()
            except Exception as exc:
                return ToolResult(
                    success=False,
                    output=str(exc),
                    metadata={"error_type": type(exc).__name__},
                )
        # No running loop: run async version in a new event loop
        try:
            return asyncio.run(self.execute_async(name, args))
        except Exception as exc:
            return ToolResult(
                success=False,
                output=str(exc),
                metadata={"error_type": type(exc).__name__},
            )

    async def execute_async(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with the given arguments (asynchronous)."""
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(
                success=False,
                output=f"Tool not found: {name}",
                metadata={},
            )
        self._trace(
            "tool_executing",
            {"name": name, "tier": entry.risk_tier.value},
        )
        # Run hook bus for high-risk tools
        if self.hook_bus is not None and entry.risk_tier in (
            ActionRiskTier.WORKSPACE_WRITE,
            ActionRiskTier.SHELL_LOCAL,
            ActionRiskTier.NETWORK,
            ActionRiskTier.EXTERNAL_WRITE,
            ActionRiskTier.DESTRUCTIVE,
        ):
            hook_result = self.hook_bus.run(
                PRE_TOOL_EXECUTE,
                {"tool_name": name, "args": args, "risk_tier": entry.risk_tier.value},
            )
            if not hook_result.allowed:
                return ToolResult(
                    success=False,
                    output=f"Hook blocked execution: {hook_result.reason}",
                    metadata={"hook_blocked": True, "reason": hook_result.reason},
                )
            if hook_result.mutated_context and "args" in hook_result.mutated_context:
                args = hook_result.mutated_context["args"]

        # Run specific hooks for command and file-write tools
        if self.hook_bus is not None:
            if name == "bash_run_command":
                cmd_hook_result = self.hook_bus.run(
                    PRE_COMMAND_EXECUTE,
                    {"command": args.get("command", ""), "args": args},
                )
                if not cmd_hook_result.allowed:
                    return ToolResult(
                        success=False,
                        output=f"Command blocked by hook: {cmd_hook_result.reason}",
                        metadata={"hook_blocked": True, "reason": cmd_hook_result.reason},
                    )
                if cmd_hook_result.mutated_context and "args" in cmd_hook_result.mutated_context:
                    args = cmd_hook_result.mutated_context["args"]
            elif name in ("fs_write_file", "edit_file"):
                file_hook_result = self.hook_bus.run(
                    PRE_FILE_WRITE,
                    {"tool_name": name, "args": args},
                )
                if not file_hook_result.allowed:
                    return ToolResult(
                        success=False,
                        output=f"File write blocked by hook: {file_hook_result.reason}",
                        metadata={"hook_blocked": True, "reason": file_hook_result.reason},
                    )
                if file_hook_result.mutated_context and "args" in file_hook_result.mutated_context:
                    args = file_hook_result.mutated_context["args"]

        if self.validation_pipeline is not None:
            validation = self.validation_pipeline.validate(name, args)
            if not validation.allowed:
                return ToolResult(
                    success=False,
                    output=f"Tool validation failed: {validation.reason}",
                    metadata={
                        "validation_error": validation.reason,
                        "warnings": validation.warnings,
                    },
                )
        else:
            validation = None
        try:
            import inspect
            if inspect.iscoroutinefunction(entry.adapter):
                result = await entry.adapter(name, args)
            elif hasattr(entry.adapter, "__call__") and inspect.iscoroutinefunction(
                getattr(type(entry.adapter), "__call__", None)
            ):
                result = await entry.adapter(name, args)
            else:
                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: entry.adapter(name, args)
                )
        except Exception as exc:
            result = ToolResult(
                success=False,
                output=str(exc),
                metadata={"error_type": type(exc).__name__},
            )
        if validation is not None:
            result.metadata = {
                **result.metadata,
                "validation_warnings": validation.warnings,
                "command_permission_class": validation.command_permission_class,
                "command_family": validation.command_family,
            }
        self._trace(
            "tool_executed",
            {
                "name": name,
                "success": result.success,
                "command_permission_class": result.metadata.get("command_permission_class"),
                "command_family": result.metadata.get("command_family"),
                "validation_warnings": result.metadata.get("validation_warnings", []),
            },
        )
        return result

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"ToolRegistry: {event}",
                payload,
            )
