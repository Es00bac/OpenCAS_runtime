"""Tool, plugin, and MCP execution helpers for AgentRuntime.

This module owns the runtime-facing tool execution seam so `AgentRuntime`
stays orchestration-shaped instead of carrying command assessment, approval,
plugin lifecycle, and MCP registration logic inline.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opencas.autonomy.models import ActionRequest, ActionRiskTier, ApprovalLevel
from opencas.somatic import AppraisalEventType
from opencas.tools import ToolUseContext
from opencas.tools.validation import assess_command

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


def build_runtime_tool_use_context(
    runtime: "AgentRuntime",
    session_id: str,
) -> ToolUseContext:
    """Create a tool-use context with the active plan, if one exists."""
    return ToolUseContext(runtime=runtime, session_id=session_id)


async def hydrate_runtime_tool_use_context(
    runtime: "AgentRuntime",
    ctx: ToolUseContext,
) -> ToolUseContext:
    """Populate plan-mode state on a freshly created tool-use context."""
    plan_store = getattr(runtime.ctx, "plan_store", None)
    if plan_store is not None:
        try:
            active_plans = await plan_store.list_active()
            if active_plans:
                ctx.plan_mode = True
                ctx.active_plan_id = active_plans[0].plan_id
        except Exception:
            pass
    return ctx


async def discover_and_register_mcp_tools(
    runtime: "AgentRuntime",
) -> List[str]:
    """Eagerly discover and register all configured MCP tools."""
    registry = getattr(runtime.ctx, "mcp_registry", None)
    if registry is None:
        return []
    registered: List[str] = []
    for server_name in list(registry._configs.keys()):
        ok = await registry.ensure_initialized(server_name)
        if not ok:
            continue
        registered.extend(_register_mcp_tools(runtime, registry, server_name))
    return registered


async def register_mcp_server_tools(
    runtime: "AgentRuntime",
    server_name: str,
) -> List[str]:
    """Lazy-register tools from a specific MCP server."""
    registry = getattr(runtime.ctx, "mcp_registry", None)
    if registry is None:
        return []
    ok = await registry.ensure_initialized(server_name)
    if not ok:
        return []
    return _register_mcp_tools(runtime, registry, server_name, skip_existing=True)


def make_mcp_list_servers_adapter(runtime: "AgentRuntime"):
    """Expose MCP server inventory through the normal tool registry."""
    from opencas.tools.models import ToolResult

    async def adapter(name: str, args: Dict[str, Any]) -> ToolResult:
        registry = getattr(runtime.ctx, "mcp_registry", None)
        if registry is None:
            return ToolResult(success=True, output="No MCP registry configured.", metadata={})
        configs = getattr(registry, "_configs", {})
        initialized = getattr(registry, "_initialized", set())
        lines = []
        for server_name, cfg in configs.items():
            status = "initialized" if server_name in initialized else "not_initialized"
            lines.append(f"{server_name}: {status} (command: {cfg.command})")
        return ToolResult(
            success=True,
            output="\n".join(lines) or "No MCP servers configured.",
            metadata={"servers": list(configs.keys()), "initialized": list(initialized)},
        )

    return adapter


def make_mcp_register_adapter(runtime: "AgentRuntime"):
    """Expose lazy MCP registration through the tool registry."""
    from opencas.tools.models import ToolResult

    async def adapter(name: str, args: Dict[str, Any]) -> ToolResult:
        server_name = str(args.get("server_name", ""))
        try:
            registered = await register_mcp_server_tools(runtime, server_name)
            return ToolResult(
                success=True,
                output=f"Registered {len(registered)} tools from server '{server_name}'.",
                metadata={"registered": registered, "server": server_name},
            )
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"server": server_name})

    return adapter


async def execute_runtime_tool(
    runtime: "AgentRuntime",
    name: str,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a tool through the registry after self-approval."""
    entry = runtime.tools.get(name)
    if entry is None:
        return {"success": False, "output": f"Tool not found: {name}"}

    plugin_lifecycle = getattr(runtime.ctx, "plugin_lifecycle", None)
    if plugin_lifecycle is not None and plugin_lifecycle.is_tool_disabled(name):
        return {
            "success": False,
            "output": f"Tool {name} is disabled because its plugin is disabled.",
        }

    request = ActionRequest(
        tier=entry.risk_tier,
        description=f"tool {name}: {entry.description}",
        tool_name=name,
        payload=_build_tool_request_payload(name, args),
    )
    approval = await handle_runtime_action(runtime, request)
    if not approval["approved"]:
        if entry.risk_tier != ActionRiskTier.READONLY:
            runtime.ctx.somatic.bump_from_work(intensity=0.05, success=False)
        await runtime.ctx.somatic.emit_appraisal_event(
            AppraisalEventType.TOOL_REJECTED,
            source_text=f"tool {name} rejected",
            trigger_event_id=str(request.action_id),
            meta={"tool_name": name, "args": args},
        )
        return {
            "success": False,
            "output": f"Tool execution blocked: {approval['decision'].reasoning}",
            "decision": approval["decision"],
        }

    result = await runtime.tools.execute_async(name, args)
    if entry.risk_tier != ActionRiskTier.READONLY:
        runtime.ctx.somatic.bump_from_work(intensity=0.1, success=result.success)
    await runtime.ctx.somatic.emit_appraisal_event(
        AppraisalEventType.TOOL_EXECUTED,
        source_text=f"tool {name} executed",
        trigger_event_id=str(request.action_id),
        meta={"tool_name": name, "success": result.success},
    )
    if result.success:
        resolved_goals = await runtime.executive.check_goal_resolution(result.output)
        for goal in resolved_goals:
            await runtime.ctx.somatic.emit_appraisal_event(
                AppraisalEventType.GOAL_ACHIEVED,
                source_text=f"Goal achieved: {goal}",
                trigger_event_id=str(request.action_id),
            )
    runtime._sync_executive_snapshot()
    return {
        "success": result.success,
        "output": result.output,
        "metadata": result.metadata,
    }


async def install_runtime_plugin(
    runtime: "AgentRuntime",
    path: Path | str,
) -> Optional[Any]:
    """Install a plugin from a directory or manifest file."""
    lifecycle = getattr(runtime.ctx, "plugin_lifecycle", None)
    if lifecycle is None:
        return None
    return await lifecycle.install(path)


async def uninstall_runtime_plugin(
    runtime: "AgentRuntime",
    plugin_id: str,
) -> None:
    """Uninstall a plugin."""
    lifecycle = getattr(runtime.ctx, "plugin_lifecycle", None)
    if lifecycle is not None:
        await lifecycle.uninstall(plugin_id)


async def enable_runtime_plugin(
    runtime: "AgentRuntime",
    plugin_id: str,
) -> None:
    """Enable a plugin."""
    lifecycle = getattr(runtime.ctx, "plugin_lifecycle", None)
    if lifecycle is not None:
        await lifecycle.enable(plugin_id)


async def disable_runtime_plugin(
    runtime: "AgentRuntime",
    plugin_id: str,
) -> None:
    """Disable a plugin."""
    lifecycle = getattr(runtime.ctx, "plugin_lifecycle", None)
    if lifecycle is not None:
        await lifecycle.disable(plugin_id)


async def submit_runtime_repair(
    runtime: "AgentRuntime",
    task: Any,
) -> Any:
    """Submit a repair task to the bounded assistant agent."""
    await runtime.baa.start()
    return await runtime.baa.submit(task)


async def handle_runtime_action(
    runtime: "AgentRuntime",
    request: ActionRequest,
) -> Dict[str, Any]:
    """Evaluate an action through the self-approval ladder."""
    decision = runtime.approval.evaluate(request)
    await runtime.approval.maybe_record(decision, request, decision.score)
    runtime._trace(
        "handle_action",
        {
            "action_id": str(request.action_id),
            "decision": decision.level.value,
            "confidence": decision.confidence,
        },
    )
    return {
        "approved": decision.level in (
            ApprovalLevel.CAN_DO_NOW,
            ApprovalLevel.CAN_DO_WITH_CAUTION,
        ),
        "decision": decision,
    }


def _register_mcp_tools(
    runtime: "AgentRuntime",
    registry: Any,
    server_name: str,
    *,
    skip_existing: bool = False,
) -> List[str]:
    from opencas.tools.mcp_adapter import make_mcp_tool_adapter

    registered: List[str] = []
    for tool_meta in registry._tools.get(server_name, {}).values():
        tool_name = tool_meta["name"]
        if skip_existing and tool_name in runtime.tools._tools:
            continue
        adapter = make_mcp_tool_adapter(registry, server_name, tool_name)
        runtime.tools.register(
            tool_name,
            tool_meta.get("description", f"MCP tool {tool_name}"),
            adapter,
            ActionRiskTier.READONLY,
            tool_meta.get("inputSchema", {"type": "object"}),
            plugin_id=f"mcp:{server_name}",
        )
        registered.append(tool_name)
    return registered


def _build_tool_request_payload(
    name: str,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    request_payload = dict(args)
    if name in {
        "bash_run_command",
        "pty_start",
        "pty_interact",
        "pty_write",
        "pty_poll",
        "pty_observe",
        "process_start",
        "workflow_supervise_session",
    }:
        command = str(args.get("command", "")).strip()
        if command:
            assessment = assess_command(command)
            request_payload.update(
                {
                    "command_family": assessment.family,
                    "command_permission_class": assessment.permission_class,
                    "command_executable": assessment.executable,
                    "command_subcommand": assessment.subcommand,
                }
            )
        elif name in {"pty_interact", "pty_write", "workflow_supervise_session"} and args.get("session_id"):
            request_payload.update(
                {
                    "command_family": "interactive_session",
                    "command_permission_class": "bounded_write",
                }
            )
        elif name in {"pty_poll", "pty_observe"} and args.get("session_id"):
            request_payload.update(
                {
                    "command_family": "interactive_session",
                    "command_permission_class": "read_only",
                }
            )
    elif name in {"pty_kill", "pty_remove"}:
        request_payload.update(
            {
                "command_family": "interactive_session",
                "command_permission_class": "bounded_write",
            }
        )
    return request_payload
