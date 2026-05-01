"""Tool, plugin, and MCP execution helpers for AgentRuntime.

This module owns the runtime-facing tool execution seam so `AgentRuntime`
stays orchestration-shaped instead of carrying command assessment, approval,
plugin lifecycle, and MCP registration logic inline.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opencas.autonomy.models import ActionRequest, ActionRiskTier, ApprovalLevel
from opencas.governance import (
    AutoReviewMode,
    classify_web_action,
    normalize_auto_review_mode,
    normalize_web_domain,
)
from opencas.infra import POST_ACTION_DECISION, POST_TOOL_EXECUTE, PRE_TOOL_EXECUTE
from opencas.platform import CapabilityDescriptor, CapabilitySource, CapabilityStatus
from opencas.somatic import AppraisalEventType
from opencas.provenance_events_adapter import (
    ProvenanceEventType,
    emit_provenance_event,
)

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


def build_runtime_tool_use_context(
    runtime: "AgentRuntime",
    session_id: str,
) -> ToolUseContext:
    """Create a tool-use context with the active plan, if one exists."""
    from opencas.tools.context import ToolUseContext

    return ToolUseContext(runtime=runtime, session_id=session_id)


async def hydrate_runtime_tool_use_context(
    runtime: "AgentRuntime",
    ctx: ToolUseContext,
) -> ToolUseContext:
    """Populate plan-mode state on a freshly created tool-use context."""
    plan_store = getattr(runtime.ctx, "plan_store", None)
    if plan_store is not None and ctx.task_id:
        try:
            active_plans = await plan_store.list_active(task_id=ctx.task_id)
            if active_plans:
                ctx.plan_mode = True
                ctx.active_plan_id = active_plans[0].plan_id
        except Exception:
            pass
    return ctx


async def resolve_runtime_tool_use_artifact_hint(
    runtime: "AgentRuntime",
    ctx: ToolUseContext,
    *,
    objective: str = "",
) -> Optional[str]:
    """Resolve the best available artifact hint for general tool-use planning."""

    explicit_hint = _clean_artifact_path(ctx.artifact_hint)
    if explicit_hint:
        return explicit_hint

    task_cache: dict[str, Any] = {}

    if ctx.task_id:
        task = await _get_task(runtime, ctx.task_id, task_cache)
        artifact = _artifact_from_task(task)
        if artifact:
            return artifact

    if ctx.active_plan_id:
        artifact = await _artifact_from_plan(runtime, ctx.active_plan_id, task_cache)
        if artifact:
            return artifact

    objective_text = str(objective or "").strip()
    if not objective_text:
        return None

    best_match = await _match_artifact_from_active_state(
        runtime,
        objective=objective_text,
        task_cache=task_cache,
    )
    return best_match


async def discover_and_register_mcp_tools(
    runtime: "AgentRuntime",
) -> List[str]:
    """Eagerly discover and register all configured MCP tools."""
    registry = getattr(runtime.ctx, "mcp_registry", None)
    if registry is None:
        return []
    registered: List[str] = []
    for server_name in list(registry._configs.keys()):
        try:
            registered.extend(await _sync_mcp_server_registration(runtime, registry, server_name))
        except Exception:
            continue
    return registered


async def register_mcp_server_tools(
    runtime: "AgentRuntime",
    server_name: str,
) -> List[str]:
    """Lazy-register tools from a specific MCP server."""
    registry = getattr(runtime.ctx, "mcp_registry", None)
    if registry is None:
        return []
    return await _sync_mcp_server_registration(runtime, registry, server_name)


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
        except Exception as exc:
            failure = _get_mcp_failure_descriptor(runtime, server_name)
            if failure is not None:
                return ToolResult(
                    success=False,
                    output=f"MCP server '{server_name}' failed validation: {', '.join(failure.validation_errors)}",
                    metadata={
                        "server": server_name,
                        "validation_errors": list(failure.validation_errors),
                    },
                )
            return ToolResult(
                success=False,
                output=str(exc),
                metadata={"server": server_name, "error_type": type(exc).__name__},
            )

        failure = _get_mcp_failure_descriptor(runtime, server_name)
        if failure is not None:
            return ToolResult(
                success=False,
                output=f"MCP server '{server_name}' failed validation: {', '.join(failure.validation_errors)}",
                metadata={
                    "server": server_name,
                    "validation_errors": list(failure.validation_errors),
                },
            )
        return ToolResult(
            success=True,
            output=f"Registered {len(registered)} tools from server '{server_name}'.",
            metadata={"registered": registered, "server": server_name},
        )

    return adapter


async def _match_artifact_from_active_state(
    runtime: "AgentRuntime",
    *,
    objective: str,
    task_cache: dict[str, Any],
) -> Optional[str]:
    objective_tokens = _tokenize_text(objective)
    if not objective_tokens:
        return None

    best_score = 0
    best_artifact: Optional[str] = None

    task_store = getattr(runtime.ctx, "tasks", None)
    if task_store is not None:
        try:
            pending_tasks = await task_store.list_pending(limit=25)
        except Exception:
            pending_tasks = []
        for task in pending_tasks:
            task_id = str(getattr(task, "task_id", "") or "")
            if task_id:
                task_cache[task_id] = task
            artifact = _artifact_from_task(task)
            if not artifact:
                continue
            score = _score_artifact_candidate(
                objective_tokens=objective_tokens,
                texts=[getattr(task, "objective", ""), artifact],
            )
            if score > best_score:
                best_score = score
                best_artifact = artifact

    plan_store = getattr(runtime.ctx, "plan_store", None)
    if plan_store is not None:
        try:
            active_plans = await plan_store.list_active()
        except Exception:
            active_plans = []
        for plan in active_plans:
            artifact = await _artifact_from_plan(runtime, plan, task_cache)
            if not artifact:
                continue
            score = _score_artifact_candidate(
                objective_tokens=objective_tokens,
                texts=[getattr(plan, "content", ""), artifact],
            )
            if score > best_score:
                best_score = score
                best_artifact = artifact

    return best_artifact if best_score > 0 else None


async def _artifact_from_plan(
    runtime: "AgentRuntime",
    plan_or_id: Any,
    task_cache: dict[str, Any],
) -> Optional[str]:
    plan_store = getattr(runtime.ctx, "plan_store", None)
    if plan_store is None:
        return None

    plan = plan_or_id
    if isinstance(plan_or_id, str):
        try:
            plan = await plan_store.get_plan(plan_or_id)
        except Exception:
            return None
    if plan is None:
        return None

    task_id = str(getattr(plan, "task_id", "") or "")
    if task_id:
        task = await _get_task(runtime, task_id, task_cache)
        artifact = _artifact_from_task(task)
        if artifact:
            return artifact

    try:
        actions = await plan_store.get_actions(getattr(plan, "plan_id", ""), limit=25)
    except Exception:
        actions = []
    candidate_paths = _artifact_candidates_from_actions(
        actions,
        state_dir=getattr(getattr(runtime.ctx, "config", None), "state_dir", None),
    )
    if not candidate_paths:
        return None
    return candidate_paths[0]


async def _get_task(
    runtime: "AgentRuntime",
    task_id: str,
    task_cache: dict[str, Any],
) -> Optional[Any]:
    cached = task_cache.get(task_id)
    if cached is not None:
        return cached

    task_store = getattr(runtime.ctx, "tasks", None)
    if task_store is None:
        return None
    try:
        task = await task_store.get(task_id)
    except Exception:
        return None
    if task is not None:
        task_cache[task_id] = task
    return task


def _artifact_from_task(task: Any) -> Optional[str]:
    if task is None:
        return None

    meta = getattr(task, "meta", {})
    if isinstance(meta, dict):
        resume_project = meta.get("resume_project")
        if isinstance(resume_project, dict):
            artifact = _clean_artifact_path(resume_project.get("canonical_artifact_path"))
            if artifact:
                return artifact
        artifact = _clean_artifact_path(meta.get("canonical_artifact_path"))
        if artifact:
            return artifact

    artifacts = [
        _clean_artifact_path(path)
        for path in getattr(task, "artifacts", []) or []
    ]
    artifacts = [path for path in artifacts if path]
    if len(artifacts) == 1:
        return artifacts[0]
    return None


def _artifact_candidates_from_actions(actions: List[Any], *, state_dir: Any) -> List[str]:
    scored: dict[str, int] = {}
    for action in actions:
        args = getattr(action, "args", {}) if action is not None else {}
        if not isinstance(args, dict):
            continue
        for key in ("canonical_artifact_path", "artifact_path", "file_path", "path"):
            candidate = _clean_artifact_path(args.get(key))
            if not candidate or _is_plan_scaffold_path(candidate, state_dir):
                continue
            scored[candidate] = scored.get(candidate, 0) + 1
    return [
        path
        for path, _count in sorted(
            scored.items(),
            key=lambda item: (item[1], len(item[0])),
            reverse=True,
        )
    ]


def _is_plan_scaffold_path(path: str, state_dir: Any) -> bool:
    normalized = str(path or "").replace("\\", "/")
    if normalized.startswith(".opencas/plans/") or "/.opencas/plans/" in normalized:
        return True
    if state_dir:
        plans_dir = str(Path(state_dir) / "plans").replace("\\", "/")
        if normalized.startswith(plans_dir):
            return True
    return False


def _score_artifact_candidate(*, objective_tokens: set[str], texts: List[str]) -> int:
    candidate_tokens: set[str] = set()
    for text in texts:
        candidate_tokens.update(_tokenize_text(text))
    return len(objective_tokens & candidate_tokens)


def _tokenize_text(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_./-]+", str(text or "").lower())
        if len(token) >= 4
    }


def _clean_artifact_path(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


async def execute_runtime_tool(
    runtime: "AgentRuntime",
    name: str,
    args: Dict[str, Any],
    *,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
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
        payload=_build_tool_request_payload(runtime, name, args),
    )
    approval = await handle_runtime_action(
        runtime,
        request,
        session_id=session_id,
        task_id=task_id,
        tool_name=name,
        args=args,
    )
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

    hook_bus = getattr(runtime.ctx, "hook_bus", None)
    hook_context: Dict[str, Any] = {
        "tool_name": name,
        "args": args,
        "risk_tier": entry.risk_tier.value,
        "session_id": session_id or getattr(getattr(runtime.ctx, "config", None), "session_id", None),
        "task_id": task_id,
    }
    if hook_bus is not None:
        pre_hook = hook_bus.run(PRE_TOOL_EXECUTE, hook_context)
        if not pre_hook.allowed:
            failure = {
                "success": False,
                "output": f"Tool execution blocked: {pre_hook.reason}",
                "metadata": {"hook_blocked": True, "reason": pre_hook.reason},
            }
            post_context = {
                **hook_context,
                "result_success": False,
                "result_output": failure["output"],
                "result_metadata": failure["metadata"],
            }
            post_hook = hook_bus.run(POST_TOOL_EXECUTE, post_context)
            if post_hook.mutated_context is not None:
                failure["metadata"] = dict(post_hook.mutated_context.get("result_metadata") or failure["metadata"])
            return failure
        if pre_hook.mutated_context is not None:
            hook_context = pre_hook.mutated_context
            args = dict(hook_context.get("args") or args)

    result = await runtime.tools.execute_async(name, args)
    await _record_web_trust_outcome(runtime, request.payload, result.success)
    if entry.risk_tier != ActionRiskTier.READONLY:
        runtime.ctx.somatic.bump_from_work(intensity=0.1, success=result.success)
    await runtime.ctx.somatic.emit_appraisal_event(
        AppraisalEventType.TOOL_EXECUTED,
        source_text=_tool_appraisal_source_text(name, result.output),
        trigger_event_id=str(request.action_id),
        meta={"tool_name": name, "success": result.success},
    )
    if result.success:
        if hook_bus is not None:
            post_context = {
                **hook_context,
                "result_success": True,
                "result_output": result.output,
                "result_metadata": dict(result.metadata or {}),
            }
            post_hook = hook_bus.run(POST_TOOL_EXECUTE, post_context)
            if post_hook.mutated_context is not None:
                result.metadata = dict(post_hook.mutated_context.get("result_metadata") or result.metadata or {})
    result.metadata = dict(result.metadata or {})
    affective_pressure = await _record_affective_tool_result(
        runtime,
        name=name,
        result=result,
        action_id=str(request.action_id),
        session_id=session_id,
        task_id=task_id,
    )
    if affective_pressure:
        result.metadata["affective_pressure"] = affective_pressure
    if result.success:
        result.metadata = dict(result.metadata or {})
        resolved_goals = await runtime.executive.check_goal_resolution(result.output)
        provenance_events = list(result.metadata.get("provenance_events") or [])
        provenance_events.append(
            emit_provenance_event(
                None,
                event_type=ProvenanceEventType.CHECK,
                triggering_artifact=f"tool|default|{name}",
                triggering_action="VERIFY",
                parent_link_id=str(request.action_id),
                linked_link_ids=[str(request.action_id)],
                details={
                    "action_id": str(request.action_id),
                    "result_success": True,
                    "resolved_goal_count": len(resolved_goals),
                },
            ).to_dict()
        )
        for goal in resolved_goals:
            await runtime.ctx.somatic.emit_appraisal_event(
                AppraisalEventType.GOAL_ACHIEVED,
                source_text=f"Goal achieved: {goal}",
                trigger_event_id=str(request.action_id),
            )
        result.metadata["provenance_events"] = provenance_events
    runtime._sync_executive_snapshot()
    return {
        "success": result.success,
        "output": result.output,
        "metadata": result.metadata,
    }


def _tool_appraisal_source_text(name: str, output: str, *, max_chars: int = 1200) -> str:
    """Use actual tool output for appraisal without feeding unbounded text."""
    text = str(output or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    if text:
        return f"tool {name} output: {text}"
    return f"tool {name} executed with empty output"


async def _record_affective_tool_result(
    runtime: "AgentRuntime",
    *,
    name: str,
    result: Any,
    action_id: str,
    session_id: Optional[str],
    task_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    service = getattr(getattr(runtime, "ctx", None), "affective_examinations", None)
    examine = getattr(service, "examine_tool_result", None)
    if not callable(examine):
        return None
    config_session = getattr(getattr(getattr(runtime, "ctx", None), "config", None), "session_id", None)
    try:
        record = await examine(
            session_id=session_id or config_session,
            source_id=action_id,
            tool_name=name,
            success=bool(result.success),
            output=str(result.output or ""),
            meta={"task_id": task_id} if task_id else None,
        )
    except Exception:
        return None
    return record.pressure_metadata()


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
    *,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate an action through the self-approval ladder."""
    args = args or {}
    decision = runtime.approval.evaluate(request)
    auto_review_meta: Dict[str, Any] = {"mode": "default", "eligible": False, "reason": "not_configured"}
    auto_review = getattr(runtime, "auto_review", None) or getattr(runtime, "auto_review_policy", None)
    if auto_review is not None and hasattr(auto_review, "review"):
        decision, auto_review_meta = await auto_review.review(request, decision)
    await runtime.approval.maybe_record(decision, request, decision.score)
    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        trace(
            "handle_action",
            {
                "action_id": str(request.action_id),
                "decision": decision.level.value,
                "confidence": decision.confidence,
                "auto_review": auto_review_meta,
            },
        )
    target_kind = "file" if any(
        str(args.get(key, "") or "").strip() for key in ("file_path", "path")
    ) else "tool"
    target_id = (
        str(args.get("file_path") or args.get("path") or "").strip()
        or str(args.get("session_id") or args.get("process_id") or args.get("plan_id") or args.get("schedule_id") or args.get("commitment_id") or "").strip()
    )
    scope_key = "workspace" if target_kind == "file" and target_id else "default"
    artifact = f"{target_kind}|{scope_key}|{target_id or tool_name or request.tool_name or 'action'}"
    approved = decision.level in (
        ApprovalLevel.CAN_DO_NOW,
        ApprovalLevel.CAN_DO_WITH_CAUTION,
    )
    provenance_events = [
        emit_provenance_event(
            None,
            event_type=ProvenanceEventType.CHECK,
            triggering_artifact=artifact,
            triggering_action="DECIDE",
            parent_link_id=str(request.action_id),
            linked_link_ids=[str(request.action_id)],
            details={
                "action_id": str(request.action_id),
                "approved": approved,
                "decision_level": decision.level.value,
                "tool_name": tool_name or request.tool_name or "",
                "auto_review": auto_review_meta,
            },
        ).to_dict()
    ]
    if not approved:
        provenance_events.append(
            emit_provenance_event(
                None,
                event_type=ProvenanceEventType.BLOCKED,
                triggering_artifact=artifact,
                triggering_action="DECIDE",
                parent_link_id=str(request.action_id),
                linked_link_ids=[str(request.action_id)],
                details={
                    "action_id": str(request.action_id),
                    "reasoning": decision.reasoning,
                    "decision_level": decision.level.value,
                    "auto_review": auto_review_meta,
                    "tool_name": tool_name or request.tool_name or "",
                },
            ).to_dict()
    )
    hook_bus = getattr(runtime.ctx, "hook_bus", None)
    if hook_bus is not None:
        hook_bus.run(
            POST_ACTION_DECISION,
            {
                "session_id": session_id or getattr(getattr(runtime.ctx, "config", None), "session_id", None),
                "task_id": task_id,
                "tool_name": tool_name or request.tool_name or "",
                "args": args or dict(request.payload),
                "risk_tier": request.tier.value,
                "artifact": artifact,
                "target_kind": target_kind,
                "target_id": target_id or tool_name or "",
                "scope_key": scope_key,
                "approved": decision.level in (
                    ApprovalLevel.CAN_DO_NOW,
                    ApprovalLevel.CAN_DO_WITH_CAUTION,
                ),
                "decision_level": decision.level.value,
                "reasoning": decision.reasoning,
                "score": decision.score,
                "auto_review": auto_review_meta,
                "source_trace": {
                    "event": "action_decision",
                    "outcome": "approved" if approved else "escalated",
                    "tool_name": tool_name or request.tool_name or "",
                    "task_id": task_id,
                },
            },
        )
    return {
        "approved": approved,
        "decision": decision,
        "auto_review": auto_review_meta,
        "provenance_events": provenance_events,
    }


async def _sync_mcp_server_registration(
    runtime: "AgentRuntime",
    registry: Any,
    server_name: str,
) -> List[str]:
    """Initialize an MCP server, project its current state, and surface failures."""

    owner_id = _mcp_owner_id(server_name)
    _clear_mcp_owner_state(runtime, owner_id)
    ok = await registry.ensure_initialized(server_name)
    if not ok:
        _register_mcp_registration_failure(
            runtime,
            server_name,
            ["MCP server initialization failed"],
        )
        return []
    try:
        return _register_mcp_tools(runtime, registry, server_name)
    except Exception as exc:
        _clear_mcp_owner_state(runtime, owner_id)
        _register_mcp_registration_failure(runtime, server_name, [str(exc)])
        raise


def _register_mcp_tools(
    runtime: "AgentRuntime",
    registry: Any,
    server_name: str,
) -> List[str]:
    from opencas.tools.mcp_adapter import make_mcp_tool_adapter

    owner_id = _mcp_owner_id(server_name)
    capability_registry = getattr(runtime, "capability_registry", None)
    if capability_registry is not None:
        capability_registry.unregister_owner(owner_id)

    registered: List[str] = []
    for tool_meta in registry._tools.get(server_name, {}).values():
        tool_name = tool_meta["name"]
        adapter = make_mcp_tool_adapter(registry, server_name, tool_name)
        runtime.tools.register(
            tool_name,
            tool_meta.get("description", f"MCP tool {tool_name}"),
            adapter,
            ActionRiskTier.READONLY,
            tool_meta.get("inputSchema", {"type": "object"}),
            plugin_id=owner_id,
        )
        registered.append(tool_name)
        if capability_registry is not None:
            capability_registry.register(
                CapabilityDescriptor(
                    capability_id=_mcp_capability_id(server_name, tool_name),
                    display_name=_mcp_capability_display_name(server_name, tool_name),
                    kind="tool",
                    source=CapabilitySource.MCP,
                    owner_id=owner_id,
                    status=CapabilityStatus.ENABLED,
                    tool_names=[tool_name],
                    metadata={"owner_name": server_name},
                )
            )
    return registered


def _clear_mcp_owner_state(runtime: "AgentRuntime", owner_id: str) -> None:
    capability_registry = getattr(runtime, "capability_registry", None)
    if capability_registry is not None:
        capability_registry.unregister_owner(owner_id)
    tools = getattr(runtime, "tools", None)
    if tools is not None:
        tools.unregister_owner(owner_id)


def _register_mcp_registration_failure(
    runtime: "AgentRuntime",
    server_name: str,
    validation_errors: List[str],
) -> None:
    owner_id = _mcp_owner_id(server_name)
    capability_registry = getattr(runtime, "capability_registry", None)
    if capability_registry is None:
        return
    capability_registry.register(
        CapabilityDescriptor(
            capability_id=owner_id,
            display_name=server_name,
            kind="mcp_server",
            source=CapabilitySource.MCP,
            owner_id=owner_id,
            status=CapabilityStatus.FAILED_VALIDATION,
            validation_errors=list(validation_errors),
            metadata={"owner_name": server_name},
        )
    )


def _mcp_capability_suffix(server_name: str, tool_name: str) -> str:
    normalized = tool_name.strip()
    for prefix in (f"mcp__{server_name}__", f"{server_name}__"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def _mcp_owner_id(server_name: str) -> str:
    return f"mcp:{server_name}"


def _mcp_capability_id(server_name: str, tool_name: str) -> str:
    return f"{_mcp_owner_id(server_name)}.{_mcp_capability_suffix(server_name, tool_name)}"


def _mcp_capability_display_name(server_name: str, tool_name: str) -> str:
    return f"{server_name}:{_mcp_capability_suffix(server_name, tool_name)}"


def _get_mcp_failure_descriptor(
    runtime: "AgentRuntime",
    server_name: str,
) -> CapabilityDescriptor | None:
    capability_registry = getattr(runtime, "capability_registry", None)
    if capability_registry is None:
        return None
    descriptor = capability_registry.get(_mcp_owner_id(server_name))
    if descriptor is None:
        return None
    if descriptor.source is not CapabilitySource.MCP:
        return None
    if descriptor.status is not CapabilityStatus.FAILED_VALIDATION:
        return None
    return descriptor


def _build_tool_request_payload(
    runtime: "AgentRuntime",
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
            from opencas.tools.validation import assess_command

            assessment = assess_command(command)
            request_payload.update(
                {
                    "command_family": assessment.family,
                    "command_permission_class": assessment.permission_class,
                    "command_executable": assessment.executable,
                    "command_subcommand": assessment.subcommand,
                }
            )
            request_payload.update(_classify_shell_command_scope(runtime, args, command))
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
    elif name in {"fs_write_file", "edit_file"}:
        request_payload.update(_classify_workspace_write_scope(runtime, args))
    elif name in {"web_fetch", "web_search", "browser_navigate"}:
        request_payload.update(_classify_web_request_target(name, args))
    elif name in {
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_wait",
        "browser_snapshot",
    }:
        request_payload.update(_classify_browser_session_target(runtime, name, args))
    request_payload.update(_approval_mode_payload(runtime, request_payload))
    return request_payload


def _approval_mode_payload(runtime: "AgentRuntime", payload: Dict[str, Any]) -> Dict[str, Any]:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    try:
        mode = normalize_auto_review_mode(getattr(config, "approval_mode", AutoReviewMode.DEFAULT.value))
    except ValueError:
        mode = AutoReviewMode.DEFAULT
    if mode is not AutoReviewMode.AUTO_REVIEW:
        return {}
    return {
        "approval_mode": mode.value,
        "approval_channel": payload.get("approval_channel") or "on_request",
    }


async def _record_web_trust_outcome(
    runtime: "AgentRuntime",
    payload: Dict[str, Any],
    success: bool,
) -> None:
    web_trust = getattr(runtime.ctx, "web_trust", None)
    if web_trust is None:
        return
    action_class = payload.get("web_action_class")
    if not action_class:
        return
    await web_trust.record_outcome(
        url=payload.get("web_url"),
        domain=payload.get("web_domain"),
        action_class=str(action_class),
        success=success,
    )


def _classify_web_request_target(
    name: str,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    action_class = classify_web_action(name)
    if action_class is None:
        return payload
    payload["web_action_class"] = action_class.value
    if name == "web_search":
        return payload
    url = str(args.get("url", "")).strip()
    if not url:
        return payload
    domain = normalize_web_domain(url)
    if domain:
        payload["web_url"] = url
        payload["web_domain"] = domain
    return payload


def _classify_browser_session_target(
    runtime: "AgentRuntime",
    name: str,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    action_class = classify_web_action(name)
    if action_class is None:
        return payload
    payload["web_action_class"] = action_class.value
    session_id = str(args.get("session_id", "")).strip()
    scope_key = str(args.get("scope_key", "default")).strip() or "default"
    if not session_id:
        return payload
    supervisor = getattr(runtime, "browser_supervisor", None)
    if supervisor is None or not hasattr(supervisor, "describe_session"):
        return payload
    details = supervisor.describe_session(scope_key, session_id)
    if not details:
        return payload
    url = str(details.get("url", "")).strip()
    if not url:
        return payload
    domain = normalize_web_domain(url)
    if not domain:
        return payload
    payload["web_url"] = url
    payload["web_domain"] = domain
    return payload


def _classify_workspace_write_scope(
    runtime: "AgentRuntime",
    args: Dict[str, Any],
) -> Dict[str, Any]:
    file_path = str(args.get("file_path") or args.get("path") or "").strip()
    if not file_path:
        return {}

    resolved = Path(file_path).expanduser().resolve()
    managed_root = runtime.ctx.config.agent_workspace_root()
    plans_root = (runtime.ctx.config.state_dir / "plans").resolve()

    try:
        resolved.relative_to(managed_root)
        return {
            "write_scope": "managed_workspace",
            "managed_workspace_root": str(managed_root),
        }
    except ValueError:
        pass

    try:
        resolved.relative_to(plans_root)
        return {
            "write_scope": "plans",
            "plans_root": str(plans_root),
        }
    except ValueError:
        pass

    return {"write_scope": "other"}


def _classify_shell_command_scope(
    runtime: "AgentRuntime",
    args: Dict[str, Any],
    command: str,
) -> Dict[str, Any]:
    managed_root = runtime.ctx.config.agent_workspace_root()
    candidates: List[Path] = []
    explicit_cwd = str(args.get("cwd") or "").strip()
    if explicit_cwd:
        try:
            candidates.append(Path(explicit_cwd).expanduser().resolve())
        except OSError:
            pass

    prefix = _extract_leading_cd_prefix(command)
    if prefix is not None:
        try:
            resolved = Path(prefix).expanduser().resolve()
            candidates.append(resolved)
        except OSError:
            pass

    for candidate in candidates:
        try:
            candidate.relative_to(managed_root)
        except ValueError:
            continue

        payload = {
            "command_scope": "managed_workspace",
            "managed_workspace_root": str(managed_root),
        }
        remainder = _command_after_leading_cd(command)
        if remainder:
            from opencas.tools.validation import assess_command

            effective = assess_command(remainder)
            payload.update(
                {
                    "command_effective_family": effective.family,
                    "command_effective_permission_class": effective.permission_class,
                    "command_effective_executable": effective.executable,
                    "command_effective_subcommand": effective.subcommand,
                }
            )
        return payload

    return {}


def _extract_leading_cd_prefix(command: str) -> Optional[str]:
    stripped = command.strip()
    match = re.match(r"^cd\s+(.+?)\s*(?:&&|;)", stripped)
    if match is None:
        return None
    raw = match.group(1).strip()
    try:
        tokenized = shlex.split(raw)
    except ValueError:
        return raw.strip("\"'")
    if not tokenized:
        return None
    return tokenized[0]


def _command_after_leading_cd(command: str) -> Optional[str]:
    stripped = command.strip()
    match = re.match(r"^cd\s+.+?\s*(?:&&|;)\s*(.+)$", stripped)
    if match is None:
        return None
    remainder = match.group(1).strip()
    return remainder or None
