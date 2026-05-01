"""Provenance hook handlers for the runtime tool pipeline.

The hook bus remains the policy seam; this module only turns durable
tool- and decision-level events into canonical registry entries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from opencas.api.provenance_entry import (
    Action,
    Risk,
    append_registry_entry_from_event_context,
    now_iso8601_ts,
)
from opencas.provenance_events_adapter import (
    ProvenanceEventType,
    append_provenance_event,
    emit_provenance_event,
)
from opencas.autonomy.models import ActionRiskTier, ApprovalLevel
from opencas.infra.hook_bus import (
    POST_ACTION_DECISION,
    POST_TOOL_EXECUTE,
    POST_SESSION_LIFECYCLE,
    PRE_TOOL_EXECUTE,
    HookResult,
)

_FILE_WRITE_TOOLS = {"fs_write_file", "edit_file"}
_SESSION_CREATE_TOOLS = {"browser_start", "process_start", "pty_start", "pty_interact"}
_SESSION_DELETE_TOOLS = {
    "browser_close",
    "process_kill",
    "process_remove",
    "process_clear",
    "pty_kill",
    "pty_remove",
    "pty_clear",
}
_WORKFLOW_CREATE_TOOLS = {
    "workflow_create_commitment",
    "workflow_create_schedule",
    "workflow_create_writing_task",
    "workflow_create_plan",
}
_WORKFLOW_UPDATE_TOOLS = {
    "workflow_update_commitment",
    "workflow_update_schedule",
    "workflow_update_plan",
}


def register_runtime_provenance_hooks(runtime: Any) -> None:
    """Attach provenance emission hooks to the runtime hook bus."""
    if getattr(runtime, "_provenance_hooks_registered", False):
        return

    hook_bus = getattr(getattr(runtime, "ctx", None), "hook_bus", None)
    if hook_bus is None:
        return

    hook_bus.register(PRE_TOOL_EXECUTE, lambda hook_name, ctx: _pre_tool_execute(runtime, hook_name, ctx), priority=-100)
    hook_bus.register(POST_TOOL_EXECUTE, lambda hook_name, ctx: _post_tool_execute(runtime, hook_name, ctx), priority=-100)
    hook_bus.register(
        POST_ACTION_DECISION,
        lambda hook_name, ctx: _post_action_decision(runtime, hook_name, ctx),
        priority=-100,
    )
    hook_bus.register(
        POST_SESSION_LIFECYCLE,
        lambda hook_name, ctx: _post_session_lifecycle(runtime, hook_name, ctx),
        priority=-100,
    )
    runtime._provenance_hooks_registered = True


def emit_runtime_session_lifecycle(
    runtime: Any,
    *,
    transition: str,
    reason: str,
    note: str | None = None,
    entrypoint: str | None = None,
    session_id: str | None = None,
) -> None:
    """Route a session lifecycle transition through the provenance hook bus."""
    hook_bus = getattr(getattr(runtime, "ctx", None), "hook_bus", None)
    if hook_bus is None:
        return

    normalized_session_id = str(session_id or _runtime_session_id(runtime)).strip() or _runtime_session_id(runtime)
    normalized_transition = str(transition or "").strip().lower()
    if not normalized_transition:
        return

    dedupe_key = (
        normalized_transition,
        normalized_session_id,
        str(entrypoint or "").strip(),
        str(note or "").strip(),
    )
    seen = getattr(runtime, "_provenance_session_lifecycle_keys", None)
    if seen is None:
        seen = set()
        runtime._provenance_session_lifecycle_keys = seen
    if dedupe_key in seen:
        return
    seen.add(dedupe_key)

    hook_bus.run(
        POST_SESSION_LIFECYCLE,
        {
            "session_id": normalized_session_id,
            "artifact": f"session|lifecycle|{normalized_session_id}",
            "action": _session_lifecycle_action(normalized_transition).value,
            "why": reason,
            "risk": Risk.LOW.value,
            "ts": now_iso8601_ts(),
            "transition": normalized_transition,
            "entrypoint": str(entrypoint or "").strip() or None,
            "note": str(note or "").strip() or None,
            "source_trace": {
                "event": "session_lifecycle",
                "transition": normalized_transition,
                "entrypoint": str(entrypoint or "").strip() or None,
                "note": str(note or "").strip() or None,
            },
        },
    )


def _runtime_session_id(runtime: Any) -> str:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    for candidate in (
        getattr(config, "session_id", None),
        getattr(runtime, "session_id", None),
        getattr(runtime, "_session_id", None),
        "default",
    ):
        value = str(candidate or "").strip()
        if value:
            return value
    return "default"


def _managed_workspace_root(runtime: Any) -> Optional[Path]:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    if config is None:
        return None
    candidate = getattr(config, "agent_workspace_root", None)
    try:
        if callable(candidate):
            root = candidate()
        else:
            root = candidate
    except Exception:
        return None
    if root is None:
        return None
    try:
        return Path(root).expanduser().resolve()
    except Exception:
        return None


def _plans_root(runtime: Any) -> Optional[Path]:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    state_dir = getattr(config, "state_dir", None)
    if state_dir is None:
        return None
    try:
        return (Path(state_dir).expanduser().resolve() / "plans").resolve()
    except Exception:
        return None


def _parse_json_object(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _to_risk(risk_tier: Any) -> Risk:
    value = str(risk_tier or "").strip().lower()
    if value == ActionRiskTier.READONLY.value:
        return Risk.LOW
    if value == ActionRiskTier.WORKSPACE_WRITE.value:
        return Risk.MEDIUM
    if value in {
        ActionRiskTier.SHELL_LOCAL.value,
        ActionRiskTier.NETWORK.value,
        ActionRiskTier.EXTERNAL_WRITE.value,
    }:
        return Risk.HIGH
    if value == ActionRiskTier.DESTRUCTIVE.value:
        return Risk.CRITICAL
    return Risk.MEDIUM


def _workspace_relative_path(runtime: Any, raw_path: Any) -> Tuple[str, str]:
    candidate = str(raw_path or "").strip()
    if not candidate:
        return "path", "unknown"
    try:
        path = Path(candidate).expanduser().resolve()
    except Exception:
        return "path", candidate

    workspace_root = _managed_workspace_root(runtime)
    if workspace_root is not None:
        try:
            rel = path.relative_to(workspace_root)
            return "workspace", rel.as_posix()
        except ValueError:
            pass

    plans_root = _plans_root(runtime)
    if plans_root is not None:
        try:
            rel = path.relative_to(plans_root)
            return "plans", rel.as_posix()
        except ValueError:
            pass

    return "path", path.as_posix()


def _result_target_id(ctx: Dict[str, Any]) -> str:
    for key in ("target_id", "session_id", "process_id", "plan_id", "commitment_id", "schedule_id"):
        value = str(ctx.get(key, "") or "").strip()
        if value:
            return value
    output = _parse_json_object(ctx.get("result_output"))
    for key in ("session_id", "process_id", "plan_id", "commitment_id", "schedule_id"):
        value = str(output.get(key, "") or "").strip()
        if value:
            return value
    metadata = ctx.get("result_metadata")
    if isinstance(metadata, dict):
        for key in ("session_id", "process_id", "plan_id", "commitment_id", "schedule_id", "path"):
            value = str(metadata.get(key, "") or "").strip()
            if value:
                return value
    return ""


def _target_scope(ctx: Dict[str, Any]) -> str:
    value = str(ctx.get("scope_key", "") or "default").strip()
    return value or "default"


def _success_why(tool_name: str, kind: str, action: Action, target_id: str, result_output: str) -> str:
    if kind == "file":
        verb = "wrote" if action == Action.CREATE else "updated"
    elif action == Action.CREATE:
        verb = "started" if kind in {"browser", "process", "pty"} else "created"
    elif action == Action.UPDATE:
        verb = "updated"
    elif action == Action.DELETE:
        verb = "closed" if kind in {"browser", "pty"} else "stopped"
    else:
        verb = action.value.lower()

    target = target_id.strip()
    if target:
        return f"{tool_name} {verb} {target}"
    output = result_output.strip()
    return output or f"{tool_name} completed"


def _artifact_for_tool(runtime: Any, ctx: Dict[str, Any]) -> Optional[Tuple[str, str, str, Action]]:
    tool_name = str(ctx.get("tool_name", "") or "").strip()
    if not tool_name:
        return None

    scope = _target_scope(ctx)
    args = ctx.get("args") if isinstance(ctx.get("args"), dict) else {}
    result_output = _parse_json_object(ctx.get("result_output"))
    result_metadata = ctx.get("result_metadata") if isinstance(ctx.get("result_metadata"), dict) else {}
    target_id = _result_target_id(ctx)

    if tool_name in _SESSION_CREATE_TOOLS:
        if tool_name in {"pty_interact"} and str(args.get("session_id", "") or "").strip():
            action = Action.UPDATE
        elif str(args.get("session_id", "") or "").strip():
            action = Action.UPDATE
        else:
            action = Action.CREATE
        target_id = target_id or str(result_output.get("session_id") or "").strip()
        if not target_id:
            target_id = str(result_metadata.get("session_id") or "").strip()
        if not target_id:
            target_id = str(args.get("session_id") or "").strip()
        if not target_id:
            return None
        kind = "browser" if tool_name.startswith("browser_") else "process" if tool_name.startswith("process_") else "pty"
        return kind, scope, target_id, action

    if tool_name in _SESSION_DELETE_TOOLS:
        kind = "browser" if tool_name.startswith("browser_") else "process" if tool_name.startswith("process_") else "pty"
        if not target_id:
            target_id = str(args.get("session_id") or args.get("process_id") or "").strip()
        if not target_id:
            target_id = "all"
        return kind, scope, target_id, Action.DELETE

    if tool_name in _FILE_WRITE_TOOLS:
        file_path = str(args.get("file_path") or args.get("path") or result_metadata.get("path") or "").strip()
        if not file_path:
            return None
        kind, rel_path = _workspace_relative_path(runtime, file_path)
        action = Action.CREATE if bool(ctx.get("artifact_preexisting") is False) or tool_name == "fs_write_file" else Action.UPDATE
        return "file", kind, rel_path, action

    if tool_name in _WORKFLOW_CREATE_TOOLS:
        artifact_id = target_id or str(result_output.get("plan_id") or result_output.get("schedule_id") or result_output.get("commitment_id") or "").strip()
        if not artifact_id:
            artifact_id = str(result_metadata.get("plan_id") or result_metadata.get("schedule_id") or result_metadata.get("commitment_id") or "").strip()
        if not artifact_id:
            artifact_id = str(args.get("title") or args.get("content") or tool_name).strip()
        kind = "plan" if "plan" in tool_name else "schedule" if "schedule" in tool_name else "commitment" if "commitment" in tool_name else "file"
        if tool_name == "workflow_create_writing_task":
            file_path = str(result_output.get("output_path") or result_metadata.get("output_path") or "").strip()
            if file_path:
                kind, rel_path = _workspace_relative_path(runtime, file_path)
                return "file", kind, rel_path, Action.CREATE
        return kind, "default", artifact_id, Action.CREATE

    if tool_name in _WORKFLOW_UPDATE_TOOLS:
        artifact_id = target_id or str(result_output.get("plan_id") or result_output.get("schedule_id") or result_output.get("commitment_id") or "").strip()
        if not artifact_id:
            artifact_id = str(result_metadata.get("plan_id") or result_metadata.get("schedule_id") or result_metadata.get("commitment_id") or "").strip()
        if not artifact_id:
            artifact_id = str(args.get("plan_id") or args.get("schedule_id") or args.get("commitment_id") or "").strip()
        if not artifact_id:
            return None
        kind = "plan" if "plan" in tool_name else "schedule" if "schedule" in tool_name else "commitment"
        return kind, "default", artifact_id, Action.UPDATE

    return None


def _record_entry(
    runtime: Any,
    *,
    artifact: str,
    action: Action,
    why: str,
    risk: Risk,
    session_id: Optional[str] = None,
) -> None:
    payload = {
        "session_id": str(session_id or _runtime_session_id(runtime)).strip() or _runtime_session_id(runtime),
        "artifact": artifact,
        "action": action.value,
        "why": why,
        "risk": risk.value,
    }
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    state_dir = getattr(config, "state_dir", None)
    if state_dir is None:
        return
    default_path = Path(state_dir) / "operator_action_history.jsonl"
    append_registry_entry_from_event_context(runtime, payload, default_path=default_path)


def _session_lifecycle_action(transition: str) -> Action:
    if transition in {"boot", "start", "resume", "restart"}:
        return Action.CREATE
    if transition in {"shutdown", "stop", "end", "complete", "commit"}:
        return Action.COMMIT
    if transition in {"interrupt", "abort", "fail", "rollback"}:
        return Action.ROLLBACK
    return Action.UPDATE


def _post_session_lifecycle(runtime: Any, _hook_name: str, ctx: Dict[str, Any]) -> HookResult:
    transition = str(ctx.get("transition", "") or "").strip().lower()
    if not transition:
        return HookResult(allowed=True)

    action = _session_lifecycle_action(transition)
    session_id = str(ctx.get("session_id", "") or "").strip() or _runtime_session_id(runtime)
    note = str(ctx.get("note", "") or "").strip()
    entrypoint = str(ctx.get("entrypoint", "") or "").strip()
    why = str(ctx.get("reason", "") or "").strip()
    if not why:
        if note:
            why = note
        elif entrypoint:
            why = f"{entrypoint} {transition}"
        else:
            why = f"{transition} session lifecycle transition"

    payload = dict(ctx)
    payload.update(
        {
            "session_id": session_id,
            "artifact": f"session|lifecycle|{session_id}",
            "action": action.value,
            "why": why,
            "risk": str(ctx.get("risk", "") or Risk.LOW.value),
            "transition": transition,
            "source_trace": {
                "event": "session_lifecycle",
                "transition": transition,
                "entrypoint": entrypoint or None,
                "note": note or None,
                "outcome": str(ctx.get("outcome", "") or "completed"),
            },
        }
    )
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    state_dir = getattr(config, "state_dir", None)
    if state_dir is None:
        return HookResult(allowed=True)
    default_path = Path(state_dir) / "operator_action_history.jsonl"
    append_registry_entry_from_event_context(runtime, payload, default_path=default_path)
    return HookResult(allowed=True)


def _pre_tool_execute(runtime: Any, _hook_name: str, ctx: Dict[str, Any]) -> HookResult:
    tool_name = str(ctx.get("tool_name", "") or "").strip()
    if not tool_name:
        return HookResult(allowed=True, mutated_context=ctx)

    enriched = dict(ctx)
    args = dict(enriched.get("args") or {})
    if tool_name in _FILE_WRITE_TOOLS:
        file_path = str(args.get("file_path") or args.get("path") or "").strip()
        if file_path:
            kind, rel_path = _workspace_relative_path(runtime, file_path)
            enriched["artifact_scope"] = kind
            enriched["artifact_name"] = rel_path
            try:
                enriched["artifact_preexisting"] = Path(file_path).expanduser().resolve().exists()
            except Exception:
                enriched["artifact_preexisting"] = None
    elif tool_name in _SESSION_CREATE_TOOLS:
        scope = _target_scope(enriched)
        session_id = str(args.get("session_id") or "").strip()
        if session_id:
            enriched["artifact_kind"] = "browser" if tool_name.startswith("browser_") else "process" if tool_name.startswith("process_") else "pty"
            enriched["artifact_scope"] = scope
            enriched["artifact_name"] = session_id
    elif tool_name in _SESSION_DELETE_TOOLS:
        scope = _target_scope(enriched)
        session_id = str(args.get("session_id") or args.get("process_id") or "").strip()
        if session_id:
            enriched["artifact_kind"] = "browser" if tool_name.startswith("browser_") else "process" if tool_name.startswith("process_") else "pty"
            enriched["artifact_scope"] = scope
            enriched["artifact_name"] = session_id
    elif tool_name in _WORKFLOW_CREATE_TOOLS | _WORKFLOW_UPDATE_TOOLS:
        target = str(args.get("plan_id") or args.get("schedule_id") or args.get("commitment_id") or "").strip()
        if target:
            enriched["artifact_kind"] = "plan" if "plan" in tool_name else "schedule" if "schedule" in tool_name else "commitment"
            enriched["artifact_scope"] = "default"
            enriched["artifact_name"] = target
        elif tool_name == "workflow_create_writing_task":
            enriched["artifact_kind"] = "file"
            enriched["artifact_scope"] = "workspace"
            output_path = str(args.get("output_path") or "").strip()
            if output_path:
                _, rel_path = _workspace_relative_path(runtime, output_path)
                enriched["artifact_name"] = rel_path

    return HookResult(allowed=True, mutated_context=enriched)


def _post_tool_execute(runtime: Any, _hook_name: str, ctx: Dict[str, Any]) -> HookResult:
    tool_name = str(ctx.get("tool_name", "") or "").strip()
    if not tool_name:
        return HookResult(allowed=True)

    result_success = bool(ctx.get("result_success", False))
    result_output = str(ctx.get("result_output", "") or "")
    risk_tier = ctx.get("risk_tier")
    risk = _to_risk(risk_tier)
    args = ctx.get("args") if isinstance(ctx.get("args"), dict) else {}
    session_id = str(ctx.get("session_id", "") or "").strip() or None

    if result_success:
        artifact = _artifact_for_tool(runtime, ctx)
        if artifact is None:
            return HookResult(allowed=True)
        kind, scope, target_id, action = artifact
        if kind == "file":
            artifact_id = f"file|{scope}|{target_id}"
        else:
            artifact_id = f"{kind}|{scope}|{target_id}"
        if action == Action.UPDATE and tool_name == "fs_write_file" and ctx.get("artifact_preexisting") is False:
            action = Action.CREATE
        if tool_name == "workflow_create_writing_task" and kind == "file":
            action = Action.CREATE
        if tool_name == "pty_interact" and str(args.get("session_id", "") or "").strip():
            action = Action.UPDATE
        why = _success_why(tool_name, kind, action, target_id, result_output)
        if len(why) > 512:
            why = why[:509].rstrip() + "..."
        _record_entry(runtime, artifact=artifact_id, action=action, why=why, risk=risk, session_id=session_id)
        event = emit_provenance_event(
            None,
            event_type=ProvenanceEventType.MUTATION,
            triggering_artifact=artifact_id,
            triggering_action=action.value,
            parent_link_id=artifact_id,
            linked_link_ids=[artifact_id, str(ctx.get("session_id", "") or "").strip() or artifact_id],
            details={
                "tool_name": tool_name,
                "session_id": session_id,
                "why": why,
                "risk": risk.value,
            },
        ).to_dict()
        enriched = dict(ctx)
        result_metadata = dict(enriched.get("result_metadata") or {})
        enriched["result_metadata"] = append_provenance_event(result_metadata, event)
        return HookResult(allowed=True, mutated_context=enriched)

    # Only record notable failures for state-changing or lifecycle actions.
    if tool_name not in (_FILE_WRITE_TOOLS | _SESSION_CREATE_TOOLS | _SESSION_DELETE_TOOLS | _WORKFLOW_CREATE_TOOLS | _WORKFLOW_UPDATE_TOOLS):
        return HookResult(allowed=True)

    artifact = _artifact_for_tool(runtime, ctx)
    if artifact is None:
        artifact_id = f"tool|default|{tool_name}"
        action = Action.ROLLBACK
    else:
        kind, scope, target_id, action = artifact
        artifact_id = f"{kind}|{scope}|{target_id}"
    why = result_output.strip() or f"{tool_name} failed"
    if len(why) > 512:
        why = why[:509].rstrip() + "..."
    _record_entry(runtime, artifact=artifact_id, action=Action.ROLLBACK, why=why, risk=risk, session_id=session_id)
    event = emit_provenance_event(
        None,
        event_type=ProvenanceEventType.BLOCKED,
        triggering_artifact=artifact_id,
        triggering_action="ROLLBACK",
        parent_link_id=artifact_id,
        linked_link_ids=[artifact_id, str(ctx.get("session_id", "") or "").strip() or artifact_id],
        details={
            "tool_name": tool_name,
            "session_id": session_id,
            "why": why,
            "risk": risk.value,
        },
    ).to_dict()
    enriched = dict(ctx)
    result_metadata = dict(enriched.get("result_metadata") or {})
    enriched["result_metadata"] = append_provenance_event(result_metadata, event)
    return HookResult(allowed=True, mutated_context=enriched)


def _post_action_decision(runtime: Any, _hook_name: str, ctx: Dict[str, Any]) -> HookResult:
    tool_name = str(ctx.get("tool_name", "") or "").strip()
    if not tool_name:
        return HookResult(allowed=True)

    approved = bool(ctx.get("approved", False))
    decision_level = str(ctx.get("decision_level", "") or "").strip().lower()
    if approved and decision_level not in {
        ApprovalLevel.CAN_DO_AFTER_MORE_EVIDENCE.value,
        ApprovalLevel.MUST_ESCALATE.value,
    }:
        return HookResult(allowed=True)

    risk_tier = ctx.get("risk_tier")
    risk = _to_risk(risk_tier)
    args = ctx.get("args") if isinstance(ctx.get("args"), dict) else {}
    session_scope = _target_scope(ctx)
    session_id = str(ctx.get("session_id", "") or "").strip() or None
    artifact = None
    for candidate in (
        str(args.get("file_path") or args.get("path") or "").strip(),
        str(args.get("session_id") or args.get("process_id") or args.get("plan_id") or args.get("schedule_id") or args.get("commitment_id") or "").strip(),
    ):
        if candidate:
            if candidate.endswith(".md") or candidate.endswith(".txt") or "/" in candidate or "\\" in candidate:
                scope, rel = _workspace_relative_path(runtime, candidate)
                artifact = f"file|{scope}|{rel}" if scope in {"workspace", "plans"} else f"path|default|{rel}"
            else:
                kind = "process" if tool_name.startswith("process_") else "browser" if tool_name.startswith("browser_") else "pty" if tool_name.startswith("pty_") else "plan" if "plan" in tool_name else "schedule" if "schedule" in tool_name else "commitment" if "commitment" in tool_name else "tool"
                artifact = f"{kind}|{session_scope}|{candidate}" if kind != "tool" else f"tool|{session_scope}|{tool_name}"
            break
    if artifact is None:
        artifact = f"tool|{session_scope}|{tool_name}"

    reasoning = str(ctx.get("reasoning", "") or ctx.get("decision_reasoning", "") or "").strip()
    why = reasoning or f"{tool_name} required escalation"
    if len(why) > 512:
        why = why[:509].rstrip() + "..."
    _record_entry(runtime, artifact=artifact, action=Action.DECIDE, why=why, risk=risk, session_id=session_id)
    return HookResult(allowed=True)
