"""ReAct-style tool-use loop for OpenCAS."""

from __future__ import annotations

import asyncio
from collections import Counter
import json
from typing import Any, Callable, Dict, List, Optional

from opencas.api import LLMClient
from opencas.autonomy.models import ActionRiskTier
from opencas.autonomy.self_approval import SelfApprovalLadder
from opencas.memory import EpisodeKind
from opencas.telemetry import EventKind, Tracer
from opencas.tools.registry import ToolRegistry
from opencas.tools.schema import build_tool_schemas

from .context import ToolUseContext, ToolUseResult, UserInputRequired
from .loop_guard import ToolLoopGuard


class ToolUseLoop:
    """Iterative tool-use loop: LLM plans, tools execute, observations feed back."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        approval: SelfApprovalLadder,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.approval = approval
        self.tracer = tracer

    async def run(
        self,
        objective: str,
        messages: List[Dict[str, Any]],
        ctx: ToolUseContext,
        payload: Optional[Dict[str, Any]] = None,
        on_focus_enter: Optional[Callable[[], None]] = None,
        on_focus_exit: Optional[Callable[[], None]] = None,
    ) -> ToolUseResult:
        """Run the loop until the LLM finishes or max iterations are reached."""
        if messages and messages[0].get("role") != "system":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an autonomous assistant. You have access to tools. "
                        "Use them only when they materially help accomplish the user's objective. "
                        "Prefer answering directly when no tool is required. "
                        "Prefer composite tools over low-level multi-step tool sequences when both are available. "
                        "Avoid repeated polling or observation loops unless the objective explicitly requires them. "
                        "When you are done, provide a concise summary of what you did."
                        + (" You are in PLAN MODE: only read files and write to the plans directory." if ctx.plan_mode else "")
                    ),
                },
                *messages,
            ]
        elif messages:
            # Inject plan mode constraint into existing system message
            first = messages[0]
            if ctx.plan_mode and "PLAN MODE" not in str(first.get("content", "")):
                first["content"] = (
                    str(first.get("content", ""))
                    + " You are in PLAN MODE: only read files and write to the plans directory."
                )

        all_tool_calls: List[Dict[str, Any]] = []
        executed_steps: List[Dict[str, Any]] = []
        iterations = 0
        guard = ToolLoopGuard()
        _in_focus_mode = False

        available_tools = self._filter_tools(ctx, objective=objective)
        schemas = build_tool_schemas(available_tools)

        def _check_guard(tc: Dict[str, Any]) -> Optional[str]:
            return guard.record_call(ctx.session_id, tc["name"], tc.get("args", {}))

        try:
            while iterations < ctx.max_iterations:
                iterations += 1
                complexity = self._select_iteration_complexity(
                    objective=objective,
                    ctx=ctx,
                    iteration=iterations,
                    guard=guard,
                )
                response = await self.llm.chat_completion(
                    messages=messages,
                    complexity=complexity,
                    tools=schemas if schemas else None,
                    payload=payload,
                    source="tool_use_loop",
                    session_id=ctx.session_id,
                    task_id=ctx.task_id,
                )

                message = response.get("choices", [{}])[0].get("message", {})
                raw_tool_calls = message.get("tool_calls") or []

                # Append assistant message (with or without tool calls)
                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                content = message.get("content")
                if content:
                    assistant_msg["content"] = content
                if raw_tool_calls:
                    assistant_msg["tool_calls"] = raw_tool_calls
                messages.append(assistant_msg)

                if not raw_tool_calls:
                    final = content or "Done."
                    return ToolUseResult(
                        final_output=final,
                        messages=messages,
                        tool_calls=all_tool_calls,
                        iterations=iterations,
                    )

                tool_calls = self._normalize_tool_calls(raw_tool_calls)
                all_tool_calls.extend(tool_calls)
                fulfilled_ids: set[str] = set()

                # Partition into concurrent (readonly) and serial (everything else)
                concurrent_calls: List[Dict[str, Any]] = []
                serial_calls: List[Dict[str, Any]] = []
                for tc in tool_calls:
                    entry = self.tools.get(tc["name"])
                    if entry and entry.risk_tier == ActionRiskTier.READONLY:
                        concurrent_calls.append(tc)
                    else:
                        serial_calls.append(tc)

                # Guard check for concurrent batch
                for tc in concurrent_calls:
                    guard_reason = _check_guard(tc)
                    if guard_reason:
                        self._trace("tool_loop_guard_fired", {"reason": guard_reason, "session_id": ctx.session_id, "tool_loop_guard_fires": 1})
                        return ToolUseResult(
                            final_output=self._build_guard_summary(
                                guard_reason=guard_reason,
                                executed_steps=executed_steps,
                                pending_calls=[
                                    _tc for _tc in tool_calls if _tc["id"] not in fulfilled_ids
                                ],
                            ),
                            messages=messages,
                            tool_calls=all_tool_calls,
                            iterations=iterations,
                            guard_fired=True,
                            guard_reason=guard_reason,
                        )

                # Execute concurrent batch
                if concurrent_calls:
                    results = await asyncio.gather(
                        *[self._execute_tool_call(tc, ctx) for tc in concurrent_calls],
                        return_exceptions=True,
                    )
                    for tc, result in zip(concurrent_calls, results):
                        if isinstance(result, Exception):
                            result = {"success": False, "output": str(result), "metadata": {}}
                        executed_steps.append(
                            {
                                "name": tc["name"],
                                "success": bool(result.get("success", False)),
                            }
                        )
                        messages.append(self._build_tool_result_message(tc, result))
                        fulfilled_ids.add(tc["id"])

                # Guard check and execute serial batch
                for tc in serial_calls:
                    guard_reason = _check_guard(tc)
                    if guard_reason:
                        self._trace("tool_loop_guard_fired", {"reason": guard_reason, "session_id": ctx.session_id, "tool_loop_guard_fires": 1})
                        return ToolUseResult(
                            final_output=self._build_guard_summary(
                                guard_reason=guard_reason,
                                executed_steps=executed_steps,
                                pending_calls=[
                                    _tc for _tc in tool_calls if _tc["id"] not in fulfilled_ids
                                ],
                            ),
                            messages=messages,
                            tool_calls=all_tool_calls,
                            iterations=iterations,
                            guard_fired=True,
                            guard_reason=guard_reason,
                        )
                    try:
                        result = await self._execute_tool_call(tc, ctx)
                    except Exception as exc:
                        result = {"success": False, "output": str(exc), "metadata": {}}
                    executed_steps.append(
                        {
                            "name": tc["name"],
                            "success": bool(result.get("success", False)),
                        }
                    )
                    messages.append(self._build_tool_result_message(tc, result))
                    fulfilled_ids.add(tc["id"])

                # Auto-enter focus mode when round depth crosses the threshold
                if not _in_focus_mode and guard.is_deep(ctx.session_id) and on_focus_enter:
                    on_focus_enter()
                    _in_focus_mode = True

            # Max iterations reached
            return ToolUseResult(
                final_output="Reached maximum number of tool-use iterations.",
                messages=messages,
                tool_calls=all_tool_calls,
                iterations=iterations,
            )
        finally:
            guard.reset(ctx.session_id)
            if _in_focus_mode and on_focus_exit:
                on_focus_exit()

    def _filter_tools(self, ctx: ToolUseContext, objective: str = "") -> List[Any]:
        """Filter available tools based on plan mode and runtime constraints."""
        tools = self.tools.list_tools()
        if not ctx.plan_mode:
            return self._select_tools_for_objective(tools, objective)
        # In plan mode: only read-only tools + write_file restricted to plans dir
        allowed: List[Any] = []
        for entry in tools:
            if entry.risk_tier == ActionRiskTier.READONLY:
                allowed.append(entry)
            elif entry.name == "fs_write_file":
                allowed.append(entry)
        return allowed

    def _select_tools_for_objective(self, tools: List[Any], objective: str) -> List[Any]:
        """Select a smaller, relevant tool subset for the current objective."""
        text = (objective or "").lower()
        selected_names: set[str] = set()
        conversational_only = any(
            phrase in text
            for phrase in (
                "how you understand your role",
                "how do you understand your role",
                "what is your role",
                "your role in this session",
                "who are you",
                "how are you",
            )
        )

        # Reflective turns should stay tool-free unless the user explicitly asks for work.
        if conversational_only:
            return []

        if any(token in text for token in ("runtime", "workflow", "status", "profile", "constraint", "operating roots")):
            selected_names.update({"runtime_status", "workflow_status"})
            selected_names.update(
                entry.name for entry in tools if entry.name.startswith("workflow_")
            )

        if any(token in text for token in ("commitment", "goal", "track", "promise", "deadline")):
            selected_names.update({
                "workflow_create_commitment",
                "workflow_update_commitment",
                "workflow_list_commitments",
            })
        if any(token in text for token in ("write", "writing", "draft", "document", "article", "note", "essay")):
            selected_names.update({
                "workflow_create_writing_task",
                "workflow_create_plan",
                "workflow_update_plan",
                "fs_read_file",
                "fs_write_file",
            })
        if any(token in text for token in ("triage", "repo", "overview", "summary", "audit")):
            selected_names.update({"workflow_repo_triage"})
        if any(token in text for token in ("supervise", "delegate", "launch claude", "launch codex", "launch kilocode", "launch kilo", "operator")):
            selected_names.update({"workflow_supervise_session", "pty_kill", "pty_remove"})

        if any(token in text for token in ("browser", "web", "page", "site", "url", "http", "https", "data:text")):
            selected_names.update(
                entry.name for entry in tools if entry.name.startswith("browser_")
            )
            selected_names.update({"web_search", "web_fetch"})
        if any(
            token in text
            for token in (
                "google workspace",
                "gmail",
                "email",
                "inbox",
                "calendar",
                "schedule",
                "drive",
                "google docs",
                "google doc",
                "docs",
                "sheets",
                "spreadsheet",
                "slides",
                "people",
                "contacts",
            )
        ):
            selected_names.update(
                entry.name
                for entry in tools
                if entry.name.startswith("google_workspace_")
            )
        if any(token in text for token in ("pty", "terminal", "tui", "claude", "codex", "kilocode", "kilo", "kilo-code", "vim", "editor", "shell session")):
            selected_names.update({"pty_interact", "pty_remove", "pty_clear"})
        if any(token in text for token in ("poll", "session_id", "resize", "control sequence")):
            selected_names.update(
                entry.name for entry in tools if entry.name.startswith("pty_")
            )
        if any(token in text for token in ("process", "server", "background", "daemon")):
            selected_names.update(
                entry.name for entry in tools if entry.name.startswith("process_")
            )
        if any(token in text for token in ("search", "grep", "find", "code", "repo", "project", "write", "file", "edit")):
            selected_names.update(
                {
                    "fs_read_file",
                    "fs_list_dir",
                    "fs_write_file",
                    "grep_search",
                    "glob_search",
                    "bash_run_command",
                    "lsp_diagnostics",
                    "agent",
                }
            )
        if any(token in text for token in ("plan", "planning", "checklist", "roadmap")):
            selected_names.update({"enter_plan_mode", "exit_plan_mode"})

        # Default exploration toolkit when no specific keywords match
        if not selected_names:
            selected_names.update({
                "fs_read_file",
                "fs_list_dir",
                "bash_run_command",
                "agent",
            })

        selected = [entry for entry in tools if entry.name in selected_names]
        return selected

    def _normalize_tool_calls(
        self, raw_tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize tool call structure from various LLM providers."""
        normalized: List[Dict[str, Any]] = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            name = func.get("name") or tc.get("name", "")
            arguments = func.get("arguments") or tc.get("arguments", "")
            if isinstance(arguments, str):
                try:
                    args = json.loads(arguments)
                except json.JSONDecodeError:
                    args = {"raw": arguments}
            else:
                args = dict(arguments)
            normalized.append(
                {
                    "id": tc.get("id", ""),
                    "name": name,
                    "args": args,
                }
            )
        return normalized

    def _select_iteration_complexity(
        self,
        *,
        objective: str,
        ctx: ToolUseContext,
        iteration: int,
        guard: ToolLoopGuard,
    ) -> str:
        """Choose the reasoning tier for the current loop iteration.

        Tool-heavy workflows start at standard and only climb when the loop
        keeps turning. This keeps simple tasks on cheaper models while still
        allowing the runtime to raise capability when a project stalls or grows.
        """
        if not getattr(self.llm.model_routing, "auto_escalation", True):
            return "standard"

        text = (objective or "").lower()
        if ctx.plan_mode or any(
            token in text
            for token in ("project", "refactor", "architecture", "multi-file")
        ):
            if iteration >= 4:
                return "extra_high"
            if iteration >= 2:
                return "high"

        if guard.is_deep(ctx.session_id):
            return "high" if iteration < 5 else "extra_high"
        if iteration >= 5:
            return "extra_high"
        if iteration >= 3:
            return "high"
        return "standard"

    async def _execute_tool_call(
        self, tc: Dict[str, Any], ctx: ToolUseContext
    ) -> Dict[str, Any]:
        """Execute a single tool call with approval gating and memory recording."""
        name = tc["name"]
        args = tc["args"]

        # Plan mode path restriction for fs_write_file
        if ctx.plan_mode and name == "fs_write_file":
            from pathlib import Path

            file_path = str(args.get("file_path", ""))
            plans_dir = ctx.runtime.ctx.config.state_dir / "plans"
            resolved = Path(file_path).expanduser().resolve()
            try:
                resolved.relative_to(plans_dir)
            except ValueError:
                return {
                    "success": False,
                    "output": f"Plan mode blocked write outside {plans_dir}: {file_path}",
                    "metadata": {},
                }

        # Use the runtime's execute_tool for full approval + somatic + goal tracking
        result = await ctx.runtime.execute_tool(name, args)

        # Plan mode state transitions
        if name == "enter_plan_mode" and result.get("success"):
            ctx.plan_mode = True
            ctx.active_plan_id = result.get("metadata", {}).get("plan_id")
        elif name == "exit_plan_mode" and result.get("success"):
            ctx.plan_mode = False
            ctx.active_plan_id = None

        # Persist plan actions when in plan mode
        if ctx.plan_mode and ctx.active_plan_id:
            plan_id = ctx.active_plan_id
            plan_store = getattr(ctx.runtime.ctx, "plan_store", None)
            if plan_store is not None:
                try:
                    await plan_store.record_action(
                        plan_id=plan_id,
                        tool_name=name,
                        args=args,
                        result_summary=str(result.get("output", ""))[:1024],
                        success=bool(result.get("success", False)),
                    )
                except Exception:
                    pass

        # Record episode for notable actions
        if result.get("success"):
            await ctx.runtime._record_episode(
                content=f"tool {name}: {json.dumps(args)}",
                kind=EpisodeKind.ACTION,
                session_id=ctx.session_id,
            )
        else:
            await ctx.runtime._record_episode(
                content=f"tool {name} failed: {result.get('output', '')}",
                kind=EpisodeKind.OBSERVATION,
                session_id=ctx.session_id,
            )

        # Interactive special case
        if name == "ask_user_question" and not result.get("success"):
            # The interactive adapter returns failure with the question when it wants to pause
            output = result.get("output", "")
            raise UserInputRequired(output)

        return result

    def _build_tool_result_message(
        self, tc: Dict[str, Any], result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build an OpenAI-compatible tool result message."""
        content = result.get("output", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        return {
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "name": tc["name"],
            "content": content,
        }

    def _build_guard_summary(
        self,
        *,
        guard_reason: str,
        executed_steps: List[Dict[str, Any]],
        pending_calls: List[Dict[str, Any]],
    ) -> str:
        """Summarize partial progress when the tool loop guard fires."""
        segments: List[str] = []
        if executed_steps:
            success_count = sum(1 for step in executed_steps if step["success"])
            failure_count = len(executed_steps) - success_count
            completed_counts = Counter(step["name"] for step in executed_steps)
            segments.append(
                "I made partial progress before pausing because the tool loop guard fired "
                f"({guard_reason})."
            )
            if failure_count:
                segments.append(
                    f"Completed {len(executed_steps)} tool calls: {success_count} succeeded, "
                    f"{failure_count} failed."
                )
            else:
                segments.append(
                    f"Completed {len(executed_steps)} tool calls successfully."
                )
            segments.append(
                "Completed tools: "
                + ", ".join(
                    f"{name} x{count}" for name, count in sorted(completed_counts.items())
                )
                + "."
            )
        else:
            segments.append(
                "I paused before executing the remaining tool plan because the tool loop guard "
                f"fired ({guard_reason})."
            )

        if pending_calls:
            pending_counts = Counter(tc["name"] for tc in pending_calls)
            segments.append(
                "Deferred pending tools: "
                + ", ".join(
                    f"{name} x{count}" for name, count in sorted(pending_counts.items())
                )
                + "."
            )
        segments.append("Continue in a follow-up turn with the next narrow step.")
        return " ".join(segments)

    def _trace(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"ToolUseLoop: {event}",
                payload or {},
            )
