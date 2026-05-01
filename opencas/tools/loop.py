"""ReAct-style tool-use loop for OpenCAS."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from opencas.api import LLMClient
from opencas.autonomy.models import ActionRiskTier
from opencas.autonomy.self_approval import SelfApprovalLadder
from opencas.memory import EpisodeKind
from opencas.runtime.tool_runtime import resolve_runtime_tool_use_artifact_hint
from opencas.telemetry import EventKind, Tracer
from opencas.tools.registry import ToolRegistry
from opencas.tools.schema import build_tool_schemas
from opencas.tools.tool_use_memory import ToolUseMemoryStore

from .context import ToolUseContext, ToolUseResult, UserInputRequired
from .loop_guard import ToolLoopGuard
from .progress_guard import MeaningfulProgressGuard
from .tool_embedding_index import ToolEmbeddingIndex


class ToolUseLoop:
    """Iterative tool-use loop: LLM plans, tools execute, observations feed back."""

    DEFAULT_TOOL_CALL_BUDGET = ToolLoopGuard.MAX_ROUNDS
    HARD_TOOL_CALL_BUDGET = 96
    _BUDGET_SIGNAL_WEIGHTS = (
        (
            24,
            (
                "tool loop circuit breaker",
                "exceeded 24",
                "continue the research",
                "continue researching",
            ),
        ),
        (24, ("research", "investigate", "deep dive")),
        (16, ("cross-reference", "source-backed", "sources", "citation", "verify", "fact-check")),
        (16, ("etymology", "name research", "linguistic", "historical context", "cultural context")),
        (12, ("compare", "comparison", "alternatives", "candidates", "options")),
        (8, ("look up", "web search", "search the web", "web_fetch", "web_search")),
        (8, ("manuscript", "chronicle", "worldbuilding", "draft")),
    )

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        approval: SelfApprovalLadder,
        tracer: Optional[Tracer] = None,
        tool_embedding_index: Optional[ToolEmbeddingIndex] = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.approval = approval
        self.tracer = tracer
        self.tool_embedding_index = tool_embedding_index
        self._index_built = False

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
        shadow_planning_context = await self._shadow_planning_context(ctx=ctx, objective=objective)
        tool_memory_context = self._tool_use_memory_context(ctx=ctx, objective=objective)
        supplemental_context = "\n\n".join(
            block for block in (shadow_planning_context, tool_memory_context) if block
        )
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
                        + (
                            " You are in PLAN MODE: only read files and write to the plans directory."
                            if ctx.plan_mode
                            else ""
                        )
                        + (f"\n\n{supplemental_context}" if supplemental_context else "")
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
            if shadow_planning_context and shadow_planning_context not in str(first.get("content", "")):
                first["content"] = (
                    str(first.get("content", "")).rstrip()
                    + f"\n\n{shadow_planning_context}"
                )
            if tool_memory_context and tool_memory_context not in str(first.get("content", "")):
                first["content"] = (
                    str(first.get("content", "")).rstrip()
                    + f"\n\n{tool_memory_context}"
                )

        all_tool_calls: List[Dict[str, Any]] = []
        executed_steps: List[Dict[str, Any]] = []
        iterations = 0
        tool_call_budget = self._select_tool_call_budget(
            objective=objective,
            messages=messages,
            ctx=ctx,
        )
        guard = ToolLoopGuard(max_rounds=tool_call_budget)
        progress_guard = MeaningfulProgressGuard()
        _in_focus_mode = False

        # --- Semantic tool routing: embed objective, lazily build index ---
        objective_vector: Optional[np.ndarray] = None
        if not self._index_built:
            await self._maybe_build_index(ctx)
            self._index_built = True
        if self.tool_embedding_index and self.tool_embedding_index.is_ready:
            objective_vector = await self._embed_objective(objective, ctx)
        available_tools = self._filter_tools(ctx, objective=objective, objective_vector=objective_vector)
        schemas = build_tool_schemas(available_tools)

        def _check_guard(tc: Dict[str, Any]) -> Optional[str]:
            return guard.record_call(ctx.session_id, tc["name"], tc.get("args", {}))

        def _halt_result(
            *,
            guard_reason: str,
            pending_calls: List[Dict[str, Any]],
        ) -> ToolUseResult:
            self._capture_guard_fire(
                objective=objective,
                ctx=ctx,
                guard_reason=guard_reason,
                executed_steps=executed_steps,
                pending_calls=pending_calls,
            )
            self._trace(
                "tool_loop_guard_fired",
                {
                    "reason": guard_reason,
                    "session_id": ctx.session_id,
                    "tool_call_budget": tool_call_budget,
                    "tool_loop_guard_fires": 1,
                },
            )
            return ToolUseResult(
                final_output=self._build_guard_summary(
                    guard_reason=guard_reason,
                    executed_steps=executed_steps,
                    pending_calls=pending_calls,
                ),
                messages=messages,
                tool_calls=all_tool_calls,
                iterations=iterations,
                guard_fired=True,
                guard_reason=guard_reason,
            )

        def _record_progress(
            tc: Dict[str, Any],
            result: Dict[str, Any],
            pending_calls: List[Dict[str, Any]],
        ) -> Optional[ToolUseResult]:
            progress_reason = progress_guard.record_result(
                tc["name"],
                tc.get("args", {}),
                result,
            )
            if progress_reason:
                return _halt_result(
                    guard_reason=progress_reason,
                    pending_calls=pending_calls,
                )
            return None

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
                        return _halt_result(
                            guard_reason=guard_reason,
                            pending_calls=[
                                _tc for _tc in tool_calls if _tc["id"] not in fulfilled_ids
                            ],
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
                        self._record_tool_use_memory(ctx, objective, tc, result)
                        executed_steps.append(
                            {
                                "name": tc["name"],
                                "success": bool(result.get("success", False)),
                            }
                        )
                        messages.append(self._build_tool_result_message(tc, result))
                        fulfilled_ids.add(tc["id"])
                        halted = _record_progress(
                            tc,
                            result,
                            [_tc for _tc in tool_calls if _tc["id"] not in fulfilled_ids],
                        )
                        if halted is not None:
                            return halted

                # Guard check and execute serial batch
                for tc in serial_calls:
                    guard_reason = _check_guard(tc)
                    if guard_reason:
                        return _halt_result(
                            guard_reason=guard_reason,
                            pending_calls=[
                                _tc for _tc in tool_calls if _tc["id"] not in fulfilled_ids
                            ],
                        )
                    try:
                        result = await self._execute_tool_call(tc, ctx)
                    except Exception as exc:
                        result = {"success": False, "output": str(exc), "metadata": {}}
                    self._record_tool_use_memory(ctx, objective, tc, result)
                    executed_steps.append(
                        {
                            "name": tc["name"],
                            "success": bool(result.get("success", False)),
                        }
                    )
                    messages.append(self._build_tool_result_message(tc, result))
                    fulfilled_ids.add(tc["id"])
                    halted = _record_progress(
                        tc,
                        result,
                        [_tc for _tc in tool_calls if _tc["id"] not in fulfilled_ids],
                    )
                    if halted is not None:
                        return halted

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

    def _select_tool_call_budget(
        self,
        *,
        objective: str,
        messages: List[Dict[str, Any]],
        ctx: ToolUseContext,
    ) -> int:
        """Select a bounded but task-sensitive tool-call budget."""
        explicit_budget = getattr(ctx, "tool_call_budget", None)
        if isinstance(explicit_budget, int) and explicit_budget > 0:
            return min(explicit_budget, self.HARD_TOOL_CALL_BUDGET)

        if ctx.plan_mode:
            return self.DEFAULT_TOOL_CALL_BUDGET

        context_text = self._budget_context_text(objective=objective, messages=messages)
        budget = self.DEFAULT_TOOL_CALL_BUDGET
        for weight, markers in self._BUDGET_SIGNAL_WEIGHTS:
            if any(marker in context_text for marker in markers):
                budget += weight
        return min(budget, self.HARD_TOOL_CALL_BUDGET)

    @staticmethod
    def _budget_context_text(*, objective: str, messages: List[Dict[str, Any]]) -> str:
        parts: List[str] = [str(objective or "")]
        for message in messages[-10:]:
            role = str(message.get("role") or "")
            if role not in {"user", "system"}:
                continue
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
        return " ".join(parts).lower()

    # ------------------------------------------------------------------
    # Semantic routing helpers
    # ------------------------------------------------------------------

    async def _maybe_build_index(self, ctx: ToolUseContext) -> None:
        """Lazily build the tool embedding index on first turn."""
        if self.tool_embedding_index is not None:
            return
        embeddings_svc = self._get_embeddings_service(ctx)
        if embeddings_svc is None:
            return
        try:
            self.tool_embedding_index = await ToolEmbeddingIndex.build(
                embeddings_svc, self.tools.list_tools(),
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "tool embedding index build failed", exc_info=True,
            )

    async def _embed_objective(
        self, objective: str, ctx: ToolUseContext,
    ) -> Optional[np.ndarray]:
        """Embed the objective text for semantic tool routing."""
        embeddings_svc = self._get_embeddings_service(ctx)
        if embeddings_svc is None:
            return None
        try:
            record = await embeddings_svc.embed(
                objective, task_type="RETRIEVAL_QUERY",
            )
            return np.array(record.vector, dtype=np.float32)
        except Exception:
            return None

    def _get_embeddings_service(self, ctx: ToolUseContext) -> Any:
        """Resolve the EmbeddingService from the runtime context."""
        rt = getattr(ctx, "runtime", None)
        if rt is None:
            return None
        inner = getattr(rt, "ctx", None)
        if inner is None:
            return None
        return getattr(inner, "embeddings", None)

    def _tool_use_memory_context(self, *, ctx: ToolUseContext, objective: str) -> str:
        """Build a compact learned tool-selection hint block for the prompt."""
        store = self._tool_use_memory_store(ctx)
        if store is None:
            return ""
        try:
            return store.build_context(
                objective=objective,
                available_tool_names=[entry.name for entry in self.tools.list_tools()],
                limit=5,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "tool-use memory context build failed", exc_info=True,
            )
            return ""

    def _record_tool_use_memory(
        self,
        ctx: ToolUseContext,
        objective: str,
        tc: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Record reusable tool-choice lessons without blocking tool execution."""
        store = self._tool_use_memory_store(ctx)
        if store is None:
            return
        try:
            store.record_result(
                objective=objective,
                tool_name=str(tc.get("name", "")),
                args=tc.get("args", {}),
                result=result,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "tool-use memory record failed", exc_info=True,
            )

    @staticmethod
    def _tool_use_memory_store(ctx: ToolUseContext) -> Optional[ToolUseMemoryStore]:
        """Resolve or lazily create the runtime's tool-use memory store."""
        runtime = getattr(ctx, "runtime", None)
        inner = getattr(runtime, "ctx", None)
        if inner is None:
            return None
        existing = getattr(inner, "tool_use_memory", None)
        if existing is not None:
            return existing
        config = getattr(inner, "config", None)
        state_dir = getattr(config, "state_dir", None)
        if state_dir is None:
            return None
        store = ToolUseMemoryStore(state_dir)
        try:
            setattr(inner, "tool_use_memory", store)
        except Exception:
            pass
        return store

    def _filter_tools(
        self,
        ctx: ToolUseContext,
        objective: str = "",
        objective_vector: Optional[np.ndarray] = None,
    ) -> List[Any]:
        """Filter available tools based on plan mode and runtime constraints."""
        tools = self.tools.list_tools()
        if not ctx.plan_mode:
            selected = self._select_tools_for_objective(tools, objective, objective_vector)
            return self._include_tool_use_memory_tools(ctx, objective, selected, tools)
        # In plan mode: only read-only tools + write_file restricted to plans dir
        allowed: List[Any] = []
        for entry in tools:
            if entry.risk_tier == ActionRiskTier.READONLY:
                allowed.append(entry)
            elif entry.name == "fs_write_file":
                allowed.append(entry)
        return allowed

    def _include_tool_use_memory_tools(
        self,
        ctx: ToolUseContext,
        objective: str,
        selected: List[Any],
        tools: List[Any],
    ) -> List[Any]:
        """Include tools that prior lessons say are relevant to this objective."""
        store = self._tool_use_memory_store(ctx)
        if store is None:
            return selected
        try:
            learned_names = store.relevant_tool_names(objective=objective)
        except Exception:
            logging.getLogger(__name__).warning(
                "tool-use memory lookup failed", exc_info=True,
            )
            return selected
        if not learned_names:
            return selected
        tool_map = {entry.name: entry for entry in tools}
        seen = {entry.name for entry in selected}
        augmented = list(selected)
        for name in learned_names:
            if name in seen:
                continue
            entry = tool_map.get(name)
            if entry is None:
                continue
            augmented.append(entry)
            seen.add(name)
        return augmented

    def _select_tools_for_objective(
        self,
        tools: List[Any],
        objective: str,
        objective_vector: Optional[np.ndarray] = None,
    ) -> List[Any]:
        """Select a smaller, relevant tool subset for the current objective."""
        # Reflective/conversational turns should stay tool-free
        text = (objective or "").lower()
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
        if conversational_only:
            return []

        # --- Semantic routing via embedding similarity (primary when available) ---
        if (
            objective_vector is not None
            and self.tool_embedding_index
            and self.tool_embedding_index.is_ready
        ):
            selected = self.tool_embedding_index.select_tools(objective_vector, tools)
            return self._include_required_context_tools(selected, tools, objective)

        # --- Keyword fallback (when embeddings unavailable) ---
        selected_names: set[str] = set()

        # Web retrieval is always available in non-conversational turns so the model
        selected_names.update({"search_memories", "recall_concepts"})
        # can research, fact-check, and look up current information without needing
        # the user to use specific keyword triggers.
        selected_names.update({"web_search", "web_fetch"})

        if any(
            token in text
            for token in (
                "runtime",
                "workflow",
                "status",
                "profile",
                "constraint",
                "operating roots",
            )
        ):
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
        if any(
            token in text
            for token in (
                "calendar",
                "due",
                "later",
                "may 1",
                "next return",
                "reminder",
                "reschedule",
                "return",
                "schedule",
                "soon",
                "tomorrow",
                "wait",
            )
        ):
            selected_names.update({
                "workflow_create_schedule",
                "workflow_update_schedule",
                "workflow_list_schedules",
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
        if any(
            token in text
            for token in (
                "supervise",
                "delegate",
                "launch claude",
                "launch codex",
                "launch kilocode",
                "launch kilo",
                "operator",
            )
        ):
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
        if any(
            token in text
            for token in (
                "pty",
                "terminal",
                "tui",
                "claude",
                "codex",
                "kilocode",
                "kilo",
                "kilo-code",
                "vim",
                "editor",
                "shell session",
            )
        ):
            selected_names.update({"pty_interact", "pty_remove", "pty_clear"})
        if any(token in text for token in ("poll", "session_id", "resize", "control sequence")):
            selected_names.update(
                entry.name for entry in tools if entry.name.startswith("pty_")
            )
        if any(token in text for token in ("process", "server", "background", "daemon")):
            selected_names.update(
                entry.name for entry in tools if entry.name.startswith("process_")
            )
        if any(
            token in text
            for token in (
                "search",
                "grep",
                "find",
                "code",
                "repo",
                "project",
                "write",
                "file",
                "edit",
                "memory",
                "recall",
                "sense of self",
            )
        ):
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
                    "search_memories",
                    "recall_concepts",
                }
            )
        if any(token in text for token in ("plan", "planning", "checklist", "roadmap")):
            selected_names.update({"enter_plan_mode", "exit_plan_mode"})

        # --- Plugin-provided tools (keyword surfacing) ---
        if any(
            token in text
            for token in (
                "time",
                "date",
                "today",
                "tomorrow",
                "yesterday",
                "duration",
                "elapsed",
                "ago",
                "until",
                "timezone",
                "timestamp",
                "age of",
            )
        ):
            selected_names.update({"time_now", "time_parse", "time_diff", "time_age"})
        if any(
            token in text
            for token in (
                "api",
                "post request",
                "put request",
                "delete request",
                "patch request",
                "rest",
                "endpoint",
                "webhook",
                "bearer",
                "authorization header",
            )
        ):
            selected_names.update({"http_request"})
        if any(
            token in text
            for token in (
                "calculate",
                "compute",
                "arithmetic",
                "math",
                "convert",
                "conversion",
                "kilometer",
                "mile",
                "celsius",
                "fahrenheit",
                "megabyte",
                "gigabyte",
            )
        ):
            selected_names.update({"calculate", "unit_convert"})
        if any(
            token in text
            for token in ("diff", "difference between", "compare files", "compare text", "what changed", "unified diff")
        ):
            selected_names.update({"diff_text", "diff_files"})
        if any(
            token in text
            for token in (
                "hash",
                "checksum",
                "sha256",
                "sha1",
                "md5",
                "base64",
                "encode",
                "decode",
                "url-encode",
                "url-decode",
                "slugify",
                "slug",
            )
        ):
            selected_names.update(
                {"hash_text", "base64_encode", "base64_decode", "url_encode", "url_decode", "slugify"}
            )
        if any(
            token in text
            for token in ("json", "validate schema", "json path", "json query")
        ):
            selected_names.update({"json_query", "json_validate"})
        if any(
            token in text
            for token in ("system", "cpu", "memory usage", "disk", "ram", "load average", "host stats", "host status")
        ):
            selected_names.update({"system_status"})
        if any(
            token in text
            for token in ("note", "save note", "remember this", "scratchpad", "jot")
        ):
            selected_names.update({"note_save", "note_list", "note_read"})

        # Default exploration toolkit when no specific keywords match
        if not selected_names:
            selected_names.update({
                "fs_read_file",
                "fs_list_dir",
                "bash_run_command",
                "agent",
                "web_search",
                "web_fetch",
            })

        selected = [entry for entry in tools if entry.name in selected_names]
        return self._include_required_context_tools(selected, tools, objective)

    def _include_required_context_tools(
        self,
        selected: List[Any],
        tools: List[Any],
        objective: str,
    ) -> List[Any]:
        """Preserve deterministic tool affordances for prompts semantic routing can miss."""
        required_names = self._required_tool_names_for_objective(objective)
        if not required_names:
            return selected
        tool_map = {entry.name: entry for entry in tools}
        seen = {entry.name for entry in selected}
        augmented = list(selected)
        for name in required_names:
            if name in seen:
                continue
            entry = tool_map.get(name)
            if entry is None:
                continue
            augmented.append(entry)
            seen.add(name)
        return augmented

    @staticmethod
    def _required_tool_names_for_objective(objective: str) -> List[str]:
        text = (objective or "").lower()
        local_artifact = (
            "chronicle" in text
            or "workspace" in text
            or "local file" in text
            or "file://" in text
            or bool(re.search(r"(?:^|\s)(?:/mnt/|/home/|~/)[^\s]+", text))
            or bool(
                re.search(
                    r"\b[\w./~-]+\.(?:md|txt|py|json|yaml|yml|html|css|js|ts|tsx|sh|toml)\b",
                    text,
                )
            )
        )
        if local_artifact:
            required = ["fs_read_file", "fs_list_dir", "grep_search", "glob_search"]
            artifact_write_intent = any(
                marker in text
                for marker in (
                    "append",
                    "apply",
                    "draft",
                    "edit",
                    "integrate",
                    "insert",
                    "manuscript progress",
                    "merge",
                    "modified",
                    "modify",
                    "persist",
                    "replace",
                    "revise",
                    "revision",
                    "save",
                    "update",
                    "write",
                )
            )
            if artifact_write_intent:
                required.extend(["fs_write_file", "edit_file"])
            if any(marker in text for marker in ("reschedule", "return", "schedule", "soon")):
                required.append("workflow_create_schedule")
            return required
        return []

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
        result = await ctx.runtime.execute_tool(
            name,
            args,
            session_id=ctx.session_id,
            task_id=ctx.task_id,
        )

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

    def _capture_guard_fire(
        self,
        *,
        objective: str,
        ctx: ToolUseContext,
        guard_reason: str,
        executed_steps: List[Dict[str, Any]],
        pending_calls: List[Dict[str, Any]],
    ) -> None:
        shadow_registry = getattr(getattr(ctx.runtime, "ctx", None), "shadow_registry", None)
        capture = getattr(shadow_registry, "capture_tool_loop_guard", None)
        if not callable(capture):
            return
        tool_counts = Counter(step.get("name", "unknown") for step in executed_steps if step.get("name"))
        dominant_tool = tool_counts.most_common(1)[0][0] if tool_counts else (
            pending_calls[0]["name"] if pending_calls else "tool_loop_guard"
        )
        capture(
            {
                "objective": objective,
                "guard_reason": guard_reason,
                "session_id": ctx.session_id,
                "task_id": ctx.task_id,
                "dominant_tool": dominant_tool,
                "executed_tool_counts": dict(tool_counts),
                "executed_steps": list(executed_steps),
                "pending_calls": [
                    {
                        "name": call.get("name"),
                        "args": call.get("args", {}),
                    }
                    for call in pending_calls
                ],
                "pending_tools": [call.get("name") for call in pending_calls if call.get("name")],
            }
        )

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

    async def _shadow_planning_context(
        self,
        *,
        ctx: ToolUseContext,
        objective: str,
    ) -> str:
        shadow_registry = getattr(getattr(ctx.runtime, "ctx", None), "shadow_registry", None)
        builder = getattr(shadow_registry, "build_planning_context", None)
        if not callable(builder):
            return ""
        artifact_hint = await resolve_runtime_tool_use_artifact_hint(
            ctx.runtime,
            ctx,
            objective=objective,
        )
        if artifact_hint and not ctx.artifact_hint:
            ctx.artifact_hint = artifact_hint
        context = builder(
            objective=objective,
            artifact=artifact_hint,
        )
        if not isinstance(context, dict) or not context.get("available"):
            return ""
        return str(context.get("prompt_block", "") or "").strip()

    def _trace(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"ToolUseLoop: {event}",
                payload or {},
            )
