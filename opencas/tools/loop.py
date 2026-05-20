"""ReAct-style tool-use loop for OpenCAS."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import numpy as np

from opencas.api import LLMClient
from opencas.autonomy.models import ActionRiskTier
from opencas.autonomy.self_approval import SelfApprovalLadder
from opencas.cognition import ToolCallTransit, build_tool_chain_transit_summary
from opencas.generation.policy import GenerationDomain, GenerationPhase, GenerationPolicyRequest
from opencas.memory import EpisodeKind
from opencas.projects.classifier import PROJECT_TYPE_SOFTWARE, PROJECT_TYPE_WRITING, classify_project_type
from opencas.runtime.capability_context import build_runtime_capability_context
from opencas.runtime.tool_runtime import resolve_runtime_tool_use_artifact_hint
from opencas.telemetry import EventKind, Tracer
from opencas.tools.action_memory import compact_action_head, compact_action_payload_metadata
from opencas.tools.registry import ToolRegistry
from opencas.tools.schema import build_tool_schemas
from opencas.tools.tool_use_memory import ToolUseMemoryStore

from .context import ToolUseContext, ToolUseResult, UserInputRequired
from .loop_guard import ToolLoopGuard
from .progress_guard import MeaningfulProgressGuard
from .tool_embedding_index import ToolEmbeddingIndex

_TURN_SCRATCHPAD_PREFIX = "Recent tool results this turn"


class _TurnScratchpad:
    """Compact per-turn evidence rendered before the model can repeat a call."""

    def __init__(self, *, max_entries: int = 8, max_output_chars: int = 360) -> None:
        self.max_entries = max(1, max_entries)
        self.max_output_chars = max(80, max_output_chars)
        self._entries: Dict[str, Dict[str, str]] = {}

    def record(self, tc: Dict[str, Any], result: Dict[str, Any]) -> None:
        key = self._key(tc)
        if key in self._entries:
            self._entries.pop(key)
        self._entries[key] = {
            "tool": str(tc.get("name") or "tool"),
            "args": self._args_summary(tc.get("args", {})),
            "status": "success" if bool(result.get("success", False)) else "failed",
            "output": self._excerpt(result.get("output", ""), self.max_output_chars),
            "metadata": self._metadata_summary(result.get("metadata")),
        }
        while len(self._entries) > self.max_entries:
            first_key = next(iter(self._entries))
            self._entries.pop(first_key, None)

    def render(self) -> Optional[str]:
        if not self._entries:
            return None
        lines = [
            (
                f"{_TURN_SCRATCHPAD_PREFIX} "
                "(dedupe aid; consult before repeating identical tool calls):"
            )
        ]
        for entry in self._entries.values():
            details = f"- {entry['tool']} {entry['args']}: {entry['status']}"
            if entry["metadata"]:
                details += f"; metadata: {entry['metadata']}"
            if entry["output"]:
                details += f"; output: {entry['output']}"
            lines.append(details)
        return "\n".join(lines)

    @staticmethod
    def _key(tc: Dict[str, Any]) -> str:
        try:
            args = json.dumps(tc.get("args", {}), sort_keys=True, default=str)
        except TypeError:
            args = str(tc.get("args", {}))
        return f"{tc.get('name', '')}:{args}"

    @staticmethod
    def _args_summary(args: Any) -> str:
        if not isinstance(args, dict) or not args:
            return "{}"
        preferred = []
        for key in ("file_path", "path", "query", "checksum", "command", "url"):
            if key in args:
                preferred.append(f"{key}={args[key]}")
        if preferred:
            return "{" + ", ".join(preferred[:3]) + "}"
        try:
            compact = json.dumps(args, sort_keys=True, default=str)
        except TypeError:
            compact = str(args)
        return _TurnScratchpad._excerpt(compact, 180)

    @staticmethod
    def _metadata_summary(metadata: Any) -> str:
        if not isinstance(metadata, dict) or not metadata:
            return ""
        parts = []
        for key in ("path", "checksum", "artifact_path", "task_id", "schedule_id"):
            if metadata.get(key):
                parts.append(f"{key}={metadata[key]}")
        return ", ".join(parts[:4])

    @staticmethod
    def _excerpt(value: Any, limit: int) -> str:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        compact = " ".join(str(text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."


class ToolUseLoop:
    """Iterative tool-use loop: LLM plans, tools execute, observations feed back."""

    DEFAULT_TOOL_CALL_BUDGET = ToolLoopGuard.MAX_ROUNDS
    HARD_TOOL_CALL_BUDGET = 96
    MAX_TOOL_RESULT_PROMPT_CHARS = 24_000
    TOOL_RESULT_TAIL_CHARS = 3_000
    MAX_MODEL_AWARE_TOOL_RESULT_PROMPT_CHARS = 256_000
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
        (8, ("manuscript", "creative_writing", "worldbuilding", "draft")),
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
        generation_request: Optional[GenerationPolicyRequest] = None,
        on_focus_enter: Optional[Callable[[], None]] = None,
        on_focus_exit: Optional[Callable[[], None]] = None,
    ) -> ToolUseResult:
        """Run the loop until the LLM finishes or max iterations are reached."""
        routing_objective = self._tool_routing_text(objective, messages)
        shadow_planning_context = await self._shadow_planning_context(ctx=ctx, objective=objective)
        tool_memory_context = self._tool_use_memory_context(ctx=ctx, objective=objective)
        cognitive_tool_context = await self._cognitive_tool_loop_context(ctx=ctx, objective=objective)
        capability_context = build_runtime_capability_context(ctx.runtime, max_items=100)
        supplemental_context = "\n\n".join(
            block
            for block in (
                capability_context,
                shadow_planning_context,
                tool_memory_context,
                cognitive_tool_context,
            )
            if block
        )
        if messages and messages[0].get("role") != "system":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an autonomous assistant. You have access to tools. "
                        "Use them only when they materially help accomplish the user's objective. "
                        "Prefer answering directly when no tool is required. "
                        "Do not answer with turn-scoped no-evidence boilerplate such as "
                        "'I do not have evidence in this turn' or 'I have not searched yet'; "
                        "when relevant evidence is missing and tools are available, use the "
                        "evidence tools before giving the final answer. "
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
            if capability_context and capability_context not in str(first.get("content", "")):
                first["content"] = (
                    str(first.get("content", "")).rstrip()
                    + f"\n\n{capability_context}"
                )
            if tool_memory_context and tool_memory_context not in str(first.get("content", "")):
                first["content"] = (
                    str(first.get("content", "")).rstrip()
                    + f"\n\n{tool_memory_context}"
                )
            if cognitive_tool_context and cognitive_tool_context not in str(first.get("content", "")):
                first["content"] = (
                    str(first.get("content", "")).rstrip()
                    + f"\n\n{cognitive_tool_context}"
                )
            no_turn_scoped_evidence_note = (
                "Do not answer with turn-scoped no-evidence boilerplate such as "
                "'I do not have evidence in this turn' or 'I have not searched yet'. "
                "When relevant evidence is missing and tools are available, use the "
                "evidence tools before giving the final answer."
            )
            if no_turn_scoped_evidence_note not in str(first.get("content", "")):
                first["content"] = (
                    str(first.get("content", "")).rstrip()
                    + f"\n\n{no_turn_scoped_evidence_note}"
                )

        all_tool_calls: List[Dict[str, Any]] = []
        executed_steps: List[Dict[str, Any]] = []
        tool_call_transits: List[ToolCallTransit] = []
        iterations = 0
        chain_id = f"toolchain-{uuid4().hex}"
        trust_context = self._infer_trust_context(ctx=ctx, objective=objective)
        call_repetitions: Counter[str] = Counter()
        tool_call_budget = self._select_tool_call_budget(
            objective=objective,
            messages=messages,
            ctx=ctx,
        )
        guard = ToolLoopGuard(max_rounds=tool_call_budget)
        progress_guard = MeaningfulProgressGuard()
        scratchpad = _TurnScratchpad()
        _in_focus_mode = False
        evidence_retry_used = False

        # --- Semantic tool routing: embed objective, lazily build index ---
        objective_vector: Optional[np.ndarray] = None
        if not self._index_built:
            await self._maybe_build_index(ctx)
            self._index_built = True
        if self.tool_embedding_index and self.tool_embedding_index.is_ready:
            objective_vector = await self._embed_objective(routing_objective, ctx)
        cognitive_tool_names = await self._cognitive_learned_tool_names(ctx=ctx, objective=routing_objective)
        evidence_requirement = self._merge_evidence_requirements(
            await self._cognitive_evidence_requirement(ctx=ctx, objective=routing_objective),
            self._artifact_lookup_evidence_requirement(routing_objective),
            self._web_research_evidence_requirement(routing_objective),
        )
        if evidence_requirement.get("required"):
            cognitive_tool_names.update(set(evidence_requirement.get("tool_names", set())))
        available_tools = self._filter_tools(
            ctx,
            objective=routing_objective,
            objective_vector=objective_vector,
            cognitive_tool_names=cognitive_tool_names,
        )
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
                tool_call_transits=tool_call_transits,
                tool_chain_summary=_chain_summary(),
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

        def _chain_summary():
            if not tool_call_transits:
                return None
            return build_tool_chain_transit_summary(
                objective=objective,
                chain_id=chain_id,
                call_transits=tool_call_transits,
            )

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
                    generation_request=generation_request
                    or GenerationPolicyRequest(
                        phase=GenerationPhase.EXECUTE,
                        domain=GenerationDomain.GENERAL,
                        risk_level="tool_execution",
                        source="tool_use_loop",
                        memory_focus=["tool_memory", "skill_memory", "project_memory"],
                    ),
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
                    if (
                        evidence_requirement.get("required")
                        and not evidence_retry_used
                        and not self._used_evidence_tool(all_tool_calls, evidence_requirement)
                        and iterations < ctx.max_iterations
                    ):
                        evidence_retry_used = True
                        messages.append(
                            {
                                "role": "user",
                                "content": self._evidence_requirement_prompt(evidence_requirement),
                            }
                        )
                        continue
                    if (
                        evidence_requirement.get("required")
                        and evidence_retry_used
                        and not self._used_evidence_tool(all_tool_calls, evidence_requirement)
                        and iterations < ctx.max_iterations
                    ):
                        auto_calls = self._auto_evidence_tool_calls(
                            evidence_requirement=evidence_requirement,
                            objective=routing_objective,
                            available_tools=available_tools,
                        )
                        if auto_calls:
                            raw_tool_calls = auto_calls
                            content = (
                                "OpenCAS evidence bridge auto-routed a required "
                                "read-only evidence tool before final answer."
                            )
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": content,
                                    "tool_calls": raw_tool_calls,
                                }
                            )
                            self._trace(
                                "tool_loop_auto_evidence_tool",
                                {
                                    "reason": evidence_requirement.get("reason"),
                                    "tools": [
                                        str(call.get("function", {}).get("name") or call.get("name") or "")
                                        for call in auto_calls
                                    ],
                                    "session_id": ctx.session_id,
                                    "task_id": ctx.task_id,
                                },
                            )
                    if not raw_tool_calls:
                        final = content or "Done."
                        return ToolUseResult(
                            final_output=final,
                            messages=messages,
                            tool_calls=all_tool_calls,
                            tool_call_transits=tool_call_transits,
                            tool_chain_summary=_chain_summary(),
                            iterations=iterations,
                        )

                tool_calls = self._normalize_tool_calls(raw_tool_calls)
                entry_intent = self._entry_intent_for_tool_calls(content, objective)
                for tc in tool_calls:
                    tc["entry_intent"] = entry_intent
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
                    concurrent_work = [
                        self._execute_tool_call_with_transit(
                            tc,
                            ctx,
                            chain_id=chain_id,
                            trust_context=trust_context,
                            retry_index=self._next_retry_index(call_repetitions, tc),
                        )
                        for tc in concurrent_calls
                    ]
                    results = await asyncio.gather(
                        *concurrent_work,
                        return_exceptions=True,
                    )
                    for tc, item in zip(concurrent_calls, results):
                        if isinstance(item, Exception):
                            result = {"success": False, "output": str(item), "metadata": {}}
                            transit = self._fallback_tool_call_transit(
                                tc,
                                chain_id=chain_id,
                                trust_context=trust_context,
                                result=result,
                            )
                        else:
                            result, transit = item
                        tool_call_transits.append(transit)
                        self._record_tool_use_memory(ctx, objective, tc, result)
                        executed_steps.append(
                            {
                                "name": tc["name"],
                                "success": bool(result.get("success", False)),
                            }
                        )
                        messages.append(self._build_tool_result_message(tc, result))
                        scratchpad.record(tc, result)
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
                        result, transit = await self._execute_tool_call_with_transit(
                            tc,
                            ctx,
                            chain_id=chain_id,
                            trust_context=trust_context,
                            retry_index=self._next_retry_index(call_repetitions, tc),
                        )
                    except Exception as exc:
                        result = {"success": False, "output": str(exc), "metadata": {}}
                        transit = self._fallback_tool_call_transit(
                            tc,
                            chain_id=chain_id,
                            trust_context=trust_context,
                            result=result,
                        )
                    tool_call_transits.append(transit)
                    self._record_tool_use_memory(ctx, objective, tc, result)
                    executed_steps.append(
                        {
                            "name": tc["name"],
                            "success": bool(result.get("success", False)),
                        }
                    )
                    messages.append(self._build_tool_result_message(tc, result))
                    scratchpad.record(tc, result)
                    fulfilled_ids.add(tc["id"])
                    halted = _record_progress(
                        tc,
                        result,
                        [_tc for _tc in tool_calls if _tc["id"] not in fulfilled_ids],
                    )
                    if halted is not None:
                        return halted

                self._refresh_turn_scratchpad_message(messages, scratchpad)

                # Auto-enter focus mode when round depth crosses the threshold
                if not _in_focus_mode and guard.is_deep(ctx.session_id) and on_focus_enter:
                    on_focus_enter()
                    _in_focus_mode = True

            # Max iterations reached
            return ToolUseResult(
                final_output="Reached maximum number of tool-use iterations.",
                messages=messages,
                tool_calls=all_tool_calls,
                tool_call_transits=tool_call_transits,
                tool_chain_summary=_chain_summary(),
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

    async def _cognitive_tool_loop_context(self, *, ctx: ToolUseContext, objective: str) -> str:
        """Build compact cognitive-state guidance for tool planning."""
        runtime = getattr(ctx, "runtime", None)
        inner = getattr(runtime, "ctx", None)
        store = getattr(runtime, "cognitive_state_store", None) or getattr(
            inner,
            "cognitive_state_store",
            None,
        )
        prompt_block = getattr(store, "prompt_block", None)
        if not callable(prompt_block):
            return ""
        try:
            block = await prompt_block(
                query=objective,
                session_id=ctx.session_id,
                char_budget=1200,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "cognitive tool-loop context build failed", exc_info=True,
            )
            return ""
        block = str(block or "").strip()
        if not block:
            return ""
        return (
            f"{block}\n"
            "For tool use, let these records affect attention, tool choice, retry strategy, "
            "and proof obligations only when the evidence is relevant."
        )

    async def _cognitive_evidence_requirement(self, *, ctx: ToolUseContext, objective: str) -> Dict[str, Any]:
        """Return an enforceable evidence requirement from active cognitive state."""

        runtime = getattr(ctx, "runtime", None)
        store = getattr(runtime, "cognitive_state_store", None) or getattr(
            getattr(runtime, "ctx", None),
            "cognitive_state_store",
            None,
        )
        if store is None:
            return {"required": False}

        objective_tokens = self._simple_tokens(objective)
        active: list[dict[str, Any]] = []
        list_working_memory = getattr(store, "list_working_memory", None)
        if callable(list_working_memory):
            try:
                items = await list_working_memory(limit=12)
            except Exception:
                items = []
            for item in items:
                slot = str(getattr(item, "slot", "") or "")
                if slot not in {
                    "uncertainty_seeking",
                    "prediction_error_review",
                    "counterfactual_retry",
                    "counterfactual_review",
                }:
                    continue
                content = str(getattr(item, "content", "") or "")
                content_tokens = self._simple_tokens(f"{slot} {content}")
                priority = float(getattr(item, "priority", 0.0) or 0.0)
                if priority >= 0.78 or not objective_tokens or objective_tokens & content_tokens:
                    active.append(
                        {
                            "source": "working_memory",
                            "slot": slot,
                            "priority": priority,
                            "content": content[:220],
                        }
                    )

        list_recent_events = getattr(store, "list_recent_events", None)
        if callable(list_recent_events) and not active:
            try:
                events = await list_recent_events(limit=12)
            except Exception:
                events = []
            for event in events:
                kind = str(getattr(getattr(event, "kind", ""), "value", getattr(event, "kind", "")))
                if kind not in {
                    "uncertainty_seeking",
                    "counterfactual",
                    "surprise",
                    "self_inspection_gap",
                    "recall_failure_memory",
                }:
                    continue
                summary = str(getattr(event, "summary", "") or "")
                summary_tokens = self._simple_tokens(f"{kind} {summary}")
                salience = float(getattr(event, "salience", 0.0) or 0.0)
                if salience >= 1.45 and (not objective_tokens or objective_tokens & summary_tokens):
                    active.append(
                        {
                            "source": "cognitive_event",
                            "kind": kind,
                            "salience": salience,
                            "summary": summary[:220],
                        }
                    )
                    break

        if not active:
            return {"required": False}

        preferred = {
            "artifact_lookup",
            "search_memories",
            "recall_autobiography",
            "recall_concepts",
            "runtime_status",
            "workflow_status",
            "workflow_list_schedules",
            "workflow_list_commitments",
            "fs_read_file",
            "fs_list_dir",
            "grep_search",
            "web_search",
            "web_fetch",
        }
        available = {entry.name for entry in self.tools.list_tools()}
        return {
            "required": True,
            "reason": "active_cognitive_evidence_obligation",
            "signals": active[:4],
            "tool_names": preferred & available,
        }

    @staticmethod
    def _used_evidence_tool(tool_calls: List[Dict[str, Any]], evidence_requirement: Dict[str, Any]) -> bool:
        names = set(evidence_requirement.get("tool_names", set()) or set())
        if not names:
            return False
        return any(str(call.get("name") or "") in names for call in tool_calls)

    def _auto_evidence_tool_calls(
        self,
        *,
        evidence_requirement: Dict[str, Any],
        objective: str,
        available_tools: List[Any],
    ) -> List[Dict[str, Any]]:
        """Create conservative read-only evidence calls when the model refuses to route tools.

        This is intentionally narrow. It only auto-routes evidence tools for explicit
        proof obligations, and for external web research it prefers a single
        read-only search query so the next model turn receives real evidence instead
        of another reminder prompt.
        """

        required_names = {
            str(name)
            for name in (evidence_requirement.get("tool_names", set()) or set())
            if name
        }
        available_names = {str(getattr(entry, "name", "")) for entry in available_tools}
        usable_names = required_names & available_names
        reason = str(evidence_requirement.get("reason") or "")
        if "external_web_research_required" in reason and "web_search" in usable_names:
            query = self._auto_web_search_query(objective)
            return [
                {
                    "id": f"auto-evidence-{uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"query": query}, ensure_ascii=False),
                    },
                }
            ]
        return []

    @staticmethod
    def _auto_web_search_query(objective: str) -> str:
        """Derive a bounded web-search query from an objective that needs current evidence."""

        text = " ".join(str(objective or "").split())
        lowered = text.lower()
        if any(token in lowered for token in ("5ch", "5ちゃん", "2channel", "2ch")):
            return "5ch 5ちゃん current thread board listing Japan discussion"
        if len(text) <= 180:
            return text
        trimmed = re.sub(
            r"\b(use|must use|do not answer|final answer|state the evidence|if .*? blocked)\b.*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if not trimmed:
            trimmed = text
        return trimmed[:180].rsplit(" ", 1)[0] or trimmed[:180]

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
        cognitive_tool_names: Optional[set[str]] = None,
    ) -> List[Any]:
        """Filter available tools based on plan mode and runtime constraints."""
        tools = self._exclude_disabled_plugin_tools(ctx, self.tools.list_tools())
        if not ctx.plan_mode:
            selected = self._select_tools_for_objective(tools, objective, objective_vector)
            selected = self._include_tool_use_memory_tools(ctx, objective, selected, tools)
            return self._include_named_tools(selected, tools, cognitive_tool_names or set())
        # In plan mode: only read-only tools + write_file restricted to plans dir
        allowed: List[Any] = []
        for entry in tools:
            if entry.risk_tier == ActionRiskTier.READONLY:
                allowed.append(entry)
            elif entry.name == "fs_write_file":
                allowed.append(entry)
        return allowed

    def _exclude_disabled_plugin_tools(self, ctx: ToolUseContext, tools: List[Any]) -> List[Any]:
        lifecycle = getattr(getattr(getattr(ctx, "runtime", None), "ctx", None), "plugin_lifecycle", None)
        is_tool_disabled = getattr(lifecycle, "is_tool_disabled", None)
        if not callable(is_tool_disabled):
            return list(tools)
        enabled: list[Any] = []
        for entry in tools:
            name = str(getattr(entry, "name", "") or "")
            try:
                disabled = bool(is_tool_disabled(name))
            except Exception:
                disabled = False
            if not disabled:
                enabled.append(entry)
        return enabled

    async def _cognitive_learned_tool_names(
        self,
        *,
        ctx: ToolUseContext,
        objective: str,
    ) -> set[str]:
        """Return learned-skill tool names relevant to the current objective."""
        runtime = getattr(ctx, "runtime", None)
        store = getattr(runtime, "cognitive_state_store", None) or getattr(
            getattr(runtime, "ctx", None),
            "cognitive_state_store",
            None,
        )
        list_skills = getattr(store, "list_learned_skills", None)
        if not callable(list_skills):
            return set()
        try:
            skills = await list_skills(limit=16)
        except Exception:
            logging.getLogger(__name__).warning("cognitive learned-skill lookup failed", exc_info=True)
            return set()
        objective_tokens = self._simple_tokens(objective)
        names: set[str] = set()
        for skill in skills:
            confidence = float(getattr(skill, "confidence", 0.0) or 0.0)
            if confidence < 0.55:
                continue
            status = str(getattr(skill, "activation_status", "") or "")
            if status not in {"evidence_gated_auto_use", "review_before_use", "candidate"}:
                continue
            skill_text = " ".join(
                [
                    str(getattr(skill, "name", "") or ""),
                    str(getattr(skill, "description", "") or ""),
                    " ".join(str(item) for item in (getattr(skill, "preconditions", []) or [])),
                ]
            )
            if objective_tokens and not (objective_tokens & self._simple_tokens(skill_text)):
                continue
            names.update(str(name) for name in (getattr(skill, "tool_sequence", []) or []) if str(name).strip())
        return names

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

    @staticmethod
    def _tool_routing_text(objective: str, messages: List[Dict[str, Any]]) -> str:
        """Keep recent carried-over intent visible to the tool selector."""

        current = " ".join(str(objective or "").split())
        chunks: list[str] = [current] if current else []
        if not ToolUseLoop._objective_needs_carried_tool_context(current.lower()):
            return current
        routing_markers = (
            "calendar",
            "cancel",
            "cancelled",
            "continue",
            "delete",
            "event",
            "reminder",
            "reschedule",
            "schedule",
            "scheduled",
            "unschedule",
            "undo",
            "work on it",
            "working on it",
        )
        candidates: list[Dict[str, Any]] = []
        if messages:
            candidates.append(messages[0])
            candidates.extend(messages[-8:])

        seen_ids: set[int] = set()
        for message in candidates:
            identity = id(message)
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            role = str(message.get("role", "") or "")
            content = str(message.get("content", "") or "")
            if not content or role == "tool":
                continue
            if "Runtime capability evidence:" in content:
                content = content.split("Runtime capability evidence:", 1)[0]
            compact = " ".join(content.split())
            if not compact:
                continue
            lowered = compact.lower()
            if role == "system":
                if not (
                    "[context:" in lowered
                    or "earlier conversation was compacted" in lowered
                    or "summary:" in lowered
                ):
                    continue
            elif role not in {"assistant", "user"}:
                continue
            if not any(marker in lowered for marker in routing_markers):
                continue
            excerpt = ToolUseLoop._excerpt(compact, 900)
            if excerpt and excerpt not in chunks:
                chunks.append(excerpt)
        return "\n".join(chunks)

    @staticmethod
    def _objective_needs_carried_tool_context(text: str) -> bool:
        """Return true for follow-up turns whose referent may live in prior context."""

        return any(
            marker in text
            for marker in (
                "are you working on it",
                "cancel it",
                "cancel that",
                "cancel those",
                "continue",
                "do it",
                "do that",
                "do those",
                "finish it",
                "go ahead",
                "keep going",
                "undo it",
                "undo that",
                "what are you doing",
                "work on it",
                "working on it",
            )
        )

    @staticmethod
    def _include_named_tools(selected: List[Any], tools: List[Any], names: set[str]) -> List[Any]:
        if not names:
            return selected
        tool_map = {entry.name: entry for entry in tools}
        seen = {entry.name for entry in selected}
        augmented = list(selected)
        for name in sorted(names):
            if name in seen:
                continue
            entry = tool_map.get(name)
            if entry is None:
                continue
            augmented.append(entry)
            seen.add(name)
        return augmented

    @staticmethod
    def _simple_tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9_./:-]{3,}", str(text or "").lower())
            if token not in {"the", "and", "for", "that", "with", "this"}
        }

    def _select_tools_for_objective(
        self,
        tools: List[Any],
        objective: str,
        objective_vector: Optional[np.ndarray] = None,
    ) -> List[Any]:
        """Select a smaller, relevant tool subset for the current objective."""
        # Reflective/conversational turns should stay tool-free
        text = (objective or "").lower()
        self_state_grounding = self._objective_needs_self_state_grounding(text)
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
        if conversational_only and not self_state_grounding:
            return []
        if self._objective_needs_full_tool_inventory(text):
            return list(tools)
        artifact_lookup_intent = self._objective_needs_artifact_lookup(text)

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
        project_type = classify_project_type(current_turn_text=text).project_type

        # Web retrieval is always available in non-conversational turns so the model
        selected_names.update({"search_memories", "recall_concepts"})
        # can research, fact-check, and look up current information without needing
        # the user to use specific keyword triggers.
        selected_names.update({"web_search", "web_fetch"})

        if self_state_grounding:
            selected_names.update(
                {
                    "runtime_status",
                    "workflow_status",
                    "self_inspection_query",
                    "wellbeing_query",
                    "cognitive_context_query",
                }
            )

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
                "workflow_get_commitment",
            })
        if any(token in text for token in ("plan", "planning", "checklist", "roadmap")):
            selected_names.update({
                "workflow_create_plan",
                "workflow_update_plan",
                "workflow_list_plans",
                "workflow_get_plan",
            })
        if any(token in text for token in ("baa task", "background task", "task id", "task_id", "work queue")):
            selected_names.update({
                "workflow_list_tasks",
                "workflow_get_task",
                "workflow_cancel_task",
            })
        if any(
            token in text
            for token in (
                "abandon",
                "cancel",
                "clean up",
                "cleanup",
                "undo",
                "unschedule",
                "compost",
                "delete",
                "discard",
                "placeholder",
                "remove",
                "restart",
                "start over",
                "throw out",
            )
        ):
            selected_names.update({
                "workflow_cancel_project",
                "workflow_cancel_schedule",
                "workflow_cancel_task",
                "workflow_list_tasks",
                "workflow_get_task",
                "workflow_update_schedule",
                "workflow_list_commitments",
                "workflow_get_commitment",
                "workflow_list_schedules",
                "workflow_get_schedule",
                "workflow_status",
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
                "workflow_get_schedule",
            })
        if (
            not artifact_lookup_intent
            and any(token in text for token in ("write", "writing", "draft", "document", "article", "note", "essay"))
        ):
            selected_names.update({
                "workflow_create_plan",
                "workflow_update_plan",
                "workflow_list_plans",
                "workflow_get_plan",
                "fs_read_file",
                "fs_write_file",
            })
            if project_type != PROJECT_TYPE_SOFTWARE and (
                project_type == PROJECT_TYPE_WRITING
                or any(token in text for token in ("document", "article", "note", "essay"))
            ):
                selected_names.add("workflow_create_writing_task")
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
                "gws",
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
                if entry.name.startswith("google_workspace_") or entry.name.startswith("himalaya_email_")
            )
        if any(
            token in text
            for token in (
                "cli",
                "command line",
                "command-line",
                "command not found",
                "--help",
                "man ",
                "manual",
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
            selected_names.update({"cli_discover_command", "pty_interact", "pty_remove", "pty_clear"})
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
                    "cli_discover_command",
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
        if artifact_lookup_intent:
            selected_names.discard("fs_write_file")
            selected_names.discard("edit_file")
            selected_names.discard("workflow_create_writing_task")

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
        required: list[str] = []
        if ToolUseLoop._objective_needs_full_tool_inventory(text):
            return []
        if ToolUseLoop._objective_needs_self_state_grounding(text):
            required.extend(
                [
                    "runtime_status",
                    "workflow_status",
                    "self_inspection_query",
                    "wellbeing_query",
                    "cognitive_context_query",
                ]
            )
        if classify_project_type(current_turn_text=text).project_type == PROJECT_TYPE_SOFTWARE:
            required.extend(["fs_read_file", "fs_list_dir", "grep_search", "glob_search", "fs_write_file"])
            if any(
                marker in text
                for marker in (
                    "build",
                    "cmake",
                    "compile",
                    "configure",
                    "launch",
                    "run",
                    "test",
                    "verify",
                )
            ):
                required.extend(["bash_run_command", "lsp_diagnostics"])
        if any(
            marker in text
            for marker in (
                "cli",
                "command line",
                "command-line",
                "command not found",
                "--help",
                "man ",
                "terminal",
                "shell",
            )
        ):
            required.extend(["bash_run_command", "cli_discover_command", "pty_start", "pty_interact"])
        if ToolUseLoop._objective_needs_external_web_research(text):
            required.extend(
                [
                    "web_search",
                    "web_fetch",
                    "http_request",
                    "browser_start",
                    "browser_navigate",
                    "browser_snapshot",
                ]
            )
        local_artifact = (
            "creative_writing" in text
            or ("writing" + " project") in text
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
            artifact_lookup_intent = ToolUseLoop._objective_needs_artifact_lookup(text)
            if artifact_lookup_intent:
                required.append("artifact_lookup")
            required.extend(["fs_read_file", "fs_list_dir", "grep_search", "glob_search"])
            artifact_write_intent = not artifact_lookup_intent and any(
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
                required.extend(["workflow_create_schedule", "workflow_list_schedules", "workflow_get_schedule"])
        if any(marker in text for marker in ("commitment", "promise", "deadline")):
            required.extend(["workflow_list_commitments", "workflow_get_commitment"])
        if any(
            marker in text
            for marker in (
                "learn this",
                "remember this workflow",
                "skill creation",
                "create skill",
                "new skill",
                "reusable skill",
                "teach bulma",
                "teach me how",
                "procedure",
            )
        ):
            required.extend(["cognitive_skill_library_search", "cognitive_skill_create"])
        if ToolUseLoop._objective_needs_commitment_scaffold(text):
            required.extend(
                [
                    "workflow_create_commitment",
                    "workflow_list_commitments",
                    "workflow_get_commitment",
                    "workflow_create_schedule",
                    "workflow_list_schedules",
                    "workflow_get_schedule",
                    "workflow_create_plan",
                    "workflow_update_plan",
                    "workflow_list_plans",
                    "cognitive_working_memory_update",
                    "cognitive_focus_set",
                    "cognitive_prospective_memory_set",
                    "cognitive_context_query",
                    "cognitive_skill_library_search",
                    "cognitive_skill_create",
                    "workflow_list_tasks",
                    "workflow_get_task",
                    "mcp_list_servers",
                    "mcp_register_server_tools",
                ]
            )
        if any(marker in text for marker in ("plan", "checklist", "roadmap")):
            required.extend(["workflow_list_plans", "workflow_get_plan"])
        if any(marker in text for marker in ("baa task", "background task", "task id", "task_id", "work queue")):
            required.extend(["workflow_list_tasks", "workflow_get_task"])
        if any(
            marker in text
            for marker in (
                "cancel",
                "delete",
                "duplicate",
                "only one",
                "unschedule",
                "undo",
            )
        ):
            required.extend([
                "workflow_list_schedules",
                "workflow_get_schedule",
                "workflow_cancel_schedule",
                "workflow_update_schedule",
                "workflow_list_tasks",
                "workflow_get_task",
            ])
        return list(dict.fromkeys(required))

    @staticmethod
    def _objective_needs_commitment_scaffold(text: str) -> bool:
        normalized = " ".join(str(text or "").lower().split())
        if not normalized:
            return False
        return any(
            marker in normalized
            for marker in (
                "be my assistant",
                "business model",
                "cash flow",
                "cash-flow",
                "complex project",
                "high priority",
                "income",
                "mission",
                "multi-step",
                "multistep",
                "proactive",
                "productive routine",
                "start making income",
                "start making money",
                "when i get home",
                "after i get home",
                "create an itinerary",
                "follow up",
                "remind me",
                "check in",
                "come back to",
                "get back to",
                "later today",
                "tomorrow",
            )
        )

    @staticmethod
    def _objective_needs_external_web_research(text: str) -> bool:
        normalized = str(text or "").lower()
        external_markers = (
            "2channel",
            "2chan",
            "2ch",
            "2ちゃん",
            "5channel",
            "5chan",
            "5ch",
            "5ちゃん",
            "board",
            "browser",
            "current",
            "forum",
            "headline",
            "http://",
            "https://",
            "internet",
            "latest",
            "news",
            "online",
            "page",
            "recent",
            "site",
            "thread",
            "today",
            "url",
            "web",
        )
        research_markers = (
            "browse",
            "explore",
            "fetch",
            "find",
            "go to",
            "inspect",
            "look up",
            "report",
            "research",
            "search",
            "summarize",
            "verify",
            "what people are talking about",
        )
        return (
            any(marker in normalized for marker in external_markers)
            and any(marker in normalized for marker in research_markers)
        )

    @staticmethod
    def _objective_needs_artifact_lookup(text: str) -> bool:
        return any(
            marker in text
            for marker in (
                "did you write",
                "did you make",
                "did you create",
                "do you recognize",
                "do you remember",
                "where did",
                "came from",
                "come from",
                "origin",
                "authorship",
                "provenance",
                "what changed",
                "who wrote",
            )
        )

    @staticmethod
    def _objective_needs_self_state_grounding(text: str) -> bool:
        return any(
            marker in text
            for marker in (
                "are you feeling",
                "daydream",
                "dream",
                "feeling",
                "feelings",
                "how are you",
                "how do you feel",
                "inner life",
                "outside this turn",
                "projects",
                "self-state",
                "somatic",
                "some kind of way",
                "what are you working on",
                "work on",
                "working on",
            )
        )

    @classmethod
    def _artifact_lookup_evidence_requirement(cls, objective: str) -> Dict[str, Any]:
        text = (objective or "").lower()
        has_path = (
            bool(re.search(r"(?:^|\s)(?:/mnt/|/home/|~/)[^\s]+", text))
            or bool(
                re.search(
                    r"\b[\w./~-]+\.(?:md|txt|py|json|yaml|yml|html|css|js|ts|tsx|sh|toml)\b",
                    text,
                )
            )
        )
        if not has_path or not cls._objective_needs_artifact_lookup(text):
            return {"required": False}
        return {
            "required": True,
            "reason": "workspace_artifact_handle_lookup_required",
            "signals": [
                {
                    "source": "objective",
                    "kind": "artifact_handle_question",
                    "summary": "The user asked about origin/authorship/recognition of a concrete local artifact.",
                }
            ],
            "tool_names": {"artifact_lookup"},
        }

    def _web_research_evidence_requirement(self, objective: str) -> Dict[str, Any]:
        text = (objective or "").lower()
        if not self._objective_needs_external_web_research(text):
            return {"required": False}
        preferred = {
            "web_search",
            "web_fetch",
            "http_request",
            "browser_start",
            "browser_navigate",
            "browser_snapshot",
        }
        available = {entry.name for entry in self.tools.list_tools()}
        return {
            "required": True,
            "reason": "external_web_research_required",
            "signals": [
                {
                    "source": "objective",
                    "kind": "external_web_research",
                    "summary": (
                        "The objective asks for current/external web evidence, "
                        "so a final answer must be grounded in search, fetch, HTTP, or browser output."
                    ),
                }
            ],
            "tool_names": preferred & available,
        }

    @staticmethod
    def _merge_evidence_requirements(*requirements: Dict[str, Any]) -> Dict[str, Any]:
        active = [item for item in requirements if item and item.get("required")]
        if not active:
            return {"required": False}
        tool_names: set[str] = set()
        reasons: list[str] = []
        signals: list[Any] = []
        for item in active:
            tool_names.update(set(item.get("tool_names", set()) or set()))
            reason = str(item.get("reason") or "").strip()
            if reason:
                reasons.append(reason)
            signals.extend(list(item.get("signals", []) or []))
        return {
            "required": True,
            "reason": "+".join(dict.fromkeys(reasons)),
            "signals": signals[:6],
            "tool_names": tool_names,
        }

    @staticmethod
    def _evidence_tool_hint(evidence_requirement: Dict[str, Any]) -> str:
        names = sorted(str(name) for name in evidence_requirement.get("tool_names", set()) if name)
        if not names:
            return ""
        return f" ({', '.join(names)})"

    @staticmethod
    def _evidence_requirement_prompt(evidence_requirement: Dict[str, Any]) -> str:
        reason = str(evidence_requirement.get("reason") or "")
        if "external_web_research_required" in reason:
            return (
                "Internal evidence requirement: this objective requires current/external web "
                "research evidence, and no search, fetch, HTTP, or browser evidence tool has "
                "been used yet. Use one relevant evidence tool from the available tools"
                f"{ToolUseLoop._evidence_tool_hint(evidence_requirement)}, "
                "then answer from the tool output. If every evidence path is unavailable, "
                "state the precise tool blocker instead of reporting unsupported findings."
            )
        if "workspace_artifact_handle_lookup_required" in reason:
            return (
                "Internal evidence requirement: the objective asks about a concrete workspace "
                "artifact, and no artifact evidence lookup has been used yet. Use one relevant "
                "evidence tool from the available tools"
                f"{ToolUseLoop._evidence_tool_hint(evidence_requirement)}, "
                "then answer from the retrieved timeline. If the evidence path is unavailable, "
                "state the precise blocker."
            )
        return (
            "Internal evidence requirement: relevant uncertainty or prediction-error state is "
            "active, and no evidence tool has been used yet. Use one relevant evidence tool "
            "from the available tools"
            f"{ToolUseLoop._evidence_tool_hint(evidence_requirement)}, "
            "or ask one focused question if the evidence path is unavailable; then answer concisely."
        )

    @staticmethod
    def _refresh_turn_scratchpad_message(messages: List[Dict[str, Any]], scratchpad: _TurnScratchpad) -> None:
        rendered = scratchpad.render()
        if not rendered:
            return
        messages[:] = [
            message
            for message in messages
            if not (
                message.get("role") == "user"
                and str(message.get("content", "")).startswith(_TURN_SCRATCHPAD_PREFIX)
            )
        ]
        messages.append({"role": "user", "content": rendered})

    @staticmethod
    def _objective_needs_full_tool_inventory(text: str) -> bool:
        """Detect prompts where the model is explicitly choosing among capabilities."""
        return any(
            marker in text
            for marker in (
                "available tools",
                "capability registry",
                "before you use tools",
                "check your capabilities",
                "check what you can do",
                "check to see what the tools do",
                "choose the right capability",
                "choose the right tool",
                "figure out what you can do",
                "inspect your capabilities",
                "read their descriptions",
                "tool descriptions",
                "understand what you can do",
                "what can you do",
                "what tools do you have",
                "what tools can you use",
            )
        )

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
        initial_complexity = getattr(ctx, "initial_complexity", None)
        if iteration == 1 and initial_complexity:
            return self._cap_iteration_complexity(str(initial_complexity), ctx)

        if not getattr(self.llm.model_routing, "auto_escalation", True):
            return self._cap_iteration_complexity("standard", ctx)

        text = (objective or "").lower()
        if ctx.plan_mode or any(
            token in text
            for token in ("project", "refactor", "architecture", "multi-file")
        ):
            if iteration >= 4:
                return self._cap_iteration_complexity("extra_high", ctx)
            if iteration >= 2:
                return self._cap_iteration_complexity("high", ctx)

        if guard.is_deep(ctx.session_id):
            selected = "high" if iteration < 5 else "extra_high"
            return self._cap_iteration_complexity(selected, ctx)
        if iteration >= 5:
            return self._cap_iteration_complexity("extra_high", ctx)
        if iteration >= 3:
            return self._cap_iteration_complexity("high", ctx)
        return self._cap_iteration_complexity("standard", ctx)

    @staticmethod
    def _cap_iteration_complexity(selected: str, ctx: ToolUseContext) -> str:
        max_complexity = getattr(ctx, "max_complexity", None)
        if not max_complexity:
            return selected
        order = {
            "light": 0,
            "standard": 1,
            "high": 2,
            "extra_high": 3,
            "xhigh": 3,
        }
        selected_rank = order.get(str(selected), order["standard"])
        max_rank = order.get(str(max_complexity), selected_rank)
        if selected_rank <= max_rank:
            return selected
        for name, rank in order.items():
            if rank == max_rank and name != "xhigh":
                return name
        return str(max_complexity)

    async def _execute_tool_call_with_transit(
        self,
        tc: Dict[str, Any],
        ctx: ToolUseContext,
        *,
        chain_id: str,
        trust_context: str,
        retry_index: int,
    ) -> tuple[Dict[str, Any], ToolCallTransit]:
        """Execute a tool and return mechanical transit evidence for that call."""
        pre_somatic = self._capture_somatic_state(ctx)
        started = time.monotonic()
        try:
            result = await self._execute_tool_call(tc, ctx)
        except UserInputRequired:
            raise
        except Exception as exc:
            result = {"success": False, "output": str(exc), "metadata": {}}
        duration_ms = max(0, int(round((time.monotonic() - started) * 1000)))
        post_somatic = self._capture_somatic_state(ctx)
        transit = ToolCallTransit(
            chain_id=chain_id,
            call_id=self._transit_call_id(tc, chain_id),
            tool_name=str(tc.get("name") or ""),
            entry_intent=self._excerpt(str(tc.get("entry_intent") or ""), 240),
            trust_context=trust_context,
            success=bool(result.get("success", False)),
            duration_ms=duration_ms,
            result_shape=self._result_shape_for_tool_result(result),
            certainty_delta=self._certainty_delta_for_tool_result(result),
            somatic_delta=self._somatic_delta(pre_somatic, post_somatic),
            task_mutation=self._tool_result_mutated_task(result),
            grounding=[
                {
                    "kind": "observed",
                    "source": "runtime",
                    "claim": "Tool-call transit captured from runtime execution evidence.",
                    "subject": "tool_call_transit",
                    "evidence_ids": [self._transit_call_id(tc, chain_id)],
                    "confidence": 0.82,
                }
            ],
            meta={
                "retry_index": retry_index,
                "arguments": tc.get("args", {}),
                "result_metadata_keys": sorted((result.get("metadata") or {}).keys()),
            },
        )
        return result, transit

    def _fallback_tool_call_transit(
        self,
        tc: Dict[str, Any],
        *,
        chain_id: str,
        trust_context: str,
        result: Dict[str, Any],
    ) -> ToolCallTransit:
        """Create transit evidence when execution failed outside the normal wrapper."""
        return ToolCallTransit(
            chain_id=chain_id,
            call_id=self._transit_call_id(tc, chain_id),
            tool_name=str(tc.get("name") or ""),
            entry_intent=self._excerpt(str(tc.get("entry_intent") or ""), 240),
            trust_context=trust_context,
            success=False,
            result_shape=self._result_shape_for_tool_result(result),
            certainty_delta=self._certainty_delta_for_tool_result(result),
            meta={"arguments": tc.get("args", {}), "fallback": True},
        )

    def _next_retry_index(self, call_repetitions: Counter[str], tc: Dict[str, Any]) -> int:
        key = self._transit_repetition_key(tc)
        retry_index = call_repetitions[key]
        call_repetitions[key] += 1
        return retry_index

    @staticmethod
    def _transit_repetition_key(tc: Dict[str, Any]) -> str:
        try:
            args = json.dumps(tc.get("args", {}), sort_keys=True, default=str)
        except TypeError:
            args = str(tc.get("args", {}))
        return f"{tc.get('name', '')}:{args}"

    @staticmethod
    def _transit_call_id(tc: Dict[str, Any], chain_id: str) -> str:
        return str(tc.get("id") or "").strip() or f"{chain_id}:{tc.get('name', 'tool')}"

    @staticmethod
    def _entry_intent_for_tool_calls(content: Any, objective: str) -> str:
        text = str(content or "").strip()
        if text:
            return ToolUseLoop._excerpt(text, 240)
        return ToolUseLoop._excerpt(objective, 240)

    @staticmethod
    def _infer_trust_context(*, ctx: ToolUseContext, objective: str) -> str:
        text = str(objective or "").lower()
        if "fallback" in text or "escalation" in text:
            return "fallback_escalation"
        if getattr(ctx, "task_id", None):
            return "runtime_task"
        if text:
            return "operator_requested"
        return "agent_initiated"

    @staticmethod
    def _capture_somatic_state(ctx: ToolUseContext) -> Any:
        somatic = getattr(getattr(ctx.runtime, "ctx", None), "somatic", None)
        state = getattr(somatic, "state", None)
        if state is None:
            return None
        copier = getattr(state, "model_copy", None)
        return copier() if callable(copier) else state

    @staticmethod
    def _somatic_delta(pre_state: Any, post_state: Any) -> Dict[str, float]:
        if pre_state is None or post_state is None:
            return {}
        delta: Dict[str, float] = {}
        for key in ("valence", "arousal", "fatigue", "tension", "focus", "energy", "certainty"):
            try:
                before = float(getattr(pre_state, key))
                after = float(getattr(post_state, key))
            except (TypeError, ValueError):
                continue
            change = round(after - before, 3)
            if abs(change) >= 0.01:
                delta[key] = change
        return delta

    @staticmethod
    def _result_shape_for_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
        output = result.get("output", "")
        output_text = output if isinstance(output, str) else json.dumps(output, default=str)
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        tags: list[str] = []
        if not bool(result.get("success", False)):
            tags.append("failed")
        if output_text.strip():
            tags.append("nonempty")
        else:
            tags.append("empty")
        if metadata.get("hook_blocked") or metadata.get("validation_error") or "blocked" in output_text.lower():
            tags.append("blocked")
        if metadata.get("truncated") or len(output_text) > 4000:
            tags.append("truncated")
        if "ambiguous" in output_text.lower() or "multiple possible" in output_text.lower():
            tags.append("ambiguous")
        return {
            "tags": tags,
            "output_chars": len(output_text),
            "metadata_keys": sorted(metadata.keys()),
        }

    @staticmethod
    def _certainty_delta_for_tool_result(result: Dict[str, Any]) -> float:
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        raw_delta = metadata.get("certainty_delta")
        if raw_delta is not None:
            try:
                return round(max(-1.0, min(1.0, float(raw_delta))), 3)
            except (TypeError, ValueError):
                pass
        shape = ToolUseLoop._result_shape_for_tool_result(result)
        tags = set(shape.get("tags", []))
        if "failed" in tags or "blocked" in tags:
            return -0.08
        if "empty" in tags:
            return -0.03
        if "ambiguous" in tags:
            return -0.02
        return 0.05

    @staticmethod
    def _tool_result_mutated_task(result: Dict[str, Any]) -> bool:
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        mutation_keys = (
            "task_id",
            "schedule_id",
            "commitment_id",
            "artifact_path",
            "plan_id",
            "provenance_events",
        )
        return any(key in metadata for key in mutation_keys)

    @staticmethod
    def _excerpt(text: str, limit: int) -> str:
        compact = " ".join(str(text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."

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
        execute_kwargs = {
            "session_id": ctx.session_id,
            "task_id": ctx.task_id,
        }
        if getattr(ctx, "audit_only", False):
            execute_kwargs["audit_only"] = True
        result = await ctx.runtime.execute_tool(name, args, **execute_kwargs)

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

        audit_only = bool(getattr(ctx, "audit_only", False))

        # Record episode for notable actions
        if result.get("success") and not audit_only:
            content_text, episode_payload = self._action_episode_payload(name, args, result)
            try:
                await ctx.runtime._record_episode(
                    content=content_text,
                    kind=EpisodeKind.ACTION,
                    session_id=ctx.session_id,
                    payload=episode_payload,
                )
            except TypeError:
                await ctx.runtime._record_episode(
                    content=content_text,
                    kind=EpisodeKind.ACTION,
                    session_id=ctx.session_id,
                )
            if name == "fs_read_file":
                await self._record_fs_read_chunk_artifact_episode(ctx, args, result)
        elif not audit_only:
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

    @classmethod
    async def _record_fs_read_chunk_artifact_episode(
        cls,
        ctx: ToolUseContext,
        args: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Create a durable ARTIFACT episode for chunked file reads."""
        if "offset" not in args and "limit" not in args:
            return

        payload = cls._safe_parse_json_payload(result.get("output"))
        if not payload:
            return
        path = str(args.get("file_path") or payload.get("path") or "").strip()
        if not path:
            return

        offset = cls._coerce_int(payload.get("offset"), default=0)
        limit = cls._coerce_int(payload.get("limit"), default=0)
        pagination_unit = cls._coerce_str(payload.get("pagination_unit"), default="lines")
        total_count = cls._coerce_int(
            payload.get("total_count") or payload.get("total_chars"),
            default=0,
        )
        total_lines = cls._coerce_int(payload.get("total_lines"), default=0)
        returned_count = cls._coerce_int(payload.get("returned_count"), default=0)
        if limit <= 0:
            return

        chunk_index = max(0, offset // limit)
        denominator = total_count if pagination_unit == "chars" else total_lines
        chunk_count = max(1, (denominator + limit - 1) // limit) if denominator else 1

        chunk_content = str(payload.get("content", ""))
        read_session_id = cls._coerce_str(args.get("read_session_id"), default="")
        concept_scope = cls._coerce_str(args.get("concept_scope"), default="")
        concept_label = cls._coerce_str(args.get("concept_label"), default="")

        lines = [
            f"Read chunk {chunk_index + 1}/{chunk_count} from {path}",
            (
                f"offset={offset} limit={limit} returned_count={returned_count} "
                f"pagination_unit={pagination_unit} total_count={denominator}"
            ),
        ]
        if read_session_id:
            lines.append(f"read_session_id={read_session_id}")
        if concept_scope:
            lines.append(f"concept_scope={concept_scope}")
        if concept_label:
            lines.append(f"concept_label={concept_label}")
        artifact_content = "\n".join(lines)
        if chunk_content:
            artifact_content += f"\n\n{chunk_content}"

        artifact_payload = {
            "path": path,
            "kind": "fs_read_file",
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "offset": offset,
            "limit": limit,
            "returned_count": returned_count,
            "pagination_unit": pagination_unit,
            "total_count": denominator,
            "total_lines": total_lines,
        }
        if read_session_id:
            artifact_payload["read_session_id"] = read_session_id
        if concept_scope:
            artifact_payload["concept_scope"] = concept_scope
        if concept_label:
            artifact_payload["concept_label"] = concept_label

        episode_payload = {
            "artifact": artifact_payload,
            "tool_name": "fs_read_file",
            "source_lane": "reflective",
            "origin_context_lane": "reflective",
            "context_authority": "interpretation",
            "context_material": "artifact",
        }
        try:
            await ctx.runtime._record_episode(
                content=artifact_content,
                kind=EpisodeKind.ARTIFACT,
                session_id=ctx.session_id,
                payload=episode_payload,
            )
        except TypeError:
            await ctx.runtime._record_episode(
                content=artifact_content,
                kind=EpisodeKind.ARTIFACT,
                session_id=ctx.session_id,
            )

    @classmethod
    def _action_episode_payload(
        cls,
        name: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        """Build the durable ACTION episode shape for a successful tool call."""
        metadata = dict(result.get("metadata") or {})
        payload = {
            "tool_name": name,
            "args": args,
            "result_metadata": metadata,
        }
        if name not in {"fs_write_file", "edit_file"}:
            compact_head = compact_action_head(name, args)
            if compact_head is not None:
                payload.update(compact_action_payload_metadata(compact_head))
                return compact_head.content, payload
            return f"tool {name}: {json.dumps(args)}", payload

        file_path = str(args.get("file_path") or args.get("path") or metadata.get("path") or "").strip()
        byte_count = cls._file_write_byte_count(name, args, file_path)
        checksum = str(metadata.get("checksum") or "").strip()
        if not checksum and file_path:
            checksum = cls._sha256_path(file_path)
        if not checksum:
            checksum = "pending"
        return f"tool {name} path={file_path} bytes={byte_count} checksum={checksum}", payload

    @staticmethod
    def _file_write_byte_count(name: str, args: Dict[str, Any], file_path: str) -> int:
        if name == "fs_write_file" and "content" in args:
            return len(str(args.get("content") or "").encode("utf-8"))
        if file_path:
            try:
                return Path(file_path).expanduser().resolve().stat().st_size
            except OSError:
                pass
        for key in ("new_string", "replacement", "content"):
            if key in args:
                return len(str(args.get(key) or "").encode("utf-8"))
        return 0

    @staticmethod
    def _sha256_path(file_path: str) -> str:
        try:
            h = hashlib.sha256()
            with Path(file_path).expanduser().resolve().open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return ""

    @staticmethod
    def _safe_parse_json_payload(value: Any) -> Dict[str, Any] | None:
        if not isinstance(value, str):
            return None
        try:
            payload = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_str(value: Any, *, default: str = "") -> str:
        text = str(value or "").strip()
        return text if text else default

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
            content = json.dumps(content, default=str)
        content = self._append_blocked_shell_fallback_guidance(tc, result, content)
        content = self._append_tool_failure_repair_guidance(tc, result, content)
        content = self._append_truncation_warning(result, content)
        content = self._bound_tool_result_content(
            content,
            limit=self._tool_result_prompt_char_limit(),
        )
        return {
            "role": "tool",
            "tool_call_id": tc.get("id", ""),
            "name": tc["name"],
            "content": content,
        }

    @staticmethod
    def _append_truncation_warning(result: Dict[str, Any], content: str) -> str:
        """Tell the agent that a partial result is not evidence of absence."""

        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        if not metadata.get("truncated"):
            return content

        parts = ["Result set is incomplete; do not infer absence from it."]
        total_count = metadata.get("total_count")
        returned_count = metadata.get("returned_count")
        if total_count is not None and returned_count is not None:
            parts.append(f"returned {returned_count} of {total_count}.")
        next_offset = metadata.get("next_offset")
        if next_offset is not None:
            parts.append(
                f"Re-run with next_offset={next_offset} (or a higher limit) before "
                "concluding anything is missing. Do not infer absence from this page."
            )
        else:
            parts.append("Re-run with a higher limit before concluding anything is missing.")
        warning = "\n\n[Tool evidence guard: " + " ".join(parts) + "]"
        return content + warning

    def _tool_result_prompt_char_limit(self) -> int:
        metadata_fn = getattr(self.llm, "model_context_metadata", None)
        if not callable(metadata_fn):
            return self.MAX_TOOL_RESULT_PROMPT_CHARS
        try:
            metadata = dict(metadata_fn() or {})
        except Exception:
            return self.MAX_TOOL_RESULT_PROMPT_CHARS
        prompt_budget = self._positive_int(metadata.get("prompt_context_budget"))
        if not prompt_budget or prompt_budget <= 12_000:
            return self.MAX_TOOL_RESULT_PROMPT_CHARS
        expanded = int(prompt_budget * 0.8)
        return max(
            self.MAX_TOOL_RESULT_PROMPT_CHARS,
            min(self.MAX_MODEL_AWARE_TOOL_RESULT_PROMPT_CHARS, expanded),
        )

    @staticmethod
    def _positive_int(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def _bound_tool_result_content(cls, content: str, *, limit: Optional[int] = None) -> str:
        """Keep individual tool observations from overwhelming the next LLM call."""

        text = str(content or "")
        applied_limit = max(1024, int(limit or cls.MAX_TOOL_RESULT_PROMPT_CHARS))
        if len(text) <= applied_limit:
            return text

        tail_chars = min(max(cls.TOOL_RESULT_TAIL_CHARS, applied_limit // 6), max(0, applied_limit // 4))
        marker = (
            "\n\n[Tool output truncated: "
            f"original_length={len(text)} chars; "
            f"prompt_bound={applied_limit} chars; "
            "showing the beginning and end only. "
            "Use narrower tool arguments, pagination, or a gist/summary tool if more detail is needed.]\n\n"
        )
        head_chars = max(0, applied_limit - len(marker) - tail_chars)
        return text[:head_chars].rstrip() + marker + text[-tail_chars:].lstrip()

    @staticmethod
    def _append_blocked_shell_fallback_guidance(
        tc: Dict[str, Any],
        result: Dict[str, Any],
        content: str,
    ) -> str:
        """Tell the agent how to recover when one shell command is denied."""

        if tc.get("name") != "bash_run_command" or bool(result.get("success", False)):
            return content
        output_text = str(content or "")
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        if metadata.get("missing_command") is True:
            guidance = (
                "\n\nRuntime guidance: this was a command lookup failure in the "
                "current execution environment, not proof that terminal access is "
                "absent. If the operator says the CLI exists, inspect the PATH and "
                "common user-local bins, try the command's help/man surface, or use "
                "pty_interact for terminal-native exploration. Do not say you are "
                "starting a PTY unless the response contains an actual PTY tool call "
                "or a durable follow-up record."
            )
            return output_text + guidance
        blocked = (
            output_text.strip().lower().startswith("tool execution blocked")
            or metadata.get("approval_denied") is True
            or metadata.get("hook_blocked") is True
            or metadata.get("validation_error") is True
            or "suggested_evidence" in output_text
        )
        if not blocked:
            return content
        guidance = (
            "\n\nRuntime guidance: this was a blocked shell action, not proof that shell "
            "or filesystem capabilities are absent. Do not repeat the same denied shell "
            "check. For file existence, directory contents, or source/proof inspection, "
            "use fs_list_dir, glob_search, or fs_read_file. To create or update proof "
            "documents, use fs_write_file or edit_file. Report the specific blocked "
            "command separately from any available fallback tools."
        )
        return output_text + guidance

    @staticmethod
    def _append_tool_failure_repair_guidance(
        tc: Dict[str, Any],
        result: Dict[str, Any],
        content: str,
    ) -> str:
        """Tell the model how to recover from common tool failures."""

        if bool(result.get("success", False)):
            return content
        output_text = str(content or "")
        if "[Tool recovery guidance:" in output_text:
            return output_text
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        tool_name = str(tc.get("name") or "")
        lowered = f"{tool_name} {output_text} {json.dumps(metadata, default=str)}".lower()
        guidance_parts: List[str] = []

        if metadata.get("auth_error") is True or any(
            token in lowered
            for token in (
                "invalid_grant",
                "expired or revoked",
                "authentication failed",
                "failed to get token",
                "autherror",
            )
        ):
            guidance_parts.append(
                "This is an authentication repair condition, not evidence that the Google "
                "Workspace capability is absent. Call google_workspace_auth_status if it is "
                "available, stop repeating the same Gmail/Calendar/Drive call until auth is "
                "healthy, and report the concrete reauthorization need once instead of "
                "spamming repeated failures."
            )

        if (
            tool_name.startswith("browser_")
            or "playwright" in lowered
            or "browser" in lowered
            or "session" in lowered and tool_name.startswith("browser_")
        ):
            guidance_parts.append(
                "For browser recovery, verify the browser session state. If there is no "
                "session, call browser_start before browser_navigate/browser_snapshot. For "
                "read-only research, fall back to web_search or web_fetch when browser "
                "automation is unavailable or blocked."
            )

        if tool_name in {"web_search", "web_fetch"} or "http" in lowered or "network" in lowered:
            guidance_parts.append(
                "For web evidence recovery, do not conclude the web is unavailable from one "
                "failed call. Try the paired web_search/web_fetch path, narrow the query or "
                "URL, and use browser navigation/snapshot if page rendering is required."
            )

        if tool_name in {"fs_read_file", "fs_list_dir", "grep_search", "glob_search", "fs_write_file", "edit_file"}:
            guidance_parts.append(
                "For filesystem recovery, locate the target with fs_list_dir, glob_search, "
                "or grep_search, read the exact file with fs_read_file, then retry edit_file "
                "or fs_write_file only after the current content is known."
            )

        if any(
            token in lowered
            for token in (
                "tool not found",
                "unknown tool",
                "missing required",
                "validation failed",
                "schema",
            )
        ):
            guidance_parts.append(
                "This is a tool-selection or argument-shape failure. Inspect the available "
                "tool schema, correct the tool name/arguments, or choose an equivalent "
                "available tool. Do not claim the whole capability is unavailable unless the "
                "available-tool evidence proves that."
            )

        if not guidance_parts:
            return output_text

        guidance = (
            "\n\n[Tool recovery guidance: "
            + " ".join(dict.fromkeys(guidance_parts))
            + "]"
        )
        return output_text + guidance

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
