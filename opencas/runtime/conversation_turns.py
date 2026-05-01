"""Conversation turn orchestration helpers for AgentRuntime."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from opencas.autonomy.models import (
    ActionRequest,
    ActionRiskTier,
    ApprovalDecision,
    ApprovalLevel,
)
from opencas.context import MessageRole, repair_tool_message_sequence
from opencas.memory import Episode, EpisodeKind
from opencas.somatic import AppraisalEventType
from opencas.somatic.models import SocialTarget
from opencas.tom import BeliefSubject
from opencas.tools import UserInputRequired

from .continuity_breadcrumbs import build_runtime_burst_breadcrumb
from .lane_metadata import build_assistant_message_meta
from .project_return import capture_project_return_from_turn
from .tom_intention_mirror import mirror_runtime_intention

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


_COMPACTION_TOKEN_THRESHOLD = 8000
_COMPACTION_MIN_REMOVED_COUNT = 4
_COMPACTION_COOLDOWN_SECONDS = 30 * 60

_USER_BELIEF_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^i\s+prefer\s+(.+)$", re.IGNORECASE), "prefers"),
    (re.compile(r"^i\s+want\s+(.+)$", re.IGNORECASE), "wants"),
    (re.compile(r"^i\s+need\s+(.+)$", re.IGNORECASE), "needs"),
    (re.compile(r"^i\s+think\s+(.+)$", re.IGNORECASE), "thinks"),
    (re.compile(r"^i\s+thought\s+(.+)$", re.IGNORECASE), "thought"),
    (re.compile(r"^i\s+believe\s+(.+)$", re.IGNORECASE), "believes"),
    (re.compile(r"^i\s+feel\s+(.+)$", re.IGNORECASE), "feels"),
)
_SELF_LOCATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:you|bulma|she)\s+lives?\s+(.+)$", re.IGNORECASE),
)
_USER_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^please\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:can|could|would)\s+you\s+(.+)$", re.IGNORECASE),
    re.compile(r"^make\s+sure\s+(.+)$", re.IGNORECASE),
)
_WORLD_FACT_PATTERN = re.compile(
    r"^(?!(?:i|we|you|please|can|could|would|make)\b)"
    r".+\b(?:is|are|uses|use|has|have|should|must|needs|need|requires|require|"
    r"returns|produces|writes|runs|connects|relies|rely)\b.+$",
    re.IGNORECASE,
)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"[.!?]+")


@dataclass
class ConversationLoopArtifacts:
    manifest: Any
    loop_result: Any | None
    content: str
    had_system: bool
    initial_message_count: int


async def handle_refusal_turn(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
    user_meta: Dict[str, Any],
    refusal: Any,
) -> str:
    user_appraisal = await runtime.ctx.somatic.emit_appraisal_event(
        AppraisalEventType.USER_INPUT_RECEIVED,
        source_text=user_input,
        trigger_event_id=session_id,
    )
    user_affect = (
        user_appraisal.affect_state.model_copy(deep=True)
        if user_appraisal.affect_state is not None
        else None
    )
    await runtime.ctx.context_store.append(
        session_id,
        MessageRole.USER,
        user_input,
        meta=user_meta,
    )
    await runtime._record_episode(
        user_input,
        EpisodeKind.TURN,
        session_id=session_id,
        role="user",
        affect=user_affect,
    )
    runtime._trace(
        "converse_refusal",
        {
            "session_id": session_id,
            "category": refusal.category.value if refusal.category else None,
            "reasoning": refusal.reasoning,
        },
    )
    if runtime.approval.ledger is not None:
        try:
            request = ActionRequest(
                tier=ActionRiskTier.READONLY,
                description=user_input,
                tool_name="conversation",
            )
            decision = ApprovalDecision(
                level=ApprovalLevel.MUST_ESCALATE,
                action_id=request.action_id,
                confidence=1.0,
                reasoning=refusal.reasoning,
                score=1.0,
            )
            await runtime.approval.ledger.record(decision, request, 1.0, None)
        except Exception:
            pass
    response_text = refusal.suggested_response or "I'm not able to respond to that."
    await runtime.ctx.context_store.append(
        session_id,
        MessageRole.ASSISTANT,
        response_text,
        meta=build_assistant_message_meta(runtime),
    )
    return response_text


async def persist_user_turn(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
    user_meta: Dict[str, Any],
) -> None:
    user_appraisal = await runtime.ctx.somatic.emit_appraisal_event(
        AppraisalEventType.USER_INPUT_RECEIVED,
        source_text=user_input,
        trigger_event_id=session_id,
    )
    user_affect = (
        user_appraisal.affect_state.model_copy(deep=True)
        if user_appraisal.affect_state is not None
        else None
    )
    await runtime._record_episode(
        user_input,
        EpisodeKind.TURN,
        session_id=session_id,
        role="user",
        affect=user_affect,
    )
    await runtime.ctx.context_store.append(
        session_id,
        MessageRole.USER,
        user_input,
        meta=user_meta,
    )


async def execute_conversation_tool_loop(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
) -> ConversationLoopArtifacts:
    # Keep manifest construction and tool-loop execution together so the caller
    # only orchestrates turn phases instead of managing intermediate loop state.
    manifest = await runtime.builder.build(user_input, session_id=session_id)
    messages = manifest.to_message_list()
    had_system = len(messages) > 0 and messages[0].get("role") == "system"
    initial_message_count = len(messages)
    loop_result: Optional[Any] = None

    try:
        payload = {"temperature": runtime.modulators.to_temperature()}
        tool_ctx = await runtime._build_tool_use_context(session_id=session_id)
        loop_result = await runtime.tool_loop.run(
            objective=user_input,
            messages=messages,
            ctx=tool_ctx,
            payload=payload,
            on_focus_enter=runtime.scheduler.enter_focus_mode if runtime.scheduler else None,
            on_focus_exit=runtime.scheduler.exit_focus_mode if runtime.scheduler else None,
        )
        content = loop_result.final_output
    except UserInputRequired as exc:
        content = exc.question
    except Exception as exc:
        import traceback as _tb
        stack = _tb.format_exc()
        try:
            debug_dir = Path(runtime.ctx.config.state_dir) / "logs"
            debug_dir.mkdir(parents=True, exist_ok=True)
            with (debug_dir / "conversation_turn_errors.log").open("a", encoding="utf-8") as _f:
                _f.write(f"[CONVERSATION_TURNS_ERROR] {exc}\n{stack}\n{'='*60}\n")
        except Exception:
            pass
        content = f"[Error generating response: {exc}]"

    return ConversationLoopArtifacts(
        manifest=manifest,
        loop_result=loop_result,
        content=content,
        had_system=had_system,
        initial_message_count=initial_message_count,
    )


async def persist_tool_loop_messages(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    artifacts: ConversationLoopArtifacts,
) -> None:
    loop_result = artifacts.loop_result
    if loop_result is None or not getattr(loop_result, "messages", None):
        return

    has_system = len(loop_result.messages) > 0 and loop_result.messages[0].get("role") == "system"
    offset = artifacts.initial_message_count
    if has_system and not artifacts.had_system:
        offset += 1

    repaired_messages = repair_tool_message_sequence(loop_result.messages[offset:])

    tool_loop_actions = 0
    for message in repaired_messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            tool_loop_actions += 1
            await runtime.ctx.context_store.append(
                session_id,
                MessageRole.ASSISTANT,
                message.get("content", ""),
                meta=build_assistant_message_meta(
                    runtime,
                    extra={"tool_calls": message["tool_calls"]},
                ),
            )
        elif role == "tool":
            tool_loop_actions += 1
            await runtime.ctx.context_store.append(
                session_id,
                MessageRole.TOOL,
                message.get("content", ""),
                meta={
                    "tool_call_id": message.get("tool_call_id", ""),
                    "name": message.get("name", ""),
                },
            )

    if tool_loop_actions:
        try:
            identity = getattr(runtime.ctx, "identity", None)
            continuity = getattr(identity, "continuity", None)
            if continuity is not None and hasattr(identity, "record_continuity_breadcrumb"):
                breadcrumb = build_runtime_burst_breadcrumb(
                    runtime,
                    phase="switch",
                    intent="Tool loop persisted intermediate messages",
                    focus="Tool loop persisted intermediate messages",
                    next_step="continue assistant finalization and executive update",
                )
                identity.record_continuity_breadcrumb(
                    intent="Tool loop persisted intermediate messages",
                    decision=f"{tool_loop_actions} tool-loop events were stored",
                    note=breadcrumb.note,
                    next_step="continue assistant finalization and executive update",
                )
        except Exception:
            runtime._trace("continuity_breadcrumb_tool_loop_error", {"session_id": session_id})


async def finalize_assistant_turn(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
    content: str,
    manifest: Any,
) -> None:
    # Persist the visible assistant turn first; every downstream subsystem
    # should be reacting to a response that already exists in session history.
    await runtime.ctx.context_store.append(
        session_id,
        MessageRole.ASSISTANT,
        content,
        meta=build_assistant_message_meta(runtime),
    )
    pre_appraisal_state = runtime.ctx.somatic.state.model_copy()
    expressed_affect = await runtime.ctx.somatic.appraise_generated(content)
    expressed_affect = expressed_affect.model_copy(
        update={"social_target": SocialTarget.SELF},
        deep=True,
    )
    await runtime._record_episode(
        content,
        EpisodeKind.TURN,
        session_id=session_id,
        role="assistant",
        affect=expressed_affect,
    )
    await runtime.ctx.somatic.reconcile(
        pre_state=pre_appraisal_state,
        expressed_affect=expressed_affect,
        tom_engine=runtime.ctx.tom if hasattr(runtime.ctx, "tom") else None,
    )

    await _apply_goal_directives(runtime, user_input, session_id=session_id)
    await runtime._capture_self_commitments(content, session_id)
    await capture_project_return_from_turn(
        runtime,
        session_id=session_id,
        user_input=user_input,
        assistant_content=content,
        manifest=manifest,
    )
    await _maybe_compact_manifest(runtime, session_id, manifest)
    await _record_tom_belief(runtime, user_input)
    await _record_relational_interaction(runtime, session_id, user_input, content)

    if getattr(runtime.ctx.config, "continuous_present_enabled", True):
        runtime.ctx.identity.recover_continuous_present()


async def _apply_goal_directives(
    runtime: "AgentRuntime",
    user_input: str,
    *,
    session_id: str,
) -> None:
    goals, intention, drops = runtime._extract_goal_directives(user_input)
    changed = False
    for goal in goals:
        runtime.executive.add_goal(goal)
        changed = True
    if intention:
        runtime.executive.set_intention(intention)
        await mirror_runtime_intention(
            runtime,
            intention,
            source="user_goal_directive",
            session_id=session_id,
        )
        changed = True
    for drop in drops:
        for goal in list(runtime.executive.active_goals):
            if any(token in goal.lower() for token in drop.split() if len(token) > 3):
                runtime.executive.remove_goal(goal)
                changed = True
                break
    if changed:
        try:
            identity = getattr(runtime.ctx, "identity", None)
            continuity = getattr(identity, "continuity", None)
            if continuity is not None and hasattr(identity, "record_continuity_breadcrumb"):
                breadcrumb = build_runtime_burst_breadcrumb(
                    runtime,
                    phase="switch",
                    intent=f"Parsed directives from user input: {user_input[:80]}",
                    focus=f"Parsed directives from user input: {user_input[:80]}",
                    next_step="capture snapshots and continue next turn",
                )
                identity.record_continuity_breadcrumb(
                    intent=f"Parsed directives from user input: {user_input[:80]}",
                    decision="executive goals/intention updated",
                    note=breadcrumb.note,
                    next_step="capture snapshots and continue next turn",
                )
        except Exception:
            runtime._trace("continuity_breadcrumb_goal_directives_error", {"session_id": runtime.ctx.config.session_id})
    if goals or intention or drops:
        runtime._sync_executive_snapshot()


async def _maybe_compact_manifest(runtime: "AgentRuntime", session_id: str, manifest: Any) -> None:
    if not getattr(manifest, "token_estimate", None) or manifest.token_estimate <= _COMPACTION_TOKEN_THRESHOLD:
        return
    cooldowns = getattr(runtime, "_session_compaction_cooldowns", None)
    if not isinstance(cooldowns, dict):
        cooldowns = {}
        runtime._session_compaction_cooldowns = cooldowns
    last_compaction = cooldowns.get(session_id)
    now = datetime.now(timezone.utc)
    if last_compaction is not None:
        try:
            elapsed = (now - last_compaction).total_seconds()
        except Exception:
            elapsed = _COMPACTION_COOLDOWN_SECONDS + 1
        if elapsed < _COMPACTION_COOLDOWN_SECONDS:
            runtime._trace(
                "compaction_skipped",
                {
                    "reason": "cooldown",
                    "session_id": session_id,
                    "elapsed_seconds": elapsed,
                },
            )
            return
    try:
        record = await runtime.maybe_compact_session(
            session_id,
            min_removed_count=_COMPACTION_MIN_REMOVED_COUNT,
        )
        if record is not None:
            cooldowns[session_id] = now
    except Exception as exc:
        runtime._trace("compaction_error", {"error": str(exc)})


async def _record_tom_belief(runtime: "AgentRuntime", user_input: str) -> None:
    extracted = _extract_user_input_beliefs(user_input)
    if extracted:
        for subject, predicate, confidence in extracted:
            await runtime.tom.record_belief(
                subject,
                predicate,
                confidence=confidence,
                meta={"source": "conversation_turn", "extractor": "rule_tier_a"},
            )
    elif _legacy_said_recorder_enabled(runtime):
        await runtime.tom.record_belief(
            BeliefSubject.USER,
            f"said: {user_input[:120]}",
            confidence=0.6,
            meta={"source": "conversation_turn", "extractor": "legacy_said"},
        )
    metacognition = runtime.tom.check_consistency()
    if metacognition.contradictions:
        runtime._trace("metacognitive_alert", {"contradictions": metacognition.contradictions})


def _legacy_said_recorder_enabled(runtime: "AgentRuntime") -> bool:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    return bool(getattr(config, "tom_legacy_said_recorder", False))


def _extract_user_input_beliefs(user_input: str) -> list[tuple[BeliefSubject, str, float]]:
    beliefs: list[tuple[BeliefSubject, str, float]] = []
    seen: set[tuple[BeliefSubject, str]] = set()

    for sentence in _SENTENCE_SPLIT.split(user_input):
        sentence = _normalize_belief_text(sentence, strip_article=False)
        if not sentence:
            continue

        extracted = _extract_user_preference_belief(sentence)
        if extracted is None:
            extracted = _extract_self_location_belief(sentence)
        if extracted is None:
            extracted = _extract_user_request_belief(sentence)
        if extracted is None:
            extracted = _extract_world_fact_belief(sentence)
        if extracted is None:
            continue

        subject, predicate, confidence = extracted
        key = (subject, predicate)
        if key not in seen:
            beliefs.append(extracted)
            seen.add(key)

    return beliefs


def _extract_user_preference_belief(sentence: str) -> tuple[BeliefSubject, str, float] | None:
    for pattern, verb in _USER_BELIEF_PATTERNS:
        match = pattern.match(sentence)
        if match is None:
            continue
        object_text = _normalize_belief_text(match.group(1), strip_article=False)
        if object_text:
            return BeliefSubject.USER, f"{verb} {object_text}", 0.72
    return None


def _extract_self_location_belief(sentence: str) -> tuple[BeliefSubject, str, float] | None:
    for pattern in _SELF_LOCATION_PATTERNS:
        match = pattern.match(sentence)
        if match is None:
            continue
        location = _normalize_self_location_text(match.group(1))
        if not location or not _looks_like_self_location(location):
            continue
        return BeliefSubject.SELF, f"lives {location}", 0.76
    return None


def _normalize_self_location_text(text: str) -> str:
    normalized = _normalize_belief_text(text, strip_article=False)
    replacements = (
        (r"\bwith me\b", "with user"),
        (r"\bin me\b", "in user"),
        (r"\bmy computer\b", "user's computer"),
        (r"\bmy machine\b", "user's machine"),
        (r"\bmy laptop\b", "user's laptop"),
        (r"\bmy desktop\b", "user's desktop"),
        (r"\bmy pc\b", "user's pc"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def _looks_like_self_location(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:with user|user's (?:computer|machine|laptop|desktop|pc)|"
            r"in|at|near|arvada|colorado|zip|zipcode|\d{5}(?:-\d{4})?)\b",
            text,
        )
    )


def _extract_user_request_belief(sentence: str) -> tuple[BeliefSubject, str, float] | None:
    for pattern in _USER_REQUEST_PATTERNS:
        match = pattern.match(sentence)
        if match is None:
            continue
        request = _normalize_belief_text(match.group(1), strip_article=False)
        if request:
            return BeliefSubject.USER, f"asked: {request}", 0.66
    return None


def _extract_world_fact_belief(sentence: str) -> tuple[BeliefSubject, str, float] | None:
    if _WORLD_FACT_PATTERN.match(sentence) is None:
        return None
    fact = _normalize_belief_text(sentence, strip_article=True)
    if len(fact.split()) < 3:
        return None
    return BeliefSubject.WORLD, fact, 0.68


def _normalize_belief_text(text: str, *, strip_article: bool) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    normalized = normalized.strip(" \t\r\n\"'`.,;:")
    if strip_article:
        normalized = _LEADING_ARTICLE.sub("", normalized)
    return normalized.lower()


async def _record_relational_interaction(
    runtime: "AgentRuntime",
    session_id: str,
    user_input: str,
    content: str,
) -> None:
    if not hasattr(runtime.ctx, "relational") or not runtime.ctx.relational:
        return
    interaction = Episode(
        kind=EpisodeKind.TURN,
        session_id=session_id,
        content=f"User: {user_input[:200]}\nAssistant: {content[:200]}",
        somatic_tag=runtime.ctx.somatic.state.somatic_tag,
    )
    await runtime.ctx.relational.record_interaction(
        episode=interaction,
        outcome="neutral",
    )
