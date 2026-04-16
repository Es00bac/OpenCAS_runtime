"""Conversation turn orchestration helpers for AgentRuntime."""

from __future__ import annotations

from dataclasses import dataclass
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
from opencas.tom import BeliefSubject
from opencas.tools import UserInputRequired

from .lane_metadata import build_assistant_message_meta

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


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
    await runtime.ctx.context_store.append(
        session_id,
        MessageRole.USER,
        user_input,
        meta=user_meta,
    )
    await runtime._record_episode(user_input, EpisodeKind.TURN, session_id=session_id, role="user")
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
    await runtime.ctx.somatic.emit_appraisal_event(
        AppraisalEventType.USER_INPUT_RECEIVED,
        source_text=user_input,
        trigger_event_id=session_id,
    )
    await runtime._record_episode(user_input, EpisodeKind.TURN, session_id=session_id, role="user")
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

    for message in repaired_messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
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
            await runtime.ctx.context_store.append(
                session_id,
                MessageRole.TOOL,
                message.get("content", ""),
                meta={
                    "tool_call_id": message.get("tool_call_id", ""),
                    "name": message.get("name", ""),
                },
            )


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
    await runtime._record_episode(content, EpisodeKind.TURN, session_id=session_id, role="assistant")

    pre_appraisal_state = runtime.ctx.somatic.state.model_copy()
    expressed_affect = await runtime.ctx.somatic.appraise_generated(content)
    await runtime.ctx.somatic.reconcile(
        pre_state=pre_appraisal_state,
        expressed_affect=expressed_affect,
        tom_engine=runtime.ctx.tom if hasattr(runtime.ctx, "tom") else None,
    )

    _apply_goal_directives(runtime, user_input)
    await runtime._capture_self_commitments(content, session_id)
    await _maybe_compact_manifest(runtime, session_id, manifest)
    await _record_tom_belief(runtime, user_input)
    await _record_relational_interaction(runtime, session_id, user_input, content)

    if getattr(runtime.ctx.config, "continuous_present_enabled", True):
        runtime.ctx.identity.recover_continuous_present()


def _apply_goal_directives(runtime: "AgentRuntime", user_input: str) -> None:
    goals, intention, drops = runtime._extract_goal_directives(user_input)
    for goal in goals:
        runtime.executive.add_goal(goal)
    if intention:
        runtime.executive.set_intention(intention)
    for drop in drops:
        for goal in list(runtime.executive.active_goals):
            if any(token in goal.lower() for token in drop.split() if len(token) > 3):
                runtime.executive.remove_goal(goal)
                break
    if goals or intention or drops:
        runtime._sync_executive_snapshot()


async def _maybe_compact_manifest(runtime: "AgentRuntime", session_id: str, manifest: Any) -> None:
    if not getattr(manifest, "token_estimate", None) or manifest.token_estimate <= 4000:
        return
    try:
        await runtime.maybe_compact_session(session_id)
    except Exception as exc:
        runtime._trace("compaction_error", {"error": str(exc)})


async def _record_tom_belief(runtime: "AgentRuntime", user_input: str) -> None:
    await runtime.tom.record_belief(
        BeliefSubject.USER,
        f"said: {user_input[:120]}",
        confidence=0.6,
    )
    metacognition = runtime.tom.check_consistency()
    if metacognition.contradictions:
        runtime._trace("metacognitive_alert", {"contradictions": metacognition.contradictions})


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
