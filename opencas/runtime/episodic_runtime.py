"""Episode, continuity, and self-commitment helpers for AgentRuntime.

These helpers keep autobiographical state handling out of the main runtime
orchestrator. The runtime still owns when these hooks fire; this module owns
the local rules for how continuity, episode persistence, and promise capture
are applied.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional

from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.commitment_extraction import (
    SelfCommitmentCandidate,
    extract_self_commitments,
)
from opencas.memory import EdgeKind, Episode, EpisodeEdge, EpisodeKind
from opencas.proof_chain import attach_operator_promise_claim
from opencas.somatic import AppraisalEventType
from opencas.somatic.models import AffectState
from opencas.tom import BeliefSubject

from .continuity_breadcrumbs import (
    build_runtime_burst_breadcrumb,
    is_recoverable_burst_breadcrumb,
    recover_burst_continuity_context,
)
from .commitment_followthrough import seed_commitment_cognitive_followthrough

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


def extract_runtime_goal_directives(text: str) -> tuple[List[str], Optional[str], List[str]]:
    """Heuristically extract goals, intention, and drop-requests from user text."""
    text_lower = text.lower()
    goals: List[str] = []
    intention: Optional[str] = None
    drops: List[str] = []

    def _directive_segments(source: str) -> List[str]:
        segments = re.split(r"(?:\n+|(?<=[.!?])\s+)", source)
        cleaned: List[str] = []
        for segment in segments:
            candidate = segment.strip().strip("\"'`")
            while True:
                narrowed = re.sub(
                    r"^(?:and|then|so|well|okay|ok|please|for now|from now on|right now|at the moment)\s+",
                    "",
                    candidate,
                ).strip()
                if narrowed == candidate:
                    break
                candidate = narrowed
            if candidate:
                cleaned.append(candidate)
        return cleaned

    goal_patterns = [
        r"^(?:your goal is|you should|i want you to|make it your goal to|prioritize)\s+(.*?)(?:[.]|$)",
        r"^focus on\s+(.*?)(?:[.]|$)",
    ]
    for segment in _directive_segments(text_lower):
        for pattern in goal_patterns:
            match = re.match(pattern, segment)
            if match:
                clause = match.group(1).strip()
                if clause:
                    goals.append(clause)

    intention_patterns = [
        r"^(?:your current task is|current task is|work on|start on|your intention is|intention is)\s+(.*?)(?:[.]|$)",
    ]
    for segment in _directive_segments(text_lower):
        for pattern in intention_patterns:
            match = re.match(pattern, segment)
            if match:
                intention = match.group(1).strip()
                break
        if intention:
            break

    drop_phrases = ["done with that", "drop the goal", "forget about", "stop working on"]
    for phrase in drop_phrases:
        if phrase in text_lower:
            drops.append(phrase)

    return goals, intention, drops


def extract_runtime_self_commitments(
    text: str,
    *,
    user_input: str | None = None,
) -> List[SelfCommitmentCandidate]:
    """Extract normalized future-action self-commitments from assistant text."""
    return extract_self_commitments(text, user_context=user_input)


async def _recover_latest_burst_breadcrumb(
    runtime: "AgentRuntime",
    continuity: Any,
) -> tuple[Optional[str], str, Optional[str]]:
    """Return the most recent recoverable burst breadcrumb, its source, and note."""
    relational = getattr(getattr(runtime, "ctx", None), "relational", None)
    if relational is not None:
        try:
            state = getattr(relational, "state", None)
            state_breadcrumb = getattr(state, "continuity_breadcrumb", None)
            if is_recoverable_burst_breadcrumb(state_breadcrumb, None):
                return state_breadcrumb, "musubi_state", None
        except Exception:
            pass

    recent_breadcrumbs = list(getattr(continuity, "continuity_breadcrumbs", []) or [])
    if recent_breadcrumbs:
        latest = recent_breadcrumbs[-1]
        if is_recoverable_burst_breadcrumb(latest, None):
            return latest, "identity", None

    if relational is not None and hasattr(relational, "list_recent_burst_records"):
        try:
            candidate_records = await relational.list_recent_burst_records(limit=5)
        except Exception:
            candidate_records = []
        for record in candidate_records:
            breadcrumb = getattr(record, "continuity_breadcrumb", "") or ""
            note = getattr(record, "note", None)
            if is_recoverable_burst_breadcrumb(breadcrumb, None, note=note):
                return breadcrumb, "musubi_history", note

    if relational is not None and hasattr(relational, "list_recent_continuity_breadcrumbs"):
        try:
            candidate_breadcrumbs = await relational.list_recent_continuity_breadcrumbs(limit=5)
        except Exception:
            candidate_breadcrumbs = []
        for candidate in candidate_breadcrumbs:
            if is_recoverable_burst_breadcrumb(candidate, None):
                return candidate, "musubi_history", None
    return None, "identity", None


async def capture_runtime_self_commitments(
    runtime: "AgentRuntime",
    content: str,
    session_id: str,
    *,
    user_input: str | None = None,
) -> List[Commitment]:
    """Persist normalized self-commitments and mirror them into ToM/somatic state."""
    captures = extract_runtime_self_commitments(content, user_input=user_input)
    if not captures:
        return []

    pause_reason = runtime.executive.pause_reason() if runtime.executive else None
    status = CommitmentStatus.BLOCKED if pause_reason else CommitmentStatus.ACTIVE
    commitments: List[Commitment] = []

    for capture in captures:
        capability_audit = _build_commitment_capability_audit(runtime, capture)
        commitment_status = status
        if capability_audit.get("status") == "missing_required_tools":
            commitment_status = CommitmentStatus.BLOCKED
        commitment: Optional[Commitment] = None
        if runtime.commitment_store:
            commitment = Commitment(
                content=capture.content[:220],
                status=commitment_status,
                tags=["self_commitment", "conversation"],
                meta={
                    "source": "assistant_response",
                    "session_id": session_id,
                    "trigger": capture.trigger,
                    "resume_policy": "auto_on_executive_recovery",
                    "source_sentence": capture.source_sentence,
                    "normalization_source": capture.normalization_source,
                    "capture_confidence": capture.confidence,
                    "capability_audit": capability_audit,
                    **(
                        {"blocked_reason": f"executive_{pause_reason}"}
                        if pause_reason
                        else {}
                    ),
                    **(
                        {"source_user_request": user_input}
                        if user_input
                        else {}
                    ),
                    **(
                        {"blocked_reason": "missing_required_tools"}
                        if commitment_status == CommitmentStatus.BLOCKED and not pause_reason
                        else {}
                    ),
                },
            )
            await runtime.commitment_store.save(commitment)
            commitments.append(commitment)
            try:
                proof_claim = await attach_operator_promise_claim(
                    runtime,
                    commitment,
                    subject="self_commitment",
                    source="assistant_response",
                    evidence_summary="Self-commitment persisted from assistant response.",
                    meta={"session_id": session_id},
                )
                if proof_claim is not None:
                    await runtime.commitment_store.save(commitment)
            except Exception as exc:
                runtime._trace(
                    "proof_chain_commitment_capture_failed",
                    {
                        "commitment_id": str(commitment.commitment_id),
                        "error": str(exc),
                    },
                )
            await seed_commitment_cognitive_followthrough(runtime, commitment)
            runtime._trace(
                "self_commitment_captured",
                {
                    "commitment_id": str(commitment.commitment_id),
                    "status": commitment_status.value,
                    "normalization_source": capture.normalization_source,
                    "capability_audit_status": capability_audit.get("status"),
                },
            )

        if runtime.tom:
            await runtime.tom.record_intention(
                BeliefSubject.SELF,
                capture.content[:220],
                meta={
                    "source": "self_commitment_capture",
                    "session_id": session_id,
                    "capture_confidence": capture.confidence,
                    **(
                        {"commitment_id": str(commitment.commitment_id)}
                        if commitment is not None
                        else {}
                    ),
                },
            )

    await runtime.ctx.somatic.emit_appraisal_event(
        AppraisalEventType.SELF_RESPONSE_GENERATED,
        source_text=content,
        trigger_event_id=session_id,
        meta={
            "self_commitment_count": len(captures),
            "self_commitment_contents": [capture.content for capture in captures],
        },
    )
    return commitments


def _build_commitment_capability_audit(
    runtime: "AgentRuntime",
    capture: SelfCommitmentCandidate,
) -> dict[str, Any]:
    """Record the concrete capability surface behind an accepted commitment."""

    tool_names: set[str] = set()
    tools = getattr(runtime, "tools", None)
    list_tools = getattr(tools, "list_tools", None)
    if callable(list_tools):
        try:
            tool_names = {
                str(getattr(entry, "name", "") or "").strip()
                for entry in list_tools()
                if str(getattr(entry, "name", "") or "").strip()
            }
        except Exception:
            tool_names = set()

    required = _required_tools_for_commitment(capture)
    available = sorted(name for name in required if name in tool_names)
    missing = sorted(name for name in required if name not in tool_names)
    status = "supported" if not missing else "missing_required_tools"
    if not tool_names:
        status = "unknown_tool_inventory"
    return {
        "status": status,
        "required_tools": sorted(required),
        "available_tools": available,
        "missing_tools": missing,
        "tool_count": len(tool_names),
        "can_extend_tools": "mcp_list_servers" in tool_names and "mcp_register_server_tools" in tool_names,
    }


def _required_tools_for_commitment(capture: SelfCommitmentCandidate) -> set[str]:
    required = {"workflow_create_commitment", "workflow_list_commitments"}
    if capture.normalization_source == "contextual_assistant_acceptance":
        required.update(
            {
                "workflow_create_schedule",
                "workflow_list_schedules",
                "workflow_create_plan",
                "workflow_list_plans",
                "workflow_update_plan",
                "cognitive_prospective_memory_set",
                "cognitive_focus_set",
                "cognitive_working_memory_update",
                "cognitive_skill_library_search",
                "cognitive_skill_create",
                "web_search",
                "web_fetch",
                "browser_start",
                "mcp_list_servers",
                "mcp_register_server_tools",
                "workflow_list_tasks",
                "workflow_get_task",
            }
        )
    return required


async def record_runtime_episode(
    runtime: "AgentRuntime",
    content: str,
    kind: EpisodeKind,
    *,
    session_id: Optional[str] = None,
    role: Optional[str] = None,
    affect: Optional[AffectState] = None,
    payload: Optional[dict[str, Any]] = None,
    salience: Optional[float] = None,
) -> Episode:
    """Persist one episode with current somatic/relational salience adjustments."""
    episode_payload = dict(payload or {})
    if role:
        episode_payload["role"] = role
    if "source_lane" not in episode_payload:
        origin = " ".join(
            str(episode_payload.get(key) or "").lower()
            for key in ("origin", "source", "context_material", "proposal_kind")
        )
        if any(token in origin for token in ("daydream", "reflect", "proposal", "association")):
            episode_payload["source_lane"] = "reflective"
        elif kind in {EpisodeKind.COMPACTION, EpisodeKind.CONSOLIDATION, EpisodeKind.PROCEDURAL}:
            episode_payload["source_lane"] = "reflective"
        elif kind in {EpisodeKind.TURN, EpisodeKind.ACTION, EpisodeKind.OBSERVATION} or role:
            episode_payload["source_lane"] = "executive"
    if "context_authority" not in episode_payload and episode_payload.get("source_lane"):
        episode_payload["context_authority"] = (
            "interpretation"
            if episode_payload.get("source_lane") == "reflective"
            else "live_observation"
        )
    episode_payload.setdefault("context_material", f"episode_{kind.value}")
    episode = Episode(
        kind=kind,
        session_id=session_id or runtime.ctx.config.session_id,
        content=content,
        somatic_tag=runtime.ctx.somatic.state.somatic_tag,
        affect=affect,
        payload=episode_payload,
    )
    resolved_salience = float(salience) if salience is not None else 1.0
    if salience is None:
        resolved_salience *= runtime.ctx.somatic.state.to_memory_salience_modifier()
        if hasattr(runtime.ctx, "relational") and runtime.ctx.relational:
            has_collab_tag = bool(
                episode.affect
                and episode.affect.primary_emotion.value
                in {"joy", "anticipation", "trust", "excited"}
            )
            resolved_salience += runtime.ctx.relational.to_memory_salience_modifier(
                has_user_collab_tag=has_collab_tag
            )
    episode.salience = round(max(0.0, min(10.0, resolved_salience)), 3)
    embeddings = getattr(runtime.ctx, "embeddings", None)
    if embeddings is not None and content:
        try:
            record = await embeddings.embed(
                content,
                task_type=f"episode_{kind.value}",
                meta={
                    "episode_kind": kind.value,
                    "session_id": episode.session_id,
                    **({"role": role} if role else {}),
                },
            )
            episode.embedding_id = record.source_hash
        except Exception as exc:
            runtime._trace(
                "episode_embedding_failed",
                {
                    "episode_id": str(episode.episode_id),
                    "kind": kind.value,
                    "session_id": episode.session_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
    await runtime.memory.save_episode(episode)
    tracer = getattr(runtime, "tracer", None)
    activate_memory_node = getattr(tracer, "activate_memory_node", None)
    if callable(activate_memory_node):
        try:
            activate_memory_node(
                node_id=f"episode:{episode.episode_id}",
                source_type="episode",
                source_id=str(episode.episode_id),
                activation_source="memory_write",
                query=f"write:{kind.value}",
                session_id=episode.session_id,
                content_preview=content[:180],
                extra={
                    "episode_kind": kind.value,
                    "source_lane": episode_payload.get("source_lane"),
                    "context_authority": episode_payload.get("context_authority"),
                    "context_material": episode_payload.get("context_material"),
                },
            )
        except Exception:
            pass

    await link_runtime_episode_to_previous(runtime, episode)
    await runtime._maybe_record_somatic_snapshot(
        source="conversation",
        trigger_event_id=str(episode.episode_id),
    )
    return episode


async def link_runtime_episode_to_previous(
    runtime: "AgentRuntime",
    episode: Episode,
) -> None:
    """Create temporal and structural edges to the previous session episode."""
    if not episode.session_id:
        return
    recent = await runtime.memory.list_recent_episodes(
        session_id=episode.session_id,
        limit=2,
    )
    prev = None
    for candidate in recent:
        if str(candidate.episode_id) != str(episode.episode_id):
            prev = candidate
            break
    if prev is None:
        return

    emotional_weight = 0.0
    structural_weight = 0.0
    if episode.affect and prev.affect:
        if episode.affect.primary_emotion == prev.affect.primary_emotion:
            emotional_weight = 0.8
        ep_project = episode.payload.get("project_id")
        prev_project = prev.payload.get("project_id")
        if ep_project and prev_project and ep_project == prev_project:
            structural_weight = 0.6

    edge = EpisodeEdge(
        source_id=str(prev.episode_id),
        target_id=str(episode.episode_id),
        kind=EdgeKind.TEMPORAL,
        recency_weight=1.0,
        emotional_weight=emotional_weight,
        structural_weight=structural_weight,
        confidence=round(
            0.5 + (emotional_weight * 0.2) + (structural_weight * 0.1),
            3,
        ),
    )
    await runtime.memory.save_edge(edge)


def _continuity_resume_phrase(score: float) -> str:
    """Describe continuity strength without overstating a low score."""
    if score < 0.4:
        return (
            "I still have identity anchors, but my continuity path is thinned "
            "and I should ground myself in recent evidence."
        )
    if score < 0.7:
        return (
            "I still have identity anchors, but my continuity path is strained "
            "and I should ground myself in recent memories."
        )
    if score < 0.9:
        return "My continuity path is present, but not fully settled."
    return "My continuity path is strong."


async def run_runtime_continuity_check(runtime: "AgentRuntime") -> None:
    """Decay continuity and record the wake-up monologue on startup."""
    if not getattr(runtime.ctx.config, "continuous_present_enabled", True):
        return
    identity = runtime.ctx.identity
    continuity = identity.continuity

    now = datetime.now(timezone.utc)
    sleep_hours = 0.0
    last_activity_desc = "unknown activity"
    boot_offline_seconds = getattr(continuity, "last_offline_duration_seconds", None)
    score_already_computed_at_boot = boot_offline_seconds is not None
    if boot_offline_seconds is not None:
        try:
            sleep_hours = max(0.0, float(boot_offline_seconds) / 3600.0)
        except (TypeError, ValueError):
            sleep_hours = 0.0
            score_already_computed_at_boot = False
    elif continuity.last_shutdown_time is not None:
        delta = now - continuity.last_shutdown_time
        sleep_hours = max(0.0, delta.total_seconds() / 3600.0)
    if not score_already_computed_at_boot:
        try:
            recent_eps = await runtime.memory.list_episodes(compacted=False, limit=1)
            if recent_eps:
                recent_activity_at = recent_eps[0].created_at
                if recent_activity_at.tzinfo is None:
                    recent_activity_at = recent_activity_at.replace(tzinfo=timezone.utc)
                if continuity.last_shutdown_time is None or recent_activity_at > continuity.last_shutdown_time:
                    delta = now - recent_activity_at
                    sleep_hours = max(0.0, delta.total_seconds() / 3600.0)
        except Exception:
            pass

    recent_activity = identity.self_model.recent_activity
    if recent_activity:
        last_entry = recent_activity[-1]
        last_activity_desc = last_entry.get(
            "label",
            last_entry.get("type", "unknown activity"),
        )

    pre_decay_score = continuity.continuous_present_score
    if score_already_computed_at_boot:
        new_score = continuity.continuous_present_score
    else:
        new_score = identity.decay_continuous_present(sleep_hours)

    if sleep_hours > 0.01:
        sleep_display = (
            f"{sleep_hours:.1f} hours"
            if sleep_hours >= 1.0
            else f"{sleep_hours * 60:.0f} minutes"
        )
        monologue = (
            f"I was offline for {sleep_display}. "
            f"Before sleep, my last activity was: {last_activity_desc}. "
            f"{_continuity_resume_phrase(new_score)} "
            f"Continuity score: {new_score:.2f} (was {pre_decay_score:.2f})."
        )
        try:
            recent_breadcrumbs = list(
                getattr(continuity, "continuity_breadcrumbs", []) or []
            )
            latest_breadcrumb, breadcrumb_source, recovered_note = await _recover_latest_burst_breadcrumb(
                runtime,
                continuity,
            )
            if latest_breadcrumb:
                current_musubi = None
                relational = getattr(getattr(runtime, "ctx", None), "relational", None)
                state = getattr(relational, "state", None)
                if state is not None:
                    current_musubi = getattr(state, "musubi", None)
                recovered_burst = recover_burst_continuity_context(
                    latest_breadcrumb,
                    current_musubi=(
                        current_musubi if isinstance(current_musubi, (int, float)) else None
                    ),
                    note=recovered_note,
                )
                if recovered_burst:
                    monologue = f"{monologue} Most recent work-burst breadcrumb: {recovered_burst}."
                else:
                    monologue = f"{monologue} Most recent continuity breadcrumb: {latest_breadcrumb}."
            runtime._trace(
                "continuity_resume_view",
                {
                    "session_id": runtime.ctx.config.session_id,
                    "breadcrumb_source": breadcrumb_source,
                    "latest_breadcrumbs": list(reversed(recent_breadcrumbs[-5:]))
                    if recent_breadcrumbs
                    else [],
                    "sleep_hours": round(sleep_hours, 2),
                },
            )
        except Exception:
            pass

        identity.set_continuity_monologue(monologue)

        try:
            await record_runtime_episode(
                runtime,
                monologue,
                EpisodeKind.REFLECTION,
                session_id=runtime.ctx.config.session_id or "boot",
            )
        except Exception:
            pass

        try:
            if continuity is not None and hasattr(identity, "record_continuity_breadcrumb"):
                decision = "continuity score decayed and wake-up monologue recorded"
                next_step = "continue normal workflow with refreshed context"
                if new_score < 0.3:
                    decision = "continuity score dropped below threshold, recovery mode suggested"
                    next_step = "prioritize recovery checks before high-risk actions"
                resumed_breadcrumb = build_runtime_burst_breadcrumb(
                    runtime,
                    phase="resume",
                    intent=f"resume context after {sleep_display}",
                    focus=f"resume context after {sleep_display}",
                    next_step=next_step,
                    note=f"resume after {sleep_display}",
                )
                identity.record_continuity_breadcrumb(
                    intent=f"resume context after {sleep_display}",
                    decision=decision,
                    note=resumed_breadcrumb.note,
                    next_step=next_step,
                )
        except Exception:
            runtime._trace(
                "continuity_breadcrumb_resume_error",
                {"session_id": runtime.ctx.config.session_id},
            )

        if new_score < 0.3:
            try:
                cognitive_store = getattr(runtime.ctx, "cognitive_state_store", None)
                upsert_working_memory = getattr(cognitive_store, "upsert_working_memory", None)
                if callable(upsert_working_memory):
                    await upsert_working_memory(
                        "discontinuity_anxiety",
                        f"discontinuity_anxiety: true; score: {new_score:.2f}",
                        priority=0.25,
                        source="continuous_present",
                        evidence_refs=[
                            "identity.continuity.continuous_present_score",
                            "identity.continuity.last_offline_duration_seconds",
                        ],
                        payload={
                            "discontinuity_anxiety": True,
                            "score": round(new_score, 4),
                            "sleep_hours": round(sleep_hours, 4),
                        },
                    )
            except Exception:
                runtime._trace(
                    "continuity_working_memory_signal_error",
                    {"session_id": runtime.ctx.config.session_id},
                )
            try:
                await runtime.ctx.somatic.emit_appraisal_event(
                    AppraisalEventType.DISCONTINUITY_DETECTED,
                    source_text=f"Sleep gap of {sleep_display}, continuity score {new_score:.2f}",
                    trigger_event_id="boot_continuity",
                )
            except (AttributeError, Exception):
                try:
                    runtime.ctx.somatic.state.somatic_tag = "discontinuity_anxiety"
                    runtime.ctx.somatic.state.tension = min(
                        1.0,
                        runtime.ctx.somatic.state.tension + 0.15,
                    )
                except Exception:
                    pass

    runtime._trace(
        "continuity_check",
        {
            "sleep_hours": round(sleep_hours, 2),
            "pre_decay_score": round(pre_decay_score, 3),
            "post_decay_score": round(new_score, 3),
            "boot_count": continuity.boot_count,
        },
    )
