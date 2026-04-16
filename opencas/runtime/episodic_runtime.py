"""Episode, continuity, and self-commitment helpers for AgentRuntime.

These helpers keep autobiographical state handling out of the main runtime
orchestrator. The runtime still owns when these hooks fire; this module owns
the local rules for how continuity, episode persistence, and promise capture
are applied.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING, List, Optional

from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.commitment_extraction import (
    SelfCommitmentCandidate,
    extract_self_commitments,
)
from opencas.memory import EdgeKind, Episode, EpisodeEdge, EpisodeKind
from opencas.somatic import AppraisalEventType
from opencas.tom import BeliefSubject

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


def extract_runtime_goal_directives(text: str) -> tuple[List[str], Optional[str], List[str]]:
    """Heuristically extract goals, intention, and drop-requests from user text."""
    text_lower = text.lower()
    goals: List[str] = []
    intention: Optional[str] = None
    drops: List[str] = []

    goal_patterns = [
        r"(?:your goal is|you should|i want you to|focus on|make it your goal to|prioritize)\s+(.*?)(?:[.]|$)",
    ]
    for pattern in goal_patterns:
        for match in re.finditer(pattern, text_lower):
            clause = match.group(1).strip()
            if clause:
                goals.append(clause)

    intention_patterns = [
        r"(?:current task is|work on|start on|intention is)\s+(.*?)(?:[.]|$)",
    ]
    for pattern in intention_patterns:
        match = re.search(pattern, text_lower)
        if match:
            intention = match.group(1).strip()
            break

    drop_phrases = ["done with that", "drop the goal", "forget about", "stop working on"]
    for phrase in drop_phrases:
        if phrase in text_lower:
            drops.append(phrase)

    return goals, intention, drops


def extract_runtime_self_commitments(text: str) -> List[SelfCommitmentCandidate]:
    """Extract normalized future-action self-commitments from assistant text."""
    return extract_self_commitments(text)


async def capture_runtime_self_commitments(
    runtime: "AgentRuntime",
    content: str,
    session_id: str,
) -> List[Commitment]:
    """Persist normalized self-commitments and mirror them into ToM/somatic state."""
    captures = extract_runtime_self_commitments(content)
    if not captures:
        return []

    pause_reason = runtime.executive.pause_reason() if runtime.executive else None
    status = CommitmentStatus.BLOCKED if pause_reason else CommitmentStatus.ACTIVE
    commitments: List[Commitment] = []

    for capture in captures:
        commitment: Optional[Commitment] = None
        if runtime.commitment_store:
            commitment = Commitment(
                content=capture.content[:220],
                status=status,
                tags=["self_commitment", "conversation"],
                meta={
                    "source": "assistant_response",
                    "session_id": session_id,
                    "trigger": capture.trigger,
                    "resume_policy": "auto_on_executive_recovery",
                    "source_sentence": capture.source_sentence,
                    "normalization_source": capture.normalization_source,
                    "capture_confidence": capture.confidence,
                    **(
                        {"blocked_reason": f"executive_{pause_reason}"}
                        if pause_reason
                        else {}
                    ),
                },
            )
            await runtime.commitment_store.save(commitment)
            commitments.append(commitment)
            runtime._trace(
                "self_commitment_captured",
                {
                    "commitment_id": str(commitment.commitment_id),
                    "status": status.value,
                    "normalization_source": capture.normalization_source,
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


async def record_runtime_episode(
    runtime: "AgentRuntime",
    content: str,
    kind: EpisodeKind,
    *,
    session_id: Optional[str] = None,
    role: Optional[str] = None,
) -> Episode:
    """Persist one episode with current somatic/relational salience adjustments."""
    episode = Episode(
        kind=kind,
        session_id=session_id or runtime.ctx.config.session_id,
        content=content,
        somatic_tag=runtime.ctx.somatic.state.somatic_tag,
        affect=None,
        payload={"role": role} if role else {},
    )
    salience = 1.0
    salience *= runtime.ctx.somatic.state.to_memory_salience_modifier()
    if hasattr(runtime.ctx, "relational") and runtime.ctx.relational:
        has_collab_tag = bool(
            episode.affect
            and episode.affect.primary_emotion.value
            in {"joy", "anticipation", "trust", "excited"}
        )
        salience += runtime.ctx.relational.to_memory_salience_modifier(
            has_user_collab_tag=has_collab_tag
        )
    episode.salience = round(max(0.0, min(10.0, salience)), 3)
    await runtime.memory.save_episode(episode)

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


async def run_runtime_continuity_check(runtime: "AgentRuntime") -> None:
    """Decay continuity and record the wake-up monologue on startup."""
    if not getattr(runtime.ctx.config, "continuous_present_enabled", True):
        return
    identity = runtime.ctx.identity
    continuity = identity.continuity

    now = datetime.now(timezone.utc)
    sleep_hours = 0.0
    last_activity_desc = "unknown activity"
    if continuity.last_shutdown_time is not None:
        delta = now - continuity.last_shutdown_time
        sleep_hours = max(0.0, delta.total_seconds() / 3600.0)
    elif continuity.boot_count > 1:
        try:
            recent_eps = await runtime.memory.list_episodes(compacted=False, limit=1)
            if recent_eps:
                delta = now - recent_eps[0].created_at
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
            f"I am still me. The thread is unbroken. "
            f"Continuity score: {new_score:.2f} (was {pre_decay_score:.2f})."
        )
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

        if new_score < 0.3:
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
