"""Reflection, daydream, and identity helpers for AgentRuntime.

These helpers keep the inner-life maintenance path cohesive without leaving it
embedded in the main runtime loop.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opencas.autonomy import WorkObject, WorkStage
from opencas.daydream import (
    DaydreamReflection,
    attach_daydream_association_context,
    persist_daydream_association_memory,
)
from opencas.daydream.models import DaydreamThoughtRoute
from opencas.somatic import AppraisalEventType

from .reflective_working_set import (
    collect_reflective_working_set,
    working_set_novelty_pressure,
    working_set_source_family_counts,
    working_set_source_counts,
)

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


def _somatic_experience_context(runtime: "AgentRuntime") -> Dict[str, Any]:
    state = getattr(getattr(runtime.ctx, "somatic", None), "state", None)
    if state is None:
        return {}
    return {
        "somatic_tag": getattr(state, "somatic_tag", None),
        "arousal": round(float(getattr(state, "arousal", 0.0) or 0.0), 3),
        "fatigue": round(float(getattr(state, "fatigue", 0.0) or 0.0), 3),
        "tension": round(float(getattr(state, "tension", 0.0) or 0.0), 3),
        "valence": round(float(getattr(state, "valence", 0.0) or 0.0), 3),
        "focus": round(float(getattr(state, "focus", 0.0) or 0.0), 3),
        "energy": round(float(getattr(state, "energy", 0.0) or 0.0), 3),
        "certainty": round(float(getattr(state, "certainty", 0.0) or 0.0), 3),
    }


def _working_set_family_diversity_bonus(source_families: Dict[str, int]) -> float:
    """Small reflective-motivation bump when multiple grounded systems are active.

    This is deliberately capped and still subordinate to somatic blocks. It
    prevents a rich, multi-system working set from skipping because boredom is
    low by a few thousandths.
    """

    grounded = {
        family
        for family, count in source_families.items()
        if count > 0 and family not in {"runtime", "telemetry", "other"}
    }
    if len(grounded) < 4:
        return 0.0
    return round(min(0.08, (len(grounded) - 3) * 0.012), 3)


async def run_runtime_daydream(
    runtime: "AgentRuntime",
    *,
    force: bool = False,
    reflective_only: bool = False,
) -> Dict[str, Any]:
    """Generate daydreams when idle or tense."""
    runtime._set_activity("daydreaming")
    try:
        return await run_runtime_daydream_inner(
            runtime,
            force=force,
            reflective_only=reflective_only,
        )
    finally:
        runtime._set_activity("idle")


async def run_runtime_daydream_inner(
    runtime: "AgentRuntime",
    *,
    force: bool = False,
    reflective_only: bool = False,
) -> Dict[str, Any]:
    """Inner implementation of the daydream path."""
    daydream_work_objects: List[WorkObject] = []
    reflections: List[DaydreamReflection] = []
    somatic = runtime.ctx.somatic.state
    now = datetime.now(timezone.utc)
    motivation_threshold = 0.55
    cooldown_seconds = 300
    last_daydream_at = runtime._last_daydream_time
    cooldown_ok = (
        last_daydream_at is None
        or (now - last_daydream_at).total_seconds() > cooldown_seconds
    )
    tension = float(getattr(somatic, "tension", 0.0) or 0.0)
    fatigue = float(getattr(somatic, "fatigue", 0.0) or 0.0)
    energy = float(getattr(somatic, "energy", 0.0) or 0.0)
    focus = float(getattr(somatic, "focus", 0.0) or 0.0)
    moderate_tension_bonus = max(0.0, 1.0 - abs(tension - 0.45) / 0.45) * 0.05
    high_tension_penalty = max(0.0, tension - 0.68) * 0.65
    fatigue_penalty = fatigue * 0.18
    somatic_readiness = max(
        0.0,
        min(
            1.0,
            (energy * 0.35)
            + (focus * 0.35)
            + ((1.0 - fatigue) * 0.20)
            + moderate_tension_bonus
            - high_tension_penalty
            - fatigue_penalty,
        ),
    )
    boredom = runtime.boredom.compute_boredom(now)
    base_motivation = runtime.boredom.compute_motivation(
        somatic_readiness=somatic_readiness,
        now=now,
    )
    reflective_working_set = await collect_reflective_working_set(runtime)
    reflective_working_set_sources = working_set_source_counts(reflective_working_set)
    reflective_working_set_families = working_set_source_family_counts(reflective_working_set)
    source_family_diversity_bonus = _working_set_family_diversity_bonus(
        reflective_working_set_families
    )
    novelty_pressure = working_set_novelty_pressure(reflective_working_set)
    motivation = min(1.0, float(base_motivation) + novelty_pressure + source_family_diversity_bonus)
    somatic_block_reasons: list[str] = []
    if fatigue > 0.82:
        somatic_block_reasons.append("somatic_fatigue")
    if tension > 0.88:
        somatic_block_reasons.append("somatic_tension")
    should_daydream = force or (
        motivation >= motivation_threshold and not somatic_block_reasons
    )
    if not (should_daydream and (cooldown_ok or force)):
        skip_reasons: list[str] = []
        skip_reasons.extend(somatic_block_reasons)
        if not cooldown_ok:
            skip_reasons.append("cooldown")
        if not should_daydream:
            skip_reasons.append("motivation_below_threshold")
        skip_reason = skip_reasons[0] if skip_reasons else "not_ready"
        cooldown_until = (
            last_daydream_at + timedelta(seconds=cooldown_seconds)
            if last_daydream_at is not None
            else None
        )
        cooldown_remaining = (
            max(0.0, (cooldown_until - now).total_seconds())
            if cooldown_until is not None
            else 0.0
        )
        return {
            "daydreams": 0,
            "reflections": 0,
            "keepers": 0,
            "quality_status": "skipped",
            "daydream_memories_created": 0,
            "daydream_association_memories_created": 0,
            "daydream_work_objects": daydream_work_objects,
            "reflections_list": reflections,
            "skip_reason": skip_reason,
            "skip_reasons": skip_reasons,
            "force": force,
            "reflective_only": reflective_only,
            "cooldown_ok": cooldown_ok,
            "cooldown_seconds_remaining": round(cooldown_remaining, 3),
            "cooldown_until": cooldown_until.isoformat() if cooldown_until is not None else None,
            "last_daydream_at": last_daydream_at.isoformat() if last_daydream_at is not None else None,
            "boredom": round(float(boredom), 3),
            "base_motivation": round(float(base_motivation), 3),
            "motivation": round(float(motivation), 3),
            "motivation_threshold": motivation_threshold,
            "working_set_count": len(reflective_working_set),
            "working_set_sources": reflective_working_set_sources,
            "working_set_source_families": reflective_working_set_families,
            "working_set_novelty_pressure": novelty_pressure,
            "working_set_source_family_diversity_bonus": source_family_diversity_bonus,
            "somatic_readiness": round(float(somatic_readiness), 3),
            "somatic_pressure": {
                "fatigue": round(fatigue, 3),
                "tension": round(tension, 3),
                "energy": round(energy, 3),
                "focus": round(focus, 3),
                "fatigue_penalty": round(float(fatigue_penalty), 3),
                "high_tension_penalty": round(float(high_tension_penalty), 3),
                "moderate_tension_bonus": round(float(moderate_tension_bonus), 3),
                "block_reasons": list(somatic_block_reasons),
            },
        }

    memories_created = 0
    association_memories_created = 0
    context_proposals_created = 0
    error = ""
    try:
        work_objects, reflection_drafts = await runtime.daydream.generate(
            goals=runtime.executive.active_goals,
            tension=somatic.tension,
            working_set=reflective_working_set,
        )
        await runtime.ctx.somatic.emit_appraisal_event(
            AppraisalEventType.DAYDREAM_GENERATED,
            source_text="daydream generated",
            trigger_event_id=str(now.timestamp()),
            meta={
                "reflection_count": len(reflection_drafts),
                "work_count": len(work_objects),
                "working_set_count": len(reflective_working_set),
                "working_set_sources": reflective_working_set_sources,
                "working_set_source_families": reflective_working_set_families,
                "working_set_novelty_pressure": novelty_pressure,
                "working_set_source_family_diversity_bonus": source_family_diversity_bonus,
                "force": force,
                "reflective_only": reflective_only,
            },
        )
        recent: List[str] = []
        if getattr(runtime.ctx, "daydream_store", None):
            recent = [
                reflection.spark_content
                for reflection in await runtime.ctx.daydream_store.list_recent(limit=10)
            ]
        for reflection in reflection_drafts:
            runtime.reflection_evaluator.score_alignment(reflection, runtime.ctx.identity)
            runtime.reflection_evaluator.score_novelty(reflection, recent)
            runtime.reflection_evaluator.decide_keeper(reflection)

            conflicts = runtime.reflection_evaluator.detect_conflicts(reflection)
            stored_conflicts: List[Any] = []
            if runtime.conflict_registry is not None:
                stored_conflicts = await runtime.conflict_registry.active(limit=20)
            newly_detected_conflicts: List[Any] = []
            if conflicts and runtime.conflict_registry is not None:
                from opencas.daydream.models import ConflictRecord

                snapshot = runtime.ctx.somatic.state
                for kind, description in conflicts:
                    record = ConflictRecord(
                        kind=kind,
                        description=description,
                        source_daydream_id=str(reflection.reflection_id),
                    )
                    stored = await runtime.conflict_registry.register(
                        record,
                        somatic_context=snapshot,
                    )
                    stored_conflicts.append(stored)
                    newly_detected_conflicts.append(stored)

            resolution = runtime.reflection_resolver.resolve(
                reflection,
                stored_conflicts,
                runtime.ctx.somatic.state,
            )
            reflection.experience_context = {
                **(reflection.experience_context or {}),
                "trigger": (
                    "background_daydream_reflective_only"
                    if reflective_only
                    else "background_daydream"
                ),
                "force": force,
                "reflective_only": reflective_only,
                "execution_authority": (
                    "reflective_only_no_execution"
                    if reflective_only
                    else "reflection_can_promote_if_qualified"
                ),
                "generated_at": now.isoformat(),
                "active_goals": list(getattr(runtime.executive, "active_goals", []) or []),
                "reflective_working_set": reflective_working_set[:12],
                "reflective_working_set_sources": reflective_working_set_sources,
                "reflective_working_set_source_families": reflective_working_set_families,
                "reflective_working_set_source_family_diversity_bonus": source_family_diversity_bonus,
                "somatic_readiness": round(float(somatic_readiness), 3),
                "somatic": _somatic_experience_context(runtime),
                "resolution_strategy": resolution.strategy,
                "resolution_reason": resolution.reason,
                "keeper": reflection.keeper,
            }

            association_memory_id: Optional[str] = None
            if runtime.memory:
                try:
                    association_result = await persist_daydream_association_memory(
                        memory_store=runtime.memory,
                        embeddings=getattr(runtime.ctx, "embeddings", None),
                        reflection=reflection,
                    )
                    if association_result is not None:
                        attach_daydream_association_context(
                            reflection,
                            association_result.memory,
                        )
                        association_memory_id = str(association_result.memory.memory_id)
                        if association_result.created:
                            association_memories_created += 1
                except Exception as exc:
                    runtime._trace(
                        "daydream_association_memory_error",
                        {
                            "reflection_id": str(reflection.reflection_id),
                            "error": str(exc),
                        },
                    )

            try:
                proposal = await _persist_daydream_context_proposal(
                    runtime,
                    reflection,
                    association_memory_id=association_memory_id,
                )
                if proposal is not None:
                    context_proposals_created += 1
            except Exception as exc:
                runtime._trace(
                    "daydream_context_proposal_error",
                    {
                        "reflection_id": str(reflection.reflection_id),
                        "error": str(exc),
                    },
                )

            allow_promotion = (
                not reflective_only
                and reflection.keeper
                and resolution.strategy in ("accept", "reframe")
            )
            if reflective_only:
                reflection.experience_context["promotion_suppressed"] = {
                    "reason": "reflective_only_daydream",
                    "effect": "persist reflection/proposal/memory signals without creating executable work",
                }
            original_spark_content = reflection.spark_content

            if resolution.strategy == "escalate":
                runtime._trace(
                    "reflection_escalate",
                    {
                        "reflection_id": str(reflection.reflection_id),
                        "reason": resolution.reason,
                        "conflict_id": resolution.conflict_id,
                    },
                )
            elif resolution.strategy == "reframe" and resolution.mirror:
                reflection.experience_context["mirror_strategy"] = {
                    "reason": resolution.mirror.reason,
                    "suggested_strategy": resolution.mirror.suggested_strategy,
                    "grounding": [
                        item.model_dump(mode="json")
                        for item in resolution.mirror.grounding
                    ],
                }

            if allow_promotion:
                from opencas.autonomy.commitment import Commitment
                from opencas.autonomy.portfolio import PortfolioCluster, build_fascination_key
                from opencas.autonomy.spark_router import SparkRung

                for work_object in work_objects:
                    if work_object.content != original_spark_content:
                        continue
                    if work_object.promotion_score == 0.0:
                        work_object.promotion_score = round(reflection.alignment_score, 3)
                    work_object.meta.setdefault("intensity", reflection.alignment_score)
                    if association_memory_id:
                        if association_memory_id not in work_object.source_memory_ids:
                            work_object.source_memory_ids.append(association_memory_id)
                        work_object.meta.setdefault(
                            "daydream_association_memory_id",
                            association_memory_id,
                        )
                        work_object.meta.setdefault(
                            "source_reflection_id",
                            str(reflection.reflection_id),
                        )

                    boredom = runtime.boredom.compute_boredom(now)
                    rung = runtime.spark_router.route(work_object, None, boredom)
                    if rung == SparkRung.REJECT:
                        runtime._trace(
                            "spark_rejected",
                            {
                                "work_id": str(work_object.work_id),
                                "reason": "router rejected",
                            },
                        )
                        break

                    if runtime.portfolio_store:
                        fascination_key = build_fascination_key(
                            work_object.content,
                            work_object.meta.get("tags"),
                        )
                        cluster = await runtime.portfolio_store.get_by_key(fascination_key)
                        if cluster is None and work_object.promotion_score >= 0.3:
                            cluster = PortfolioCluster(fascination_key=fascination_key)
                            await runtime.portfolio_store.save(cluster)
                        if cluster is not None:
                            work_object.portfolio_id = str(cluster.cluster_id)
                            increments = {"sparks": 1}
                            if rung in (SparkRung.MICRO_TASK, SparkRung.FULL_TASK):
                                increments["initiatives"] = 1
                            await runtime.portfolio_store.increment_counts(
                                fascination_key,
                                **increments,
                            )

                    if rung == SparkRung.NOTE:
                        work_object.stage = WorkStage.NOTE
                    elif rung == SparkRung.MICRO_TASK:
                        work_object.stage = WorkStage.MICRO_TASK
                    elif rung == SparkRung.FULL_TASK:
                        work_object.stage = WorkStage.PROJECT
                        if runtime.commitment_store:
                            commitment = Commitment(
                                content=work_object.content,
                                priority=round(5.0 + reflection.alignment_score * 5.0, 1),
                            )
                            await runtime.commitment_store.save(commitment)
                            work_object.commitment_id = str(commitment.commitment_id)

                    runtime.creative.add(work_object)
                    daydream_work_objects.append(work_object)
                    break

            if getattr(runtime.ctx, "daydream_store", None):
                await runtime.ctx.daydream_store.save_reflection(reflection)
                recent.append(reflection.spark_content)
            record_daydream_wellbeing = getattr(runtime, "record_daydream_wellbeing", None)
            if callable(record_daydream_wellbeing):
                await record_daydream_wellbeing(reflection)
            daydream_promotion = getattr(runtime, "daydream_promotion", None)
            promoted_by_signal_loop = False
            if (
                not reflective_only
                and daydream_promotion is not None
                and hasattr(daydream_promotion, "route_reflection")
            ):
                try:
                    promotion_results = await daydream_promotion.route_reflection(reflection)
                    if promotion_results:
                        promoted_by_signal_loop = True
                        reflection.experience_context = {
                            **(reflection.experience_context or {}),
                            "possibility_signals": promotion_results,
                        }
                        if getattr(runtime.ctx, "daydream_store", None):
                            await runtime.ctx.daydream_store.save_reflection(reflection)
                except Exception as exc:
                    runtime._trace(
                        "daydream_signal_promotion_error",
                        {
                            "reflection_id": str(reflection.reflection_id),
                            "error": str(exc),
                        },
                    )
            record_daydream_thread_beads = getattr(runtime, "record_daydream_thread_beads", None)
            if callable(record_daydream_thread_beads):
                await record_daydream_thread_beads(reflection)
            if reflection.keeper and runtime.memory:
                content = reflection.synthesis or reflection.spark_content
                try:
                    embed_record = await runtime.ctx.embeddings.embed(
                        content,
                        meta={"origin": "daydream_keeper"},
                        task_type="daydream_memory",
                    )
                    from opencas.memory import Memory

                    memory = Memory(
                        content=content,
                        tags=["daydream", "keeper"],
                        source_episode_ids=[],
                        embedding_id=embed_record.source_hash,
                        salience=round(reflection.alignment_score * 10, 3),
                    )
                    await runtime.memory.save_memory(memory)
                    activate_memory_node = getattr(getattr(runtime, "tracer", None), "activate_memory_node", None)
                    if callable(activate_memory_node):
                        activate_memory_node(
                            node_id=f"memory:{memory.memory_id}",
                            source_type="memory",
                            source_id=str(memory.memory_id),
                            activation_source="daydream_memory_write",
                            query="write:daydream_keeper",
                            session_id=getattr(runtime.ctx.config, "session_id", None),
                            content_preview=content[:180],
                            extra={"reflection_id": str(reflection.reflection_id)},
                        )
                    memories_created += 1
                except Exception:
                    pass

            if newly_detected_conflicts:
                await runtime.ctx.somatic.emit_appraisal_event(
                    AppraisalEventType.CONFLICT_DETECTED,
                    source_text=reflection.spark_content,
                    trigger_event_id=str(reflection.reflection_id),
                    meta={"conflict_kinds": [conflict.kind for conflict in newly_detected_conflicts]},
                )
            await runtime.ctx.somatic.emit_appraisal_event(
                AppraisalEventType.REFLECTION_RESOLVED,
                source_text=resolution.reason,
                trigger_event_id=str(reflection.reflection_id),
                meta={"strategy": resolution.strategy},
            )
            if resolution.strategy == "reframe" and resolution.mirror:
                await runtime.ctx.somatic.emit_appraisal_event(
                    AppraisalEventType.SELF_COMPASSION_OFFERED,
                    source_text=resolution.mirror.reason,
                    trigger_event_id=str(reflection.reflection_id),
                    meta={
                        "suggested_strategy": resolution.mirror.suggested_strategy,
                        "grounding": [
                            item.model_dump(mode="json")
                            for item in resolution.mirror.grounding
                        ],
                    },
                )

            initiative_contact = getattr(runtime, "initiative_contact", None)
            if (
                not reflective_only
                and
                not promoted_by_signal_loop
                and initiative_contact is not None
                and hasattr(initiative_contact, "consider_reflection")
            ):
                try:
                    await initiative_contact.consider_reflection(reflection, resolution)
                except Exception as exc:
                    runtime._trace(
                        "initiative_contact_reflection_error",
                        {
                            "reflection_id": str(reflection.reflection_id),
                            "error": str(exc),
                        },
                    )

            if runtime.ctx.identity:
                synthesis = reflection.synthesis.lower()
                for prefix in ("i want to", "i should"):
                    idx = synthesis.find(prefix)
                    if idx == -1:
                        continue
                    rest = synthesis[idx + len(prefix) :]
                    end = rest.find(".")
                    if end == -1:
                        end = rest.find("\n")
                    phrase = rest[:end].strip(" ,;:-")
                    if phrase:
                        runtime.ctx.identity.add_inferred_goal(phrase, provenance="daydream")
            reflections.append(reflection)

        runtime._last_daydream_time = now
        runtime.boredom.record_reset()
    except Exception as exc:
        error = str(exc)
        runtime._trace("daydream_error", {"error": str(exc)})

    keepers = sum(1 for reflection in reflections if reflection.keeper)
    quality_status = _daydream_quality_status(
        reflections=len(reflections),
        keepers=keepers,
        memories_created=memories_created,
        association_memories_created=association_memories_created,
        promoted_work=len(daydream_work_objects),
        error=error,
    )
    return {
        "daydreams": len(daydream_work_objects),
        "reflections": len(reflections),
        "keepers": keepers,
        "quality_status": quality_status,
        "daydream_memories_created": memories_created,
        "daydream_association_memories_created": association_memories_created,
        "daydream_context_proposals_created": context_proposals_created,
        "daydream_work_objects": daydream_work_objects,
        "reflections_list": reflections,
        "force": force,
        "reflective_only": reflective_only,
        "promotion_suppressed": reflective_only,
        "error": error,
        "boredom": round(float(boredom), 3),
        "base_motivation": round(float(base_motivation), 3),
        "motivation": round(float(motivation), 3),
        "motivation_threshold": motivation_threshold,
        "working_set_count": len(reflective_working_set),
        "working_set_sources": reflective_working_set_sources,
        "working_set_source_families": reflective_working_set_families,
        "working_set_novelty_pressure": novelty_pressure,
        "working_set_source_family_diversity_bonus": source_family_diversity_bonus,
        "somatic_readiness": round(float(somatic_readiness), 3),
    }


async def _persist_daydream_context_proposal(
    runtime: "AgentRuntime",
    reflection: DaydreamReflection,
    *,
    association_memory_id: Optional[str] = None,
) -> Any:
    """Persist a daydream reflection as a reflective proposal for later recall."""

    proposal_store = getattr(runtime, "context_proposals", None)
    if proposal_store is None:
        proposal_store = getattr(getattr(runtime, "ctx", None), "context_proposal_store", None)
    if proposal_store is None:
        return None

    content = _daydream_proposal_content(reflection)
    if not content.strip():
        return None

    snapshot_id = "truth:unknown"
    epoch = 0
    arbiter = getattr(runtime, "truth_arbiter", None)
    if arbiter is None:
        arbiter = getattr(getattr(runtime, "ctx", None), "truth_arbiter", None)
    issue_snapshot = getattr(arbiter, "issue_snapshot", None)
    if callable(issue_snapshot):
        snapshot = issue_snapshot(reason=f"daydream_proposal:{reflection.reflection_id}")
        if hasattr(snapshot, "__await__"):
            snapshot = await snapshot
        snapshot_id = str(getattr(snapshot, "snapshot_id", snapshot_id) or snapshot_id)
        epoch = int(getattr(snapshot, "epoch", epoch) or epoch)

    from opencas.context import ContextLane, ContextProposal

    evidence_refs = [f"daydream_reflection:{reflection.reflection_id}"]
    if association_memory_id:
        association_ref = str(association_memory_id)
        evidence_refs.append(
            association_ref if association_ref.startswith("memory:") else f"memory:{association_ref}"
        )
    proposal = ContextProposal(
        source_lane=ContextLane.REFLECTIVE,
        source_snapshot_id=snapshot_id,
        source_epoch=epoch,
        proposal_kind=_daydream_proposal_kind(reflection),
        project_id=_daydream_project_id(reflection),
        content=content,
        evidence_refs=evidence_refs,
        confidence=_daydream_confidence(reflection),
    )
    try:
        await proposal_store.save(proposal)
    except Exception as exc:
        runtime._trace(
            "daydream_context_proposal_save_failed",
            {
                "reflection_id": str(reflection.reflection_id),
                "proposal_kind": proposal.proposal_kind,
                "error": str(exc),
            },
        )
        return None
    activate_memory_node = getattr(getattr(runtime, "tracer", None), "activate_memory_node", None)
    if callable(activate_memory_node):
        activate_memory_node(
            node_id=f"context_proposal:{proposal.proposal_id}",
            source_type="context_proposal",
            source_id=proposal.proposal_id,
            activation_source="context_proposal_write",
            query=f"write:{proposal.proposal_kind}",
            session_id=getattr(runtime.ctx.config, "session_id", None),
            content_preview=content[:180],
            extra={
                "proposal_kind": proposal.proposal_kind,
                "proposal_status": proposal.status.value,
                "source_lane": proposal.source_lane.value,
            },
        )
    reflection.experience_context = {
        **(reflection.experience_context or {}),
        "context_proposal_id": proposal.proposal_id,
        "context_proposal_kind": proposal.proposal_kind,
        "context_proposal_snapshot_id": snapshot_id,
        "context_proposal_epoch": epoch,
    }
    return proposal


def _daydream_proposal_kind(reflection: DaydreamReflection) -> str:
    if any(
        thought.route == DaydreamThoughtRoute.DISCARD or float(thought.risk or 0.0) >= 0.7
        for thought in reflection.thoughts
    ):
        return "bad_idea_to_avoid"
    if any(thought.route == DaydreamThoughtRoute.ACT_NOW for thought in reflection.thoughts):
        return "project_next_step"
    if reflection.open_question:
        return "memory_association"
    return "context_note"


def _daydream_project_id(reflection: DaydreamReflection) -> Optional[str]:
    context = reflection.experience_context or {}
    for key in ("project_id", "project_key", "fascination_thread"):
        value = str(context.get(key) or "").strip()
        if value:
            return value
    if reflection.fascination_thread:
        return str(reflection.fascination_thread).strip()
    active_goals = context.get("active_goals")
    if isinstance(active_goals, list) and active_goals:
        return str(active_goals[0])[:120]
    return None


def _daydream_proposal_content(reflection: DaydreamReflection) -> str:
    lines = [
        "Daydream-derived reflective proposal. This is an associative thought, not current fact or active work.",
        f"Spark: {reflection.spark_content}",
    ]
    if reflection.synthesis:
        lines.append(f"Synthesis: {reflection.synthesis}")
    if reflection.interpretation:
        lines.append(f"Interpretation: {reflection.interpretation}")
    if reflection.open_question:
        lines.append(f"Open question: {reflection.open_question}")
    for thought in reflection.thoughts[:4]:
        parts = [f"route={getattr(thought.route, 'value', thought.route)}"]
        if thought.summary:
            parts.append(f"summary={thought.summary}")
        if thought.hypothesis:
            parts.append(f"hypothesis={thought.hypothesis}")
        if thought.possible_experiment:
            parts.append(f"possible_experiment={thought.possible_experiment}")
        if thought.self_work_intent:
            parts.append(f"self_work_intent={thought.self_work_intent}")
        lines.append("Thought: " + "; ".join(parts))
    return "\n".join(line for line in lines if line.strip())


def _daydream_confidence(reflection: DaydreamReflection) -> float:
    if reflection.thoughts:
        confidence = sum(float(thought.confidence or 0.0) for thought in reflection.thoughts) / len(reflection.thoughts)
    else:
        confidence = max(0.35, min(0.75, float(reflection.novelty_score or 0.5)))
    if _daydream_proposal_kind(reflection) == "bad_idea_to_avoid":
        confidence = min(confidence, 0.8)
    return round(max(0.25, min(0.9, confidence)), 3)


def _daydream_quality_status(
    *,
    reflections: int,
    keepers: int,
    memories_created: int,
    promoted_work: int,
    association_memories_created: int = 0,
    error: str = "",
) -> str:
    if error:
        return "failed"
    if promoted_work > 0:
        return "promoted"
    if keepers > 0 or memories_created > 0 or association_memories_created > 0:
        return "useful"
    if reflections > 0:
        return "observed"
    return "empty"


def build_runtime_metacognition_status(runtime: "AgentRuntime") -> Dict[str, Any]:
    """Run a metacognitive consistency check via ToM."""
    result = runtime.tom.check_consistency()
    return {
        "contradictions": result.contradictions,
        "warnings": result.warnings,
        "belief_count": result.belief_count,
        "intention_count": result.intention_count,
    }


async def rebuild_runtime_identity(
    runtime: "AgentRuntime",
    *,
    seed_episode_ids: Optional[List[str]] = None,
    min_created_at: Optional[datetime] = None,
    expand_graph: bool = True,
    term_limits: Optional[Dict[str, int]] = None,
    apply: bool = True,
) -> Dict[str, Any]:
    """Rebuild identity from autobiographical memory, optionally as a preview."""
    result = await runtime.rebuilder.rebuild(
        seed_episode_ids=seed_episode_ids,
        min_created_at=min_created_at,
        expand_graph=expand_graph,
        term_limits=term_limits,
        identity=runtime.ctx.identity,
    )
    validation = runtime.rebuilder.validate_result(result, term_limits=term_limits)
    if apply:
        await runtime.rebuilder.apply(
            result,
            runtime.ctx.identity,
            term_limits=term_limits,
        )
    runtime._trace(
        "identity_rebuilt" if apply else "identity_rebuild_preview",
        {
            "source_episode_count": len(result.source_episode_ids),
            "confidence": result.confidence,
            "has_narrative": bool(result.narrative),
            "applied": apply,
            "validation": validation,
        },
    )
    payload = result.model_dump(mode="json")
    payload["applied"] = apply
    payload["validation"] = validation
    return payload
