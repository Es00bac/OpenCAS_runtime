"""Reflection, daydream, and identity helpers for AgentRuntime.

These helpers keep the inner-life maintenance path cohesive without leaving it
embedded in the main runtime loop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

from opencas.autonomy import WorkObject, WorkStage
from opencas.daydream import DaydreamReflection
from opencas.somatic import AppraisalEventType

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


async def run_runtime_daydream(runtime: "AgentRuntime") -> Dict[str, Any]:
    """Generate daydreams when idle or tense."""
    runtime._set_activity("daydreaming")
    try:
        return await run_runtime_daydream_inner(runtime)
    finally:
        runtime._set_activity("idle")


async def run_runtime_daydream_inner(runtime: "AgentRuntime") -> Dict[str, Any]:
    """Inner implementation of the daydream path."""
    daydream_work_objects: List[WorkObject] = []
    reflections: List[DaydreamReflection] = []
    somatic = runtime.ctx.somatic.state
    now = datetime.now(timezone.utc)
    cooldown_ok = (
        runtime._last_daydream_time is None
        or (now - runtime._last_daydream_time).total_seconds() > 300
    )
    somatic_readiness = (somatic.energy + somatic.focus) / 2.0
    if not (
        runtime.boredom.should_daydream(somatic_readiness=somatic_readiness)
        and cooldown_ok
    ):
        return {
            "daydreams": 0,
            "reflections": 0,
            "keepers": 0,
            "daydream_memories_created": 0,
            "daydream_work_objects": daydream_work_objects,
            "reflections_list": reflections,
        }

    memories_created = 0
    try:
        work_objects, reflection_drafts = await runtime.daydream.generate(
            goals=runtime.executive.active_goals,
            tension=somatic.tension,
        )
        await runtime.ctx.somatic.emit_appraisal_event(
            AppraisalEventType.DAYDREAM_GENERATED,
            source_text="daydream generated",
            trigger_event_id=str(now.timestamp()),
            meta={
                "reflection_count": len(reflection_drafts),
                "work_count": len(work_objects),
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

            allow_promotion = reflection.keeper and resolution.strategy in ("accept", "reframe")
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
                reflection.spark_content = (
                    f"{resolution.mirror.affirmation}\n\n{reflection.spark_content}"
                )

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
            if reflection.keeper and runtime.memory:
                content = reflection.synthesis or reflection.spark_content
                try:
                    embed_record = await runtime.ctx.embeddings.embed(
                        content,
                        meta={"origin": "daydream_keeper"},
                        task_type="daydream_memory",
                    )
                    from opencas.memory import Memory

                    await runtime.memory.save_memory(
                        Memory(
                            content=content,
                            tags=["daydream", "keeper"],
                            source_episode_ids=[],
                            embedding_id=embed_record.source_hash,
                            salience=round(reflection.alignment_score * 10, 3),
                        )
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
                    source_text=resolution.mirror.affirmation,
                    trigger_event_id=str(reflection.reflection_id),
                    meta={"suggested_strategy": resolution.mirror.suggested_strategy},
                )

            if reflection.keeper and runtime.ctx.identity:
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
                        runtime.ctx.identity.add_inferred_goal(phrase)
            reflections.append(reflection)

        runtime._last_daydream_time = now
        runtime.boredom.record_reset()
    except Exception as exc:
        runtime._trace("daydream_error", {"error": str(exc)})

    keepers = sum(1 for reflection in reflections if reflection.keeper)
    return {
        "daydreams": len(daydream_work_objects),
        "reflections": len(reflections),
        "keepers": keepers,
        "daydream_memories_created": memories_created,
        "daydream_work_objects": daydream_work_objects,
        "reflections_list": reflections,
    }


def build_runtime_metacognition_status(runtime: "AgentRuntime") -> Dict[str, Any]:
    """Run a metacognitive consistency check via ToM."""
    result = runtime.tom.check_consistency()
    return {
        "contradictions": result.contradictions,
        "warnings": result.warnings,
        "belief_count": result.belief_count,
        "intention_count": result.intention_count,
    }


async def rebuild_runtime_identity(runtime: "AgentRuntime") -> Dict[str, Any]:
    """Rebuild identity from autobiographical memory and apply it to self-model."""
    result = await runtime.rebuilder.rebuild()
    await runtime.rebuilder.apply(result, runtime.ctx.identity)
    runtime._trace(
        "identity_rebuilt",
        {
            "source_episode_count": len(result.source_episode_ids),
            "confidence": result.confidence,
            "has_narrative": bool(result.narrative),
        },
    )
    return result.model_dump(mode="json")
