"""End-to-end tests for daydream integration in AgentRuntime."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from opencas.autonomy import WorkObject, WorkStage
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.daydream import (
    DaydreamReflection,
    DaydreamThought,
    DaydreamThoughtKind,
    DaydreamThoughtRoute,
)
from opencas.daydream.association_memory import DAYDREAM_ASSOCIATION_TAG
from opencas.runtime import AgentRuntime


def _backdate_boredom(runtime: AgentRuntime, hours: float = 2.0) -> None:
    """Backdate boredom clocks so should_daydream() returns True."""
    now = datetime.now(timezone.utc)
    runtime.boredom._last_activity_at = now - timedelta(hours=hours)
    runtime.boredom._last_reset_at = now - timedelta(hours=hours)


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(state_dir=tmp_path)
    ctx = await BootstrapPipeline(config).run()
    rt = AgentRuntime(ctx)
    return rt


@pytest.mark.asyncio
async def test_run_cycle_daydream_keeper_gate(runtime: AgentRuntime) -> None:
    # Force somatic state to allow daydreaming
    runtime.ctx.somatic.state.fatigue = 0.0
    runtime.ctx.somatic.state.tension = 0.5

    # Mock generate to return controlled reflections
    reflection_keeper = DaydreamReflection(
        spark_content="keeper spark about growth and clarity",
        synthesis="Growth and agency drive me forward.",
        alignment_score=0.8,
        keeper=True,
    )
    reflection_reject = DaydreamReflection(
        spark_content="reject spark about nothing relevant",
        synthesis="Completely unrelated.",
        alignment_score=0.1,
    )
    work_keeper = WorkObject(
        content="keeper spark about growth and clarity", stage=WorkStage.SPARK
    )
    work_reject = WorkObject(
        content="reject spark about nothing relevant", stage=WorkStage.SPARK
    )

    _backdate_boredom(runtime)
    runtime.daydream.generate = AsyncMock(return_value=(
        [work_keeper, work_reject],
        [reflection_keeper, reflection_reject],
    ))

    result = await runtime.run_daydream()

    assert result["daydreams"] == 1
    assert result["reflections"] == 2
    assert result["keepers"] == 1

    # Keeper spark should be on the creative ladder (routed to PROJECT as full_task)
    ladder_contents = [w.content for w in runtime.creative.list_by_stage(WorkStage.PROJECT)]
    assert "keeper spark about growth and clarity" in ladder_contents
    assert "reject spark about nothing relevant" not in ladder_contents

    # Both reflections should be saved without injecting canned mirror text
    recent = await runtime.ctx.daydream_store.list_recent(limit=10)
    sparks = {r.spark_content for r in recent}
    assert any("keeper spark about growth and clarity" in s for s in sparks)
    assert any("reject spark about nothing relevant" in s for s in sparks)
    keeper_record = next(r for r in recent if "keeper spark about growth and clarity" in r.spark_content)
    assert keeper_record.experience_context["trigger"] == "background_daydream"
    assert keeper_record.experience_context["resolution_strategy"] in {"accept", "reframe"}
    assert keeper_record.experience_context["somatic"]["tension"] == 0.5


@pytest.mark.asyncio
async def test_run_cycle_conflict_detection(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.state.fatigue = 0.0
    runtime.ctx.somatic.state.tension = 0.5
    runtime.ctx.somatic.state.valence = 0.0
    runtime.ctx.somatic.state.arousal = 0.0

    reflection = DaydreamReflection(
        spark_content="I should study but I want to play",
        synthesis="I want to play.",
        alignment_score=0.6,
        novelty_score=0.5,
        keeper=True,
    )
    work = WorkObject(content="I should study but I want to play", stage=WorkStage.SPARK)

    _backdate_boredom(runtime)
    runtime.daydream.generate = AsyncMock(return_value=([work], [reflection]))

    await runtime.run_daydream()

    active = await runtime.ctx.conflict_store.list_active_conflicts()
    kinds = {c.kind for c in active}
    assert "obligation_vs_curiosity" in kinds


@pytest.mark.asyncio
async def test_run_cycle_inferred_goal_extraction(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.state.fatigue = 0.0
    runtime.ctx.somatic.state.tension = 0.5

    reflection = DaydreamReflection(
        spark_content="spark about growth",
        synthesis="I want to learn Japanese.",
    )
    work = WorkObject(content="spark about growth", stage=WorkStage.SPARK)

    _backdate_boredom(runtime)
    runtime.daydream.generate = AsyncMock(return_value=([work], [reflection]))

    await runtime.run_daydream()

    inferred = [
        goal.get("text") if isinstance(goal, dict) else goal
        for goal in runtime.ctx.identity.user_model.inferred_goals
    ]
    assert "learn japanese" in inferred


@pytest.mark.asyncio
async def test_run_cycle_cooldown_blocks_second_call(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.state.fatigue = 0.0
    runtime.ctx.somatic.state.tension = 0.5

    reflection = DaydreamReflection(
        spark_content="spark",
        synthesis="synth",
        alignment_score=0.6,
        novelty_score=0.5,
        keeper=True,
    )
    work = WorkObject(content="spark", stage=WorkStage.SPARK)

    _backdate_boredom(runtime)
    runtime.daydream.generate = AsyncMock(return_value=([work], [reflection]))

    result1 = await runtime.run_daydream()
    assert result1["reflections"] == 1

    result2 = await runtime.run_daydream()
    # Cooldown should block second daydream generation
    assert result2["reflections"] == 0
    assert result2["daydreams"] == 0
    assert result2["skip_reason"] == "cooldown"
    assert result2["cooldown_ok"] is False
    assert isinstance(result2["motivation"], float)
    assert result2["last_daydream_at"] is not None
    assert result2["cooldown_seconds_remaining"] > 0
    assert result2["cooldown_until"] is not None
    assert isinstance(result2["boredom"], float)


@pytest.mark.asyncio
async def test_run_daydream_reports_low_motivation_noop(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.state.energy = 0.3
    runtime.ctx.somatic.state.focus = 0.3
    runtime.daydream.generate = AsyncMock()

    result = await runtime.run_daydream()

    assert result["daydreams"] == 0
    assert result["reflections"] == 0
    assert result["skip_reason"] == "motivation_below_threshold"
    assert result["cooldown_ok"] is True
    assert result["motivation"] < result["motivation_threshold"]
    assert result["boredom"] >= 0.0
    runtime.daydream.generate.assert_not_called()


@pytest.mark.asyncio
async def test_run_daydream_reports_somatic_pressure_noop(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.state.energy = 0.9
    runtime.ctx.somatic.state.focus = 0.9
    runtime.ctx.somatic.state.fatigue = 0.91
    runtime.ctx.somatic.state.tension = 0.56
    _backdate_boredom(runtime)
    runtime.daydream.generate = AsyncMock()

    result = await runtime.run_daydream()

    assert result["daydreams"] == 0
    assert result["reflections"] == 0
    assert result["skip_reason"] == "somatic_fatigue"
    assert "somatic_fatigue" in result["skip_reasons"]
    assert result["somatic_pressure"]["fatigue"] == 0.91
    assert result["somatic_pressure"]["block_reasons"] == ["somatic_fatigue"]
    runtime.daydream.generate.assert_not_called()


@pytest.mark.asyncio
async def test_forced_reflective_daydream_persists_without_promotion(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.state.energy = 0.2
    runtime.ctx.somatic.state.focus = 0.2
    runtime.ctx.somatic.state.fatigue = 0.91
    runtime.ctx.somatic.state.tension = 0.9

    reflection = DaydreamReflection(
        spark_content="keeper spark about growth and clarity under overload",
        synthesis="Growth and clarity can remain useful later without becoming work while overloaded.",
        alignment_score=0.9,
        novelty_score=0.7,
        keeper=True,
    )
    work = WorkObject(content=reflection.spark_content, stage=WorkStage.SPARK)
    runtime.daydream.generate = AsyncMock(return_value=([work], [reflection]))

    result = await runtime.run_daydream(force=True, reflective_only=True)

    assert result["reflections"] == 1
    assert result["keepers"] == 1
    assert result["daydreams"] == 0
    assert result["force"] is True
    assert result["reflective_only"] is True
    assert result["promotion_suppressed"] is True
    runtime.daydream.generate.assert_awaited_once()

    promoted = [
        item
        for item in runtime.creative.list_by_stage(WorkStage.PROJECT)
        if item.content == reflection.spark_content
    ]
    assert promoted == []
    stored = await runtime.ctx.daydream_store.list_recent(limit=5)
    stored_reflection = next(
        item for item in stored if item.spark_content == reflection.spark_content
    )
    assert stored_reflection.experience_context["reflective_only"] is True
    assert stored_reflection.experience_context["execution_authority"] == (
        "reflective_only_no_execution"
    )
    assert stored_reflection.experience_context["promotion_suppressed"]["reason"] == (
        "reflective_only_daydream"
    )


@pytest.mark.asyncio
async def test_run_daydream_creates_keeper_memory(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.state.fatigue = 0.0
    runtime.ctx.somatic.state.tension = 0.5

    reflection = DaydreamReflection(
        spark_content="keeper spark about growth",
        synthesis="Growth and clarity drive me to remember what matters.",
        alignment_score=0.8,
        novelty_score=0.5,
        keeper=True,
    )
    work = WorkObject(
        content="keeper spark about growth", stage=WorkStage.SPARK
    )

    _backdate_boredom(runtime)
    runtime.daydream.generate = AsyncMock(return_value=([work], [reflection]))

    result = await runtime.run_daydream()
    assert result["keepers"] == 1

    memories = await runtime.ctx.memory.list_memories(limit=10)
    dm = [m for m in memories if "daydream" in m.tags and "keeper" in m.tags]
    assert len(dm) >= 1
    assert "growth and clarity" in dm[0].content.lower()


@pytest.mark.asyncio
async def test_run_daydream_creates_non_keeper_association_memory(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.state.fatigue = 0.0
    runtime.ctx.somatic.state.tension = 0.5
    runtime.ctx.identity.self_model.values = []
    runtime.ctx.identity.self_model.traits = []
    runtime.ctx.identity.self_model.current_goals = ["Japanese grammar practice"]
    runtime.ctx.identity.self_model.current_intention = ""

    reflection = DaydreamReflection(
        spark_content="A bad Chapter 3 idea: make the Void Node explain everything directly.",
        synthesis="This is probably too explicit, but it marks a path to avoid.",
        open_question="Can the Void Node stay technical without overexplaining?",
        alignment_score=0.12,
        novelty_score=0.82,
        keeper=False,
        thoughts=[
            DaydreamThought(
                kind=DaydreamThoughtKind.STORY_SEED,
                route=DaydreamThoughtRoute.DISCARD,
                summary="Overexplaining the Void Node would flatten Chapter 3's tension.",
                usefulness=0.35,
                novelty=0.74,
                confidence=0.62,
                risk=0.7,
            )
        ],
    )
    work = WorkObject(content=reflection.spark_content, stage=WorkStage.SPARK)

    _backdate_boredom(runtime)
    runtime.daydream.generate = AsyncMock(return_value=([work], [reflection]))

    result = await runtime.run_daydream()

    assert result["keepers"] == 0
    assert result["daydream_association_memories_created"] == 1
    assert result["quality_status"] == "useful"

    associations = await runtime.ctx.memory.list_memories_by_tag(
        DAYDREAM_ASSOCIATION_TAG,
        limit=10,
    )
    assert len(associations) == 1
    assert "Void Node" in associations[0].content
    assert "not a factual claim" in associations[0].content
    assert "daydream_non_keeper" in associations[0].tags
    assert "daydream_caution" in associations[0].tags

    stored = await runtime.ctx.daydream_store.list_recent(limit=5)
    stored_reflection = next(
        item
        for item in stored
        if item.spark_content == reflection.spark_content
    )
    assert stored_reflection.experience_context["association_memory_id"] == str(
        associations[0].memory_id
    )


@pytest.mark.asyncio
async def test_promoted_daydream_work_links_association_memory(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.state.fatigue = 0.0
    runtime.ctx.somatic.state.tension = 0.5
    runtime.ctx.identity.self_model.values = ["Writing Project", "Chapter 3", "sensory texture"]
    runtime.ctx.identity.self_model.traits = []
    runtime.ctx.identity.self_model.current_goals = ["Writing Project Chapter 3 sensory texture"]
    runtime.ctx.identity.self_model.current_intention = "Writing Project Chapter 3 sensory texture"

    reflection = DaydreamReflection(
        spark_content="keeper spark about Writing Project Chapter 3 sensory texture",
        synthesis="The Writing Project Chapter 3 texture should stay technical and tactile.",
        alignment_score=0.8,
        novelty_score=0.64,
        keeper=True,
    )
    work = WorkObject(content=reflection.spark_content, stage=WorkStage.SPARK)

    _backdate_boredom(runtime)
    runtime.daydream.generate = AsyncMock(return_value=([work], [reflection]))

    await runtime.run_daydream()

    associations = await runtime.ctx.memory.list_memories_by_tag(
        DAYDREAM_ASSOCIATION_TAG,
        limit=10,
    )
    assert len(associations) == 1
    association_id = str(associations[0].memory_id)

    promoted = [
        item
        for item in runtime.creative.list_by_stage(WorkStage.PROJECT)
        if item.content == reflection.spark_content
    ]
    assert len(promoted) == 1
    assert association_id in promoted[0].source_memory_ids
    assert promoted[0].meta["daydream_association_memory_id"] == association_id
