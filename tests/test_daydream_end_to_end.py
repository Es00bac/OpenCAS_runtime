"""End-to-end tests for daydream integration in AgentRuntime."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from opencas.autonomy import WorkObject, WorkStage
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.daydream import DaydreamReflection
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

    # Both reflections should be saved (reframe may prepend mirror affirmation)
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

    assert "learn japanese" in runtime.ctx.identity.user_model.inferred_goals


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
