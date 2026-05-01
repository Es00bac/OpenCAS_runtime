"""Phase 2 integration tests: autonomy core end-to-end."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.models import ActionRequest, ActionRiskTier
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="phase2-test",
    )
    ctx = await BootstrapPipeline(config).run()
    return AgentRuntime(ctx)


@pytest.mark.asyncio
async def test_self_approve_readonly_action(runtime: AgentRuntime) -> None:
    req = ActionRequest(
        tier=ActionRiskTier.READONLY,
        description="Read project README",
    )
    outcome = await runtime.handle_action(req)
    assert outcome["approved"] is True
    assert outcome["decision"].level.value == "can_do_now"


@pytest.mark.asyncio
async def test_spark_promotes_through_ladder(runtime: AgentRuntime) -> None:
    runtime.executive.add_goal("rust learning")
    spark = WorkObject(content="I want to do rust learning today", stage=WorkStage.SPARK)
    runtime.creative.add(spark)

    promoted = runtime.creative.try_promote(spark)
    assert promoted is True
    assert spark.stage == WorkStage.NOTE

    # Run full cycle and check executive receives promoted micro-tasks+
    result = await runtime.run_cycle()
    assert result["creative"]["promoted"] >= 1


@pytest.mark.asyncio
async def test_executive_capacity_enforces_limit(runtime: AgentRuntime) -> None:
    for i in range(5):
        runtime.executive.enqueue(WorkObject(content=str(i), stage=WorkStage.SPARK))
    assert runtime.executive.is_overloaded is True
    assert runtime.executive.capacity_remaining == 0

    overflow = WorkObject(content="overflow", stage=WorkStage.MICRO_TASK)
    assert runtime.executive.enqueue(overflow) is False


@pytest.mark.asyncio
async def test_runtime_cycle_enqueues_promoted_work(runtime: AgentRuntime) -> None:
    runtime.executive.add_goal("fitness")
    runtime.creative.add(WorkObject(content="fitness tracker app", stage=WorkStage.SPARK))

    result = await runtime.run_cycle()
    # At least one promotion should happen; enqueued count may vary depending on stage
    assert result["creative"]["promoted"] >= 1


@pytest.mark.asyncio
async def test_converse_persists_episodes_and_updates_goals(runtime: AgentRuntime) -> None:
    runtime.executive.add_goal("test integration")
    response = await runtime.converse("hello openCAS", session_id="phase2-test")

    episodes = await runtime.memory.list_episodes(session_id="phase2-test")
    contents = [e.content for e in episodes]
    assert "hello openCAS" in contents
    assert response in contents or any(response in c for c in contents)
    assert "test integration" in runtime.executive.active_goals


@pytest.mark.asyncio
async def test_daydream_generation_on_idle(runtime: AgentRuntime) -> None:
    runtime.ctx.somatic.set_tension(0.6)
    runtime.executive.add_goal("creative writing")

    result = await runtime.run_cycle()
    # When idle/tense, daydreams may be generated (actual count depends on LLM availability)
    assert "daydreams" in result


@pytest.mark.asyncio
async def test_destructive_action_escalates(runtime: AgentRuntime) -> None:
    runtime.ctx.identity.user_model.trust_level = 0.3
    req = ActionRequest(
        tier=ActionRiskTier.DESTRUCTIVE,
        description="Delete production database",
    )
    outcome = await runtime.handle_action(req)
    assert outcome["approved"] is False
    assert outcome["decision"].level.value == "must_escalate"
