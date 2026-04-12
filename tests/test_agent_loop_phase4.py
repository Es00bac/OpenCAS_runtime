"""Integration tests for Phase 4 Inner Life wiring in AgentRuntime."""

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapContext
from opencas.daydream.models import ConflictRecord, DaydreamReflection
from opencas.runtime.agent_loop import AgentRuntime
from opencas.somatic import AppraisalEventType


@pytest_asyncio.fixture
async def runtime(tmp_path_factory):
    from opencas.bootstrap.pipeline import BootstrapPipeline
    from opencas.bootstrap import BootstrapConfig
    config = BootstrapConfig(
        state_dir=tmp_path_factory.mktemp("state"),
        session_id="phase4-test",
    )
    ctx = await BootstrapPipeline(config).run()
    rt = AgentRuntime(ctx)
    yield rt
    await rt._close_stores()


@pytest.mark.asyncio
async def test_converse_emits_appraisal_event(runtime):
    runtime.ctx.somatic.set_valence(0.0)
    # Send a greeting that won't be refused
    _ = await runtime.converse("Hello, how are you?")
    recent = await runtime.ctx.somatic.store.list_recent(limit=5)
    sources = [r.source for r in recent]
    assert "user_input_received" in sources


@pytest.mark.asyncio
async def test_execute_tool_emits_appraisal_event(runtime):
    # Register a harmless mock tool so it can be found
    from opencas.autonomy.models import ActionRiskTier
    from opencas.tools import ToolRegistry

    class MockTool:
        async def execute(self, args):
            class Result:
                success = True
                output = "done"
                metadata = {}
            return Result()

    runtime.tools.register(
        "mock_hello",
        "Say hello",
        MockTool(),
        ActionRiskTier.READONLY,
        {"type": "object", "properties": {}},
    )
    # Bypass self-approval by pre-approving
    from opencas.autonomy.models import ApprovalDecision, ApprovalLevel
    runtime.approval.evaluate = lambda req: ApprovalDecision(
        level=ApprovalLevel.CAN_DO_NOW,
        action_id=req.action_id,
        confidence=1.0,
        reasoning="mock",
        score=0.0,
    )
    await runtime.execute_tool("mock_hello", {})
    recent = await runtime.ctx.somatic.store.list_recent(limit=5)
    sources = [r.source for r in recent]
    assert "tool_executed" in sources


@pytest.mark.asyncio
async def test_run_daydream_resolver_accept_allows_promotion(runtime):
    # Force boredom high enough to daydream
    from datetime import datetime, timezone, timedelta
    past = datetime.now(timezone.utc) - timedelta(hours=3)
    runtime.boredom._last_activity_at = past
    runtime.boredom._last_reset_at = past

    # Mock daydream generator
    from opencas.autonomy import WorkObject, WorkStage
    from uuid import uuid4
    wo = WorkObject(content="fitness app idea")
    reflection = DaydreamReflection(
        spark_content="fitness app idea",
        alignment_score=0.8,
        novelty_score=0.8,
        keeper=True,
    )

    async def _mock_generate(*args, **kwargs):
        return [wo], [reflection]

    runtime.daydream.generate = _mock_generate
    runtime.reflection_evaluator.score_alignment = lambda reflection, identity: setattr(reflection, "alignment_score", 0.8) or 0.8

    result = await runtime.run_daydream()
    assert result["daydreams"] >= 1
    assert wo in result["daydream_work_objects"]


@pytest.mark.asyncio
async def test_run_daydream_resolver_escalate_blocks_promotion(runtime):
    runtime.boredom.record_activity()
    from datetime import datetime, timezone, timedelta
    past = datetime.now(timezone.utc) - timedelta(hours=3)
    runtime.boredom._last_activity_at = past
    runtime.boredom._last_reset_at = past

    from opencas.autonomy import WorkObject
    from opencas.somatic.models import SomaticState
    wo = WorkObject(content="overwhelming task")
    reflection = DaydreamReflection(
        spark_content="overwhelming task",
        alignment_score=0.5,
        novelty_score=0.5,
        keeper=True,
    )
    async def _mock_generate(*args, **kwargs):
        return [wo], [reflection]

    runtime.daydream.generate = _mock_generate
    runtime.reflection_evaluator.score_alignment = lambda reflection, identity: setattr(reflection, "alignment_score", 0.5) or 0.5
    # Force high tension/fatigue for escalate
    runtime.ctx.somatic._state = SomaticState(tension=0.8, fatigue=0.8, arousal=0.8)
    # Pre-seed an acute conflict
    if runtime.conflict_registry:
        await runtime.conflict_registry.register(
            ConflictRecord(kind="energy_vs_ambition", description="too much", occurrence_count=3)
        )

    result = await runtime.run_daydream()
    assert result["daydreams"] == 0
    assert wo not in result["daydream_work_objects"]


@pytest.mark.asyncio
async def test_run_daydream_resolver_reframe(runtime):
    runtime.boredom.record_activity()
    from datetime import datetime, timezone, timedelta
    past = datetime.now(timezone.utc) - timedelta(hours=3)
    runtime.boredom._last_activity_at = past
    runtime.boredom._last_reset_at = past

    from opencas.autonomy import WorkObject
    from opencas.somatic.models import SomaticState
    wo = WorkObject(content="stressed idea")
    reflection = DaydreamReflection(
        spark_content="stressed idea",
        alignment_score=0.6,
        novelty_score=0.6,
        keeper=True,
    )
    async def _mock_generate(*args, **kwargs):
        return [wo], [reflection]

    runtime.daydream.generate = _mock_generate
    runtime.reflection_evaluator.score_alignment = lambda reflection, identity: setattr(reflection, "alignment_score", 0.6) or 0.6
    # Moderate tension triggers reframe
    runtime.ctx.somatic._state = SomaticState(tension=0.6, fatigue=0.3, arousal=0.5)

    result = await runtime.run_daydream()
    # Reframe allows promotion
    assert result["daydreams"] >= 1
    # Spark content should be prefixed with mirror affirmation
    assert "Pacing is wisdom" in reflection.spark_content or "Stay with the process" in reflection.spark_content
