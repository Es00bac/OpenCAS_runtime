from pathlib import Path

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime
from opencas.tools.models import ToolResult


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="wellbeing-tool-session",
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    try:
        yield runtime
    finally:
        await runtime._close_stores()


@pytest.mark.asyncio
async def test_wellbeing_query_returns_latest_state(runtime: AgentRuntime):
    runtime.ctx.somatic.state.fatigue = 0.82
    await runtime.run_wellbeing_maintenance()

    result = await runtime.tools.execute_async(
        "wellbeing_query",
        {
            "include_recent_events": True,
            "include_recommendations": True,
            "include_maintenance_outcomes": True,
            "limit": 3,
        },
    )

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert "overall_risk" in result.output
    assert result.metadata["latest_state"]["recovery_need"] >= 0.7
    assert result.metadata["event_count"] >= 1
    assert result.metadata["recommendation_count"] >= 1
    assert result.metadata["maintenance_outcome_count"] >= 1
    outcome = result.metadata["maintenance_outcomes"][0]
    assert outcome["meta"]["effect_basis"] in {"estimated", "record_only"}
    assert "risk_delta" in outcome["meta"]
