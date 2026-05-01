"""Tests for tool-result affective examination integration."""

from pathlib import Path

import pytest
import pytest_asyncio

from opencas.autonomy.models import ActionRiskTier, ApprovalDecision, ApprovalLevel
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime
from opencas.tools.models import ToolResult


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    ctx = await BootstrapPipeline(
        BootstrapConfig(state_dir=tmp_path, session_id="tool-affect-test")
    ).run()
    rt = AgentRuntime(ctx)
    try:
        yield rt
    finally:
        await rt._close_stores()


@pytest.mark.asyncio
async def test_execute_tool_records_affective_pressure_from_actual_result(runtime):
    async def uncertain_tool(name, args):
        return ToolResult(
            success=True,
            output="warning: output is stale and uncertain; verify before relying on it",
            metadata={},
        )

    runtime.tools.register(
        "mock_uncertain",
        "Return uncertain evidence",
        uncertain_tool,
        ActionRiskTier.READONLY,
        {"type": "object", "properties": {}},
    )
    runtime.approval.evaluate = lambda req: ApprovalDecision(
        level=ApprovalLevel.CAN_DO_NOW,
        action_id=req.action_id,
        confidence=1.0,
        reasoning="mock",
        score=0.0,
    )

    result = await runtime.execute_tool("mock_uncertain", {})
    records = await runtime.ctx.affective_examinations.store.list_recent(limit=3)

    assert result["success"] is True
    assert result["metadata"]["affective_pressure"]["action_pressure"] == "verify"
    assert records[0].source_excerpt == (
        "warning: output is stale and uncertain; verify before relying on it"
    )
    assert records[0].source_id
    assert records[0].meta["tool_name"] == "mock_uncertain"
