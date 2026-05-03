from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from opencas.api.server import create_app
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="wellbeing-api-session",
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    try:
        yield runtime
    finally:
        await runtime._close_stores()


@pytest.mark.asyncio
async def test_wellbeing_snapshot_endpoint_returns_grounded_state(runtime: AgentRuntime):
    runtime.ctx.somatic.state.fatigue = 0.8
    await runtime.run_wellbeing_maintenance()
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/wellbeing/snapshot")

    assert response.status_code == 200
    payload = response.json()
    assert "state" in payload
    assert payload["state"]["recovery_need"] >= 0.7
    assert payload["grounding_count"] >= 1
