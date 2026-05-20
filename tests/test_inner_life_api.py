from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from opencas.api.server import create_app
from opencas.api.routes.inner_life import build_inner_life_router
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.context import ContextLane, ContextProposal
from opencas.runtime import AgentRuntime
from fastapi import FastAPI


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="inner-life-api-session",
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    try:
        yield runtime
    finally:
        await runtime._close_stores()


@pytest.mark.asyncio
async def test_inner_life_summary_marks_configured_empty_stores(
    runtime: AgentRuntime,
) -> None:
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/inner-life/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["daydreams"]["available"] is True
    assert data["daydreams"]["configured_empty"] is True
    assert data["context_proposals"]["available"] is True
    assert data["context_proposals"]["configured_empty"] is True
    assert data["thread_registry"]["available"] is True
    assert data["thread_registry"]["configured_empty"] is True
    assert data["wellbeing"]["available"] is True
    assert data["wellbeing"]["maintenance_outcomes"] == 0
    assert data["dreaming"]["available"] is True
    assert data["dreaming"]["configured_empty"] is True
    assert data["proof_chain"]["available"] is True
    assert data["proof_chain"]["configured_empty"] is True


@pytest.mark.asyncio
async def test_inner_life_runtime_truth_packet_exposes_current_surface(
    runtime: AgentRuntime,
) -> None:
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/inner-life/runtime-truth")

    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["agent"]["profile_id"]
    assert data["workspace"]["state_dir"]
    assert data["capabilities"]["tool_count"] > 0
    assert len(data["capabilities"]["tools"]) <= 40
    assert data["inner_life"]["proof_chain"]["available"] is True
    assert data["inner_life"]["context_proposals"]["available"] is True

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        full_response = await client.get("/api/inner-life/runtime-truth?include_all=true")

    assert full_response.status_code == 200
    full_data = full_response.json()
    assert len(full_data["capabilities"]["tools"]) == full_data["capabilities"]["tool_count"]
    assert (
        len(full_data["capabilities"]["capabilities"])
        == full_data["capabilities"]["capability_count"]
    )
    assert full_data["tool_count"] == full_data["capabilities"]["tool_count"]
    assert len(full_data["tools"]) == full_data["tool_count"]
    assert full_data["capability_count"] == full_data["capabilities"]["capability_count"]
    assert len(full_data["accessible_capabilities"]) == full_data["capability_count"]
    tool_by_name = {tool["name"]: tool for tool in full_data["tools"]}
    assert "artifact_lookup" in tool_by_name
    assert tool_by_name["artifact_lookup"]["description"]


@pytest.mark.asyncio
async def test_inner_life_surfaces_context_proposal_status(
    runtime: AgentRuntime,
) -> None:
    proposal = ContextProposal(
        source_lane=ContextLane.REFLECTIVE,
        source_snapshot_id="truth:1:abc",
        source_epoch=1,
        proposal_kind="context_note",
        content="A remembered reflective proposal.",
    )
    await runtime.ctx.context_proposal_store.save(proposal)

    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/inner-life/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["context_proposals"]["configured_empty"] is False
    assert data["context_proposals"]["recent_proposals"] == 1
    assert data["context_proposals"]["status_counts"]["pending"] == 1


@pytest.mark.asyncio
async def test_inner_life_can_trigger_manual_nightly_dream_with_proof(
    runtime: AgentRuntime,
) -> None:
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/inner-life/nightly-dream",
            json={"mode": "medium", "reason": "manual proof check"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["mode"] == "medium"
    assert data["dream_id"]
    assert data["proof"]["dream_claims"] >= 1
    assert data["artifact_path"]
    assert data["side_effectful"] is True
    assert "dream_record" in data["side_effects"]


@pytest.mark.asyncio
async def test_inner_life_manual_nightly_dream_uses_idempotent_manual_bucket() -> None:
    seen_results = []

    async def _runner(*, mode, consolidation_result):
        seen_results.append(dict(consolidation_result))
        return {"available": True, "mode": mode, "dream_id": "dream-test"}

    runtime = type(
        "Runtime",
        (),
        {
            "run_nightly_dream": staticmethod(_runner),
            "_last_consolidation_result": None,
            "proof_store": None,
        },
    )()
    app = FastAPI()
    app.include_router(build_inner_life_router(runtime))
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/api/inner-life/nightly-dream", json={"mode": "light"})
        second = await client.post("/api/inner-life/nightly-dream", json={"mode": "light"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(seen_results) == 2
    assert seen_results[0]["result_id"] == seen_results[1]["result_id"]
    assert seen_results[0]["manual_truth_check"] is True
