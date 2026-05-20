from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from opencas.api.server import create_app
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime
from opencas.wellbeing import MaintenanceActionType, MaintenanceOutcome, WellbeingState


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
    assert payload["maintenance_outcome_count"] >= 1
    assert payload["maintenance_outcomes"][0]["meta"]["effect_basis"] in {
        "estimated",
        "record_only",
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        outcomes_response = await client.get("/api/wellbeing/outcomes")

    assert outcomes_response.status_code == 200
    outcomes_payload = outcomes_response.json()
    assert outcomes_payload["available"] is True
    assert outcomes_payload["count"] >= 1
    assert outcomes_payload["items"][0]["meta"]["effect_basis"] in {
        "estimated",
        "record_only",
    }
    assert outcomes_payload["outcomes"] == outcomes_payload["items"]

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        run_response = await client.post("/api/wellbeing/maintenance", json={})

    assert run_response.status_code == 200
    run_payload = run_response.json()
    assert run_payload["available"] is True
    assert run_payload["maintenance_outcomes_recorded"] >= 1


@pytest.mark.asyncio
async def test_wellbeing_snapshot_surfaces_repeated_record_only_deep_think_followup(
    runtime: AgentRuntime,
):
    await runtime.wellbeing_store.save_state(WellbeingState(coherence=0.48, drift_load=0.62))
    for _ in range(3):
        await runtime.wellbeing_store.save_maintenance_outcome(
            MaintenanceOutcome(
                action_type=MaintenanceActionType.DEEP_THINK.value,
                before_risk=0.4,
                after_risk=0.4,
                outcome="deep_think_seed_recorded",
                meta={
                    "effect_basis": "record_only",
                    "effect_direction": "unmeasured",
                    "followup_required": True,
                    "risk_delta": 0.0,
                },
            )
        )

    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/wellbeing/snapshot")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stuck_loop_count"] == 1
    assert payload["repeat_count"] == 3
    assert payload["next_followup_action"]["type"] == "self_modification_proposal"
    assert payload["next_followup_action"]["effect_remains_unmeasured"] is True
    assert payload["linked_followup_ids"]

    recommendations = await runtime.wellbeing_store.list_recommendations(status="proposed", limit=5)
    assert recommendations
    assert recommendations[0].meta["followup_kind"] == "repeated_record_only_maintenance"


@pytest.mark.asyncio
async def test_wellbeing_snapshot_surfaces_pending_drift_drain(runtime: AgentRuntime):
    await runtime.wellbeing_store.save_state(WellbeingState(coherence=0.62, drift_load=0.84))
    await runtime.wellbeing_store.save_maintenance_outcome(
        MaintenanceOutcome(
            action_type=MaintenanceActionType.RECOVER.value,
            before_risk=0.52,
            after_risk=0.48,
            outcome="background_throttle_recorded",
            meta={
                "effect_basis": "estimated",
                "effect_direction": "expected_lower",
                "risk_delta": -0.04,
                "drift_drain_status": "pending_validation",
            },
        )
    )

    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/wellbeing/snapshot")

    assert response.status_code == 200
    payload = response.json()
    assert payload["drift_drain_pending"] == 1
