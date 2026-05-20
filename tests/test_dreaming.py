from pathlib import Path

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.dreaming import DreamMode, DreamStore, NightlyDreamingService
from opencas.proof_chain import ProofClaimType, ProofEvidenceKind
from opencas.runtime import AgentRuntime


@pytest.mark.asyncio
async def test_nightly_dreaming_records_light_medium_heavy_modes(tmp_path: Path) -> None:
    store = await DreamStore(tmp_path / "dreaming.db").connect()
    service = NightlyDreamingService(
        store=store,
        workspace_root=tmp_path / "workspace",
    )

    records = []
    for mode in (DreamMode.LIGHT, DreamMode.MEDIUM, DreamMode.HEAVY):
        record = await service.run(
            mode=mode,
            consolidation_result={
                "result_id": f"consolidation-{mode.value}",
                "dream_for_date": "2026-05-12",
                "required_nightly_dream": mode is DreamMode.MEDIUM,
                "scheduler_trigger": "required_nightly_dream" if mode is DreamMode.MEDIUM else "",
                "candidate_episodes": 4,
                "clusters_formed": 2,
                "memories_created": 1,
                "commitments_consolidated": 1,
            },
        )
        records.append(record)

    recent = await store.list_recent(limit=5)

    assert [record.mode for record in records] == [
        DreamMode.LIGHT,
        DreamMode.MEDIUM,
        DreamMode.HEAVY,
    ]
    assert len(recent) == 3
    assert recent[0].mode == DreamMode.HEAVY
    assert records[0].source == "nightly_consolidation"
    assert records[0].meta["dream_for_date"] == "2026-05-12"
    assert records[1].meta["required_nightly_dream"] is True
    assert "2026-05-12" in records[1].summary
    assert records[0].artifact_path
    assert Path(records[0].artifact_path).exists()
    assert records[1].insights
    assert records[2].narrative

    await store.close()


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="dreaming-runtime-session",
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    try:
        yield runtime
    finally:
        await runtime._close_stores()


def test_fresh_runtime_exposes_nightly_dreaming_components(runtime: AgentRuntime) -> None:
    assert runtime.dream_store is not None
    assert runtime.nightly_dreaming is not None


@pytest.mark.asyncio
async def test_runtime_runs_nightly_dream_from_fresh_bootstrap(runtime: AgentRuntime) -> None:
    result = await runtime.run_nightly_dream(
        mode=DreamMode.LIGHT,
        consolidation_result={"result_id": "fresh-bootstrap-consolidation"},
    )

    assert result["available"] is True
    assert result["mode"] == "light"
    assert result["dream_id"]
    assert result["required_nightly_dream"] is False

    recent = await runtime.dream_store.list_recent(limit=1)
    assert recent
    assert recent[0].consolidation_result_id == "fresh-bootstrap-consolidation"
    proof_claims = await runtime.proof_store.list_claims(
        claim_type=ProofClaimType.DREAM,
        limit=5,
    )
    assert proof_claims
    assert proof_claims[0].evidence_links[0].evidence_kind == ProofEvidenceKind.DREAM_RECORD
