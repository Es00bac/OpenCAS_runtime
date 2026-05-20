from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from opencas.api.server import create_app
from opencas.autonomy.commitment import Commitment
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.execution.models import ExecutionReceipt, RepairTask
from opencas.proof_chain import (
    ProofChainService,
    ProofClaimStatus,
    ProofClaimType,
    ProofEvidenceKind,
    ProofStore,
)
from opencas.runtime import AgentRuntime
from opencas.scheduling import (
    ScheduleAction,
    ScheduleKind,
    ScheduleRun,
    ScheduleRunStatus,
    ScheduleStatus,
)


@pytest.mark.asyncio
async def test_proof_chain_links_claims_to_evidence(tmp_path: Path) -> None:
    store = await ProofStore(tmp_path / "proof_chain.db").connect()
    service = ProofChainService(store)

    claim = await service.record_claim(
        claim_type=ProofClaimType.ACTION,
        claim="The build completed successfully.",
        subject="kPony build",
    )
    linked = await service.link_evidence(
        claim.claim_id,
        evidence_kind=ProofEvidenceKind.RECEIPT,
        evidence_id="receipt:build-123",
        summary="pytest and cmake passed",
        supports_claim=True,
    )

    recent = await store.list_claims(limit=5)

    assert linked.status == ProofClaimStatus.VERIFIED
    assert recent[0].claim_id == claim.claim_id
    assert recent[0].evidence_links[0].evidence_id == "receipt:build-123"
    await store.close()


@pytest_asyncio.fixture
async def runtime(tmp_path: Path):
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="proof-chain-runtime-session",
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    try:
        yield runtime
    finally:
        await runtime._close_stores()


def test_fresh_runtime_exposes_proof_chain(runtime: AgentRuntime) -> None:
    assert runtime.proof_store is not None
    assert runtime.proof_chain is not None


@pytest.mark.asyncio
async def test_self_commitment_capture_records_verifiable_promise_claim(
    runtime: AgentRuntime,
) -> None:
    commitments = await runtime._capture_self_commitments(
        "The next step is run the build and report back with proof. I'll come back to this after I rest.",
        "proof-chain-runtime-session",
    )

    assert commitments
    claims = await runtime.proof_store.list_claims(
        claim_type=ProofClaimType.PROMISE,
        limit=5,
    )
    assert claims
    assert claims[0].subject == "self_commitment"
    assert claims[0].status == ProofClaimStatus.EVIDENCE_LINKED
    assert str(commitments[0].commitment_id) in claims[0].evidence_links[0].evidence_id
    saved = await runtime.commitment_store.get(str(commitments[0].commitment_id))
    assert saved is not None
    assert saved.meta["claim_id"] == str(claims[0].claim_id)


@pytest.mark.asyncio
async def test_proof_api_exposes_unverified_claims(runtime: AgentRuntime) -> None:
    await runtime.proof_chain.record_claim(
        claim_type=ProofClaimType.CAPABILITY,
        claim="The agent can inspect a tool capability.",
        subject="capability_grounding",
    )
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        summary = await client.get("/api/proof/summary")
        claims = await client.get(
            "/api/proof/claims",
            params={"status": ProofClaimStatus.UNVERIFIED.value},
        )

    assert summary.status_code == 200
    assert claims.status_code == 200
    assert summary.json()["unverified_count"] >= 1
    assert claims.json()["claims"][0]["status"] == ProofClaimStatus.UNVERIFIED.value


@pytest.mark.asyncio
async def test_proof_summary_counts_actual_evidence_links(runtime: AgentRuntime) -> None:
    verified = await runtime.proof_chain.record_claim(
        claim_type=ProofClaimType.ACTION,
        claim="A receipt-backed action completed.",
        subject="proof_summary",
    )
    await runtime.proof_chain.link_evidence(
        verified.claim_id,
        evidence_kind=ProofEvidenceKind.RECEIPT,
        evidence_id="receipt:proof-summary-1",
        summary="Receipt exists for the completed action.",
    )
    await runtime.proof_chain.record_claim(
        claim_type=ProofClaimType.CAPABILITY,
        claim="A currently unsupported claim should remain unverified.",
        subject="proof_summary",
    )
    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        summary = await client.get("/api/proof/summary")

    data = summary.json()
    assert data["evidence_linked_count"] >= 1
    assert data["claims_with_evidence_count"] >= 1
    assert data["verified_with_evidence_count"] >= 1
    assert data["verified_without_evidence_count"] == 0


@pytest.mark.asyncio
async def test_promise_lookup_api_returns_joined_commitment_schedule_task_receipt(
    runtime: AgentRuntime,
) -> None:
    claim = await runtime.proof_chain.record_claim(
        claim_type=ProofClaimType.PROMISE,
        claim="I will verify the promise-to-proof spine.",
        subject="self_commitment",
    )
    commitment = Commitment(
        content="Verify the promise-to-proof spine.",
        meta={"source": "assistant_response", "claim_id": str(claim.claim_id)},
    )
    await runtime.commitment_store.save(commitment)
    await runtime.proof_chain.link_evidence(
        claim.claim_id,
        evidence_kind=ProofEvidenceKind.COMMITMENT,
        evidence_id=f"commitment:{commitment.commitment_id}",
        summary="Promise commitment captured.",
    )

    start = commitment.created_at
    schedule = await runtime.schedule_service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Verify promise spine",
        objective="Verify the promise-to-proof spine.",
        start_at=start,
        status=ScheduleStatus.ACTIVE,
        commitment_id=str(commitment.commitment_id),
        meta={"claim_id": str(claim.claim_id)},
    )
    task = RepairTask(
        objective="Verify the promise-to-proof spine.",
        commitment_id=str(commitment.commitment_id),
        meta={"claim_id": str(claim.claim_id), "schedule_id": str(schedule.schedule_id)},
    )
    await runtime.baa.store.save(task)
    await runtime.ctx.schedule_store.record_run(
        ScheduleRun(
            schedule_id=schedule.schedule_id,
            scheduled_for=start,
            status=ScheduleRunStatus.SUBMITTED,
            task_id=str(task.task_id),
            meta={"claim_id": str(claim.claim_id)},
        )
    )
    receipt = await runtime.ctx.receipt_store.save_direct(
        ExecutionReceipt(
            task_id=task.task_id,
            objective=task.objective,
            completed_at=start,
            success=True,
            output="Promise spine verified.",
            meta={"claim_id": str(claim.claim_id)},
        )
    )
    await runtime.proof_chain.link_evidence(
        claim.claim_id,
        evidence_kind=ProofEvidenceKind.RECEIPT,
        evidence_id=f"receipt:{receipt.receipt_id}",
        summary="Scheduled task receipt completed.",
    )

    app = create_app(runtime)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/api/proof-chain/promise-lookup",
            params={"since": commitment.created_at.isoformat()},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["available"] is True
    assert data["count"] == 1
    record = data["items"][0]
    assert record["claim"]["claim_id"] == str(claim.claim_id)
    assert record["commitment"]["commitment_id"] == str(commitment.commitment_id)
    assert record["schedules"][0]["schedule_id"] == str(schedule.schedule_id)
    assert record["tasks"][0]["task_id"] == str(task.task_id)
    assert record["receipts"][0]["receipt_id"] == str(receipt.receipt_id)


@pytest.mark.asyncio
async def test_promise_lookup_tool_returns_joined_records(runtime: AgentRuntime) -> None:
    claim = await runtime.proof_chain.record_claim(
        claim_type=ProofClaimType.PROMISE,
        claim="I will expose promise lookup as a tool.",
        subject="self_commitment",
    )
    commitment = Commitment(
        content="Expose promise lookup as a tool.",
        meta={"source": "assistant_response", "claim_id": str(claim.claim_id)},
    )
    await runtime.commitment_store.save(commitment)

    tool = runtime.tools.get("proof_chain_promise_lookup")
    assert tool is not None

    result = await tool.adapter(
        "proof_chain_promise_lookup",
        {"since": commitment.created_at.isoformat()},
    )

    assert result.success is True
    payload = result.output
    assert str(claim.claim_id) in payload
    assert str(commitment.commitment_id) in payload
