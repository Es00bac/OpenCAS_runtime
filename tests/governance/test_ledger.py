"""Tests for the governance / approval ledger subsystem."""

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from opencas.autonomy.models import ActionRequest, ActionRiskTier, ApprovalDecision, ApprovalLevel
from opencas.governance import ApprovalLedger, ApprovalLedgerStore


@pytest_asyncio.fixture
async def ledger(tmp_path):
    store = ApprovalLedgerStore(tmp_path / "governance.db")
    await store.connect()
    yield ApprovalLedger(store)
    await store.close()


@pytest.mark.asyncio
async def test_ledger_persists_entry(ledger: ApprovalLedger) -> None:
    request = ActionRequest(
        tier=ActionRiskTier.READONLY,
        description="read file",
        tool_name="fs_read_file",
    )
    decision = ApprovalDecision(
        level=ApprovalLevel.CAN_DO_NOW,
        action_id=request.action_id,
        confidence=0.9,
        reasoning="safe",
        score=0.1,
    )
    entry = await ledger.record(decision, request, score=0.1)
    assert entry.action_id == request.action_id
    assert entry.level == "can_do_now"
    assert entry.tier == ActionRiskTier.READONLY

    fetched = await ledger.store.get(str(entry.entry_id))
    assert fetched is not None
    assert fetched.reasoning == "safe"


@pytest.mark.asyncio
async def test_ledger_list_by_action(ledger: ApprovalLedger) -> None:
    request = ActionRequest(
        tier=ActionRiskTier.SHELL_LOCAL,
        description="run shell",
    )
    decision = ApprovalDecision(
        level=ApprovalLevel.CAN_DO_WITH_CAUTION,
        action_id=request.action_id,
        confidence=0.7,
        reasoning="caution",
        score=0.3,
    )
    await ledger.record(decision, request, score=0.3)
    await ledger.record(decision, request, score=0.3)

    items = await ledger.store.list_by_action(str(request.action_id))
    assert len(items) == 2


@pytest.mark.asyncio
async def test_ledger_query_stats(ledger: ApprovalLedger) -> None:
    request = ActionRequest(
        tier=ActionRiskTier.DESTRUCTIVE,
        description="danger",
    )
    decision = ApprovalDecision(
        level=ApprovalLevel.MUST_ESCALATE,
        action_id=request.action_id,
        confidence=0.5,
        reasoning="dangerous",
        score=0.9,
    )
    await ledger.record(decision, request, score=0.9)

    stats = await ledger.query_stats(window_days=7)
    assert stats["window_days"] == 7
    assert len(stats["breakdown"]) >= 1
    row = stats["breakdown"][0]
    assert row["tier"] == "destructive"
    assert row["level"] == "must_escalate"
    assert row["count"] == 1


@pytest.mark.asyncio
async def test_self_approval_records_to_ledger() -> None:
    from opencas.identity import IdentityManager, IdentityStore
    from opencas.autonomy.self_approval import SelfApprovalLadder

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        identity_store = IdentityStore(tmp_path / "identity")
        identity = IdentityManager(identity_store)
        identity.load()

        store = ApprovalLedgerStore(tmp_path / "governance.db")
        await store.connect()
        ledger = ApprovalLedger(store)

        ladder = SelfApprovalLadder(identity=identity, ledger=ledger)
        request = ActionRequest(
            tier=ActionRiskTier.READONLY,
            description="read",
        )
        decision = ladder.evaluate(request)
        await ladder.maybe_record(decision, request, decision.score)

        items = await store.list_recent()
        assert len(items) >= 1
        assert items[0].action_id == request.action_id

        await store.close()
