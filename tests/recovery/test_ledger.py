from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from opencas.recovery.ledger import RecoveryLedger
from opencas.recovery.models import RecoveryDecision


@pytest.mark.asyncio
async def test_ledger_records_and_loads_decision(tmp_path) -> None:
    ledger = RecoveryLedger(tmp_path / "recovery.db")
    await ledger.connect()

    decision = RecoveryDecision(
        decision_id="decision-1",
        candidate_id="task:abc",
        classification="resume_now",
        strategy="deterministic_review",
        result="submitted",
        created_task_id="task:new",
        evidence_refs=["task:abc", "salvage:def"],
    )

    await ledger.record_decision(decision)
    loaded = await ledger.latest_decision("task:abc")

    assert loaded == decision
    await ledger.close()


@pytest.mark.asyncio
async def test_ledger_blocks_reconsideration_until_due(tmp_path) -> None:
    ledger = RecoveryLedger(tmp_path / "recovery.db")
    await ledger.connect()
    future = datetime.now(timezone.utc) + timedelta(hours=4)

    await ledger.record_decision(
        RecoveryDecision(
            decision_id="decision-2",
            candidate_id="task:abc",
            classification="unsafe_or_external_blocker",
            strategy="leave_blocked_with_reconsideration",
            result="blocked",
            evidence_refs=["task:abc"],
            next_reconsideration_at=future,
        )
    )

    assert await ledger.should_skip_candidate("task:abc") is True
    assert await ledger.should_skip_candidate("task:other") is False
    await ledger.close()


@pytest.mark.asyncio
async def test_ledger_allows_reconsideration_after_due_time(tmp_path) -> None:
    ledger = RecoveryLedger(tmp_path / "recovery.db")
    await ledger.connect()
    past = datetime.now(timezone.utc) - timedelta(minutes=1)

    await ledger.record_decision(
        RecoveryDecision(
            decision_id="decision-3",
            candidate_id="task:abc",
            classification="resume_now",
            strategy="deterministic_review",
            result="no_progress",
            evidence_refs=["task:abc"],
            next_reconsideration_at=past,
        )
    )

    assert await ledger.should_skip_candidate("task:abc") is False
    await ledger.close()
