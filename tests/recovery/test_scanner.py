from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.execution.models import AttemptOutcome, AttemptSalvagePacket, ExecutionStage, RepairTask, RetryMode
from opencas.recovery.models import RecoveryCandidateKind
from opencas.recovery.scanner import RecoveryScanner


@dataclass
class FakeCommitmentStore:
    active: list[Commitment] = field(default_factory=list)
    blocked: list[Commitment] = field(default_factory=list)

    async def list_active(self, limit: int = 50):
        return self.active[:limit]

    async def list_by_status(self, status: CommitmentStatus, limit: int = 50):
        if status == CommitmentStatus.BLOCKED:
            return self.blocked[:limit]
        return []


@dataclass
class FakeTaskStore:
    tasks: list[RepairTask] = field(default_factory=list)
    packets: dict[str, AttemptSalvagePacket] = field(default_factory=dict)

    async def list_all(self, limit: int = 100):
        return self.tasks[:limit]

    async def get_latest_salvage_packet(self, task_id):
        return self.packets.get(str(task_id))


@dataclass
class FakeHarnessStore:
    loops: list[SimpleNamespace] = field(default_factory=list)

    async def list_loops(self, limit: int = 100):
        return self.loops[:limit]


@dataclass
class FakeScheduleStore:
    schedules: list[SimpleNamespace] = field(default_factory=list)

    async def list_schedules(self, limit: int = 100):
        return self.schedules[:limit]


@pytest.mark.asyncio
async def test_scanner_finds_active_blocked_failed_and_paused_candidates() -> None:
    now = datetime.now(timezone.utc)
    failed_task = RepairTask(task_id=uuid4(), objective="Finish app", stage=ExecutionStage.FAILED)
    packet = AttemptSalvagePacket(
        packet_id=uuid4(),
        task_id=failed_task.task_id,
        attempt=2,
        objective="Finish app",
        divergence_signature="same_tool_loop",
        outcome=AttemptOutcome.FAILED,
        best_next_step="Review existing artifact and narrow repair.",
        recommended_mode=RetryMode.DETERMINISTIC_REVIEW,
        created_at=now,
    )
    loop = SimpleNamespace(
        loop_id="loop-1",
        title="Draft chapter 3",
        status="paused",
        updated_at=now,
        meta={"project_type": "creative_writing"},
    )

    scanner = RecoveryScanner(
        commitment_store=FakeCommitmentStore(
            active=[Commitment(content="Keep app build moving")],
            blocked=[Commitment(content="Return to writing project", status=CommitmentStatus.BLOCKED)],
        ),
        task_store=FakeTaskStore(tasks=[failed_task], packets={str(failed_task.task_id): packet}),
        harness_store=FakeHarnessStore(loops=[loop]),
    )

    candidates = await scanner.scan(limit_per_store=10)
    kinds = {candidate.kind for candidate in candidates}

    assert RecoveryCandidateKind.ACTIVE_COMMITMENT in kinds
    assert RecoveryCandidateKind.BLOCKED_COMMITMENT in kinds
    assert RecoveryCandidateKind.FAILED_TASK in kinds
    assert RecoveryCandidateKind.PAUSED_OBJECTIVE_LOOP in kinds
    failed = next(candidate for candidate in candidates if candidate.kind == RecoveryCandidateKind.FAILED_TASK)
    assert failed.payload["salvage"]["recommended_mode"] == "deterministic_review"


@pytest.mark.asyncio
async def test_scanner_finds_recurring_schedule_candidates() -> None:
    now = datetime.now(timezone.utc)
    schedule = SimpleNamespace(
        schedule_id="daily-checkin",
        title="Daily check-in",
        status="active",
        updated_at=now,
        next_run_at=now,
        recurrence="daily",
    )
    scanner = RecoveryScanner(schedule_store=FakeScheduleStore(schedules=[schedule]))

    candidates = await scanner.scan(limit_per_store=10)

    candidate = next(candidate for candidate in candidates if candidate.kind == RecoveryCandidateKind.RECURRING_SCHEDULE)
    assert candidate.candidate_id == "schedule:daily-checkin"
    assert candidate.evidence_refs == ["schedule:daily-checkin"]
    assert candidate.title == "Daily check-in"
    assert candidate.status == "active"
    assert candidate.payload["next_run_at"] == now
