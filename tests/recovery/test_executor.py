from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from opencas.recovery.executor import RecoveryExecutorAdapter
from opencas.recovery.models import RecoveryPlan, RecoveryStrategy


@dataclass
class FakeBAA:
    submitted: list[object] = field(default_factory=list)

    async def submit(self, task):
        self.submitted.append(task)
        return "future"


@dataclass
class FakeHarness:
    loops: list[dict] = field(default_factory=list)

    async def create_objective_loop(self, **kwargs):
        self.loops.append(kwargs)
        return SimpleNamespace(loop_id="loop-new")


@pytest.mark.asyncio
async def test_executor_submits_repair_task_for_deterministic_review() -> None:
    baa = FakeBAA()
    adapter = RecoveryExecutorAdapter(baa=baa)

    result = await adapter.execute(
        RecoveryPlan(
            candidate_id="task:old",
            strategy=RecoveryStrategy.DETERMINISTIC_REVIEW,
            objective="Review existing artifact.",
            evidence_refs=["task:old"],
        )
    )

    assert result.result == "submitted"
    assert result.created_task_id is not None
    assert baa.submitted[0].description == "Review existing artifact."
    assert baa.submitted[0].metadata["recovery_candidate_id"] == "task:old"


@pytest.mark.asyncio
async def test_executor_creates_objective_loop_for_continuity_resume() -> None:
    harness = FakeHarness()
    adapter = RecoveryExecutorAdapter(harness=harness)

    result = await adapter.execute(
        RecoveryPlan(
            candidate_id="loop:old",
            strategy=RecoveryStrategy.CREATE_CONTINUITY_THEN_RESUME,
            objective="Draft next scene.",
            evidence_refs=["loop:old", "continuity:abc"],
        )
    )

    assert result.result == "submitted"
    assert result.created_loop_id == "loop-new"
    assert harness.loops[0]["title"] == "Recovery: Draft next scene."
    assert harness.loops[0]["meta"]["recovery_candidate_id"] == "loop:old"


@pytest.mark.asyncio
async def test_executor_records_blocked_plan_without_submission() -> None:
    adapter = RecoveryExecutorAdapter()

    result = await adapter.execute(
        RecoveryPlan(
            candidate_id="task:old",
            strategy=RecoveryStrategy.LEAVE_BLOCKED_WITH_RECONSIDERATION,
            objective="OAuth token expired",
            evidence_refs=["task:old"],
        )
    )

    assert result.result == "blocked"
    assert result.created_task_id is None


@pytest.mark.asyncio
async def test_executor_confirms_scheduled_recurrence_without_submission() -> None:
    baa = FakeBAA()
    harness = FakeHarness()
    adapter = RecoveryExecutorAdapter(baa=baa, harness=harness)

    result = await adapter.execute(
        RecoveryPlan(
            candidate_id="schedule:daily-digest",
            strategy=RecoveryStrategy.CONFIRM_SCHEDULED_RECURRENCE,
            objective="Confirm recurring work remains scheduled.",
            evidence_refs=["schedule:daily-digest"],
        )
    )

    assert result.result == "scheduled"
    assert result.created_task_id is None
    assert result.created_loop_id is None
    assert result.next_reconsideration_at is not None
    assert result.strategy == "confirm_scheduled_recurrence"
    assert baa.submitted == []
    assert harness.loops == []
