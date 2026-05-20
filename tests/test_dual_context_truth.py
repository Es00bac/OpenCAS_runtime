from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from opencas.context import TruthArbiter


class _Executive:
    def __init__(self) -> None:
        self.intention = "Return to project: Writing Project 4"
        self.intention_source = "active_work"
        self.active_goals: list[str] = []
        self.queue_size = 0

    def snapshot(self) -> dict:
        return {
            "intention": self.intention,
            "intention_source": self.intention_source,
            "active_goals": list(self.active_goals),
            "queue_size": self.queue_size,
            "capacity_remaining": 5,
        }


class _ScheduleService:
    def __init__(self) -> None:
        self.seconds_until = 10.0
        self.next_run_at = "2026-05-09T10:00:00+00:00"

    def temporal_agenda(self) -> dict:
        return {
            "now": f"2026-05-09T09:59:{int(10 - self.seconds_until):02d}+00:00",
            "horizon_end": "2026-05-10T09:59:00+00:00",
            "counts": {"active": 1, "due_now": 0},
            "upcoming": [
                {
                    "schedule_id": "schedule-1",
                    "title": "Scheduled work",
                    "next_run_at": self.next_run_at,
                    "seconds_until": self.seconds_until,
                    "status": "active",
                }
            ],
        }


@pytest.mark.asyncio
async def test_truth_snapshot_epoch_is_stable_until_live_truth_changes() -> None:
    executive = _Executive()
    baa = SimpleNamespace(_live_tasks={})
    runtime = SimpleNamespace(executive=executive, baa=baa)
    arbiter = TruthArbiter(runtime)

    first = await arbiter.issue_snapshot(reason="initial")
    second = await arbiter.issue_snapshot(reason="same")

    assert first.epoch == 1
    assert second.epoch == 1
    assert second.source_hash == first.source_hash

    executive.active_goals.append("Write Chapter 3")
    executive.queue_size = 1
    changed = await arbiter.issue_snapshot(reason="goal added")

    assert changed.epoch == 2
    assert changed.source_hash != first.source_hash


@pytest.mark.asyncio
async def test_truth_snapshot_ignores_volatile_schedule_clock_fields() -> None:
    executive = _Executive()
    executive.intention = None
    executive.intention_source = None
    schedule_service = _ScheduleService()
    runtime = SimpleNamespace(
        executive=executive,
        baa=SimpleNamespace(_live_tasks={}),
        schedule_service=schedule_service,
    )
    arbiter = TruthArbiter(runtime)

    first = await arbiter.issue_snapshot(reason="schedule initial")
    schedule_service.seconds_until = 9.0
    second = await arbiter.issue_snapshot(reason="clock tick")

    assert second.epoch == first.epoch
    assert second.source_hash == first.source_hash

    schedule_service.next_run_at = "2026-05-09T11:00:00+00:00"
    changed = await arbiter.issue_snapshot(reason="schedule changed")

    assert changed.epoch == first.epoch + 1
    assert changed.source_hash != first.source_hash


@pytest.mark.asyncio
async def test_truth_snapshot_ignores_runtime_activity_label_churn() -> None:
    executive = _Executive()
    executive.intention = None
    executive.intention_source = None
    runtime = SimpleNamespace(
        executive=executive,
        baa=SimpleNamespace(_live_tasks={}),
        _activity="idle",
    )
    arbiter = TruthArbiter(runtime)

    first = await arbiter.issue_snapshot(reason="idle")
    runtime._activity = "converse"
    second = await arbiter.issue_snapshot(reason="activity label changed")

    assert second.epoch == first.epoch
    assert second.source_hash == first.source_hash
    assert second.runtime_activity["activity"] == "converse"


@pytest.mark.asyncio
async def test_truth_snapshot_epoch_is_serialized_under_concurrency() -> None:
    executive = _Executive()
    executive.intention = None
    executive.intention_source = None
    runtime = SimpleNamespace(executive=executive, baa=SimpleNamespace(_live_tasks={}))
    arbiter = TruthArbiter(runtime)

    snapshots = await asyncio.gather(
        *(arbiter.issue_snapshot(reason=f"concurrent-{index}") for index in range(12))
    )

    assert {snapshot.epoch for snapshot in snapshots} == {1}
    assert len({snapshot.snapshot_id for snapshot in snapshots}) == 1


@pytest.mark.asyncio
async def test_truth_snapshot_normalizes_stale_active_work_intention() -> None:
    runtime = SimpleNamespace(executive=_Executive(), baa=SimpleNamespace(_live_tasks={}))
    arbiter = TruthArbiter(runtime)

    snapshot = await arbiter.issue_snapshot(reason="stale check")

    assert snapshot.executive["intention"] is None
    assert snapshot.executive["intention_source"] == "stale_active_work"


@pytest.mark.asyncio
async def test_truth_snapshot_tracks_baa_live_task_ids() -> None:
    executive = _Executive()
    executive.intention = None
    executive.intention_source = None
    baa = SimpleNamespace(_live_tasks={})
    runtime = SimpleNamespace(executive=executive, baa=baa)
    arbiter = TruthArbiter(runtime)

    first = await arbiter.issue_snapshot(reason="initial")
    baa._live_tasks["task-1"] = SimpleNamespace(task_id="task-1")
    changed = await arbiter.issue_snapshot(reason="task started")

    assert changed.epoch == first.epoch + 1
    assert changed.baa["live_task_ids"] == ["task-1"]
