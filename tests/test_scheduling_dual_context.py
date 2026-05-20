from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.scheduling import (
    ScheduleAction,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleService,
    ScheduleStore,
)


class _FakeBAA:
    def __init__(self) -> None:
        self.submitted = []

    async def submit(self, task):
        self.submitted.append(task)
        return None


class _FakeTruthArbiter:
    async def issue_snapshot(self, *, reason: str = ""):
        return SimpleNamespace(
            snapshot_id="truth:5:abc",
            epoch=5,
            source_hash="abc",
            reason=reason,
        )


@pytest_asyncio.fixture
async def schedule_store(tmp_path):
    store = ScheduleStore(tmp_path / "schedules.db")
    await store.connect()
    try:
        yield store
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_submit_baa_schedule_attaches_truth_metadata(schedule_store: ScheduleStore) -> None:
    baa = _FakeBAA()
    runtime = SimpleNamespace(baa=baa, truth_arbiter=_FakeTruthArbiter())
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 9, 8, 0, tzinfo=timezone.utc)

    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Return to writing project 4246",
        objective="Use accepted reflective notes while revising Chapter 3.",
        start_at=start,
        recurrence=ScheduleRecurrence.NONE,
        meta={"accepted_proposal_ids": ["proposal:chapter-3-note"]},
    )

    await service.process_due(now=start + timedelta(minutes=1))

    assert len(baa.submitted) == 1
    task = baa.submitted[0]
    assert task.meta["source"] == "schedule"
    assert task.meta["schedule_id"] == str(item.schedule_id)
    assert task.meta["origin_context_lane"] == "executive"
    assert task.meta["authority"] == "schedule_due"
    assert task.meta["context_truth_snapshot_id"] == "truth:5:abc"
    assert task.meta["context_truth_epoch"] == 5
    assert task.meta["source_snapshot_id"] == "truth:5:abc"
    assert task.meta["accepted_proposal_ids"] == ["proposal:chapter-3-note"]
