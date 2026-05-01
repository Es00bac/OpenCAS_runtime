"""Tests for scheduled tasks, events, and recurrence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opencas.api.routes.schedule import build_schedule_router
from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.commitment_store import CommitmentStore
from opencas.scheduling import (
    ScheduleAction,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleRun,
    ScheduleRunStatus,
    ScheduleService,
    ScheduleStore,
)


class FakeBAA:
    def __init__(self) -> None:
        self.submitted = []

    async def submit(self, task):
        self.submitted.append(task)
        return None


@pytest_asyncio.fixture
async def schedule_store(tmp_path):
    store = ScheduleStore(tmp_path / "schedules.db")
    await store.connect()
    try:
        yield store
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_interval_schedule_catches_up_once(schedule_store: ScheduleStore) -> None:
    baa = FakeBAA()
    runtime = SimpleNamespace(baa=baa)
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 4, 10, 8, 0, tzinfo=timezone.utc)
    now = start + timedelta(hours=16, minutes=10)

    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Check release state",
        objective="Check release state",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=5,
    )

    result = await service.process_due(now=now)
    updated = await schedule_store.get(str(item.schedule_id))
    runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))

    assert result["submitted"] == 1
    assert len(baa.submitted) == 1
    assert runs[0].scheduled_for == start + timedelta(hours=15)
    assert updated is not None
    assert updated.next_run_at == start + timedelta(hours=20)


@pytest.mark.asyncio
async def test_schedule_stops_when_linked_commitment_is_finished(schedule_store: ScheduleStore, tmp_path) -> None:
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    try:
        commitment = Commitment(
            content="Return to project: Chronicle 4246",
            status=CommitmentStatus.COMPLETED,
            tags=["project_return"],
        )
        await commitment_store.save(commitment)
        baa = FakeBAA()
        runtime = SimpleNamespace(baa=baa, commitment_store=commitment_store)
        service = ScheduleService(schedule_store, runtime=runtime)
        start = datetime(2026, 4, 10, 8, 0, tzinfo=timezone.utc)

        item = await service.create_schedule(
            kind=ScheduleKind.TASK,
            action=ScheduleAction.SUBMIT_BAA,
            title="Return to Chronicle 4246",
            objective="Return to Chronicle 4246",
            start_at=start,
            recurrence=ScheduleRecurrence.INTERVAL_HOURS,
            interval_hours=24,
            commitment_id=str(commitment.commitment_id),
        )

        result = await service.process_due(now=start + timedelta(minutes=1))
        updated = await schedule_store.get(str(item.schedule_id))
        runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))

        assert result["skipped"] == 1
        assert baa.submitted == []
        assert runs[0].status == ScheduleRunStatus.SKIPPED
        assert runs[0].meta["skip_reason"] == "linked_commitment_finished"
        assert updated is not None
        assert updated.status.value == "completed"
        assert updated.next_run_at is None
    finally:
        await commitment_store.close()


@pytest.mark.asyncio
async def test_reminder_event_records_without_baa(schedule_store: ScheduleStore) -> None:
    baa = FakeBAA()
    runtime = SimpleNamespace(baa=baa)
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 4, 10, 12, 0, tzinfo=timezone.utc)

    await service.create_schedule(
        kind=ScheduleKind.EVENT,
        action=ScheduleAction.REMINDER_ONLY,
        title="Lunch",
        start_at=start,
        recurrence=ScheduleRecurrence.NONE,
    )

    result = await service.process_due(now=start + timedelta(minutes=1))
    runs = await schedule_store.list_runs()

    assert result["recorded"] == 1
    assert baa.submitted == []
    assert runs[0].status.value == "recorded"


@pytest.mark.asyncio
async def test_weekday_calendar_range(schedule_store: ScheduleStore) -> None:
    service = ScheduleService(schedule_store)
    monday = datetime(2026, 4, 6, 9, 0, tzinfo=timezone.utc)
    item = await service.create_schedule(
        kind=ScheduleKind.EVENT,
        action=ScheduleAction.REMINDER_ONLY,
        title="Weekday check",
        start_at=monday,
        recurrence=ScheduleRecurrence.WEEKDAYS,
    )

    occurrences = service.occurrences_between(
        item,
        start=monday,
        end=monday + timedelta(days=7),
    )

    assert [dt.weekday() for dt in occurrences] == [0, 1, 2, 3, 4, 0]


@pytest.mark.asyncio
async def test_temporal_agenda_surfaces_due_upcoming_and_recent_runs(schedule_store: ScheduleStore) -> None:
    service = ScheduleService(schedule_store)
    now = datetime(2026, 4, 29, 18, 0, tzinfo=timezone.utc)
    due = await service.create_schedule(
        kind=ScheduleKind.EVENT,
        action=ScheduleAction.REMINDER_ONLY,
        title="Check temporal awareness",
        start_at=now - timedelta(minutes=15),
        recurrence=ScheduleRecurrence.NONE,
        priority=8.0,
        tags=["temporal"],
    )
    await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Run calendar-backed follow-up",
        objective="Run calendar-backed follow-up",
        start_at=now + timedelta(hours=2),
        recurrence=ScheduleRecurrence.NONE,
        priority=6.0,
    )
    await schedule_store.record_run(
        ScheduleRun(
            schedule_id=due.schedule_id,
            scheduled_for=now - timedelta(hours=1),
            status=ScheduleRunStatus.RECORDED,
        )
    )

    agenda = await service.temporal_agenda(now=now, horizon_hours=24)

    assert agenda["counts"]["active"] == 2
    assert agenda["counts"]["due_now"] == 1
    assert agenda["counts"]["upcoming"] == 1
    assert agenda["next"]["title"] == "Check temporal awareness"
    assert agenda["due_now"][0]["is_due"] is True
    assert agenda["upcoming"][0]["title"] == "Run calendar-backed follow-up"
    assert agenda["recent_runs"][0]["status"] == "recorded"


@pytest.mark.asyncio
async def test_schedule_api_create_list_trigger(schedule_store: ScheduleStore) -> None:
    baa = FakeBAA()
    runtime = SimpleNamespace(
        baa=baa,
        ctx=SimpleNamespace(
            schedule_store=schedule_store,
            schedule_service=ScheduleService(schedule_store, runtime=SimpleNamespace(baa=baa)),
        ),
    )
    app = FastAPI()
    app.include_router(build_schedule_router(runtime))
    client = TestClient(app)
    start = datetime.now(timezone.utc) + timedelta(hours=1)

    created = client.post(
        "/api/schedule/items",
        json={
            "kind": "task",
            "title": "API task",
            "objective": "Run API task",
            "start_at": start.isoformat(),
            "recurrence": "none",
        },
    )
    assert created.status_code == 200
    schedule_id = created.json()["item"]["schedule_id"]

    listed = client.get("/api/schedule/items")
    assert listed.status_code == 200
    assert listed.json()["count"] == 1

    triggered = client.post(f"/api/schedule/items/{schedule_id}/trigger")
    assert triggered.status_code == 200
    assert triggered.json()["triggered"] is True
    assert len(baa.submitted) == 1

    detail = client.get(f"/api/schedule/items/{schedule_id}")
    assert detail.json()["item"]["status"] == "active"

    agenda = client.get("/api/schedule/agenda?horizon_hours=24")
    assert agenda.status_code == 200
    assert agenda.json()["available"] is True
    assert "counts" in agenda.json()


def test_dashboard_contains_schedule_surface() -> None:
    dashboard = Path("opencas/dashboard/static/index.html")
    html = dashboard.read_text(encoding="utf-8")
    assert "tab==='schedule'" in html
    assert "scheduleApp()" in html
    assert "/api/schedule/items" in html
    assert "/api/schedule/calendar" in html
    assert "/api/schedule/runs" in html
    assert "/api/schedule/agenda" in html
    assert "Temporal Awareness" in html
