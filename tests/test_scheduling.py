"""Tests for scheduled tasks, events, and recurrence."""

from __future__ import annotations

import json
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
from opencas.execution.models import ExecutionReceipt
from opencas.execution.receipt_store import ExecutionReceiptStore
from opencas.scheduling import (
    ScheduleAction,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleRun,
    ScheduleRunStatus,
    ScheduleService,
    ScheduleStatus,
    ScheduleStore,
)


class FakeBAA:
    def __init__(self) -> None:
        self.submitted = []

    async def submit(self, task):
        self.submitted.append(task)
        return None


class FlakyBAA:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.submitted = []

    async def submit(self, task):
        self.submitted.append(task)
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class PausedExecutive:
    def recommend_pause(self) -> bool:
        return True

    def pause_reason(self) -> str:
        return "fatigue"


class OverloadedExecutive:
    def recommend_pause(self) -> bool:
        return True

    def pause_reason(self) -> str:
        return "overload"


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
            content="Return to project: writing project 4246",
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
            title="Return to writing project 4246",
            objective="Return to writing project 4246",
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
async def test_schedule_submit_baa_reaches_baa_while_executive_paused(
    schedule_store: ScheduleStore,
) -> None:
    baa = FakeBAA()
    runtime = SimpleNamespace(baa=baa, executive=PausedExecutive())
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 10, 4, 46, tzinfo=timezone.utc)

    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Return to index",
        objective="Return to project index",
        start_at=start,
        recurrence=ScheduleRecurrence.NONE,
    )

    result = await service.process_due(now=start + timedelta(minutes=1))
    updated = await schedule_store.get(str(item.schedule_id))
    runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))

    assert result["submitted"] == 1
    assert result["skipped"] == 0
    assert len(baa.submitted) == 1
    assert runs[0].status == ScheduleRunStatus.SUBMITTED
    assert runs[0].meta.get("skip_reason") != "executive_recommended_pause"
    assert updated is not None
    assert updated.status == ScheduleStatus.COMPLETED
    assert updated.next_run_at is None


@pytest.mark.asyncio
async def test_operator_project_return_schedule_bypasses_overload_pause(
    schedule_store: ScheduleStore,
) -> None:
    baa = FakeBAA()
    runtime = SimpleNamespace(baa=baa, executive=OverloadedExecutive())
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 17, 15, 24, tzinfo=timezone.utc)

    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Return to The Orchard of Second Species",
        objective="Return to the workspace novel and continue revision until completion evidence exists.",
        start_at=start,
        recurrence=ScheduleRecurrence.NONE,
        tags=["project_return", "self_directed"],
        meta={
            "source": "project_return_capture",
            "source_session_id": "chat-session-123",
            "project_key": "the-orchard-of-second-species",
            "project_title": "The Orchard of Second Species",
            "workspace_abs_path": "/mnt/xtra/OpenCAS/workspace/novels/the-orchard-of-second-species",
            "workspace_rel_path": "novels/the-orchard-of-second-species",
            "workspace_project_confidence": 1.0,
        },
    )

    result = await service.process_due(now=start + timedelta(minutes=1))
    updated = await schedule_store.get(str(item.schedule_id))
    runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))

    assert result["submitted"] == 1
    assert result["skipped"] == 0
    assert len(baa.submitted) == 1
    assert runs[0].status == ScheduleRunStatus.SUBMITTED
    assert runs[0].meta.get("skip_reason") != "executive_recommended_pause"
    assert baa.submitted[0].meta["source"] == "schedule"
    assert baa.submitted[0].meta["authority"] == "schedule_due"
    assert baa.submitted[0].meta["source_session_id"] == "chat-session-123"
    assert baa.submitted[0].meta["workspace_project_confidence"] == 1.0
    assert updated is not None
    assert updated.status == ScheduleStatus.COMPLETED


@pytest.mark.asyncio
async def test_immediate_operator_project_start_schedule_bypasses_overload_without_existing_workspace(
    schedule_store: ScheduleStore,
) -> None:
    baa = FakeBAA()
    runtime = SimpleNamespace(baa=baa, executive=OverloadedExecutive())
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 18, 0, 41, tzinfo=timezone.utc)

    await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Return to Writing Project 20260518-004126",
        objective="Start the requested writing project under the requested workspace parent.",
        start_at=start,
        recurrence=ScheduleRecurrence.NONE,
        tags=["project_return", "self_directed"],
        meta={
            "source": "project_return_capture",
            "source_session_id": "chat-session-123",
            "project_key": "writing-project-20260518-004126",
            "project_title": "Writing Project 20260518-004126",
            "requested_workspace_abs_path": "/mnt/xtra/OpenCAS/workspace/novels",
            "requested_workspace_rel_path": "workspace/novels",
            "requested_workspace_kind": "parent",
            "start_policy": "immediate_operator_work_request",
        },
    )

    result = await service.process_due(now=start + timedelta(seconds=5))
    runs = await schedule_store.list_runs()

    assert result["submitted"] == 1
    assert result["skipped"] == 0
    assert len(baa.submitted) == 1
    assert runs[0].status == ScheduleRunStatus.SUBMITTED
    assert baa.submitted[0].meta["requested_workspace_rel_path"] == "workspace/novels"
    assert baa.submitted[0].meta["start_policy"] == "immediate_operator_work_request"


@pytest.mark.asyncio
async def test_operator_project_return_continuation_schedule_bypasses_overload_with_copied_authority(
    schedule_store: ScheduleStore,
) -> None:
    baa = FakeBAA()
    runtime = SimpleNamespace(baa=baa, executive=OverloadedExecutive())
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 18, 1, 10, tzinfo=timezone.utc)

    await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Continue North Star Draft",
        objective="Continue the linked project until completion evidence exists.",
        start_at=start,
        recurrence=ScheduleRecurrence.NONE,
        tags=["project_followthrough", "project_return", "self_directed", "auto_continue"],
        meta={
            "source": "project_followthrough_recovery",
            "source_session_id": "chat-session-456",
            "workspace_rel_path": "workspace/projects/north-star-draft",
            "workspace_project_confidence": 0.95,
            "project_intent": "finish the requested draft",
            "next_step": "resume from saved notes",
        },
    )

    result = await service.process_due(now=start + timedelta(minutes=1))
    runs = await schedule_store.list_runs()

    assert result["submitted"] == 1
    assert result["skipped"] == 0
    assert runs[0].status == ScheduleRunStatus.SUBMITTED
    assert baa.submitted[0].meta["source"] == "schedule"
    assert baa.submitted[0].meta["source_session_id"] == "chat-session-456"
    assert baa.submitted[0].meta["workspace_rel_path"] == "workspace/projects/north-star-draft"


@pytest.mark.asyncio
async def test_weak_project_followthrough_continuation_schedule_reaches_baa_overload_gate(
    schedule_store: ScheduleStore,
) -> None:
    baa = FakeBAA()
    runtime = SimpleNamespace(baa=baa, executive=OverloadedExecutive())
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 18, 1, 12, tzinfo=timezone.utc)

    await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Continue weak project followthrough",
        objective="Continue a project without preserved operator evidence.",
        start_at=start,
        recurrence=ScheduleRecurrence.NONE,
        tags=["project_followthrough", "self_directed", "auto_continue"],
        meta={
            "source": "project_followthrough_recovery",
            "workspace_rel_path": "workspace/projects/weak",
            "workspace_project_confidence": 0.95,
        },
    )

    result = await service.process_due(now=start + timedelta(minutes=1))
    runs = await schedule_store.list_runs()

    assert result["submitted"] == 1
    assert result["skipped"] == 0
    assert len(baa.submitted) == 1
    assert runs[0].status == ScheduleRunStatus.SUBMITTED
    assert runs[0].meta.get("skip_reason") != "executive_recommended_pause"


@pytest.mark.asyncio
async def test_operator_project_return_schedule_reaches_baa_fatigue_gate(
    schedule_store: ScheduleStore,
) -> None:
    baa = FakeBAA()
    runtime = SimpleNamespace(baa=baa, executive=PausedExecutive())
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 17, 15, 24, tzinfo=timezone.utc)

    await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Return to The Orchard of Second Species",
        objective="Return to the workspace novel and continue revision until completion evidence exists.",
        start_at=start,
        recurrence=ScheduleRecurrence.NONE,
        tags=["project_return", "self_directed"],
        meta={
            "source": "project_return_capture",
            "source_session_id": "chat-session-123",
            "project_key": "the-orchard-of-second-species",
            "project_title": "The Orchard of Second Species",
            "workspace_abs_path": "/mnt/xtra/OpenCAS/workspace/novels/the-orchard-of-second-species",
            "workspace_rel_path": "novels/the-orchard-of-second-species",
            "workspace_project_confidence": 1.0,
        },
    )

    result = await service.process_due(now=start + timedelta(minutes=1))
    runs = await schedule_store.list_runs()

    assert result["submitted"] == 1
    assert result["skipped"] == 0
    assert len(baa.submitted) == 1
    assert runs[0].status == ScheduleRunStatus.SUBMITTED
    assert runs[0].meta.get("skip_reason") != "executive_recommended_pause"


@pytest.mark.asyncio
async def test_schedule_copies_claim_id_from_linked_commitment_into_task(
    schedule_store: ScheduleStore,
    tmp_path: Path,
) -> None:
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    try:
        commitment = Commitment(
            content="Verify the promise spine.",
            status=CommitmentStatus.ACTIVE,
            meta={"source": "assistant_response", "claim_id": "claim-spine-1"},
        )
        await commitment_store.save(commitment)
        baa = FakeBAA()
        runtime = SimpleNamespace(baa=baa, commitment_store=commitment_store)
        service = ScheduleService(schedule_store, runtime=runtime)
        start = datetime(2026, 5, 6, 18, 0, tzinfo=timezone.utc)

        item = await service.create_schedule(
            kind=ScheduleKind.TASK,
            action=ScheduleAction.SUBMIT_BAA,
            title="Verify promise spine",
            objective="Verify the promise spine.",
            start_at=start,
            recurrence=ScheduleRecurrence.NONE,
            commitment_id=str(commitment.commitment_id),
        )
        result = await service.process_due(now=start + timedelta(minutes=1))
        updated = await schedule_store.get(str(item.schedule_id))

        assert result["submitted"] == 1
        assert updated is not None
        assert updated.meta["claim_id"] == "claim-spine-1"
        assert len(baa.submitted) == 1
        assert baa.submitted[0].meta["claim_id"] == "claim-spine-1"
        assert baa.submitted[0].commitment_id == str(commitment.commitment_id)
    finally:
        await commitment_store.close()


@pytest.mark.asyncio
async def test_failed_schedule_submission_stays_active_and_defers_retry(schedule_store: ScheduleStore) -> None:
    contacts = []

    async def _contact_owner(**kwargs):
        contacts.append(kwargs)
        return {"status": "sent", "channel": "telegram"}

    service = ScheduleService(
        schedule_store,
        runtime=SimpleNamespace(baa=None, initiative_contact_owner=_contact_owner),
    )
    start = datetime(2026, 5, 3, 20, 0, tzinfo=timezone.utc)
    now = start + timedelta(minutes=1)

    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Return to kPony",
        objective="Return to software project kPony.",
        start_at=start,
        recurrence=ScheduleRecurrence.NONE,
        tags=["project_return"],
        meta={"project_key": "kpony", "project_title": "kPony", "project_type": "software"},
    )

    result = await service.process_due(now=now)
    updated = await schedule_store.get(str(item.schedule_id))
    runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))

    assert result["failed"] == 1
    assert runs[0].status == ScheduleRunStatus.FAILED
    assert "runtime BAA is not available" in (runs[0].error or "")
    assert updated is not None
    assert updated.status.value == "active"
    assert updated.next_run_at == now + timedelta(minutes=5)
    assert contacts
    assert contacts[0]["channel"] == "telegram"
    assert "Return to kPony" in contacts[0]["message"]
    assert "runtime BAA is not available" in contacts[0]["message"]


@pytest.mark.asyncio
async def test_baa_submit_failure_leaves_claimed_run_and_advances_schedule(
    schedule_store: ScheduleStore,
) -> None:
    baa = FlakyBAA([RuntimeError("submit unavailable")])
    service = ScheduleService(schedule_store, runtime=SimpleNamespace(baa=baa))
    start = datetime(2026, 5, 8, 8, 0, tzinfo=timezone.utc)
    now = start + timedelta(minutes=1)

    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Run slow task",
        objective="Run slow task",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=1,
    )

    result = await service.process_due(now=now)
    updated = await schedule_store.get(str(item.schedule_id))
    runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))

    assert result["failed"] == 1
    assert len(baa.submitted) == 1
    assert runs[0].status.value == "claimed"
    assert runs[0].error is None
    assert updated is not None
    assert updated.next_run_at == start + timedelta(hours=1)

    await service.process_due(now=now)
    assert len(baa.submitted) == 1


@pytest.mark.asyncio
async def test_baa_advance_partial_failure_keeps_claimed_run_without_submit(
    schedule_store: ScheduleStore,
) -> None:
    baa = FlakyBAA([None])
    service = ScheduleService(schedule_store, runtime=SimpleNamespace(baa=baa))
    start = datetime(2026, 5, 8, 8, 0, tzinfo=timezone.utc)
    now = start + timedelta(minutes=1)

    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Run partial advance task",
        objective="Run partial advance task",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=1,
    )
    original_advance = service._advance_item

    async def advance_then_fail(item, scheduled_for, now):
        await original_advance(item, scheduled_for, now)
        raise RuntimeError("advance write failed after save")

    service._advance_item = advance_then_fail

    result = await service.process_due(now=now)
    updated = await schedule_store.get(str(item.schedule_id))
    runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))

    assert result["failed"] == 1
    assert baa.submitted == []
    assert runs[0].status.value == "claimed"
    assert updated is not None
    assert updated.next_run_at == start + timedelta(hours=1)


@pytest.mark.asyncio
async def test_process_due_recovers_old_claimed_baa_run_once(
    schedule_store: ScheduleStore,
) -> None:
    baa = FlakyBAA([RuntimeError("first submit failed"), None])
    service = ScheduleService(schedule_store, runtime=SimpleNamespace(baa=baa))
    start = datetime(2026, 5, 8, 8, 0, tzinfo=timezone.utc)
    now = start + timedelta(minutes=1)

    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Recover claimed task",
        objective="Recover claimed task",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=1,
    )

    await service.process_due(now=now)
    await service.process_due(now=now + timedelta(seconds=61))

    runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))
    assert len(baa.submitted) == 2
    assert len(runs) == 1
    assert runs[0].status == ScheduleRunStatus.SUBMITTED
    assert runs[0].task_id == str(baa.submitted[-1].task_id)


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
async def test_gmail_alert_schedule_contacts_owner_once_for_new_messages(
    schedule_store: ScheduleStore,
) -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.tool_calls = []
            self.contacts = []
            self.messages = [
                {
                    "id": "msg-1",
                    "from": "DataAnnotation <noreply@mail.dataannotation.tech>",
                    "subject": "Qualification available",
                    "date": "Mon, 04 May 2026 14:00:00 +0000",
                },
                {
                    "id": "msg-2",
                    "from": "DataAnnotation <noreply@mail.dataannotation.tech>",
                    "subject": "New projects",
                    "date": "Mon, 04 May 2026 15:00:00 +0000",
                },
            ]

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            self.tool_calls.append((name, args))
            return {
                "success": True,
                "output": json.dumps(
                    {
                        "count": len(self.messages),
                        "messages": self.messages,
                        "query": args["query"],
                    }
                ),
                "metadata": {"result_count": len(self.messages)},
            }

        async def initiative_contact_owner(self, **kwargs):
            self.contacts.append(kwargs)
            return {"status": "sent", "channel": "telegram"}

    runtime = FakeRuntime()
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 4, 16, 0, tzinfo=timezone.utc)
    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action="gmail_alert",
        title="Gmail DataAnnotations check",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=6,
        tags=["gmail-monitor", "dataannotations"],
        meta={
            "gmail_query": "from:(dataannotations OR dataannotation) newer_than:30d",
            "max_results": 5,
            "alert_channel": "telegram",
            "alert_urgency": "high",
        },
    )

    first = await service.process_due(now=start + timedelta(minutes=1))
    updated = await schedule_store.get(str(item.schedule_id))
    first_runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))

    assert first["recorded"] == 1
    assert runtime.tool_calls[0] == (
        "google_workspace_gmail_headlines",
        {
            "query": "from:(dataannotations OR dataannotation) newer_than:30d",
            "max_results": 5,
            "include_snippet": False,
        },
    )
    assert len(runtime.contacts) == 1
    assert runtime.contacts[0]["channel"] == "telegram"
    assert runtime.contacts[0]["urgency"] == "high"
    assert "Qualification available" in runtime.contacts[0]["message"]
    assert "New projects" in runtime.contacts[0]["message"]
    assert first_runs[0].status == ScheduleRunStatus.RECORDED
    assert first_runs[0].meta["gmail_alert"]["new_count"] == 2
    assert updated is not None
    seen = updated.meta["seen_message_ids"]
    assert set(seen) == {"msg-1", "msg-2"}
    assert all(seen[message_id]["delivered_at"] for message_id in seen)
    assert all(seen[message_id]["persisted_at"] for message_id in seen)
    assert updated.next_run_at == start + timedelta(hours=6)

    second = await service.process_due(now=start + timedelta(hours=6, minutes=1))
    second_runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))
    updated_again = await schedule_store.get(str(item.schedule_id))

    assert second["recorded"] == 1
    assert len(runtime.contacts) == 1
    assert second_runs[0].meta["gmail_alert"]["new_count"] == 0
    assert updated_again is not None
    assert updated_again.meta["seen_message_ids"] == seen
    assert updated_again.next_run_at == start + timedelta(hours=12)


@pytest.mark.asyncio
async def test_gmail_alert_redelivers_when_final_seen_persist_fails(
    schedule_store: ScheduleStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.contacts = []

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            return {
                "success": True,
                "output": json.dumps(
                    {
                        "count": 1,
                        "messages": [
                            {
                                "id": "msg-new",
                                "from": "DataAnnotation <noreply@mail.dataannotation.tech>",
                                "subject": "New project",
                                "date": "Mon, 04 May 2026 15:00:00 +0000",
                            }
                        ],
                    }
                ),
            }

        async def initiative_contact_owner(self, **kwargs):
            self.contacts.append(kwargs)
            return {"status": "sent", "channel": "telegram"}

    runtime = FakeRuntime()
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 4, 16, 0, tzinfo=timezone.utc)
    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.GMAIL_ALERT,
        title="Gmail DataAnnotations check",
        start_at=start,
        meta={"gmail_query": "from:(dataannotations OR dataannotation) newer_than:30d"},
    )
    original_save = schedule_store.save
    failed_final_persist = False

    async def flaky_save(candidate):
        nonlocal failed_final_persist
        seen = (candidate.meta or {}).get("seen_message_ids")
        if (
            not failed_final_persist
            and isinstance(seen, dict)
            and seen.get("msg-new", {}).get("persisted_at")
        ):
            failed_final_persist = True
            raise RuntimeError("final seen persist failed")
        await original_save(candidate)

    monkeypatch.setattr(schedule_store, "save", flaky_save)

    first = await service.trigger(str(item.schedule_id), now=start + timedelta(minutes=1), manual=True)
    assert first.status == ScheduleRunStatus.FAILED
    assert len(runtime.contacts) == 1

    second = await service.trigger(str(item.schedule_id), now=start + timedelta(minutes=2), manual=True)
    updated = await schedule_store.get(str(item.schedule_id))

    assert second.status == ScheduleRunStatus.RECORDED
    assert len(runtime.contacts) == 2
    assert updated is not None
    seen_entry = updated.meta["seen_message_ids"]["msg-new"]
    assert seen_entry["delivered_at"]
    assert seen_entry["persisted_at"]


@pytest.mark.asyncio
async def test_gmail_alert_does_not_mark_seen_when_owner_contact_is_suppressed(
    schedule_store: ScheduleStore,
) -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.contacts = []

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            return {
                "success": True,
                "output": json.dumps(
                    {
                        "count": 1,
                        "messages": [
                            {
                                "id": "msg-new",
                                "from": "DataAnnotation <noreply@mail.dataannotation.tech>",
                                "subject": "New project",
                                "date": "Mon, 04 May 2026 15:00:00 +0000",
                            }
                        ],
                    }
                ),
            }

        async def initiative_contact_owner(self, **kwargs):
            self.contacts.append(kwargs)
            return {"status": "suppressed", "reason": "disabled", "channel": "telegram"}

    runtime = FakeRuntime()
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 4, 16, 0, tzinfo=timezone.utc)
    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.GMAIL_ALERT,
        title="Gmail DataAnnotations check",
        start_at=start,
        meta={
            "gmail_query": "from:(dataannotations OR dataannotation) newer_than:30d",
            "alert_channel": "telegram",
        },
    )

    run = await service.trigger(str(item.schedule_id), now=start + timedelta(minutes=1), manual=True)
    updated = await schedule_store.get(str(item.schedule_id))

    assert run.status == ScheduleRunStatus.FAILED
    assert run.meta["gmail_alert"]["new_count"] == 1
    assert run.meta["gmail_alert"]["contact_result"]["status"] == "suppressed"
    assert "disabled" in (run.error or "")
    assert len(runtime.contacts) == 1
    assert updated is not None
    assert updated.meta.get("seen_message_ids") is None


@pytest.mark.asyncio
async def test_gmail_alert_auth_failure_preflights_and_suppresses_repeat_owner_noise(
    schedule_store: ScheduleStore,
) -> None:
    auth_error = {
        "error": {
            "code": 401,
            "message": (
                "Authentication failed: Failed to get token: Server error: "
                "invalid_grant: Token has been expired or revoked."
            ),
            "reason": "authError",
        }
    }

    class FakeRuntime:
        def __init__(self) -> None:
            self.tool_calls = []
            self.contacts = []

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            self.tool_calls.append((name, args, session_id))
            return {
                "success": False,
                "output": json.dumps(auth_error),
                "metadata": {"auth_error": True},
            }

        async def initiative_contact_owner(self, **kwargs):
            self.contacts.append(kwargs)
            return {"status": "sent", "channel": "telegram"}

    runtime = FakeRuntime()
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 4, 16, 0, tzinfo=timezone.utc)
    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.GMAIL_ALERT,
        title="Gmail DataAnnotations check",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=6,
        meta={"gmail_query": "from:(dataannotations OR dataannotation) newer_than:30d"},
    )

    first = await service.process_due(now=start + timedelta(minutes=1))
    updated = await schedule_store.get(str(item.schedule_id))
    first_runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))

    assert first["failed"] == 1
    assert len(runtime.contacts) == 1
    assert "Google Workspace authentication failed" in runtime.contacts[0]["message"]
    assert "gws auth" in runtime.contacts[0]["message"]
    assert updated is not None
    assert updated.next_run_at == start + timedelta(minutes=1, hours=6)
    assert updated.meta["auth_failure"]["kind"] == "google_workspace_auth"
    assert updated.meta["auth_failure"]["failure_count"] == 1
    assert updated.meta["auth_failure"]["notification_count"] == 1
    assert first_runs[0].meta["gmail_alert"]["auth_failure"]["reason"] == "invalid_grant"
    assert [call[0] for call in runtime.tool_calls] == [
        "google_workspace_gmail_headlines",
        "google_workspace_auth_status",
    ]

    second_now = start + timedelta(hours=6, minutes=2)
    second = await service.process_due(now=second_now)
    updated_again = await schedule_store.get(str(item.schedule_id))
    runs_after_second = await schedule_store.list_runs(schedule_id=str(item.schedule_id))

    assert second["skipped"] == 1
    assert second["failed"] == 0
    assert len(runtime.contacts) == 1
    assert updated_again is not None
    assert updated_again.next_run_at == second_now + timedelta(hours=6)
    assert updated_again.meta["auth_failure"]["failure_count"] == 2
    assert updated_again.meta["auth_failure"]["preflight_count"] == 1
    assert updated_again.meta["auth_failure"]["notification_count"] == 1
    assert updated_again.meta["auth_failure"]["reauth_required"] is True
    assert runs_after_second[0].status == ScheduleRunStatus.SKIPPED
    assert runs_after_second[0].meta["skip_reason"] == "google_workspace_auth_unresolved"
    assert [call[0] for call in runtime.tool_calls] == [
        "google_workspace_gmail_headlines",
        "google_workspace_auth_status",
        "google_workspace_auth_status",
    ]


@pytest.mark.asyncio
async def test_gmail_alert_recovers_after_auth_preflight_succeeds(
    schedule_store: ScheduleStore,
) -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.tool_calls = []

        async def execute_tool(self, name, args, *, session_id=None, task_id=None):
            self.tool_calls.append((name, args, session_id))
            if name == "google_workspace_auth_status":
                return {
                    "success": True,
                    "output": json.dumps({"authenticated": True, "account": "user@example.com"}),
                    "metadata": {},
                }
            return {
                "success": True,
                "output": json.dumps({"count": 0, "messages": [], "query": args["query"]}),
                "metadata": {"result_count": 0},
            }

    runtime = FakeRuntime()
    service = ScheduleService(schedule_store, runtime=runtime)
    start = datetime(2026, 5, 4, 16, 0, tzinfo=timezone.utc)
    item = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.GMAIL_ALERT,
        title="Gmail DataAnnotations check",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=6,
        meta={
            "gmail_query": "from:(dataannotations OR dataannotation) newer_than:30d",
            "auth_failure": {
                "kind": "google_workspace_auth",
                "reason": "invalid_grant",
                "message": "Token has been expired or revoked.",
                "first_seen_at": start.isoformat(),
                "last_seen_at": start.isoformat(),
                "failure_count": 1,
                "notification_count": 1,
                "reauth_required": True,
            },
        },
    )
    item.next_run_at = start + timedelta(hours=6)
    await schedule_store.save(item)

    result = await service.process_due(now=start + timedelta(hours=6, minutes=1))
    updated = await schedule_store.get(str(item.schedule_id))
    runs = await schedule_store.list_runs(schedule_id=str(item.schedule_id))

    assert result["recorded"] == 1
    assert runs[0].status == ScheduleRunStatus.RECORDED
    assert [call[0] for call in runtime.tool_calls] == [
        "google_workspace_auth_status",
        "google_workspace_gmail_headlines",
    ]
    assert updated is not None
    assert "auth_failure" not in updated.meta
    assert updated.meta["auth_recovered_at"]
    assert updated.next_run_at == start + timedelta(hours=12)


@pytest.mark.asyncio
async def test_duplicate_gmail_alert_schedule_creation_merges_into_existing_schedule(
    schedule_store: ScheduleStore,
) -> None:
    service = ScheduleService(schedule_store)
    start = datetime(2026, 5, 4, 16, 0, tzinfo=timezone.utc)
    existing = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.GMAIL_ALERT,
        title="Gmail DataAnnotations check",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=6,
        tags=["gmail-monitor"],
        meta={
            "gmail_query": "from:(dataannotations OR dataannotation) newer_than:30d",
            "seen_message_ids": ["msg-old"],
        },
    )

    duplicate = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.GMAIL_ALERT,
        title="gmail data annotations check",
        objective="Watch Gmail for new DataAnnotations projects and qualifications.",
        start_at=start + timedelta(minutes=5),
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=6,
        tags=["dataannotations"],
        meta={
            "gmail_query": "from:(dataannotations OR dataannotation) newer_than:30d",
            "max_results": 10,
            "alert_channel": "telegram",
            "alert_urgency": "high",
        },
    )

    active_items = await schedule_store.list_items(status=ScheduleStatus.ACTIVE, limit=100)
    updated = await schedule_store.get(str(existing.schedule_id))

    assert duplicate.schedule_id == existing.schedule_id
    assert len(active_items) == 1
    assert updated is not None
    assert updated.objective == "Watch Gmail for new DataAnnotations projects and qualifications."
    assert sorted(updated.tags) == ["dataannotations", "gmail-monitor"]
    assert updated.meta["gmail_query"] == "from:(dataannotations OR dataannotation) newer_than:30d"
    assert updated.meta["seen_message_ids"] == ["msg-old"]
    assert updated.meta["max_results"] == 10
    assert updated.meta["dedupe_action"] == "merged"
    assert updated.meta["survivor_schedule_id"] == str(existing.schedule_id)
    assert updated.meta["duplicate_schedule_ids"] == []
    assert updated.meta["merged_schedule_ids"] == []


@pytest.mark.asyncio
async def test_malformed_dataannotation_gmail_alert_without_query_reuses_complete_schedule(
    schedule_store: ScheduleStore,
) -> None:
    service = ScheduleService(schedule_store)
    start = datetime(2026, 5, 4, 16, 0, tzinfo=timezone.utc)
    existing = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.GMAIL_ALERT,
        title="Gmail DataAnnotations check",
        objective="Watch Gmail for new DataAnnotations projects.",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=6,
        meta={
            "gmail_query": "from:(dataannotations OR dataannotation) newer_than:30d",
            "seen_message_ids": ["msg-old"],
        },
    )

    duplicate = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.GMAIL_ALERT,
        title="Gmail DataAnnotation check",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=6,
    )

    active_items = await schedule_store.list_items(status=ScheduleStatus.ACTIVE, limit=100)
    updated = await schedule_store.get(str(existing.schedule_id))

    assert duplicate.schedule_id == existing.schedule_id
    assert len(active_items) == 1
    assert updated is not None
    assert updated.meta["gmail_query"] == "from:(dataannotations OR dataannotation) newer_than:30d"
    assert updated.meta["seen_message_ids"] == ["msg-old"]
    assert updated.meta["dedupe_action"] == "reused"


@pytest.mark.asyncio
async def test_dataannotations_gmail_alerts_with_distinct_queries_do_not_dedupe(
    schedule_store: ScheduleStore,
) -> None:
    service = ScheduleService(schedule_store)
    start = datetime(2026, 5, 4, 16, 0, tzinfo=timezone.utc)
    first = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.GMAIL_ALERT,
        title="Gmail DataAnnotations check",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=6,
        meta={
            "gmail_query": "from:dataannotations@example.com newer_than:30d",
        },
    )
    second = await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.GMAIL_ALERT,
        title="Gmail DataAnnotations check",
        start_at=start,
        recurrence=ScheduleRecurrence.INTERVAL_HOURS,
        interval_hours=6,
        meta={
            "gmail_query": "from:different-dataannotations@example.com newer_than:30d",
        },
    )

    active_items = await schedule_store.list_items(status=ScheduleStatus.ACTIVE, limit=100)

    assert second.schedule_id != first.schedule_id
    assert len(active_items) == 2
    assert first.meta["dedupe_action"] == "created"
    assert second.meta["dedupe_action"] == "created"


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
async def test_schedule_agenda_api_surfaces_scheduler_processing_status(schedule_store: ScheduleStore) -> None:
    service = ScheduleService(schedule_store)
    now = datetime.now(timezone.utc)
    await service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Return to due work",
        objective="Pick up due scheduled work.",
        start_at=now - timedelta(minutes=5),
    )

    class FakeScheduler:
        def schedule_processing_status(self, due_now=None):
            return {
                "scheduler_running": True,
                "can_process": True,
                "blocked_reason": None,
                "due_now": due_now,
                "will_process_due_now": bool(due_now),
            }

    runtime = SimpleNamespace(
        scheduler=FakeScheduler(),
        schedule_service=service,
        ctx=SimpleNamespace(schedule_store=schedule_store, schedule_service=service),
    )
    app = FastAPI()
    app.include_router(build_schedule_router(runtime))
    client = TestClient(app)

    response = client.get("/api/schedule/agenda?horizon_hours=24")

    assert response.status_code == 200
    payload = response.json()
    assert payload["counts"]["due_now"] == 1
    assert payload["processing"]["scheduler_running"] is True
    assert payload["processing"]["can_process"] is True
    assert payload["processing"]["due_now"] == 1
    assert payload["processing"]["will_process_due_now"] is True


@pytest.mark.asyncio
async def test_temporal_agenda_binds_submitted_schedule_runs_to_baa_receipts(
    schedule_store: ScheduleStore,
    tmp_path: Path,
) -> None:
    receipt_store = await ExecutionReceiptStore(tmp_path / "receipts.db").connect()
    try:
        service = ScheduleService(
            schedule_store,
            runtime=SimpleNamespace(ctx=SimpleNamespace(receipt_store=receipt_store)),
        )
        now = datetime(2026, 5, 6, 18, 0, tzinfo=timezone.utc)
        item = await service.create_schedule(
            kind=ScheduleKind.TASK,
            action=ScheduleAction.SUBMIT_BAA,
            title="Metacognitive follow-up: verify proof loop",
            objective="Resolve this repeated self-inspection gap and create proof.",
            start_at=now - timedelta(minutes=5),
            recurrence=ScheduleRecurrence.NONE,
            tags=["cognitive-maintenance", "self_inspection_gap"],
            meta={
                "source": "cognitive_maintenance",
                "promotion_kind": "self_inspection_gap",
                "promotion_key": "gap-key",
                "cognitive_event_id": "event-1",
            },
        )
        task_id = "11111111-1111-4111-8111-111111111111"
        await schedule_store.record_run(
            ScheduleRun(
                schedule_id=item.schedule_id,
                scheduled_for=now - timedelta(minutes=5),
                status=ScheduleRunStatus.SUBMITTED,
                task_id=task_id,
                meta={"action": "submit_baa", "manual": False, "task_objective": "x" * 1500},
            )
        )
        receipt = await receipt_store.save_direct(
            ExecutionReceipt(
                task_id=task_id,
                objective="Resolve this repeated self-inspection gap and create proof.",
                completed_at=now,
                success=True,
                output="Proof note created for the repeated self-inspection gap. " + ("y" * 800),
            )
        )

        agenda = await service.temporal_agenda(now=now, horizon_hours=24)

        recent_run = agenda["recent_runs"][0]
        assert recent_run["task_id"] == task_id
        assert recent_run["meta"]["task_objective"].endswith("chars]")
        assert len(recent_run["meta"]["task_objective"]) < 1100
        assert recent_run["receipt_count"] == 1
        assert recent_run["receipts"][0]["receipt_id"] == str(receipt.receipt_id)
        assert recent_run["receipts"][0]["success"] is True
        assert "self-inspection gap" in recent_run["receipts"][0]["output"]
        assert recent_run["receipts"][0]["output"].endswith("y")
    finally:
        await receipt_store.close()


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


@pytest.mark.asyncio
async def test_schedule_api_item_detail_binds_runs_to_baa_receipts(
    schedule_store: ScheduleStore,
    tmp_path: Path,
) -> None:
    receipt_store = await ExecutionReceiptStore(tmp_path / "receipts.db").connect()
    try:
        service = ScheduleService(
            schedule_store,
            runtime=SimpleNamespace(ctx=SimpleNamespace(receipt_store=receipt_store)),
        )
        runtime = SimpleNamespace(
            ctx=SimpleNamespace(
                schedule_store=schedule_store,
                schedule_service=service,
                receipt_store=receipt_store,
            ),
        )
        app = FastAPI()
        app.include_router(build_schedule_router(runtime))
        client = TestClient(app)
        now = datetime(2026, 5, 6, 19, 0, tzinfo=timezone.utc)
        item = await service.create_schedule(
            kind=ScheduleKind.TASK,
            action=ScheduleAction.SUBMIT_BAA,
            title="Metacognitive follow-up: prove schedule receipt",
            objective="Resolve this repeated self-inspection gap and create proof.",
            start_at=now - timedelta(minutes=5),
            recurrence=ScheduleRecurrence.NONE,
            tags=["cognitive-maintenance", "self_inspection_gap"],
            meta={"promotion_kind": "self_inspection_gap", "cognitive_event_id": "event-2"},
        )
        task_id = "22222222-2222-4222-8222-222222222222"
        await schedule_store.record_run(
            ScheduleRun(
                schedule_id=item.schedule_id,
                scheduled_for=now - timedelta(minutes=5),
                status=ScheduleRunStatus.SUBMITTED,
                task_id=task_id,
            )
        )
        receipt = await receipt_store.save_direct(
            ExecutionReceipt(
                task_id=task_id,
                objective="Resolve this repeated self-inspection gap and create proof.",
                completed_at=now,
                success=True,
                output="Schedule-generated BAA work produced its proof receipt.",
            )
        )

        detail = client.get(f"/api/schedule/items/{item.schedule_id}")
        runs = client.get(f"/api/schedule/runs?schedule_id={item.schedule_id}")

        assert detail.status_code == 200
        assert detail.json()["runs"][0]["receipt_count"] == 1
        assert detail.json()["runs"][0]["receipts"][0]["receipt_id"] == str(receipt.receipt_id)
        assert runs.status_code == 200
        assert runs.json()["items"][0]["receipt_count"] == 1
    finally:
        await receipt_store.close()


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
