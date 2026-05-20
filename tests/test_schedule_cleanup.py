"""Tests for duplicate schedule maintenance utilities."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from opencas.scheduling import (
    ScheduleAction,
    ScheduleItem,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleStatus,
    ScheduleStore,
)
from scripts.cleanup_duplicate_schedules import cleanup_duplicate_dataannotations_schedules


@pytest.mark.asyncio
async def test_cleanup_duplicate_dataannotations_schedules_dry_run_and_apply(tmp_path) -> None:
    db_path = tmp_path / "schedules.db"
    store = await ScheduleStore(db_path).connect()
    start = datetime(2026, 5, 4, 16, 0, tzinfo=timezone.utc)
    try:
        complete = ScheduleItem(
            kind=ScheduleKind.TASK,
            action=ScheduleAction.GMAIL_ALERT,
            title="Gmail DataAnnotations check",
            objective="Watch Gmail for new DataAnnotations projects and qualifications.",
            start_at=start,
            recurrence=ScheduleRecurrence.INTERVAL_HOURS,
            interval_hours=6,
            tags=["dataannotations"],
            meta={
                "gmail_query": "from:(dataannotations OR dataannotation) newer_than:30d",
                "seen_message_ids": ["msg-old"],
                "alert_channel": "telegram",
            },
        )
        incomplete = ScheduleItem(
            kind=ScheduleKind.TASK,
            action=ScheduleAction.GMAIL_ALERT,
            title="gmail data annotation check",
            start_at=start,
            recurrence=ScheduleRecurrence.INTERVAL_HOURS,
            interval_hours=6,
            meta={},
        )
        await store.save(complete)
        await store.save(incomplete)
    finally:
        await store.close()

    dry_run = await cleanup_duplicate_dataannotations_schedules(db_path, apply=False)
    assert dry_run["apply"] is False
    assert dry_run["groups"] == 1
    assert dry_run["cancelled"] == []
    assert dry_run["groups_detail"][0]["survivor_schedule_id"] == str(complete.schedule_id)
    assert dry_run["groups_detail"][0]["duplicate_schedule_ids"] == [str(incomplete.schedule_id)]

    store = await ScheduleStore(db_path).connect()
    try:
        assert (await store.get(str(incomplete.schedule_id))).status == ScheduleStatus.ACTIVE
    finally:
        await store.close()

    applied = await cleanup_duplicate_dataannotations_schedules(db_path, apply=True)
    assert applied["apply"] is True
    assert applied["cancelled"] == [str(incomplete.schedule_id)]

    store = await ScheduleStore(db_path).connect()
    try:
        survivor = await store.get(str(complete.schedule_id))
        duplicate = await store.get(str(incomplete.schedule_id))
        assert survivor is not None
        assert duplicate is not None
        assert survivor.status == ScheduleStatus.ACTIVE
        assert survivor.meta["seen_message_ids"] == ["msg-old"]
        assert duplicate.status == ScheduleStatus.CANCELLED
        assert duplicate.next_run_at is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_cleanup_duplicate_dataannotations_schedules_keeps_distinct_queries(tmp_path) -> None:
    db_path = tmp_path / "schedules.db"
    store = await ScheduleStore(db_path).connect()
    start = datetime(2026, 5, 4, 16, 0, tzinfo=timezone.utc)
    try:
        first = ScheduleItem(
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
        second = ScheduleItem(
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
        await store.save(first)
        await store.save(second)
    finally:
        await store.close()

    dry_run = await cleanup_duplicate_dataannotations_schedules(db_path, apply=False)

    assert dry_run["groups"] == 0
    assert dry_run["cancelled"] == []
