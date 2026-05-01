"""Tests for retry-governor state exposed through operations API surfaces."""

from __future__ import annotations

import pytest
import pytest_asyncio

from opencas.api.operations_activity import ActivityOperationsService
from opencas.api.operations_models import TaskEntry
from opencas.autonomy.models import ActionRiskTier
from opencas.execution import RepairExecutor, RepairTask
from opencas.execution.models import ExecutionStage
from opencas.execution.store import TaskStore
from opencas.tools import ShellToolAdapter, ToolRegistry


def _make_runtime_with_store(store: TaskStore):
    from types import SimpleNamespace
    ctx = SimpleNamespace(tasks=store)
    return SimpleNamespace(ctx=ctx)


def _activity_service(runtime):
    def _human_title(s, fallback=""):
        return s or fallback

    def _task_ui_status(stage, status):
        return status

    return ActivityOperationsService(
        runtime,
        human_title=_human_title,
        task_ui_status=_task_ui_status,
    )


# ── _extract_retry_governor ────────────────────────────────────────────────


def test_extract_retry_governor_inactive_when_no_key():
    result = ActivityOperationsService._extract_retry_governor({})
    assert result == {"active": False}


def test_extract_retry_governor_blocked():
    meta = {
        "retry_governor": {
            "allowed": False,
            "reason": "low-divergence broad retry without new evidence",
            "mode": "resume_existing_artifact",
            "attempt": 2,
            "packet_id": "abc123",
            "reuse_packet_id": "prev456",
        }
    }
    result = ActivityOperationsService._extract_retry_governor(meta)
    assert result["active"] is True
    assert result["blocked"] is True
    assert result["allowed"] is False
    assert result["reason"] == "low-divergence broad retry without new evidence"
    assert result["attempt"] == 2


def test_extract_retry_governor_allowed():
    meta = {
        "retry_governor": {
            "allowed": True,
            "reason": "retry meaningfully diverged",
            "mode": "continue_retry",
            "attempt": 1,
            "packet_id": "abc123",
        }
    }
    result = ActivityOperationsService._extract_retry_governor(meta)
    assert result["active"] is True
    assert result["blocked"] is False
    assert result["allowed"] is True


# ── list_tasks: retry_blocked field ───────────────────────────────────────


@pytest.mark.asyncio
async def test_list_tasks_retry_blocked_flag(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()

    # Task blocked by governor
    blocked = RepairTask(
        objective="blocked task",
        meta={"retry_governor": {"allowed": False, "reason": "no new evidence"}},
    )
    blocked.stage = ExecutionStage.FAILED
    blocked.status = "failed"
    await store.save(blocked)

    # Task that failed normally (no governor entry)
    normal_fail = RepairTask(objective="normal failure")
    normal_fail.stage = ExecutionStage.FAILED
    normal_fail.status = "failed"
    await store.save(normal_fail)

    runtime = _make_runtime_with_store(store)
    svc = _activity_service(runtime)
    response = await svc.list_tasks(limit=50)

    entries = {e.task_id: e for e in response.items}
    assert entries[str(blocked.task_id)].retry_blocked is True
    assert entries[str(normal_fail.task_id)].retry_blocked is False
    await store.close()


# ── get_task: retry_governor section ──────────────────────────────────────


@pytest.mark.asyncio
async def test_get_task_includes_retry_governor_section(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()

    task = RepairTask(
        objective="governor-blocked task",
        meta={
            "retry_governor": {
                "allowed": False,
                "reason": "low-divergence broad retry without new evidence",
                "mode": "resume_existing_artifact",
                "attempt": 2,
                "packet_id": "pkt-001",
            }
        },
    )
    task.stage = ExecutionStage.FAILED
    task.status = "failed"
    await store.save(task)

    runtime = _make_runtime_with_store(store)
    svc = _activity_service(runtime)
    detail = await svc.get_task(str(task.task_id))

    assert detail["found"] is True
    gov = detail["task"]["retry_governor"]
    assert gov["active"] is True
    assert gov["blocked"] is True
    assert gov["attempt"] == 2
    await store.close()


@pytest.mark.asyncio
async def test_get_task_retry_governor_inactive_without_meta(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()

    task = RepairTask(objective="plain failed task")
    task.stage = ExecutionStage.FAILED
    task.status = "failed"
    await store.save(task)

    runtime = _make_runtime_with_store(store)
    svc = _activity_service(runtime)
    detail = await svc.get_task(str(task.task_id))

    assert detail["task"]["retry_governor"] == {"active": False}
    await store.close()


# ── list_salvage_packets store method ─────────────────────────────────────


@pytest.mark.asyncio
async def test_list_salvage_packets_returns_all_attempts(tmp_path):
    from unittest.mock import patch

    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()

    tools = ToolRegistry()
    shell = ShellToolAdapter(cwd=str(tmp_path), timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)

    task = RepairTask(
        objective="Continue Chronicle 4246.",
        verification_command="exit 1",
        max_attempts=3,
        retry_backoff_seconds=0.0,
        meta={
            "resume_project": {
                "signature": "chronicle-4246",
                "canonical_artifact_path": "workspace/Chronicles/4246/chronicle_4246.md",
            }
        },
    )
    from unittest.mock import AsyncMock
    executor = RepairExecutor(tools=tools, store=store)
    with patch("asyncio.sleep", new=AsyncMock()):
        await executor.run(task)   # attempt 1 → RECOVERING, packet saved
        await executor.run(task)   # attempt 2 → FAILED (governor blocks), packet saved

    packets = await store.list_salvage_packets(str(task.task_id))
    assert len(packets) == 2
    assert packets[0].attempt == 1
    assert packets[1].attempt == 2
    await store.close()


@pytest.mark.asyncio
async def test_list_salvage_packets_empty_for_unknown_task(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()
    packets = await store.list_salvage_packets("no-such-task")
    assert packets == []
    await store.close()


def test_compact_provenance_entry_preserves_linked_event_fields():
    entry = {
        "v": "1",
        "event_type": "CHECK",
        "triggering_artifact": "tool|default|health-check",
        "triggering_action": "VERIFY",
        "source_link": "opencas://provenance/check/tool%7Cdefault%7Chealth-check?action=VERIFY",
        "recorded_at": "2026-04-21T12:00:00+00:00",
        "details": {"status": "ok"},
        "session_id": "session-1",
        "artifact": "tool|default|health-check",
        "action": "VERIFY",
        "why": "health check",
        "risk": "LOW",
        "ts": "2026-04-21T12:00:00+00:00",
    }

    compact = ActivityOperationsService._compact_provenance_entry(entry)
    assert compact["event_type"] == "CHECK"
    assert compact["triggering_artifact"] == "tool|default|health-check"
    assert compact["source_link"].startswith("opencas://provenance/check/")
