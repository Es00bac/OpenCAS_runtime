"""Tests for higher-level operator workflow tools."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from opencas.api import provenance_store as ps
from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.models import WorkObject, WorkStage
from opencas.execution.models import ExecutionStage, RepairTask
from opencas.planning import PlanAction, PlanEntry
from opencas.proof_chain import ProofClaim, ProofClaimType
from opencas.scheduling import (
    ScheduleAction,
    ScheduleItem,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleRun,
    ScheduleRunStatus,
    ScheduleStatus,
)
from opencas.tools.adapters.workflow import WorkflowToolAdapter


def _make_mock_runtime(tmp_path: Path):
    """Create a minimal mock runtime for workflow adapter tests."""
    runtime = MagicMock()
    runtime.ctx = MagicMock()
    runtime.ctx.config.primary_workspace_root.return_value = tmp_path
    runtime.ctx.config.agent_workspace_root.return_value = tmp_path / "workspace"
    runtime.ctx.config.state_dir = tmp_path

    # Commitment store
    commitment_store = MagicMock()
    commitment_store.save = AsyncMock()
    commitment_store.update_status = AsyncMock(return_value=True)
    commitment_store.list_by_status = AsyncMock(return_value=[])
    commitment_store.count_by_status = AsyncMock(return_value=0)
    commitment_store.get = AsyncMock(return_value=None)
    runtime.commitment_store = commitment_store

    # Work store
    work_store = MagicMock()
    work_store.summary_counts = AsyncMock(return_value={"total": 0, "ready": 0, "blocked": 0})
    work_store.get = AsyncMock(return_value=None)
    work_store.list_by_commitment = AsyncMock(return_value=[])
    work_store.save = AsyncMock()
    runtime.ctx.work_store = work_store
    runtime.executive = MagicMock()
    runtime.executive.reconcile_work_update = MagicMock()
    runtime.ctx.schedule_store = None
    runtime.schedule_service = None

    # Plan store
    plan_store = MagicMock()
    plan_entry = MagicMock()
    plan_entry.plan_id = "plan-001"
    plan_entry.status = "active"
    plan_store.create_plan = AsyncMock(return_value=plan_entry)
    plan_store.set_status = AsyncMock(return_value=True)
    plan_store.update_content = AsyncMock(return_value=True)
    plan_store.count_active = AsyncMock(return_value=0)
    plan_store.list_active = AsyncMock(return_value=[])
    plan_store.get_plan = AsyncMock(return_value=None)
    plan_store.get_actions = AsyncMock(return_value=[])
    runtime.ctx.plan_store = plan_store

    # BAA task store
    runtime.ctx.tasks = None

    # Execute tool (for repo triage and supervise)
    runtime.execute_tool = AsyncMock(return_value={
        "success": True,
        "output": json.dumps({"ok": True}),
        "metadata": {},
    })

    return runtime


@pytest.mark.asyncio
async def test_create_commitment(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_create_commitment", {"content": "Ship v1.0"})
    assert result.success is True
    payload = json.loads(result.output)
    assert payload["content"] == "Ship v1.0"
    assert payload["status"] == "active"
    assert "commitment_id" in payload
    runtime.commitment_store.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_commitment_with_priority_and_tags(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_create_commitment", {
        "content": "Fix bug",
        "priority": 8.0,
        "tags": ["bug", "urgent"],
    })
    assert result.success is True
    saved = runtime.commitment_store.save.call_args[0][0]
    assert saved.priority == 8.0
    assert saved.tags == ["bug", "urgent"]


@pytest.mark.asyncio
async def test_create_commitment_classifies_software_project(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_commitment",
        {"content": "Build and verify the kPony Qt6 email client with cmake."},
    )

    assert result.success is True
    saved = runtime.commitment_store.save.call_args[0][0]
    assert saved.meta["project_type"] == "software"
    assert "software" in saved.tags


@pytest.mark.asyncio
async def test_create_commitment_records_operator_promise_claim(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    claim = ProofClaim(
        claim_type=ProofClaimType.PROMISE,
        claim="Ship the promise spine.",
        subject="operator_commitment",
    )
    runtime.proof_chain = SimpleNamespace(
        record_claim=AsyncMock(return_value=claim),
        link_evidence=AsyncMock(return_value=claim),
    )
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_commitment",
        {"content": "Ship the promise spine."},
    )

    assert result.success is True
    saved = runtime.commitment_store.save.call_args[0][0]
    assert saved.meta["source"] == "workflow_create_commitment"
    assert saved.meta["claim_id"] == str(claim.claim_id)
    runtime.proof_chain.record_claim.assert_awaited_once()
    runtime.proof_chain.link_evidence.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_commitment_audit_only_does_not_persist(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_commitment",
        {"content": "Audit-only commitment probe.", "_audit_only": True},
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["audit_only"] is True
    assert "commitment_id" not in payload
    runtime.commitment_store.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_commitment_missing_content(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_create_commitment", {})
    assert result.success is False
    assert "content" in result.output.lower()


@pytest.mark.asyncio
async def test_update_commitment(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_update_commitment", {
        "commitment_id": "abc-123",
        "status": "completed",
    })
    assert result.success is True
    payload = json.loads(result.output)
    assert payload["status"] == "completed"


@pytest.mark.asyncio
async def test_update_commitment_rejects_completed_software_work_without_evidence(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.commitment_store.get = AsyncMock(
        return_value=Commitment(
            content=(
                "Build and verify kPony Qt6 email client. Source code is written. "
                "Next action: run cmake configure and build."
            ),
            tags=["kpony", "software", "build"],
            meta={"project_type": "software"},
        )
    )
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_update_commitment",
        {"commitment_id": "abc-123", "status": "completed"},
    )

    assert result.success is False
    assert "completion_evidence" in result.output
    runtime.commitment_store.update_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_commitment_rejects_ongoing_income_support_without_evidence(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.commitment_store.get = AsyncMock(
        return_value=Commitment(
            content=(
                "Support Jarrod in developing a productive routine and a realistic "
                "non-employee income/business model"
            ),
            tags=["jarrod_support", "routine", "income", "business_model", "follow_through"],
            meta={"source": "workflow_create_commitment"},
        )
    )
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_update_commitment",
        {"commitment_id": "abc-123", "status": "completed"},
    )

    assert result.success is False
    assert "completion_evidence" in result.output
    runtime.commitment_store.update_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_commitment_allows_completed_software_work_with_positive_evidence(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.commitment_store.get = AsyncMock(
        return_value=Commitment(
            content="Build and verify kPony Qt6 email client with cmake.",
            tags=["software"],
            meta={"project_type": "software"},
        )
    )
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_update_commitment",
        {
            "commitment_id": "abc-123",
            "status": "completed",
            "completion_evidence": "cmake configure and build completed with exit code 0.",
        },
    )

    assert result.success is True
    runtime.commitment_store.update_status.assert_not_awaited()
    saved = runtime.commitment_store.save.call_args.args[0]
    assert saved.status == CommitmentStatus.COMPLETED
    assert saved.meta["completion_evidence"] == "cmake configure and build completed with exit code 0."
    assert saved.meta["completed_at"]


@pytest.mark.asyncio
async def test_update_commitment_completes_linked_active_work(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    commitment_id = "00000000-0000-0000-0000-000000000123"
    linked_work = WorkObject(
        stage=WorkStage.MICRO_TASK,
        content="Create writing project 4246 audiobook.",
        commitment_id=commitment_id,
    )
    runtime.commitment_store.get = AsyncMock(
        return_value=Commitment(
            commitment_id=commitment_id,
            content="Create writing project 4246 audiobook with EdgeTTS.",
            status=CommitmentStatus.ACTIVE,
            linked_work_ids=[str(linked_work.work_id)],
            tags=["writing"],
            meta={"project_type": "writing"},
        )
    )
    runtime.ctx.work_store.get = AsyncMock(return_value=linked_work)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_update_commitment",
        {
            "commitment_id": commitment_id,
            "status": "completed",
            "completion_evidence": "Final MP3, manifest, and build log verified.",
        },
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["linked_work_settled"] == [str(linked_work.work_id)]
    saved_work = runtime.ctx.work_store.save.call_args.args[0]
    assert saved_work.stage == WorkStage.ARTIFACT
    assert saved_work.blocked_by == []
    assert saved_work.meta["terminal_commitment_status"] == "completed"
    assert saved_work.meta["terminal_commitment_id"] == commitment_id
    runtime.executive.reconcile_work_update.assert_called_once_with(saved_work)


@pytest.mark.asyncio
async def test_update_commitment_settles_work_with_commitment_backpointer(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    commitment_id = "00000000-0000-0000-0000-000000000124"
    linked_work = WorkObject(
        stage=WorkStage.MICRO_TASK,
        content="Return to the current writing project.",
        commitment_id=commitment_id,
    )
    runtime.commitment_store.get = AsyncMock(
        return_value=Commitment(
            commitment_id=commitment_id,
            content="Return to the current writing project.",
            status=CommitmentStatus.ACTIVE,
            linked_work_ids=[],
            tags=["writing"],
            meta={"project_type": "writing"},
        )
    )
    runtime.ctx.work_store.list_by_commitment = AsyncMock(return_value=[linked_work])
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_update_commitment",
        {
            "commitment_id": commitment_id,
            "status": "abandoned",
        },
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["linked_work_settled"] == [str(linked_work.work_id)]
    saved_work = runtime.ctx.work_store.save.call_args.args[0]
    assert saved_work.stage == WorkStage.NOTE
    assert saved_work.meta["terminal_commitment_status"] == "abandoned"
    assert saved_work.meta["terminal_commitment_id"] == commitment_id
    runtime.executive.reconcile_work_update.assert_called_once_with(saved_work)


@pytest.mark.asyncio
async def test_update_commitment_completes_linked_active_schedules(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.commitment_store.get = AsyncMock(
        return_value=Commitment(
            content="Build and verify kPony Qt6 email client with cmake.",
            tags=["software"],
            meta={"project_type": "software"},
        )
    )
    schedule = SimpleNamespace(
        schedule_id="schedule-1",
        commitment_id="abc-123",
        status=ScheduleStatus.ACTIVE,
        next_run_at=datetime.now(timezone.utc) + timedelta(hours=1),
        meta={},
    )
    runtime.ctx.schedule_store = MagicMock()
    runtime.ctx.schedule_store.list_items = AsyncMock(return_value=[schedule])
    runtime.ctx.schedule_store.save = AsyncMock()
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_update_commitment",
        {
            "commitment_id": "abc-123",
            "status": "completed",
            "completion_evidence": "cmake configure and build completed with exit code 0.",
        },
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["linked_schedules_settled"] == ["schedule-1"]
    assert schedule.status == ScheduleStatus.COMPLETED
    assert schedule.next_run_at is None
    assert schedule.meta["settled_by_commitment_status"] == "completed"
    runtime.ctx.schedule_store.save.assert_awaited_once_with(schedule)


@pytest.mark.asyncio
async def test_update_commitment_invalid_status(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_update_commitment", {
        "commitment_id": "abc-123",
        "status": "exploded",
    })
    assert result.success is False
    assert "invalid" in result.output.lower()


@pytest.mark.asyncio
async def test_create_schedule_rejects_finished_commitment_link(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    future_start = datetime.now(timezone.utc) + timedelta(days=7)
    runtime.commitment_store.get = AsyncMock(
        return_value=Commitment(content="already done", status=CommitmentStatus.COMPLETED)
    )
    runtime.schedule_service = MagicMock()
    runtime.schedule_service.create_schedule = AsyncMock(
        return_value=SimpleNamespace(
            schedule_id="schedule-1",
            title="Continue work",
            kind=SimpleNamespace(value="task"),
            action=SimpleNamespace(value="submit_baa"),
            next_run_at=future_start,
        )
    )
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "Continue work",
            "start_at": future_start.isoformat(),
            "commitment_id": "commitment-1",
        },
    )

    assert result.success is False
    assert "completed commitment" in result.output.lower()
    runtime.schedule_service.create_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_schedule_preserves_project_return_authority_from_linked_commitment(
    tmp_path: Path,
) -> None:
    runtime = _make_mock_runtime(tmp_path)
    future_start = datetime.now(timezone.utc) + timedelta(minutes=10)
    runtime.schedule_service = MagicMock()
    runtime.schedule_service.create_schedule = AsyncMock(
        return_value=SimpleNamespace(
            schedule_id="schedule-1",
            title="Continue The Mercy of the Architect",
            kind=SimpleNamespace(value="task"),
            action=SimpleNamespace(value="submit_baa"),
            next_run_at=future_start,
            meta={},
        )
    )
    commitment = Commitment(
        content="Return to project: Writing Project 20260518-004126 — The Mercy of the Architect",
        status=CommitmentStatus.ACTIVE,
        tags=["project_return", "conversation", "self_directed"],
        meta={
            "source": "project_return_capture",
            "source_session_id": "telegram:private:8451790949:15",
            "project_key": "writing-project-20260518-004126-the-mercy-of-the-architect",
            "project_title": "Writing Project 20260518-004126 — The Mercy of the Architect",
            "project_type": "writing",
            "workspace_rel_path": "workspace/novels/writing-project-20260518-004126",
            "workspace_abs_path": str(
                tmp_path / "workspace" / "novels" / "writing-project-20260518-004126"
            ),
            "workspace_project_confidence": 1.0,
            "target_word_count": 100000,
            "project_intent": "revise and finish the manuscript until completion evidence exists",
        },
    )
    runtime.commitment_store.get = AsyncMock(return_value=commitment)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "Continue The Mercy of the Architect after Chapter 30",
            "description": "Return to canonical root and continue manuscript drafting.",
            "objective": "Continue the active writing project until explicit completion evidence exists.",
            "action": "submit_baa",
            "start_at": future_start.isoformat(),
            "commitment_id": str(commitment.commitment_id),
        },
    )

    assert result.success is True
    created_kwargs = runtime.schedule_service.create_schedule.await_args.kwargs
    assert "project_return" in created_kwargs["tags"]
    assert "self_directed" in created_kwargs["tags"]
    assert created_kwargs["meta"]["source_session_id"] == "telegram:private:8451790949:15"
    assert (
        created_kwargs["meta"]["workspace_rel_path"]
        == "workspace/novels/writing-project-20260518-004126"
    )
    assert created_kwargs["meta"]["workspace_project_confidence"] == 1.0
    assert created_kwargs["meta"]["target_word_count"] == 100000


@pytest.mark.asyncio
async def test_create_schedule_rejects_accidental_past_start(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.schedule_service = MagicMock()
    runtime.schedule_service.create_schedule = AsyncMock()
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "Continue writing later",
            "objective": "Draft the next chapter.",
            "start_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
        },
    )

    assert result.success is False
    assert "start_at is in the past" in result.output.lower()
    assert "choose a future time" in result.output.lower()
    runtime.schedule_service.create_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_schedule_rejects_reminder_only_for_unfinished_writing_return(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.schedule_service = MagicMock()
    runtime.schedule_service.create_schedule = AsyncMock()
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "writing project 4246 — Continue manuscript revision",
            "description": (
                "Return to the writing project 4246 manuscript. Next steps: prose-level "
                "revision of Chapters 1-2 and review thin spots in Chapters 4, 5, 7, 8."
            ),
            "action": "reminder_only",
            "start_at": (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(),
            "tags": ["writing-project-4246", "creative-writing", "manuscript-revision"],
        },
    )

    assert result.success is False
    assert "submit_baa" in result.output
    assert "unfinished writing/project return" in result.output.lower()
    runtime.schedule_service.create_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_schedule_rejects_far_future_active_project_return_without_delay_reason(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.schedule_service = MagicMock()
    runtime.schedule_service.create_schedule = AsyncMock()
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "Return to writing project 4246 Chapter 3 Expansion",
            "description": "Return to continue an unfinished active writing project.",
            "objective": "Persist expanded Chapter 3 prose and continue manuscript revision.",
            "start_at": (datetime.now(timezone.utc) + timedelta(days=90)).isoformat(),
            "priority": 9.0,
        },
    )

    assert result.success is False
    assert "too far in the future" in result.output.lower()
    assert "delay_reason" in result.output
    runtime.schedule_service.create_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_schedule_allows_far_future_active_project_return_with_delay_reason(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.schedule_service = MagicMock()
    runtime.schedule_service.create_schedule = AsyncMock(
        return_value=SimpleNamespace(
            schedule_id="schedule-1",
            title="Return to writing project 4246 Chapter 3 Expansion",
            kind=SimpleNamespace(value="task"),
            action=SimpleNamespace(value="submit_baa"),
            next_run_at=datetime.now(timezone.utc) + timedelta(days=90),
        )
    )
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "Return to writing project 4246 Chapter 3 Expansion",
            "description": "Return to continue an unfinished active writing project.",
            "objective": "Persist expanded Chapter 3 prose and continue manuscript revision.",
            "start_at": (datetime.now(timezone.utc) + timedelta(days=90)).isoformat(),
            "priority": 9.0,
            "delay_reason": "Wait for a planned consolidation window before returning.",
        },
    )

    assert result.success is True
    created_kwargs = runtime.schedule_service.create_schedule.await_args.kwargs
    assert created_kwargs["meta"]["delay_reason"] == "Wait for a planned consolidation window before returning."


@pytest.mark.asyncio
async def test_create_schedule_classifies_software_project(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.schedule_service = MagicMock()
    runtime.schedule_service.create_schedule = AsyncMock(
        return_value=SimpleNamespace(
            schedule_id="schedule-1",
            title="Build and verify kPony",
            kind=SimpleNamespace(value="task"),
            action=SimpleNamespace(value="submit_baa"),
            next_run_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
    )
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "Build and verify kPony",
            "objective": "Run cmake for the Qt6 email client and fix compile errors.",
            "start_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        },
    )

    assert result.success is True
    created_kwargs = runtime.schedule_service.create_schedule.await_args.kwargs
    assert created_kwargs["meta"]["project_type"] == "software"


@pytest.mark.asyncio
async def test_create_schedule_output_exposes_dedupe_metadata(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    run_at = datetime.now(timezone.utc) + timedelta(hours=1)
    runtime.schedule_service = MagicMock()
    runtime.schedule_service.create_schedule = AsyncMock(
        return_value=SimpleNamespace(
            schedule_id="schedule-1",
            title="Gmail DataAnnotations check",
            kind=SimpleNamespace(value="task"),
            action=SimpleNamespace(value="gmail_alert"),
            next_run_at=run_at,
            meta={
                "dedupe_action": "reused",
                "survivor_schedule_id": "schedule-1",
                "merged_schedule_ids": [],
                "duplicate_schedule_ids": ["schedule-2"],
            },
        )
    )
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "Gmail DataAnnotations check",
            "action": "gmail_alert",
            "start_at": run_at.isoformat(),
            "recurrence": "interval_hours",
            "interval_hours": 6,
            "meta": {
                "gmail_query": "from:(dataannotations OR dataannotation) newer_than:30d",
            },
        },
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["dedupe_action"] == "reused"
    assert payload["survivor_schedule_id"] == "schedule-1"
    assert payload["merged_schedule_ids"] == []
    assert payload["duplicate_schedule_ids"] == ["schedule-2"]
    assert result.metadata["dedupe_action"] == "reused"


@pytest.mark.asyncio
async def test_cancel_schedule_marks_schedule_cancelled(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    schedule = SimpleNamespace(
        schedule_id="schedule-1",
        status=ScheduleStatus.ACTIVE,
        next_run_at=datetime.now(timezone.utc) + timedelta(hours=1),
        title="Daily writing block",
        meta={},
    )
    runtime.ctx.schedule_store = MagicMock()
    runtime.ctx.schedule_store.get = AsyncMock(return_value=schedule)
    runtime.ctx.schedule_store.save = AsyncMock()
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_cancel_schedule",
        {"schedule_id": "schedule-1", "reason": "Operator cancelled the future work."},
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload == {
        "schedule_id": "schedule-1",
        "cancelled": True,
        "status": "cancelled",
    }
    assert schedule.status == ScheduleStatus.CANCELLED
    assert schedule.next_run_at is None
    assert schedule.meta["cancel_reason"] == "Operator cancelled the future work."
    runtime.ctx.schedule_store.save.assert_awaited_once_with(schedule)


@pytest.mark.asyncio
async def test_create_schedule_rejects_writing_return_outside_managed_workspace(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.schedule_service = MagicMock()
    runtime.schedule_service.create_schedule = AsyncMock()
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "Return to writing project 4246 Chapter 3 Expansion",
            "description": "Return to continue an unfinished active writing project.",
            "objective": (
                "Persist expanded prose to "
                f"{tmp_path}/writing/4246/story_4246.md before continuing."
            ),
            "start_at": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
            "priority": 9.0,
        },
    )

    assert result.success is False
    assert "outside managed workspace root" in result.output.lower()
    runtime.schedule_service.create_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_list_commitments(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    c = Commitment(content="Test commitment", priority=7.0, tags=["test"])
    runtime.commitment_store.list_by_status = AsyncMock(return_value=[c])
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_list_commitments", {"status": "active"})
    assert result.success is True
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["items"][0]["content"] == "Test commitment"


@pytest.mark.asyncio
async def test_get_commitment_returns_detail(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    commitment = Commitment(
        content="Finish the tool companion audit",
        priority=8.0,
        tags=["tools"],
        meta={"source": "unit-test"},
    )
    runtime.commitment_store.get = AsyncMock(return_value=commitment)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_get_commitment",
        {"commitment_id": str(commitment.commitment_id)},
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["found"] is True
    assert payload["commitment"]["commitment_id"] == str(commitment.commitment_id)
    assert payload["commitment"]["content"] == "Finish the tool companion audit"
    assert payload["commitment"]["meta"] == {"source": "unit-test"}


@pytest.mark.asyncio
async def test_get_schedule_returns_detail_and_recent_runs(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    schedule = ScheduleItem(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Return to project",
        description="Pick up the work",
        objective="Continue implementation",
        start_at=datetime.now(timezone.utc) + timedelta(hours=1),
        recurrence=ScheduleRecurrence.DAILY,
        tags=["return"],
        meta={"source": "unit-test"},
    )
    run = ScheduleRun(
        schedule_id=schedule.schedule_id,
        scheduled_for=schedule.start_at,
        status=ScheduleRunStatus.SUBMITTED,
        task_id="task-001",
    )
    runtime.ctx.schedule_store = MagicMock()
    runtime.ctx.schedule_store.get = AsyncMock(return_value=schedule)
    runtime.ctx.schedule_store.list_runs = AsyncMock(return_value=[run])
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_get_schedule",
        {"schedule_id": str(schedule.schedule_id), "run_limit": 5},
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["found"] is True
    assert payload["schedule"]["schedule_id"] == str(schedule.schedule_id)
    assert payload["schedule"]["objective"] == "Continue implementation"
    assert payload["schedule"]["meta"] == {"source": "unit-test"}
    assert payload["recent_runs"][0]["task_id"] == "task-001"
    runtime.ctx.schedule_store.list_runs.assert_awaited_once_with(
        schedule_id=str(schedule.schedule_id),
        limit=5,
    )


@pytest.mark.asyncio
async def test_create_writing_task(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_create_writing_task", {
        "title": "Architecture Overview",
        "description": "High-level system design document",
        "outline": ["Introduction", "Components", "Data Flow", "Conclusion"],
    })
    assert result.success is True
    payload = json.loads(result.output)
    assert payload["scaffold_written"] is True
    assert payload["plan_id"].startswith("plan-")
    assert payload["managed_workspace_root"] == str(tmp_path / "workspace")
    runtime.ctx.plan_store.set_status.assert_awaited_once_with(payload["plan_id"], "active")

    # Verify file was written
    output_path = Path(payload["output_path"])
    assert output_path.exists()
    assert output_path.is_relative_to(tmp_path / "workspace")
    content = output_path.read_text()
    assert "# Architecture Overview" in content
    assert "## Introduction" in content
    assert "## Conclusion" in content


@pytest.mark.asyncio
async def test_create_writing_task_records_mutation_provenance(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_create_writing_task", {
        "title": "Architecture Overview",
        "description": "High-level system design document",
        "outline": ["Introduction"],
    })
    assert result.success is True
    payload = json.loads(result.output)

    records_path = tmp_path / "provenance.transitions.jsonl"
    records = [
        ps.parse_provenance_transition(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = records[-1]

    assert record.kind == ps.ProvenanceTransitionKind.MUTATION
    assert record.status == "mutated"
    assert record.details["source_artifact"] == "workflow|writing-task|Architecture Overview"
    assert record.details["trigger_action"] == "workflow_create_writing_task"
    assert record.details["target_entity"] == (
        Path(payload["output_path"]).relative_to(tmp_path / "workspace").as_posix()
    )
    assert record.details["origin_action_id"] == payload["output_path"]
    assert record.details["parent_transition_id"] == payload["output_path"]
    assert record.details["linked_transition_ids"] == [payload["output_path"], record.details["target_entity"]]


@pytest.mark.asyncio
async def test_create_writing_task_scaffold_failure_creates_no_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    def _raise_write_error(self: Path, *_args, **_kwargs) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", _raise_write_error)

    result = await adapter(
        "workflow_create_writing_task",
        {
            "title": "Scaffold Failure",
            "outline": ["Summary"],
        },
    )

    assert result.success is False
    assert not (tmp_path / "workspace" / "notes" / "scaffold_failure.md").exists()
    runtime.commitment_store.save.assert_not_awaited()
    runtime.ctx.plan_store.create_plan.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_writing_task_plan_failure_keeps_scaffold_and_commitment(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.ctx.plan_store.create_plan = AsyncMock(side_effect=RuntimeError("plan db down"))
    adapter = WorkflowToolAdapter(runtime)

    caplog.set_level("WARNING", logger="opencas.tools.adapters.workflow_tasking")

    result = await adapter(
        "workflow_create_writing_task",
        {
            "title": "Plan Failure",
            "outline": ["Summary"],
        },
    )

    assert result.success is True
    payload = json.loads(result.output)
    output_path = Path(payload["output_path"])
    assert output_path.exists()
    assert payload["commitment_id"] is not None
    assert payload["plan_id"] is None
    runtime.commitment_store.save.assert_awaited_once()
    runtime.ctx.plan_store.set_status.assert_not_awaited()
    assert "Failed to persist writing-task plan" in caplog.text


@pytest.mark.asyncio
async def test_create_writing_task_custom_path(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)
    custom = "docs/custom.md"

    result = await adapter("workflow_create_writing_task", {
        "title": "Custom Doc",
        "output_path": custom,
    })
    assert result.success is True
    assert Path(json.loads(result.output)["output_path"]) == (tmp_path / "workspace" / custom).resolve()


@pytest.mark.asyncio
async def test_create_writing_task_rejects_path_outside_managed_workspace(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)
    outside = tmp_path.parent / "outside.md"

    result = await adapter("workflow_create_writing_task", {
        "title": "Outside Doc",
        "output_path": str(outside),
    })

    assert result.success is False
    assert "managed workspace root" in result.output


@pytest.mark.asyncio
async def test_create_plan(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_create_plan", {
        "content": "Step 1: Do X\nStep 2: Do Y",
    })
    assert result.success is True
    payload = json.loads(result.output)
    assert payload["plan_id"].startswith("plan-")
    assert payload["status"] == "active"
    runtime.ctx.plan_store.set_status.assert_awaited_once_with(payload["plan_id"], "active")


@pytest.mark.asyncio
async def test_update_plan(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_update_plan", {
        "plan_id": "plan-001",
        "content": "Revised step 1: Do Z",
    })
    assert result.success is True
    payload = json.loads(result.output)
    assert payload["updated"] is True


@pytest.mark.asyncio
async def test_list_plans(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    plan = PlanEntry(
        plan_id="plan-001",
        status="active",
        content="Step 1\nStep 2",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        project_id="project-001",
    )
    runtime.ctx.plan_store.list_active = AsyncMock(return_value=[plan])
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_list_plans", {"project_id": "project-001", "limit": 10})

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["items"][0]["plan_id"] == "plan-001"
    assert payload["items"][0]["content_preview"] == "Step 1\nStep 2"
    runtime.ctx.plan_store.list_active.assert_awaited_once_with(
        project_id="project-001",
        task_id=None,
    )


@pytest.mark.asyncio
async def test_get_plan_returns_actions(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    now = datetime.now(timezone.utc)
    plan = PlanEntry(
        plan_id="plan-001",
        status="active",
        content="Step 1\nStep 2",
        created_at=now,
        updated_at=now,
        task_id="task-001",
    )
    action = PlanAction(
        action_id="action-001",
        plan_id="plan-001",
        tool_name="fs_read_file",
        args={"file_path": "README.md"},
        result_summary="read ok",
        success=True,
        timestamp=now,
    )
    runtime.ctx.plan_store.get_plan = AsyncMock(return_value=plan)
    runtime.ctx.plan_store.get_actions = AsyncMock(return_value=[action])
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_get_plan", {"plan_id": "plan-001", "action_limit": 3})

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["found"] is True
    assert payload["plan"]["plan_id"] == "plan-001"
    assert payload["plan"]["content"] == "Step 1\nStep 2"
    assert payload["actions"][0]["tool_name"] == "fs_read_file"
    runtime.ctx.plan_store.get_actions.assert_awaited_once_with("plan-001", limit=3)


@pytest.mark.asyncio
async def test_list_tasks(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    task = RepairTask(
        objective="Fix missing companion workflow tools",
        stage=ExecutionStage.PLANNING,
        status="planning",
        meta={"title": "Tool audit"},
        project_id="project-001",
    )
    runtime.ctx.tasks = MagicMock()
    runtime.ctx.tasks.list_all = AsyncMock(return_value=[task])
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_list_tasks", {"limit": 5})

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["count"] == 1
    assert payload["items"][0]["task_id"] == str(task.task_id)
    assert payload["items"][0]["objective"] == "Fix missing companion workflow tools"
    runtime.ctx.tasks.list_all.assert_awaited_once_with(limit=5)


@pytest.mark.asyncio
async def test_get_task_returns_detail(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    task = RepairTask(
        objective="Fix missing companion workflow tools",
        stage=ExecutionStage.EXECUTING,
        status="running",
        meta={"title": "Tool audit"},
        commitment_id="commitment-001",
    )
    runtime.ctx.tasks = MagicMock()
    runtime.ctx.tasks.get = AsyncMock(return_value=task)
    runtime.ctx.tasks.get_result = AsyncMock(return_value=None)
    runtime.ctx.tasks.list_lifecycle_transitions = AsyncMock(return_value=[
        {
            "from_stage": "planning",
            "to_stage": "executing",
            "reason": "plan accepted",
            "timestamp": datetime.now(timezone.utc),
            "context": {"source": "unit-test"},
        }
    ])
    runtime.ctx.tasks.list_salvage_packets = AsyncMock(return_value=[])
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_get_task", {"task_id": str(task.task_id)})

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["found"] is True
    assert payload["task"]["task_id"] == str(task.task_id)
    assert payload["task"]["commitment_id"] == "commitment-001"
    assert payload["transitions"][0]["to_stage"] == "executing"


@pytest.mark.asyncio
async def test_repo_triage(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.execute_tool = AsyncMock(return_value={
        "success": True,
        "output": "M  file.py\n",
        "metadata": {},
    })
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_repo_triage", {})
    assert result.success is True
    payload = json.loads(result.output)
    assert "workspace" in payload
    assert "git_status" in payload
    assert "work_items" in payload
    assert "active_commitments" in payload


@pytest.mark.asyncio
async def test_supervise_session(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.execute_tool = AsyncMock(side_effect=[
        {
            "success": True,
            "output": json.dumps({
                "session_id": "sess-001",
                "running": True,
                "cleaned_combined_output": "ready",
                "screen_state": {"app": "claude", "mode": "interactive", "ready_for_input": True},
                "idle_reached": True,
                "elapsed_ms": 300,
            }),
            "metadata": {},
        },
        {
            "success": True,
            "output": json.dumps({
                "session_id": "sess-001",
                "running": True,
                "cleaned_combined_output": "hello from claude",
                "screen_state": {"app": "claude", "mode": "interactive", "ready_for_input": True},
                "idle_reached": True,
                "elapsed_ms": 500,
            }),
            "metadata": {},
        },
    ])
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_supervise_session", {
        "command": "claude",
        "task": "Hello",
        "max_rounds": 1,
    })
    assert result.success is True
    payload = json.loads(result.output)
    assert payload["session_id"] == "sess-001"
    assert payload["cleaned_output"] == "hello from claude"
    assert payload["screen_state"]["app"] == "claude"
    assert payload["supervision_advisory"]["action"] == "observe_briefly"
    assert payload["rounds_used"] == 2
    assert runtime.execute_tool.await_count == 2
    first_args = runtime.execute_tool.await_args_list[0].args[1]
    second_args = runtime.execute_tool.await_args_list[1].args[1]
    assert "input" not in first_args
    assert second_args["input"] == "Hello\r"
    assert second_args["session_id"] == "sess-001"


@pytest.mark.asyncio
async def test_supervise_session_can_verify_file_across_rounds(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    expected = tmp_path / "note.md"

    async def _fake_execute_tool(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "pty_interact" and "command" in args:
            return {
                "success": True,
                "output": json.dumps({
                    "session_id": "sess-verify",
                    "running": True,
                    "cleaned_combined_output": "ready",
                    "screen_state": {"app": "kilocode", "mode": "interactive", "ready_for_input": True},
                    "idle_reached": True,
                }),
                "metadata": {},
            }
        if name == "pty_interact":
            return {
                "success": True,
                "output": json.dumps({
                    "session_id": "sess-verify",
                    "running": True,
                    "cleaned_combined_output": "working",
                    "screen_state": {"app": "kilocode", "mode": "interactive", "ready_for_input": True},
                    "idle_reached": True,
                }),
                "metadata": {},
            }
        if name == "pty_observe":
            expected.write_text("done", encoding="utf-8")
            return {
                "success": True,
                "output": json.dumps({
                    "session_id": "sess-verify",
                    "running": True,
                    "cleaned_combined_output": "done",
                    "screen_state": {"app": "kilocode", "mode": "interactive", "ready_for_input": True},
                    "idle_reached": True,
                }),
                "metadata": {},
            }
        raise AssertionError(f"unexpected tool call: {name}")

    runtime.execute_tool = AsyncMock(side_effect=_fake_execute_tool)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_supervise_session", {
        "command": "kilocode",
        "task": "Create the note",
        "verification_path": str(expected),
        "max_rounds": 3,
    })
    assert result.success is True
    payload = json.loads(result.output)
    assert payload["verification_exists"] is True
    assert payload["session_id"] == "sess-verify"
    assert payload["screen_state"]["app"] == "kilocode"
    assert payload["supervision_advisory"]["reason"] == "verification_satisfied"
    assert payload["rounds_used"] == 3
    assert runtime.execute_tool.await_count == 3


@pytest.mark.asyncio
async def test_supervise_session_stops_on_auth_gate_before_submit(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.execute_tool = AsyncMock(return_value={
        "success": True,
        "output": json.dumps({
            "session_id": "sess-auth",
            "running": True,
            "cleaned_combined_output": "Please sign in to continue",
            "screen_state": {
                "app": "kilocode",
                "mode": "auth_required",
                "ready_for_input": False,
                "blocked": True,
            },
            "idle_reached": True,
        }),
        "metadata": {},
    })
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_supervise_session", {
        "command": "kilocode",
        "task": "Create the note",
        "max_rounds": 3,
    })

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["session_id"] == "sess-auth"
    assert payload["supervision_advisory"]["reason"] == "auth_or_gate_blocked"
    assert payload["rounds_used"] == 1
    runtime.execute_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_supervise_session_records_waiting_provenance_when_blocked(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.execute_tool = AsyncMock(return_value={
        "success": True,
        "output": json.dumps({
            "session_id": "sess-auth",
            "running": True,
            "cleaned_combined_output": "Please sign in to continue",
            "screen_state": {
                "app": "kilocode",
                "mode": "auth_required",
                "ready_for_input": False,
                "blocked": True,
            },
            "idle_reached": True,
        }),
        "metadata": {},
    })
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_supervise_session", {
        "command": "kilocode",
        "task": "Create the note",
        "max_rounds": 3,
    })

    assert result.success is True
    records_path = tmp_path / "provenance.transitions.jsonl"
    records = [
        ps.parse_provenance_transition(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = records[-1]

    assert record.kind == ps.ProvenanceTransitionKind.WAITING
    assert record.status == "blocked"
    assert record.details["source_artifact"] == "workflow|supervision|workflow-supervision"
    assert record.details["trigger_action"] == "workflow_supervise_session"
    assert record.details["target_entity"] == "sess-auth"
    assert record.details["origin_action_id"] == "sess-auth"
    assert record.details["parent_transition_id"] == "sess-auth"
    assert record.details["linked_transition_ids"] == ["sess-auth"]


@pytest.mark.asyncio
async def test_supervise_session_uses_adaptive_observe_waits(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)

    async def _fake_execute_tool(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "pty_interact" and "command" in args:
            return {
                "success": True,
                "output": json.dumps({
                    "session_id": "sess-adaptive",
                    "running": True,
                    "cleaned_combined_output": "ready",
                    "screen_state": {"app": "kilocode", "mode": "interactive", "ready_for_input": True},
                    "idle_reached": True,
                }),
                "metadata": {},
            }
        if name == "pty_interact":
            return {
                "success": True,
                "output": json.dumps({
                    "session_id": "sess-adaptive",
                    "running": True,
                    "cleaned_combined_output": "awaiting follow-up",
                    "screen_state": {"app": "kilocode", "mode": "interactive", "ready_for_input": True},
                    "idle_reached": True,
                }),
                "metadata": {},
            }
        if name == "pty_observe":
            assert args["idle_seconds"] == pytest.approx(0.35)
            assert args["max_wait_seconds"] == pytest.approx(2.5)
            return {
                "success": True,
                "output": json.dumps({
                    "session_id": "sess-adaptive",
                    "running": True,
                    "cleaned_combined_output": "still awaiting follow-up",
                    "screen_state": {"app": "kilocode", "mode": "interactive", "ready_for_input": True},
                    "idle_reached": True,
                }),
                "metadata": {},
            }
        raise AssertionError(f"unexpected tool call: {name}")

    runtime.execute_tool = AsyncMock(side_effect=_fake_execute_tool)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_supervise_session", {
        "command": "kilocode",
        "task": "Create the note",
        "idle_seconds": 1.0,
        "continue_wait_seconds": 10.0,
        "max_rounds": 3,
    })

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["supervision_advisory"]["action"] == "observe_briefly"
    assert payload["rounds_used"] == 4


@pytest.mark.asyncio
async def test_supervise_session_sends_single_enter_follow_up_for_stalled_kilocode(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    expected = tmp_path / "note.md"

    async def _fake_execute_tool(name: str, args: dict[str, object]) -> dict[str, object]:
        if name == "pty_interact" and "command" in args:
            return {
                "success": True,
                "output": json.dumps({
                    "session_id": "sess-kilo-follow-up",
                    "running": True,
                    "cleaned_combined_output": "ready",
                    "screen_state": {"app": "kilocode", "mode": "interactive", "ready_for_input": True},
                    "idle_reached": True,
                }),
                "metadata": {},
            }
        if name == "pty_interact" and args.get("input") == "Create the note\r":
            return {
                "success": True,
                "output": json.dumps({
                    "session_id": "sess-kilo-follow-up",
                    "running": True,
                    "cleaned_combined_output": "composer staged",
                    "screen_state": {"app": "kilocode", "mode": "interactive", "ready_for_input": True},
                    "idle_reached": True,
                }),
                "metadata": {},
            }
        if name == "pty_observe":
            return {
                "success": True,
                "output": json.dumps({
                    "session_id": "sess-kilo-follow-up",
                    "running": True,
                    "cleaned_combined_output": "composer staged",
                    "screen_state": {"app": "kilocode", "mode": "interactive", "ready_for_input": True},
                    "idle_reached": True,
                }),
                "metadata": {},
            }
        if name == "pty_interact" and args.get("input") == "\r":
            expected.write_text("done", encoding="utf-8")
            return {
                "success": True,
                "output": json.dumps({
                    "session_id": "sess-kilo-follow-up",
                    "running": True,
                    "cleaned_combined_output": "submitted",
                    "screen_state": {"app": "kilocode", "mode": "interactive", "ready_for_input": False},
                    "idle_reached": True,
                }),
                "metadata": {},
            }
        raise AssertionError(f"unexpected tool call: {name} {args}")

    runtime.execute_tool = AsyncMock(side_effect=_fake_execute_tool)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_supervise_session", {
        "command": "kilocode",
        "task": "Create the note",
        "verification_path": str(expected),
        "max_rounds": 4,
    })

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["verification_exists"] is True
    assert payload["supervision_advisory"]["reason"] == "verification_satisfied"
    assert payload["rounds_used"] == 4
    assert runtime.execute_tool.await_args_list[3].args[1]["input"] == "\r"


@pytest.mark.asyncio
async def test_supervise_session_missing_args(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_supervise_session", {})
    assert result.success is False


@pytest.mark.asyncio
async def test_unknown_tool(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter("workflow_nonexistent", {})
    assert result.success is False
    assert "Unknown" in result.output
