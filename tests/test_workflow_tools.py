"""Tests for higher-level operator workflow tools."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from opencas.api import provenance_store as ps
from opencas.autonomy.commitment import Commitment, CommitmentStatus
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
    runtime.commitment_store = commitment_store

    # Work store
    work_store = MagicMock()
    work_store.summary_counts = AsyncMock(return_value={"total": 0, "ready": 0, "blocked": 0})
    runtime.ctx.work_store = work_store

    # Plan store
    plan_store = MagicMock()
    plan_entry = MagicMock()
    plan_entry.plan_id = "plan-001"
    plan_entry.status = "active"
    plan_store.create_plan = AsyncMock(return_value=plan_entry)
    plan_store.set_status = AsyncMock(return_value=True)
    plan_store.update_content = AsyncMock(return_value=True)
    plan_store.count_active = AsyncMock(return_value=0)
    runtime.ctx.plan_store = plan_store

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
            next_run_at=datetime(2026, 5, 1, 10, tzinfo=timezone.utc),
        )
    )
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "Continue work",
            "start_at": "2026-05-01T10:00:00+00:00",
            "commitment_id": "commitment-1",
        },
    )

    assert result.success is False
    assert "completed commitment" in result.output.lower()
    runtime.schedule_service.create_schedule.assert_not_awaited()


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
            "title": "Chronicle 4246 — Continue manuscript revision",
            "description": (
                "Return to the Chronicle 4246 manuscript. Next steps: prose-level "
                "revision of Chapters 1-2 and review thin spots in Chapters 4, 5, 7, 8."
            ),
            "action": "reminder_only",
            "start_at": (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat(),
            "tags": ["chronicle-4246", "creative-writing", "manuscript-revision"],
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
            "title": "Return to Chronicle 4246 Chapter 3 Expansion",
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
            title="Return to Chronicle 4246 Chapter 3 Expansion",
            kind=SimpleNamespace(value="task"),
            action=SimpleNamespace(value="submit_baa"),
            next_run_at=datetime.now(timezone.utc) + timedelta(days=90),
        )
    )
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "Return to Chronicle 4246 Chapter 3 Expansion",
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
async def test_create_schedule_rejects_writing_return_outside_managed_workspace(tmp_path: Path) -> None:
    runtime = _make_mock_runtime(tmp_path)
    runtime.schedule_service = MagicMock()
    runtime.schedule_service.create_schedule = AsyncMock()
    adapter = WorkflowToolAdapter(runtime)

    result = await adapter(
        "workflow_create_schedule",
        {
            "title": "Return to Chronicle 4246 Chapter 3 Expansion",
            "description": "Return to continue an unfinished active writing project.",
            "objective": (
                "Persist expanded prose to "
                f"{tmp_path}/Chronicles/4246/chronicle_4246.md before continuing."
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
