"""Tests for project lifecycle cancel/compost operations."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.commitment_store import CommitmentStore
from opencas.execution.models import ExecutionStage, RepairTask
from opencas.execution.store import TaskStore
from opencas.governance import (
    BlockReason,
    DecompositionStage,
    ShadowRegistry,
    ShadowRegistryStore,
)
from opencas.projects.lifecycle import cancel_project, cancel_task
from opencas.scheduling import ScheduleAction, ScheduleKind, ScheduleService, ScheduleStatus, ScheduleStore


class _Config:
    def __init__(self, root):
        self.state_dir = root / ".opencas"
        self._root = root
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def primary_workspace_root(self):
        return self._root

    def agent_workspace_root(self):
        return self._root / "workspace"


@pytest_asyncio.fixture
async def lifecycle_runtime(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    schedule_store = ScheduleStore(tmp_path / "schedules.db")
    await schedule_store.connect()
    task_store = TaskStore(tmp_path / "tasks.db")
    await task_store.connect()
    runtime = SimpleNamespace(
        commitment_store=commitment_store,
        schedule_service=ScheduleService(schedule_store),
        ctx=SimpleNamespace(
            config=_Config(tmp_path),
            schedule_store=schedule_store,
            shadow_registry=ShadowRegistry(ShadowRegistryStore(tmp_path / "shadow_registry")),
            task_store=task_store,
        ),
        executive=SimpleNamespace(remove_goal=lambda _goal: None),
        _trace=lambda *_args, **_kwargs: None,
    )
    try:
        yield runtime
    finally:
        await task_store.close()
        await schedule_store.close()
        await commitment_store.close()


@pytest.mark.asyncio
async def test_cancel_project_composts_workspace_and_stops_linked_work(lifecycle_runtime) -> None:
    project_dir = lifecycle_runtime.ctx.config.agent_workspace_root() / "kPony"
    project_dir.mkdir()
    (project_dir / "README.md").write_text("# kPony\n", encoding="utf-8")

    commitment = Commitment(
        content="Return to project: kPony",
        tags=["project_return", "self_directed"],
        meta={"source": "project_return_capture", "project_key": "kpony", "project_title": "kPony"},
    )
    await lifecycle_runtime.commitment_store.save(commitment)
    schedule = await lifecycle_runtime.schedule_service.create_schedule(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        title="Return to kPony",
        objective="Continue kPony.",
        start_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        commitment_id=str(commitment.commitment_id),
        meta={"source": "project_return_capture", "project_key": "kpony", "project_title": "kPony"},
    )
    task = RepairTask(
        objective="Return to project: kPony",
        commitment_id=str(commitment.commitment_id),
        meta={"source": "project_return_capture", "project_key": "kpony", "project_title": "kPony"},
    )
    await lifecycle_runtime.ctx.task_store.save(task)

    result = await cancel_project(
        lifecycle_runtime,
        project_key="kpony",
        project_title="kPony",
        workspace_path="kPony",
        reason="broken first attempt",
    )

    assert result.success is True
    assert result.project_key == "kpony"
    assert project_dir.exists() is False
    assert result.compost_path is not None
    assert result.compost_path.exists()
    assert (result.compost_path / "README.md").read_text(encoding="utf-8") == "# kPony\n"
    assert result.receipt_path is not None
    assert result.receipt_path.exists()
    assert result.salvage_index_path is not None
    assert result.salvage_index_path.exists()
    assert result.shadow_registry_id is not None

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["shadow_registry_id"] == result.shadow_registry_id
    assert "README.md" in receipt["salvage_candidates"]
    shadow_item = lifecycle_runtime.ctx.shadow_registry.store.get(result.shadow_registry_id)
    assert shadow_item is not None
    assert shadow_item.block_reason == BlockReason.PROJECT_COMPOSTED
    assert shadow_item.decomposition_stage == DecompositionStage.FERMENTING
    assert shadow_item.raw_parameters["project_key"] == "kpony"
    assert shadow_item.raw_parameters["salvage_candidates"] == ["README.md"]

    updated_commitment = await lifecycle_runtime.commitment_store.get(str(commitment.commitment_id))
    assert updated_commitment is not None
    assert updated_commitment.status == CommitmentStatus.ABANDONED
    assert updated_commitment.meta["cancel_reason"] == "broken first attempt"

    updated_schedule = await lifecycle_runtime.ctx.schedule_store.get(str(schedule.schedule_id))
    assert updated_schedule is not None
    assert updated_schedule.status == ScheduleStatus.CANCELLED
    assert updated_schedule.next_run_at is None

    updated_task = await lifecycle_runtime.ctx.task_store.get(str(task.task_id))
    assert updated_task is not None
    assert updated_task.stage == ExecutionStage.FAILED
    assert updated_task.status == "cancelled"
    assert updated_task.meta["cancel_reason"] == "broken first attempt"


@pytest.mark.asyncio
async def test_cancel_task_can_soft_cancel_or_delete(lifecycle_runtime) -> None:
    task = RepairTask(objective="Bad task")
    await lifecycle_runtime.ctx.task_store.save(task)

    soft = await cancel_task(lifecycle_runtime, task_id=str(task.task_id), reason="wrong objective")

    assert soft.success is True
    updated = await lifecycle_runtime.ctx.task_store.get(str(task.task_id))
    assert updated is not None
    assert updated.stage == ExecutionStage.FAILED
    assert updated.status == "cancelled"

    hard = await cancel_task(lifecycle_runtime, task_id=str(task.task_id), reason="discard", hard_delete=True)

    assert hard.success is True
    assert await lifecycle_runtime.ctx.task_store.get(str(task.task_id)) is None
