from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.commitment_store import CommitmentStore
from opencas.execution.receipt_store import ExecutionReceiptStore
from opencas.execution.store import TaskStore
from opencas.memory.models import Episode, EpisodeKind
from opencas.memory.store import MemoryStore
from opencas.planning.store import PlanStore
from opencas.scheduling.store import ScheduleStore
from opencas.workspace.artifact_lookup import ArtifactLookupService
from opencas.workspace.service import WorkspaceIndexService
from opencas.workspace.store import WorkspaceStore


async def _stores(tmp_path: Path, target: Path, content: str):
    state = tmp_path / ".opencas"
    workspace_store = await WorkspaceStore(state / "workspace.db").connect()
    service = WorkspaceIndexService(
        store=workspace_store,
        embeddings_client=None,
        llm_client=None,
        workspace_roots=[tmp_path / "workspace"],
        llm_model="test",
        embedding_model="test",
    )
    await service.record_write(target)

    memory_store = await MemoryStore(state / "memory.db").connect()
    action = Episode(
        kind=EpisodeKind.ACTION,
        created_at=datetime(2026, 5, 1, 18, 3, 33, tzinfo=timezone.utc),
        content=f"tool fs_write_file path={target} bytes={len(content)} checksum={hashlib.sha256(content.encode()).hexdigest()}",
        payload={"tool_name": "fs_write_file", "args": {"file_path": str(target)}},
    )
    turn = Episode(
        kind=EpisodeKind.TURN,
        created_at=datetime(2026, 5, 1, 18, 4, 33, tzinfo=timezone.utc),
        content=f"asking about {target}",
    )
    await memory_store.save_episode(action)
    await memory_store.save_episode(turn)

    schedule_store = await ScheduleStore(state / "schedules.db").connect()
    await schedule_store._db.execute(  # type: ignore[union-attr]
        """
        INSERT INTO schedules (
            schedule_id, created_at, updated_at, kind, action, status, title,
            description, objective, start_at, timezone, recurrence, tags, meta
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "schedule-1",
            "2026-05-01T18:00:00+00:00",
            "2026-05-01T18:00:00+00:00",
            "task",
            "submit_baa",
            "active",
            "If I Am The Operator - Expansion Block",
            "",
            f"write {target}",
            "2026-05-01T18:00:00+00:00",
            "America/Denver",
            "none",
            "[]",
            "{}",
        ),
    )
    await schedule_store._db.execute(  # type: ignore[union-attr]
        """
        INSERT INTO schedule_runs (
            run_id, schedule_id, scheduled_for, started_at, status, task_id, meta
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "run-1",
            "schedule-1",
            "2026-05-01T18:00:00+00:00",
            "2026-05-01T18:00:01+00:00",
            "success",
            "task-1",
            json.dumps({"result_artifacts": [str(target)]}),
        ),
    )
    await schedule_store._db.commit()  # type: ignore[union-attr]

    task_store = await TaskStore(state / "tasks.db").connect()
    await task_store._db.execute(  # type: ignore[union-attr]
        """
        INSERT INTO tasks (
            task_id, created_at, updated_at, objective, stage, status,
            artifacts, meta, phases, convergence_hashes, depends_on,
            result_output, result_timestamp, result_stage, success
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task-1",
            "2026-05-01T18:00:01+00:00",
            "2026-05-01T18:03:40+00:00",
            f"write manuscript to {target}",
            "done",
            "completed",
            json.dumps([str(target)]),
            "{}",
            "[]",
            "[]",
            "[]",
            f"wrote {target}",
            "2026-05-01T18:03:40+00:00",
            "done",
            1,
        ),
    )
    await task_store._db.commit()  # type: ignore[union-attr]

    receipt_store = await ExecutionReceiptStore(state / "receipts.db").connect()
    await receipt_store._db.execute(  # type: ignore[union-attr]
        """
        INSERT INTO receipts (
            receipt_id, task_id, objective, phases, created_at, success, output
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "receipt-1",
            "task-1",
            f"receipt for {target}",
            "[]",
            "2026-05-01T18:03:41+00:00",
            1,
            f"output path {target}",
        ),
    )
    await receipt_store._db.commit()  # type: ignore[union-attr]

    commitment_store = await CommitmentStore(state / "commitments.db").connect()
    commitment = Commitment(
        commitment_id=UUID("11111111-1111-4111-8111-111111111111"),
        created_at=datetime(2026, 5, 1, 17, 55, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 1, 17, 55, tzinfo=timezone.utc),
        content=f"finish {target}",
        status=CommitmentStatus.ACTIVE,
    )
    await commitment_store.save(commitment)

    plan_store = await PlanStore(state / "plans.db").connect()
    await plan_store.create_plan("plan-1", content=f"Plan around {target}", task_id="task-1")

    provenance_path = state / "provenance.transitions.jsonl"
    provenance_path.write_text(
        json.dumps(
            {
                "created_at": "2026-05-01T18:02:00+00:00",
                "event_type": "MUTATION",
                "triggering_artifact": f"file|workspace|{target.name}",
                "target_entity": str(target),
                "source_artifact": "writing-task",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "workspace_store": workspace_store,
        "memory_store": memory_store,
        "schedule_store": schedule_store,
        "task_store": task_store,
        "receipt_store": receipt_store,
        "commitment_store": commitment_store,
        "plan_store": plan_store,
        "provenance_path": provenance_path,
    }


@pytest.mark.asyncio
async def test_artifact_lookup_joins_workspace_memory_and_workflow_stores(tmp_path: Path) -> None:
    target = tmp_path / "workspace" / "if_i_am_the_operator" / "manuscript_draft.md"
    target.parent.mkdir(parents=True)
    content = "operator manuscript"
    target.write_text(content, encoding="utf-8")
    stores = await _stores(tmp_path, target, content)

    result = await ArtifactLookupService(**stores).lookup(path=str(target))

    assert result.current["exists_on_disk"] is True
    assert result.current["checksum"] == hashlib.sha256(content.encode()).hexdigest()
    sources = {entry.source for entry in result.timeline}
    assert {"filesystem", "memory", "schedule", "task", "receipt", "commitment", "plan", "provenance"} <= sources
    timestamps = [entry.timestamp for entry in result.timeline]
    assert timestamps == sorted(timestamps)
    assert all(entry.source_table and entry.event_kind and entry.summary and entry.ref_id for entry in result.timeline)


@pytest.mark.asyncio
async def test_artifact_lookup_by_checksum_finds_same_artifact(tmp_path: Path) -> None:
    target = tmp_path / "workspace" / "artifact.md"
    target.parent.mkdir(parents=True)
    content = "same checksum"
    target.write_text(content, encoding="utf-8")
    stores = await _stores(tmp_path, target, content)
    checksum = hashlib.sha256(content.encode()).hexdigest()

    result = await ArtifactLookupService(**stores).lookup(checksum=checksum)

    assert str(target.resolve()) in result.sibling_paths
    assert result.query["checksum"] == checksum
    assert result.timeline


@pytest.mark.asyncio
async def test_artifact_lookup_unknown_path_returns_empty_success(tmp_path: Path) -> None:
    workspace_store = await WorkspaceStore(tmp_path / ".opencas" / "workspace.db").connect()
    memory_store = await MemoryStore(tmp_path / ".opencas" / "memory.db").connect()

    result = await ArtifactLookupService(
        workspace_store=workspace_store,
        memory_store=memory_store,
    ).lookup(path=str(tmp_path / "workspace" / "missing.md"))

    assert result.current["exists_on_disk"] is False
    assert result.timeline == []
    assert result.summary_counts == {}
