"""Tests for deterministic execution salvage packets."""

import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from opencas.execution import PhaseRecord, RepairTask
from opencas.execution.models import AttemptOutcome, AttemptSalvagePacket, RetryMode
from opencas.execution.salvage import build_salvage_packet
from opencas.execution.store import TaskStore


def _sample_task() -> RepairTask:
    task = RepairTask(
        objective="Continue Chronicle 4246 from the existing manuscript.",
        project_id="loop-4246",
        meta={
            "resume_project": {
                "signature": "chronicle-4246",
                "canonical_artifact_path": "workspace/Chronicles/4246/chronicle_4246.md",
            }
        },
    )
    task.attempt = 2
    task.artifacts = [
        "plan:continue chapter repair from existing manuscript",
        "exec:wrote chapter 9 bridge scene and noted missing fallout scene",
    ]
    task.phases = [
        PhaseRecord(phase="plan", success=True, output="continue chapter repair from existing manuscript"),
        PhaseRecord(phase="execute", success=True, output="wrote chapter 9 bridge scene and noted missing fallout scene"),
        PhaseRecord(phase="verify", success=False, output="continuity check still missing fallout scene"),
    ]
    return task


def test_build_salvage_packet_captures_partial_attempt_shape() -> None:
    task = _sample_task()

    packet = build_salvage_packet(
        task,
        outcome=AttemptOutcome.VERIFY_FAILED,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=[
            "workspace/Chronicles/4246/chronicle_4246.md",
            "workspace/Chronicles/4246/chronicle_4246.md",
        ],
        tool_calls=[{"name": "fs_read_file"}, {"name": "fs_write_file"}],
    )

    assert packet.project_signature == "chronicle-4246"
    assert packet.project_id == "loop-4246"
    assert packet.canonical_artifact_path == "workspace/Chronicles/4246/chronicle_4246.md"
    assert packet.outcome == AttemptOutcome.VERIFY_FAILED
    assert packet.recommended_mode == RetryMode.RESUME_EXISTING_ARTIFACT
    assert packet.artifact_paths_touched == ["workspace/Chronicles/4246/chronicle_4246.md"]
    assert packet.plan_digest
    assert packet.execution_digest
    assert packet.verification_digest
    assert packet.tool_signature
    assert packet.best_next_step
    assert packet.divergence_signature


def test_build_salvage_packet_marks_no_meaningful_progress_attempt() -> None:
    task = RepairTask(objective="Try broad repair again without changing artifact.")
    task.attempt = 2
    task.phases = [
        PhaseRecord(phase="plan", success=True, output="try the broad repair again"),
        PhaseRecord(phase="execute", success=False, output="failed"),
        PhaseRecord(phase="verify", success=False, output=""),
    ]

    packet = build_salvage_packet(
        task,
        outcome=AttemptOutcome.FAILED,
        canonical_artifact_path=None,
        artifact_paths_touched=[],
        tool_calls=[{"name": "agent", "args": {"prompt": "try again"}}],
    )

    assert packet.meaningful_progress_signal == "no_meaningful_progress"
    assert packet.recommended_mode == RetryMode.DETERMINISTIC_REVIEW
    assert "no meaningful progress" in packet.best_next_step.lower()
    assert "no meaningful progress" in packet.discovered_constraints


def test_build_salvage_packet_is_deterministic_for_equivalent_inputs() -> None:
    task = _sample_task()

    packet1 = build_salvage_packet(
        task,
        outcome=AttemptOutcome.VERIFY_FAILED,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=[
            "workspace/Chronicles/4246/chronicle_4246.md",
            "workspace/Chronicles/4246/chronicle_4246.md",
            "workspace/Chronicles/4246/notes.md",
        ],
        tool_calls=[{"name": "fs_read_file"}, {"name": "fs_write_file"}],
    )
    packet2 = build_salvage_packet(
        task,
        outcome=AttemptOutcome.VERIFY_FAILED,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=[
            "workspace/Chronicles/4246/notes.md",
            "workspace/Chronicles/4246/chronicle_4246.md",
        ],
        tool_calls=[{"name": "fs_read_file"}, {"name": "fs_write_file"}],
    )

    assert packet1.artifact_paths_touched == packet2.artifact_paths_touched
    assert packet1.packet_id == packet2.packet_id
    assert packet1.created_at == packet2.created_at
    assert packet1.divergence_signature == packet2.divergence_signature
    assert packet1.tool_signature == packet2.tool_signature


@pytest.mark.asyncio
async def test_store_round_trips_latest_salvage_packet(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()
    task = _sample_task()

    packet1 = build_salvage_packet(
        task,
        outcome=AttemptOutcome.FAILED,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=["workspace/Chronicles/4246/chronicle_4246.md"],
        tool_calls=[{"name": "fs_read_file"}],
    )
    await store.save_salvage_packet(packet1)

    task.attempt = 3
    later_started_at = task.phases[-1].started_at + timedelta(seconds=5)
    task.phases[-1] = PhaseRecord(
        phase="verify",
        success=False,
        output="continuity check still missing fallout scene",
        started_at=later_started_at,
        ended_at=later_started_at,
    )
    packet2 = build_salvage_packet(
        task,
        outcome=AttemptOutcome.VERIFY_FAILED,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=[
            "workspace/Chronicles/4246/chronicle_4246.md",
            "workspace/Chronicles/4246/notes.md",
        ],
        tool_calls=[{"name": "fs_read_file"}, {"name": "fs_write_file"}],
    )
    await store.save_salvage_packet(packet2)

    fetched = await store.get_latest_salvage_packet(str(task.task_id))

    assert fetched is not None
    assert fetched.model_dump(mode="json") == packet2.model_dump(mode="json")
    await store.close()


@pytest.mark.asyncio
async def test_store_prefers_highest_attempt_when_salvage_timestamps_match(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()
    task = _sample_task()
    created_at = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    packet1 = AttemptSalvagePacket(
        packet_id=UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
        task_id=task.task_id,
        attempt=2,
        project_signature="chronicle-4246",
        project_id=task.project_id,
        objective=task.objective,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=["workspace/Chronicles/4246/chronicle_4246.md"],
        plan_digest="plan-a",
        execution_digest="exec-a",
        verification_digest="verify-a",
        tool_signature="tool-a",
        divergence_signature="divergence-a",
        outcome=AttemptOutcome.FAILED,
        partial_value="failed draft",
        discovered_constraints=["missing fallout scene"],
        unresolved_questions=[],
        best_next_step="resume the existing manuscript",
        recommended_mode=RetryMode.RESUME_EXISTING_ARTIFACT,
        llm_spend_class="broad",
        created_at=created_at,
    )
    await store.save_salvage_packet(packet1)
    packet2 = AttemptSalvagePacket(
        packet_id=UUID("00000000-0000-0000-0000-000000000000"),
        task_id=task.task_id,
        attempt=3,
        project_signature="chronicle-4246",
        project_id=task.project_id,
        objective=task.objective,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=[
            "workspace/Chronicles/4246/chronicle_4246.md",
            "workspace/Chronicles/4246/notes.md",
        ],
        plan_digest="plan-b",
        execution_digest="exec-b",
        verification_digest="verify-b",
        tool_signature="tool-b",
        divergence_signature="divergence-b",
        outcome=AttemptOutcome.VERIFY_FAILED,
        partial_value="bridge scene drafted",
        discovered_constraints=["still missing fallout scene"],
        unresolved_questions=[],
        best_next_step="repair the remaining gap and rerun verification",
        recommended_mode=RetryMode.RESUME_EXISTING_ARTIFACT,
        llm_spend_class="broad",
        created_at=created_at,
    )
    await store.save_salvage_packet(packet2)

    fetched = await store.get_latest_salvage_packet(str(task.task_id))

    assert fetched is not None
    assert fetched.attempt == 3
    assert fetched.packet_id == packet2.packet_id
    await store.close()


@pytest.mark.asyncio
async def test_store_replaces_salvage_packet_for_same_attempt(tmp_path) -> None:
    store = TaskStore(tmp_path / "tasks.db")
    await store.connect()
    task = _sample_task()
    created_at = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)

    first = AttemptSalvagePacket(
        packet_id=UUID("11111111-1111-1111-1111-111111111111"),
        task_id=task.task_id,
        attempt=2,
        project_signature="chronicle-4246",
        project_id=task.project_id,
        objective=task.objective,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=["workspace/Chronicles/4246/chronicle_4246.md"],
        plan_digest="plan-a",
        execution_digest="exec-a",
        verification_digest="verify-a",
        tool_signature="tool-a",
        divergence_signature="divergence-a",
        outcome=AttemptOutcome.FAILED,
        partial_value="failed draft",
        discovered_constraints=["missing fallout scene"],
        unresolved_questions=[],
        best_next_step="resume the existing manuscript",
        recommended_mode=RetryMode.RESUME_EXISTING_ARTIFACT,
        llm_spend_class="broad",
        created_at=created_at,
    )
    second = AttemptSalvagePacket(
        packet_id=UUID("22222222-2222-2222-2222-222222222222"),
        task_id=task.task_id,
        attempt=2,
        project_signature="chronicle-4246",
        project_id=task.project_id,
        objective=task.objective,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=[
            "workspace/Chronicles/4246/chronicle_4246.md",
            "workspace/Chronicles/4246/notes.md",
        ],
        plan_digest="plan-b",
        execution_digest="exec-b",
        verification_digest="verify-b",
        tool_signature="tool-b",
        divergence_signature="divergence-b",
        outcome=AttemptOutcome.VERIFY_FAILED,
        partial_value="bridge scene drafted",
        discovered_constraints=["still missing fallout scene"],
        unresolved_questions=[],
        best_next_step="repair the remaining gap and rerun verification",
        recommended_mode=RetryMode.RESUME_EXISTING_ARTIFACT,
        llm_spend_class="broad",
        created_at=created_at,
    )

    await store.save_salvage_packet(first)
    await store.save_salvage_packet(second)

    fetched = await store.get_latest_salvage_packet(str(task.task_id))
    assert fetched is not None
    assert fetched.attempt == 2
    assert fetched.packet_id == second.packet_id

    assert store._db is not None
    cursor = await store._db.execute(
        "SELECT COUNT(*) AS count FROM attempt_salvage_packets WHERE task_id = ? AND attempt = ?",
        (str(task.task_id), 2),
    )
    row = await cursor.fetchone()
    assert row["count"] == 1
    await store.close()


@pytest.mark.asyncio
async def test_store_migrates_legacy_duplicate_salvage_rows_before_unique_index(tmp_path) -> None:
    db_path = tmp_path / "tasks.db"
    legacy_db = sqlite3.connect(db_path)
    legacy_db.executescript(
        """
        CREATE TABLE attempt_salvage_packets (
            packet_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            project_id TEXT,
            project_signature TEXT,
            outcome TEXT NOT NULL,
            divergence_signature TEXT NOT NULL,
            canonical_artifact_path TEXT,
            recommended_mode TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX idx_salvage_task_created
            ON attempt_salvage_packets(task_id, created_at DESC);
        CREATE INDEX idx_salvage_project_created
            ON attempt_salvage_packets(project_signature, created_at DESC);
        """
    )
    task = _sample_task()
    created_at = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    older = AttemptSalvagePacket(
        packet_id=UUID("11111111-1111-1111-1111-111111111111"),
        task_id=task.task_id,
        attempt=2,
        project_signature="chronicle-4246",
        project_id=task.project_id,
        objective=task.objective,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=["workspace/Chronicles/4246/chronicle_4246.md"],
        plan_digest="plan-a",
        execution_digest="exec-a",
        verification_digest="verify-a",
        tool_signature="tool-a",
        divergence_signature="divergence-a",
        outcome=AttemptOutcome.FAILED,
        partial_value="failed draft",
        discovered_constraints=["missing fallout scene"],
        unresolved_questions=[],
        best_next_step="resume the existing manuscript",
        recommended_mode=RetryMode.RESUME_EXISTING_ARTIFACT,
        llm_spend_class="broad",
        created_at=created_at,
    )
    newer = AttemptSalvagePacket(
        packet_id=UUID("22222222-2222-2222-2222-222222222222"),
        task_id=task.task_id,
        attempt=2,
        project_signature="chronicle-4246",
        project_id=task.project_id,
        objective=task.objective,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=[
            "workspace/Chronicles/4246/chronicle_4246.md",
            "workspace/Chronicles/4246/notes.md",
        ],
        plan_digest="plan-b",
        execution_digest="exec-b",
        verification_digest="verify-b",
        tool_signature="tool-b",
        divergence_signature="divergence-b",
        outcome=AttemptOutcome.VERIFY_FAILED,
        partial_value="bridge scene drafted",
        discovered_constraints=["still missing fallout scene"],
        unresolved_questions=[],
        best_next_step="repair the remaining gap and rerun verification",
        recommended_mode=RetryMode.RESUME_EXISTING_ARTIFACT,
        llm_spend_class="broad",
        created_at=created_at + timedelta(seconds=1),
    )
    legacy_db.execute(
        """
        INSERT INTO attempt_salvage_packets (
            packet_id, task_id, attempt, project_id, project_signature, outcome,
            divergence_signature, canonical_artifact_path, recommended_mode,
            created_at, payload
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(older.packet_id),
            str(older.task_id),
            older.attempt,
            older.project_id,
            older.project_signature,
            older.outcome.value,
            older.divergence_signature,
            older.canonical_artifact_path,
            older.recommended_mode.value,
            older.created_at.isoformat(),
            older.model_dump_json(),
        ),
    )
    legacy_db.execute(
        """
        INSERT INTO attempt_salvage_packets (
            packet_id, task_id, attempt, project_id, project_signature, outcome,
            divergence_signature, canonical_artifact_path, recommended_mode,
            created_at, payload
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(newer.packet_id),
            str(newer.task_id),
            newer.attempt,
            newer.project_id,
            newer.project_signature,
            newer.outcome.value,
            newer.divergence_signature,
            newer.canonical_artifact_path,
            newer.recommended_mode.value,
            newer.created_at.isoformat(),
            newer.model_dump_json(),
        ),
    )
    legacy_db.commit()
    legacy_db.close()

    store = TaskStore(db_path)
    await store.connect()

    fetched = await store.get_latest_salvage_packet(str(task.task_id))
    assert fetched is not None
    assert fetched.packet_id == newer.packet_id

    assert store._db is not None
    cursor = await store._db.execute(
        "SELECT COUNT(*) AS count FROM attempt_salvage_packets WHERE task_id = ? AND attempt = ?",
        (str(task.task_id), 2),
    )
    row = await cursor.fetchone()
    assert row["count"] == 1
    await store.close()
