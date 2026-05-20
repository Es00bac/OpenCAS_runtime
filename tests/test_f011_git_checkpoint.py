"""Regression coverage for F-011 git checkpoint failure boundaries."""

from pathlib import Path

import pytest

from opencas.execution import ExecutionPhase, ExecutionStage, RepairExecutor, RepairTask
from opencas.execution.git_checkpoint import GitCheckpointManager
from opencas.tools import ToolRegistry


def _isolate_git_identity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "missing-global-gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(tmp_path / "missing-system-gitconfig"))
    monkeypatch.delenv("GIT_AUTHOR_NAME", raising=False)
    monkeypatch.delenv("GIT_AUTHOR_EMAIL", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_NAME", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_EMAIL", raising=False)


def test_git_checkpoint_snapshot_returns_none_when_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_git_identity(monkeypatch, tmp_path)
    target = tmp_path / "scratch" / "file.txt"
    target.parent.mkdir()
    target.write_text("original")

    checkpoint = GitCheckpointManager(target.parent)

    assert checkpoint.snapshot([str(target)]) is None


def test_git_checkpoint_restore_rejects_invalid_head_hash(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    checkpoint = GitCheckpointManager(scratch)

    with pytest.raises(Exception, match="HEAD"):
        checkpoint.restore("HEAD")


@pytest.mark.asyncio
async def test_executor_records_snapshot_and_rollback_failure_when_checkpoint_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_git_identity(monkeypatch, tmp_path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    target = scratch / "file.txt"
    target.write_text("original")

    executor = RepairExecutor(tools=ToolRegistry())

    async def detect(_task: RepairTask) -> str:
        return str(target)

    async def plan(_task: RepairTask) -> str:
        return "modify file"

    async def execute(_task: RepairTask, _plan: str) -> str:
        target.write_text("modified")
        return "modified file"

    async def verify(_task: RepairTask) -> bool:
        return False

    executor._detect = detect
    executor._plan = plan
    executor._execute_plan = execute
    executor._verify = verify

    task = RepairTask(objective=f"modify {target}", scratch_dir=str(scratch), max_attempts=1)

    result = await executor.run(task)

    snapshot_phase = next(phase for phase in task.phases if phase.phase == ExecutionPhase.SNAPSHOT)
    rollback_phases = [
        phase for phase in task.phases if getattr(phase.phase, "value", phase.phase) == "rollback"
    ]

    assert snapshot_phase.success is False
    assert task.checkpoint_commit is None
    assert target.read_text() == "modified"
    assert rollback_phases
    assert rollback_phases[-1].success is False
    assert "missing checkpoint commit" in (rollback_phases[-1].output or "")
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED
