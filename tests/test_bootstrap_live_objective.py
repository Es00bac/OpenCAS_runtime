"""Focused bootstrap-store tests for live-objective alignment."""

from pathlib import Path

import pytest

from opencas.bootstrap import BootstrapConfig
from opencas.bootstrap.live_objective import read_tasklist_live_objective
from opencas.bootstrap.pipeline_stores import initialize_runtime_stores
from opencas.identity import IdentityManager, IdentityStore
from opencas.telemetry import TelemetryStore, Tracer


def test_read_tasklist_live_objective(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-137` Continuity surface reconciliation decision bead\n"
        "  - owner: Codex\n"
        "  - status: in progress\n",
        encoding="utf-8",
    )

    assert (
        read_tasklist_live_objective(workspace_root)
        == "Continuity surface reconciliation decision bead"
    )


def test_read_tasklist_live_objective_handles_missing_workspace() -> None:
    assert read_tasklist_live_objective(None) is None


@pytest.mark.asyncio
async def test_initialize_runtime_stores_prefers_tasklist_live_objective(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-137` Continuity surface reconciliation decision bead\n"
        "  - owner: Codex\n"
        "  - status: in progress\n",
        encoding="utf-8",
    )
    config = BootstrapConfig(
        state_dir=tmp_path / "state",
        session_id="bootstrap-live-objective",
        workspace_root=workspace_root,
    ).resolve_paths()

    identity = IdentityManager(IdentityStore(config.state_dir / "identity"))
    identity.load()
    telemetry_store = TelemetryStore(config.state_dir / "telemetry")
    tracer = Tracer(telemetry_store)
    stages: list[str] = []

    first = await initialize_runtime_stores(
        config,
        identity=identity,
        tracer=tracer,
        stage=lambda name, _meta=None: stages.append(name),
    )
    first.executive.set_intention("stale poetic residue")
    first.executive.save_snapshot(config.state_dir / "executive.json")
    await first.memory.close()
    await first.tasks.close()
    await first.receipt_store.close()
    await first.context_store.close()
    await first.work_store.close()
    await first.commitment_store.close()
    await first.portfolio_store.close()

    second_identity = IdentityManager(IdentityStore(config.state_dir / "identity"))
    second_identity.load()
    second = await initialize_runtime_stores(
        config,
        identity=second_identity,
        tracer=tracer,
        stage=lambda name, _meta=None: stages.append(name),
    )

    assert second.executive.intention == "Continuity surface reconciliation decision bead"
    assert second.executive.intention_source == "tasklist_live_objective"
    assert "executive_online" in stages

    await second.memory.close()
    await second.tasks.close()
    await second.receipt_store.close()
    await second.context_store.close()
    await second.work_store.close()
    await second.commitment_store.close()
    await second.portfolio_store.close()
