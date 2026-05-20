"""Focused tests for runtime status snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from opencas.runtime.consolidation_state import persist_consolidation_runtime_state
from opencas.runtime.consolidation_worker import consolidation_worker_status_path
from opencas.runtime.status_views import build_consolidation_status


def test_build_consolidation_status_falls_back_to_persisted_state(tmp_path: Path) -> None:
    persist_consolidation_runtime_state(
        tmp_path,
        {
            "last_run_at": "2026-04-21T08:30:00+00:00",
            "last_result_id": "result-123",
        },
    )
    runtime = SimpleNamespace(
        _last_consolidation_result=None,
        ctx=SimpleNamespace(config=SimpleNamespace(state_dir=tmp_path)),
    )

    status = build_consolidation_status(runtime)

    assert status["available"] is True
    assert status["timestamp"] == "2026-04-21T08:30:00+00:00"
    assert status["result_id"] == "result-123"
    assert status["persisted_only"] is True


def test_build_consolidation_status_loads_completed_worker_result_after_restart(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "consolidation_worker" / "results" / "run-1.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-12T14:44:54.040208Z",
                "result_id": "result-worker-1",
                "clusters_formed": 27,
                "memories_created": 6,
                "commitment_clusters_formed": 3,
                "commitment_work_objects_created": 2,
                "budget_exhausted": True,
                "budget_reason": "cluster_summaries",
            }
        ),
        encoding="utf-8",
    )
    status_path = consolidation_worker_status_path(tmp_path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "run-1",
                "result_path": str(result_path),
            }
        ),
        encoding="utf-8",
    )
    persist_consolidation_runtime_state(
        tmp_path,
        {
            "last_run_at": "2026-05-12T14:44:54.040208Z",
            "last_result_id": "result-worker-1",
        },
    )
    runtime = SimpleNamespace(
        _last_consolidation_result=None,
        ctx=SimpleNamespace(config=SimpleNamespace(state_dir=tmp_path)),
    )

    status = build_consolidation_status(runtime)

    assert status["persisted_only"] is True
    assert status["result_id"] == "result-worker-1"
    assert status["clusters_formed"] == 27
    assert status["memories_created"] == 6
    assert status["commitment_clusters_formed"] == 3
    assert status["commitment_work_objects_created"] == 2
    assert status["budget_reason"] == "cluster_summaries"


def test_build_consolidation_status_exposes_last_nightly_dream() -> None:
    runtime = SimpleNamespace(
        _last_consolidation_result={
            "timestamp": "2026-05-04T03:00:00+00:00",
            "result_id": "result-456",
            "nightly_dream": {
                "dream_id": "dream-123",
                "mode": "light",
                "artifact_path": "/tmp/dream.md",
            },
        },
        ctx=SimpleNamespace(config=SimpleNamespace(state_dir=None)),
    )

    status = build_consolidation_status(runtime)

    assert status["available"] is True
    assert status["nightly_dream"]["dream_id"] == "dream-123"
    assert status["nightly_dream"]["mode"] == "light"
