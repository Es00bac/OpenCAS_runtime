"""Focused tests for runtime status snapshots."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from opencas.runtime.consolidation_state import persist_consolidation_runtime_state
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
