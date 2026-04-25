from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_scenario9_memory_continuity import _run


@pytest.mark.asyncio
async def test_run_scenario9_memory_continuity(tmp_path: Path) -> None:
    report = await _run(tmp_path)

    assert report["material_success"] is True
    assert report["anchor_episode_retrieved"] is True
    assert report["distilled_memory_retrieved"] is True
    assert report["artifact_verified"] is True
    assert report["episode_access_count"] >= 1
    assert report["episode_success_count"] >= 1
    assert report["memory_access_count"] >= 1
    assert report["memory_value_snapshot"]["evidence_level"] == "grounded"

    artifact_path = Path(str(report["artifact_path"]))
    body = artifact_path.read_text(encoding="utf-8")
    assert "# Redwood Launch Notes" in body
    assert "Gating issue: R-17" in body

    report_json = tmp_path / "scenario9_memory_continuity_report.json"
    report_json.write_text(json.dumps(report), encoding="utf-8")
    assert report_json.exists()
