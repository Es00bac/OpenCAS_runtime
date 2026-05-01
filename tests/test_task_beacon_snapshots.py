from __future__ import annotations

import json
import re
import subprocess
import os
from pathlib import Path

from opencas.bootstrap.task_beacon import build_task_beacon, public_task_beacon_payload


def _read_fixture(name: str) -> str:
    return Path(__file__).with_name("fixtures").joinpath(name).read_text(encoding="utf-8")


def _render_task_beacon(task_beacon: dict[str, object]) -> str:
    helper_path = Path(__file__).resolve().parents[1] / "opencas" / "dashboard" / "static" / "js" / "task_beacon.js"
    script = f"""
const {{ renderTaskBeaconSummary }} = require({json.dumps(str(helper_path))});
const beacon = JSON.parse(process.env.TASK_BEACON_JSON);
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (m) => ({{
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
}}[m]));
process.stdout.write(renderTaskBeaconSummary(beacon, escapeHtml));
"""
    env = dict(os.environ)
    env["TASK_BEACON_JSON"] = json.dumps(task_beacon)
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return re.sub(r"\s+", " ", result.stdout).strip()


def _project_public_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        "available": payload["available"],
        "matched_only": payload["matched_only"],
        "headline": payload["headline"],
        "counts": payload["counts"],
        "bucket_signature": payload["bucket_signature"],
        "view_model": {
            "buckets": [
                {"state": bucket["state"], "count": bucket["count"]}
                for bucket in payload["view_model"]["buckets"]
            ]
        },
        "rules": payload["rules"],
        "model": payload["model"],
    }


def test_task_beacon_public_payload_snapshot_for_noisy_dozens_fixture(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir(exist_ok=True)
    (workspace_root / "TaskList.md").write_text(_read_fixture("task_beacon_dozens.md"), encoding="utf-8")

    first = _project_public_payload(public_task_beacon_payload(build_task_beacon(workspace_root)))
    second = _project_public_payload(public_task_beacon_payload(build_task_beacon(workspace_root)))

    snapshot_path = Path(__file__).with_name("snapshots").joinpath("task_beacon_noisy_public_payload.json")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert first == second
    assert first == snapshot


def test_task_beacon_render_snapshot_for_noisy_dozens_fixture(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir(exist_ok=True)
    (workspace_root / "TaskList.md").write_text(_read_fixture("task_beacon_dozens.md"), encoding="utf-8")

    first = _render_task_beacon(
        public_task_beacon_payload(
            build_task_beacon(workspace_root),
            include_details=True,
            include_items=True,
        )
    )
    second = _render_task_beacon(
        public_task_beacon_payload(
            build_task_beacon(workspace_root),
            include_details=True,
            include_items=True,
        )
    )

    snapshot_path = Path(__file__).with_name("snapshots").joinpath("task_beacon_noisy_render.html")
    snapshot = snapshot_path.read_text(encoding="utf-8").strip()

    assert first == second
    assert first == snapshot
    assert "task-beacon-top-item" in first
    assert "task-beacon-bucket-details" in first
    assert "task-beacon-fragment-list" in first
