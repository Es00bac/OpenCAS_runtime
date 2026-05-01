from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from opencas.bootstrap.task_beacon import build_task_beacon, public_task_beacon_payload


def _normalize_html(html: str) -> str:
    return re.sub(r"\s+", " ", html).strip()


def _read_fixture(name: str) -> str:
    return Path(__file__).with_name("fixtures").joinpath(name).read_text(encoding="utf-8")


def _render_task_beacon(task_beacon: dict[str, object]) -> str:
    helper_path = Path(__file__).resolve().parents[1] / "opencas" / "dashboard" / "static" / "js" / "task_beacon.js"
    script = f"""
const {{ renderTaskBeaconSummary }} = require({json.dumps(str(helper_path))});
const beacon = JSON.parse(process.argv[2]);
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (m) => ({{
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
}}[m]));
process.stdout.write(renderTaskBeaconSummary(beacon, escapeHtml));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cjs", delete=False) as handle:
        handle.write(script)
        script_path = handle.name
    try:
        result = subprocess.run(
            ["node", script_path, json.dumps(task_beacon)],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        Path(script_path).unlink(missing_ok=True)
    return result.stdout


def _merge_task_beacon_summary(
    current: dict[str, object] | None,
    incoming: dict[str, object] | None,
) -> dict[str, object] | None:
    helper_path = Path(__file__).resolve().parents[1] / "opencas" / "dashboard" / "static" / "js" / "task_beacon.js"
    script = f"""
const {{ mergeTaskBeaconSummary }} = require({json.dumps(str(helper_path))});
const current = JSON.parse(process.argv[2]);
const incoming = JSON.parse(process.argv[3]);
process.stdout.write(JSON.stringify(mergeTaskBeaconSummary(current, incoming)));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cjs", delete=False) as handle:
        handle.write(script)
        script_path = handle.name
    try:
        result = subprocess.run(
            ["node", script_path, json.dumps(current), json.dumps(incoming)],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        Path(script_path).unlink(missing_ok=True)
    return json.loads(result.stdout)


def test_task_beacon_render_stays_compact_and_three_state_only(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir(exist_ok=True)
    (workspace_root / "TaskList.md").write_text(_read_fixture("task_beacon_noisy_fragments.md"), encoding="utf-8")
    task_beacon = build_task_beacon(workspace_root)

    html = _normalize_html(_render_task_beacon(task_beacon))

    assert "<h4>Task Beacon</h4>" in html
    assert '<span class="badge">now</span>' in html
    assert '<span class="badge">next</span>' in html
    assert '<span class="badge">later</span>' in html
    assert 'task-beacon-bucket-summary' in html
    assert 'task-beacon-bucket-details' in html
    assert 'task-beacon-fragment-list' in html
    assert 'TASK-901' in html
    assert 'TASK-902' in html
    assert 'TASK-904' in html
    assert 'Build/test active now candidate' in html
    assert 'Build/test noisy duplicate' in html
    assert 'Build/test archived follow-up' in html


def test_task_beacon_render_is_repeatable_for_noisy_fragment_sets(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir(exist_ok=True)
    (workspace_root / "TaskList.md").write_text(_read_fixture("task_beacon_noisy_fragments.md"), encoding="utf-8")

    first = _normalize_html(_render_task_beacon(build_task_beacon(workspace_root)))
    second = _normalize_html(_render_task_beacon(build_task_beacon(workspace_root)))

    assert first == second
    assert '<span class="badge">now</span>' in first
    assert '<span class="badge">next</span>' in first
    assert '<span class="badge">later</span>' in first
    assert 'task-beacon-bucket-summary' in first
    assert 'task-beacon-bucket-details' in first
    assert 'task-beacon-fragment-list' in first
    assert 'TASK-901' in first
    assert 'TASK-902' in first
    assert 'TASK-904' in first
    assert 'Build/test noisy duplicate' in first
    assert 'Build/test archived follow-up' in first


def test_task_beacon_render_stays_terse_for_mixed_live_fragments(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir(exist_ok=True)
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-501` Build/test live now base\n"
        "  - owner: Codex\n"
        "  - status: in progress\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-502` Build/test live next base\n"
        "  - owner: Codex\n"
        "  - status: pending\n\n"
        "## Recently Completed\n\n"
        "- `TASK-503` Build/test live later base\n"
        "  - owner: Codex\n"
        "  - status: completed\n",
        encoding="utf-8",
    )

    task_beacon = build_task_beacon(
        workspace_root,
        live_fragments=[
            {
                "task_id": "TASK-501",
                "title": "Build/test live now base",
                "section": "Next Up / Backlog",
                "status": "blocked",
                "content": "- `TASK-501` Build/test live now base\n  - owner: Codex\n  - status: blocked\n  - result:\n    - blocked by the flaky pytest shard",
            },
            {
                "task_id": "TASK-502",
                "title": "Build/test live next base",
                "section": "In Progress",
                "status": "in progress",
                "content": "- `TASK-502` Build/test live next base\n  - owner: Codex\n  - status: in progress\n  - result:\n    - second live pass should still collapse quietly",
            },
            {
                "task_id": "TASK-503",
                "title": "Build/test live later base",
                "section": "Recently Completed",
                "status": "queued",
                "content": "- `TASK-503` Build/test live later base\n  - owner: Codex\n  - status: queued\n  - result:\n    - queued follow-up should stay later",
            },
            {
                "task_id": "TASK-504",
                "title": "Build/test live only candidate",
                "section": "In Progress",
                "status": "in progress",
                "content": "- `TASK-504` Build/test live only candidate\n  - owner: Codex\n  - status: in progress\n  - result:\n    - live-only now candidate",
            },
        ],
    )

    html = _normalize_html(_render_task_beacon(task_beacon))

    assert '<span class="badge">now</span>' in html
    assert '<span class="badge">next</span>' in html
    assert '<span class="badge">later</span>' in html
    assert 'task-beacon-bucket-summary' in html
    assert 'task-beacon-bucket-details' in html
    assert 'task-beacon-fragment-list' in html
    assert 'TASK-504' in html
    assert 'TASK-501' in html
    assert 'TASK-503' in html
    assert 'Build/test live only candidate' in html
    assert 'Build/test live now base' in html
    assert 'Build/test live later base' in html


def test_task_beacon_render_collapses_dozens_of_fragments_into_three_states_only(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir(exist_ok=True)
    (workspace_root / "TaskList.md").write_text(_read_fixture("task_beacon_dozens.md"), encoding="utf-8")

    task_beacon = build_task_beacon(workspace_root)
    html = _normalize_html(_render_task_beacon(task_beacon))

    assert [bucket["state"] for bucket in task_beacon["view_model"]["buckets"]] == ["now", "next", "later"]
    assert [bucket["count"] for bucket in task_beacon["view_model"]["buckets"]] == [3, 5, 3]
    assert [set(bucket) for bucket in task_beacon["view_model"]["buckets"]] == [
        {"state", "count", "item", "items"},
        {"state", "count", "item", "items"},
        {"state", "count", "item", "items"},
    ]
    assert '<span class="badge">now</span>' in html
    assert '<span class="badge">next</span>' in html
    assert '<span class="badge">later</span>' in html
    assert 'task-beacon-bucket-summary' in html
    assert 'task-beacon-top-item' in html
    assert 'task-beacon-bucket-details' in html
    assert 'task-beacon-fragment-list' in html
    assert 'TASK-103' in html
    assert 'TASK-201' in html
    assert 'TASK-303' in html


def test_task_beacon_render_uses_minimal_empty_state(tmp_path: Path) -> None:
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir(exist_ok=True)
    (workspace_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-900` General maintenance\n"
        "  - owner: Codex\n"
        "  - status: in progress\n",
        encoding="utf-8",
    )

    html = _normalize_html(_render_task_beacon(build_task_beacon(workspace_root)))

    assert "<h4>Task Beacon</h4>" in html
    assert "No matching build/test fragments." in html
    assert "task-beacon-buckets" not in html
    assert "task-beacon-bucket-item" not in html
    assert "task-beacon-bucket-reason" not in html


def test_task_beacon_render_stays_quiet_for_same_bucket_shape(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir(exist_ok=True)
    (first_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-901` Build/test quiet now\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - active fragments should collapse into now\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-902` Build/test quiet next\n"
        "  - owner: Codex\n"
        "  - status: blocked\n"
        "  - result:\n"
        "    - blocked fragments should collapse into next\n\n"
        "## Recently Completed\n\n"
        "- `TASK-903` Build/test quiet later\n"
        "  - owner: Codex\n"
        "  - status: completed\n"
        "  - result:\n"
        "    - completed fragments should collapse into later\n",
        encoding="utf-8",
    )

    second_root = tmp_path / "second"
    second_root.mkdir(exist_ok=True)
    (second_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-901` Build/test quiet now changed\n"
        "  - owner: Codex\n"
        "  - status: running\n"
        "  - result:\n"
        "    - the default surface still only cares about now\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-902` Build/test quiet next changed\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - fragment wording changes should not change the render\n\n"
        "## Recently Completed\n\n"
        "- `TASK-903` Build/test quiet later changed\n"
        "  - owner: Codex\n"
        "  - status: queued\n"
        "  - result:\n"
        "    - later stays later\n",
        encoding="utf-8",
    )

    first_beacon = build_task_beacon(first_root)
    second_beacon = build_task_beacon(second_root)
    first_html = _normalize_html(_render_task_beacon(first_beacon))
    second_html = _normalize_html(_render_task_beacon(second_beacon))

    assert [bucket["state"] for bucket in first_beacon["view_model"]["buckets"]] == ["now", "next", "later"]
    assert [bucket["state"] for bucket in second_beacon["view_model"]["buckets"]] == ["now", "next", "later"]
    assert [bucket["count"] for bucket in first_beacon["view_model"]["buckets"]] == [1, 1, 1]
    assert [bucket["count"] for bucket in second_beacon["view_model"]["buckets"]] == [1, 1, 1]
    assert first_html != second_html
    assert 'task-beacon-bucket-summary' in first_html
    assert 'task-beacon-bucket-summary' in second_html
    assert 'task-beacon-bucket-details' in first_html
    assert 'task-beacon-bucket-details' in second_html
    assert 'task-beacon-fragment-list' in first_html
    assert 'task-beacon-fragment-list' in second_html
    assert 'Build/test quiet now' in first_html
    assert 'Build/test quiet now changed' in second_html
    assert '<span class="badge">now</span>' in first_html
    assert '<span class="badge">next</span>' in first_html
    assert '<span class="badge">later</span>' in first_html
    assert first_beacon["bucket_signature"] != second_beacon["bucket_signature"]


def test_task_beacon_summary_merge_preserves_current_state_when_bucket_signature_is_unchanged(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir(exist_ok=True)
    (first_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-101` Build/test now alpha\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - active alpha should stay in now\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-201` Build/test next beta\n"
        "  - owner: Codex\n"
        "  - status: blocked\n"
        "  - result:\n"
        "    - blocked by the flaky pytest shard\n\n"
        "## Recently Completed\n\n"
        "- `TASK-301` Build/test later gamma\n"
        "  - owner: Codex\n"
        "  - status: completed\n"
        "  - result:\n"
        "    - completed fragment should stay in later\n",
        encoding="utf-8",
    )

    second_root = tmp_path / "second"
    second_root.mkdir(exist_ok=True)
    (second_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-101` Build/test now alpha\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - active alpha should stay in now\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-201` Build/test next beta\n"
        "  - owner: Codex\n"
        "  - status: blocked\n"
        "  - result:\n"
        "    - blocked by the flaky pytest shard\n\n"
        "- `TASK-201` Build/test next beta\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - alternate hidden fragment text should not matter\n\n"
        "## Recently Completed\n\n"
        "- `TASK-301` Build/test later gamma\n"
        "  - owner: Codex\n"
        "  - status: completed\n"
        "  - result:\n"
        "    - completed fragment should stay in later\n",
        encoding="utf-8",
    )

    first = build_task_beacon(first_root)
    second = build_task_beacon(second_root)
    merged = _merge_task_beacon_summary(
        public_task_beacon_payload(first),
        public_task_beacon_payload(second),
    )

    assert merged == public_task_beacon_payload(first)
    assert merged["bucket_signature"] == public_task_beacon_payload(first)["bucket_signature"]


def test_task_beacon_summary_merge_updates_when_representative_item_changes(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir(exist_ok=True)
    (first_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-101` Build/test now alpha\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - active alpha should stay in now\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-201` Build/test next beta\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - live blockers should stay visible\n\n"
        "- `TASK-202` Build/test next gamma\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - the quieter duplicate should stay behind the blocked one\n\n"
        "## Recently Completed\n\n"
        "- `TASK-301` Build/test later omega\n"
        "  - owner: Codex\n"
        "  - status: completed\n"
        "  - result:\n"
        "    - completed fragment should stay in later\n",
        encoding="utf-8",
    )

    second_root = tmp_path / "second"
    second_root.mkdir(exist_ok=True)
    (second_root / "TaskList.md").write_text(
        "# OpenCAS Task List\n\n"
        "## In Progress\n\n"
        "- `TASK-101` Build/test now alpha\n"
        "  - owner: Codex\n"
        "  - status: in progress\n"
        "  - result:\n"
        "    - active alpha should stay in now\n\n"
        "## Next Up / Backlog\n\n"
        "- `TASK-201` Build/test next beta revised\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - live blockers should stay visible\n\n"
        "- `TASK-202` Build/test next gamma\n"
        "  - owner: Codex\n"
        "  - status: pending\n"
        "  - result:\n"
        "    - the quieter duplicate should stay behind the blocked one\n\n"
        "## Recently Completed\n\n"
        "- `TASK-301` Build/test later omega\n"
        "  - owner: Codex\n"
        "  - status: completed\n"
        "  - result:\n"
        "    - completed fragment should stay in later\n",
        encoding="utf-8",
    )

    first = build_task_beacon(first_root)
    second = build_task_beacon(second_root)
    merged = _merge_task_beacon_summary(
        public_task_beacon_payload(first),
        public_task_beacon_payload(second),
    )

    assert merged["headline"] == public_task_beacon_payload(second)["headline"]
    assert merged["bucket_signature"] == public_task_beacon_payload(second)["bucket_signature"]
