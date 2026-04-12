from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_qualification_remediation import build_rollup, render_markdown


def _write_run(runs_dir: Path, run_id: str, label: str, success: bool, outcome: str) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "agent_checks": [
            {
                "label": label,
                "material_success": success,
                "outcome": outcome,
                "response": outcome,
            }
        ],
    }
    (run_dir / "live_debug_validation_report.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_rollup_classifies_runner_issue(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "run-older", "integrated_operator_workflow", True, "artifact_verified")
    history_path = tmp_path / "qualification_rerun_history.jsonl"
    history_path.write_text(
        json.dumps({
            "event": "completed",
            "request_id": "req-runner",
            "labels": ["integrated_operator_workflow"],
            "returncode": 1,
            "generated_run_ids": [],
            "latest_run_id": None,
        })
        + "\n",
        encoding="utf-8",
    )

    payload = build_rollup(runs_dir, history_path)

    assert payload["items"][0]["recommended_action"] == "investigate_runner"
    assert payload["items"][0]["latest_run"] is None


def test_build_rollup_classifies_code_change_vs_continue_testing(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "run-1", "integrated_operator_workflow", False, "artifact_missing")
    _write_run(runs_dir, "run-2", "integrated_operator_workflow", True, "artifact_verified")
    _write_run(runs_dir, "run-3", "kilocode_supervised_work", False, "artifact_missing")
    history_path = tmp_path / "qualification_rerun_history.jsonl"
    history_path.write_text(
        "\n".join([
            json.dumps({
                "event": "completed",
                "request_id": "req-ok",
                "labels": ["integrated_operator_workflow"],
                "returncode": 0,
                "generated_run_ids": ["run-2"],
                "latest_run_id": "run-2",
            }),
            json.dumps({
                "event": "completed",
                "request_id": "req-bad",
                "labels": ["kilocode_supervised_work"],
                "returncode": 0,
                "generated_run_ids": ["run-3"],
                "latest_run_id": "run-3",
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    payload = build_rollup(runs_dir, history_path)

    by_request = {item["request_id"]: item for item in payload["items"]}
    assert by_request["req-ok"]["recommended_action"] == "continue_testing"
    assert by_request["req-ok"]["after_rate"] == 0.5
    assert by_request["req-bad"]["recommended_action"] == "code_change_justified"


def test_render_markdown_mentions_recommended_action(tmp_path: Path) -> None:
    payload = {
        "count": 1,
        "history_path": str(tmp_path / "qualification_rerun_history.jsonl"),
        "items": [
            {
                "request_id": "req-1",
                "label": "integrated_operator_workflow",
                "returncode": 0,
                "latest_run_id": "run-1",
                "before_rate": 0.0,
                "after_rate": 0.5,
                "latest_run": {"outcome": "artifact_verified", "success": True},
                "previous_run": {"outcome": "artifact_missing", "success": False},
                "recommended_action": "continue_testing",
            }
        ],
    }

    rendered = render_markdown(payload)

    assert "Recommended action" in rendered
    assert "`continue_testing`" in rendered
