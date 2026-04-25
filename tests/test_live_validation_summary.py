"""Tests for live validation qualification summary tooling."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.summarize_live_validations import (
    aggregate_reports,
    load_reports,
    render_markdown,
)


def _write_report(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "live_debug_validation_report.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_load_reports_and_aggregate(tmp_path: Path) -> None:
    _write_report(
        tmp_path / "run-a",
        {
            "run_id": "run-a",
            "started_at": "2026-04-09T00:00:00+00:00",
            "finished_at": "2026-04-09T00:01:00+00:00",
            "model": "kimi-coding/k2p5",
            "embedding_model": "google/gemini-embedding-2-preview",
            "direct_checks": {
                "runtime_status": {"success": True},
                "browser_probe": {"success": False},
            },
            "agent_checks": [
                {
                    "label": "writing_workflow",
                    "material_success": True,
                    "timed_out": False,
                    "tool_message_delta": 6,
                    "outcome": "artifact_verified",
                },
                {
                    "label": "integrated_operator_workflow",
                    "material_success": False,
                    "timed_out": False,
                    "tool_message_delta": 18,
                    "outcome": "artifact_missing",
                    "response": "first attempt failed",
                },
            ],
        },
    )
    _write_report(
        tmp_path / "run-b",
        {
            "run_id": "run-b",
            "started_at": "2026-04-10T00:00:00+00:00",
            "finished_at": "2026-04-10T00:02:30+00:00",
            "model": "kimi-coding/k2p5",
            "embedding_model": "google/gemini-embedding-2-preview",
            "direct_checks": {
                "runtime_status": {"success": True},
            },
            "agent_checks": [
                {
                    "label": "writing_workflow",
                    "material_success": True,
                    "timed_out": False,
                    "tool_message_delta": 5,
                    "outcome": "artifact_verified",
                },
                {
                    "label": "integrated_operator_workflow",
                    "material_success": True,
                    "timed_out": False,
                    "tool_message_delta": 20,
                    "outcome": "artifact_verified",
                },
            ],
        },
    )

    reports = load_reports(tmp_path)
    assert [payload["run_id"] for _, payload in reports] == ["run-b", "run-a"]

    summary = aggregate_reports(reports)
    assert summary["summary_scope"] == "retained_runs_dir_snapshot"
    assert summary["total_runs"] == 2
    assert summary["direct_success_rate"] == 0.667
    assert summary["agent_success_rate"] == 0.75
    assert summary["agent_checks"]["writing_workflow"]["success_rate"] == 1.0
    assert summary["agent_checks"]["integrated_operator_workflow"]["success_rate"] == 0.5
    assert summary["agent_checks"]["integrated_operator_workflow"]["recent_failures"][0]["run_id"] == "run-a"


def test_render_markdown_includes_summary_fields(tmp_path: Path) -> None:
    _write_report(
        tmp_path / "run-a",
        {
            "run_id": "run-a",
            "started_at": "2026-04-09T00:00:00+00:00",
            "finished_at": "2026-04-09T00:01:00+00:00",
            "model": "kimi-coding/k2p5",
            "embedding_model": "google/gemini-embedding-2-preview",
            "direct_checks": {},
            "agent_checks": [
                {
                    "label": "project_management_workflow",
                    "material_success": True,
                    "timed_out": False,
                    "tool_message_delta": 4,
                    "outcome": "artifact_verified",
                }
            ],
        },
    )
    summary = aggregate_reports(load_reports(tmp_path))
    rendered = render_markdown(summary)
    assert "OpenCAS Live Validation Qualification Summary" in rendered
    assert "Scope: `current retained run folders`" in rendered
    assert "qualification_remediation_rollup.md" in rendered
    assert "project_management_workflow" in rendered
    assert "artifact_verified" in rendered


def test_render_markdown_formats_missing_rates_without_none(tmp_path: Path) -> None:
    _write_report(
        tmp_path / "run-a",
        {
            "run_id": "run-a",
            "started_at": "2026-04-09T00:00:00+00:00",
            "finished_at": "2026-04-09T00:01:00+00:00",
            "model": "kimi-coding/k2p5",
            "embedding_model": "google/gemini-embedding-2-preview",
            "direct_checks": {},
            "agent_checks": [],
        },
    )
    summary = aggregate_reports(load_reports(tmp_path))
    rendered = render_markdown(summary)

    assert "Direct success rate: `-`" in rendered
    assert "Agent success rate: `-`" in rendered
    assert "`None`" not in rendered


def test_legacy_agent_checks_are_inferred_correctly(tmp_path: Path) -> None:
    _write_report(
        tmp_path / "run-a",
        {
            "run_id": "run-a",
            "started_at": "2026-04-09T00:00:00+00:00",
            "finished_at": "2026-04-09T00:01:00+00:00",
            "model": "kimi-coding/k2p5",
            "embedding_model": "google/gemini-embedding-2-preview",
            "direct_checks": {},
            "agent_checks": [
                {
                    "label": "browser_probe",
                    "timed_out": False,
                    "tool_message_delta": 3,
                    "response": "Loaded page successfully.",
                },
                {
                    "label": "write_project_note",
                    "timed_out": False,
                    "tool_message_delta": 2,
                    "expected_file": "/tmp/note.md",
                    "expected_file_exists": True,
                    "response": "Created file.",
                },
            ],
        },
    )
    summary = aggregate_reports(load_reports(tmp_path))
    assert summary["agent_success_rate"] == 1.0
    assert summary["agent_checks"]["browser_probe"]["success_rate"] == 1.0
    assert summary["agent_checks"]["browser_probe"]["outcomes"] == {"completed": 1}
    assert summary["agent_checks"]["write_project_note"]["outcomes"] == {"artifact_verified": 1}
