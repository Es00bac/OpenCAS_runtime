"""Tests for operations API routes."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from opencas.api.routes.operations import (
    QUALIFICATION_RERUN_HISTORY_PATH,
    QUALIFICATION_SUMMARY_PATH,
    VALIDATION_RUNS_DIR,
    build_operations_router,
)
from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.models import WorkObject, WorkStage
from opencas.planning.models import PlanAction, PlanEntry


def _make_mock_runtime():
    runtime = MagicMock()
    runtime.ctx = MagicMock()
    runtime.ctx.config = MagicMock()
    runtime.ctx.config.state_dir = "/tmp/opencas-test-state"

    # Process supervisor
    runtime.process_supervisor = MagicMock()
    runtime.process_supervisor.snapshot.return_value = {
        "total_count": 1,
        "running_count": 1,
        "completed_count": 0,
        "scope_count": 1,
        "entries": [
            {
                "process_id": "proc-001",
                "pid": 4242,
                "scope_key": "qualification",
                "command": "python scripts/run_qualification_cycle.py --agent-check-label integrated_operator_workflow",
                "cwd": "(workspace_root)",
                "metadata": {
                    "kind": "qualification_rerun",
                    "source_label": "integrated_operator_workflow",
                    "source_note": "Treat this as a coordination-budget issue first.",
                    "requested_at": 1700000001.0,
                },
                "running": True,
                "returncode": None,
                "created_at": 1700000000.0,
                "stdout_preview": "cycle output",
                "stderr_preview": "",
                "last_polled_at": 1700000005.0,
            }
        ],
    }
    runtime.process_supervisor.poll.return_value = {
        "found": True,
        "pid": 4242,
        "command": "python scripts/run_qualification_cycle.py --agent-check-label integrated_operator_workflow",
        "metadata": {
            "kind": "qualification_rerun",
            "source_label": "integrated_operator_workflow",
            "source_note": "Treat this as a coordination-budget issue first.",
            "requested_at": 1700000001.0,
        },
        "running": True,
        "returncode": None,
        "stdout": "cycle output",
        "stderr": "",
    }
    runtime.process_supervisor.start.return_value = "proc-001"
    runtime.process_supervisor.kill.return_value = True
    runtime.process_supervisor.remove.return_value = True

    # PTY supervisor
    runtime.pty_supervisor = MagicMock()
    runtime.pty_supervisor.snapshot.return_value = {
        "total_count": 1,
        "running_count": 1,
        "completed_count": 0,
        "scope_count": 1,
        "entries": [
            {
                "session_id": "pty-001",
                "pid": 12345,
                "scope_key": "test",
                "command": "claude",
                "cwd": "/tmp",
                "running": True,
                "returncode": None,
                "rows": 24,
                "cols": 80,
                "created_at": 1700000000.0,
                "last_observed_at": 1700000100.0,
                "last_screen_state": {"app": "claude", "mode": "interactive", "ready_for_input": True},
                "last_cleaned_output": "Ready for input",
            }
        ],
    }
    runtime.pty_supervisor.kill.return_value = True
    runtime.pty_supervisor.remove.return_value = True
    runtime.pty_supervisor.clear.return_value = 1
    runtime.pty_supervisor.observe_until_quiet.return_value = {
        "found": True,
        "session_id": "pty-001",
        "command": "claude",
        "running": True,
        "returncode": None,
        "cleaned_combined_output": "Refreshed output",
        "screen_state": {"app": "claude", "mode": "interactive", "ready_for_input": True},
        "elapsed_ms": 200,
    }

    # Browser supervisor
    runtime.browser_supervisor = MagicMock()
    runtime.browser_supervisor.snapshot.return_value = {
        "total_count": 1,
        "entries": [
            {
                "session_id": "browser-001",
                "scope_key": "test-browser",
                "url": "https://example.com",
                "title": "Example Domain",
                "headless": True,
                "viewport": "1280x900",
                "created_at": 1700000200.0,
                "last_observed_at": 1700000210.0,
                "last_snapshot_text": "Example Domain body",
                "last_snapshot_links": [{"text": "More information", "href": "https://iana.org"}],
                "last_snapshot_screenshot": None,
            }
        ],
    }
    runtime.browser_supervisor.snapshot_page = AsyncMock(return_value={
        "found": True,
        "url": "https://example.com",
        "title": "Example Domain",
        "text": "Example Domain body",
        "links": [{"text": "More information", "href": "https://iana.org"}],
    })
    runtime.browser_supervisor.navigate = AsyncMock(return_value={
        "found": True,
        "url": "https://example.com/next",
        "title": "Next Page",
        "status": 200,
    })
    runtime.browser_supervisor.click = AsyncMock(return_value={
        "found": True,
        "url": "https://example.com/next",
        "title": "Next Page",
    })
    runtime.browser_supervisor.type_text = AsyncMock(return_value={
        "found": True,
        "url": "https://example.com/form",
        "title": "Form Page",
    })
    runtime.browser_supervisor.press = AsyncMock(return_value={
        "found": True,
        "url": "https://example.com/form",
        "title": "Form Page",
    })
    runtime.browser_supervisor.wait = AsyncMock(return_value={
        "found": True,
        "url": "https://example.com/form",
        "title": "Form Page",
    })
    runtime.browser_supervisor.close = AsyncMock(return_value=True)
    runtime.browser_supervisor.clear = AsyncMock(return_value=1)

    # Receipt store
    receipt_store = MagicMock()
    receipt_store.list_recent = AsyncMock(return_value=[])
    receipt_store.get = AsyncMock(return_value=None)
    runtime.ctx.receipt_store = receipt_store

    # Work store
    work_store = MagicMock()
    work_store.summary_counts = AsyncMock(return_value={"total": 0, "ready": 0, "blocked": 0})
    work_store.list_all = AsyncMock(return_value=[])
    work_store.get = AsyncMock(return_value=None)
    work_store.save = AsyncMock(return_value=None)
    runtime.ctx.work_store = work_store

    # Commitment store
    commitment_store = MagicMock()
    commitment_store.list_by_status = AsyncMock(return_value=[])
    commitment_store.get = AsyncMock(return_value=None)
    commitment_store.save = AsyncMock(return_value=None)
    runtime.commitment_store = commitment_store

    # Plan store
    plan_store = MagicMock()
    plan_store.list_active = AsyncMock(return_value=[])
    plan_store.get_plan = AsyncMock(return_value=None)
    plan_store.get_actions = AsyncMock(return_value=[])
    plan_store.update_content = AsyncMock(return_value=True)
    plan_store.set_status = AsyncMock(return_value=True)
    runtime.ctx.plan_store = plan_store

    return runtime


def _make_test_app(runtime):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(build_operations_router(runtime))
    return app


def test_list_sessions() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_processes"] == 1
    assert data["processes"][0]["process_id"] == "proc-001"
    assert data["total_pty"] == 1
    assert len(data["pty"]) == 1
    assert data["pty"][0]["session_id"] == "pty-001"
    assert data["pty"][0]["command"] == "claude"
    assert data["pty"][0]["running"] is True
    assert data["pty"][0]["cwd"] == "/tmp"
    assert data["pty"][0]["last_screen_state"]["app"] == "claude"
    assert data["pty"][0]["last_cleaned_output"] == "Ready for input"
    assert data["total_browser"] == 1
    assert data["browser"][0]["session_id"] == "browser-001"
    assert data["scopes"] == [
        {"scope_key": "qualification", "process_count": 1, "pty_count": 0, "browser_count": 0},
        {"scope_key": "test", "process_count": 0, "pty_count": 1, "browser_count": 0},
        {"scope_key": "test-browser", "process_count": 0, "pty_count": 0, "browser_count": 1},
    ]
    assert data["current_scope"] is None


def test_get_qualification_summary() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    payload = {
        "total_runs": 3,
        "total_direct_checks": 10,
        "total_agent_checks": 8,
        "direct_success_rate": 0.9,
        "agent_success_rate": 0.75,
        "average_run_duration_seconds": 42.5,
        "models": ["kimi-coding/k2p5"],
        "embedding_models": ["google/gemini-embedding-2-preview"],
        "agent_checks": {
            "integrated_operator_workflow": {
                "runs": 2,
                "successes": 1,
                "failures": 1,
                "success_rate": 0.5,
                "timeouts": 0,
            },
            "writing_workflow": {
                "runs": 1,
                "successes": 1,
                "failures": 0,
                "success_rate": 1.0,
                "timeouts": 0,
            },
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "live_validation_summary.json"
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        runs_dir = Path(tmpdir) / "runs"
        history_path = Path(tmpdir) / "qualification_rerun_history.jsonl"
        remediation_path = Path(tmpdir) / "qualification_remediation_rollup.json"
        history_path.write_text(
            (
                json.dumps(
                    {
                        "event": "requested",
                        "request_id": "req-000",
                        "process_id": "proc-000",
                        "label": "integrated_operator_workflow",
                        "source_label": "integrated_operator_workflow",
                        "source_note": "Treat this as a coordination-budget issue first.",
                        "requested_at": 1700000001.0,
                        "command": "python scripts/run_qualification_cycle.py --agent-check-label integrated_operator_workflow",
                        "iterations": 2,
                        "include_direct_checks": False,
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "event": "completed",
                        "request_id": "req-000",
                        "labels": ["integrated_operator_workflow"],
                        "completed_at": 1700000100.0,
                        "returncode": 0,
                        "iterations_executed": 2,
                        "generated_run_ids": ["debug-validation-20260409-020000"],
                        "latest_run_id": "debug-validation-20260409-020000",
                    }
                )
                + "\n"
            ),
            encoding="utf-8",
        )
        remediation_path.write_text(
            json.dumps(
                {
                    "count": 1,
                    "items": [
                        {
                            "request_id": "req-000",
                            "label": "integrated_operator_workflow",
                            "returncode": 0,
                            "latest_run_id": "debug-validation-20260409-020000",
                            "before_rate": 0.0,
                            "after_rate": 0.5,
                            "recommended_action": "continue_testing",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        run_a = runs_dir / "debug-validation-20260409-010000"
        run_a.mkdir(parents=True, exist_ok=True)
        (run_a / "live_debug_validation_report.json").write_text(
            json.dumps(
                {
                    "run_id": "debug-validation-20260409-010000",
                    "started_at": "2026-04-09T01:00:00+00:00",
                    "finished_at": "2026-04-09T01:01:30+00:00",
                    "agent_checks": [
                        {"label": "integrated_operator_workflow", "material_success": False, "outcome": "artifact_missing"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        run_b = runs_dir / "debug-validation-20260409-020000"
        run_b.mkdir(parents=True, exist_ok=True)
        (run_b / "live_debug_validation_report.json").write_text(
            json.dumps(
                {
                    "run_id": "debug-validation-20260409-020000",
                    "started_at": "2026-04-09T02:00:00+00:00",
                    "finished_at": "2026-04-09T02:01:30+00:00",
                    "agent_checks": [
                        {"label": "integrated_operator_workflow", "material_success": True, "outcome": "artifact_verified"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        with (
            patch("opencas.api.routes.operations.QUALIFICATION_SUMMARY_PATH", summary_path),
            patch("opencas.api.routes.operations.VALIDATION_RUNS_DIR", runs_dir),
            patch("opencas.api.routes.operations.QUALIFICATION_RERUN_HISTORY_PATH", history_path),
            patch("opencas.api.routes.operations.QUALIFICATION_REMEDIATION_PATH", remediation_path),
        ):
            resp = client.get("/api/operations/qualification")

    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["summary"]["total_runs"] == 3
    assert data["weakest_checks"][0]["label"] == "integrated_operator_workflow"
    assert data["recommended_reruns"][0]["label"] == "integrated_operator_workflow"
    assert data["recommended_reruns"][0]["command"][-4:] == ["--agent-check-label", "integrated_operator_workflow", "--iterations", "2"]
    assert "coordination-budget" in data["recommended_reruns"][0]["note"]
    assert data["recommended_reruns"][0]["comparison"]["trend"] == "improved"
    assert data["recommended_reruns"][0]["comparison"]["latest"]["run_id"] == "debug-validation-20260409-020000"
    assert data["recommended_reruns"][0]["comparison"]["previous"]["run_id"] == "debug-validation-20260409-010000"
    assert data["recommended_reruns"][0]["rate_window"]["recent_success_rate"] == 0.5
    assert data["recommended_reruns"][0]["rate_window"]["previous_success_rate"] is None
    assert data["recommended_reruns"][0]["active_rerun"]["process_id"] == "proc-001"
    assert data["recommended_reruns"][0]["last_completed_run"]["run_id"] == "debug-validation-20260409-020000"
    assert data["recommended_reruns"][0]["last_completed_run"]["outcome"] == "artifact_verified"
    assert data["recommended_reruns"][0]["last_request"]["process_id"] == "proc-000"
    assert data["recommended_reruns"][0]["last_request"]["label"] == "integrated_operator_workflow"
    assert data["recommended_reruns"][0]["last_completion_event"]["request_id"] == "req-000"
    assert data["recommended_reruns"][0]["last_completion_event"]["latest_run_id"] == "debug-validation-20260409-020000"
    assert data["active_reruns"][0]["process_id"] == "proc-001"
    assert data["active_reruns"][0]["metadata"]["source_label"] == "integrated_operator_workflow"
    assert [item["run_id"] for item in data["recent_runs"]] == [
        "debug-validation-20260409-020000",
        "debug-validation-20260409-010000",
    ]
    assert data["recent_rerun_history"][0]["event"] == "completed"
    assert data["recent_rerun_history"][0]["request_id"] == "req-000"
    assert data["recent_rerun_history"][0]["comparison"]["trend"] == "improved"
    assert data["recent_rerun_history"][0]["comparison"]["latest"]["outcome"] == "artifact_verified"
    assert data["recent_rerun_history"][0]["rate_window"]["recent_success_rate"] == 0.5
    assert data["recent_rerun_history"][1]["event"] == "requested"
    assert data["recent_rerun_history"][1]["process_id"] == "proc-000"
    assert data["remediation_rollup"]["found"] is True
    assert data["remediation_rollup"]["items"][0]["recommended_action"] == "continue_testing"


def test_get_qualification_label_detail() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    payload = {
        "total_runs": 3,
        "total_direct_checks": 10,
        "total_agent_checks": 8,
        "direct_success_rate": 0.9,
        "agent_success_rate": 0.75,
        "average_run_duration_seconds": 42.5,
        "models": ["kimi-coding/k2p5"],
        "embedding_models": ["google/gemini-embedding-2-preview"],
        "agent_checks": {
            "integrated_operator_workflow": {
                "runs": 2,
                "successes": 1,
                "failures": 1,
                "success_rate": 0.5,
                "timeouts": 0,
                "recent_failures": [],
            },
            "writing_workflow": {
                "runs": 1,
                "successes": 1,
                "failures": 0,
                "success_rate": 1.0,
                "timeouts": 0,
            },
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "live_validation_summary.json"
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        runs_dir = Path(tmpdir) / "runs"
        run_a = runs_dir / "debug-validation-20260409-010000"
        run_a.mkdir(parents=True, exist_ok=True)
        (run_a / "live_debug_validation_report.json").write_text(
            json.dumps(
                {
                    "run_id": "debug-validation-20260409-010000",
                    "started_at": "2026-04-09T01:00:00+00:00",
                    "finished_at": "2026-04-09T01:01:30+00:00",
                    "agent_checks": [
                        {"label": "integrated_operator_workflow", "material_success": False, "outcome": "artifact_missing"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        run_b = runs_dir / "debug-validation-20260409-020000"
        run_b.mkdir(parents=True, exist_ok=True)
        (run_b / "live_debug_validation_report.json").write_text(
            json.dumps(
                {
                    "run_id": "debug-validation-20260409-020000",
                    "started_at": "2026-04-09T02:00:00+00:00",
                    "finished_at": "2026-04-09T02:01:30+00:00",
                    "agent_checks": [
                        {"label": "integrated_operator_workflow", "material_success": True, "outcome": "artifact_verified"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        history_path = Path(tmpdir) / "qualification_rerun_history.jsonl"
        history_path.write_text(
            json.dumps({"event": "requested", "request_id": "req-1", "process_id": "proc-1", "label": "integrated_operator_workflow", "labels": ["integrated_operator_workflow"], "requested_at": 1700000001.0})
            + "\n"
            + json.dumps({"event": "completed", "request_id": "req-1", "labels": ["integrated_operator_workflow"], "completed_at": 1700000100.0, "returncode": 0, "generated_run_ids": ["debug-validation-20260409-020000"], "latest_run_id": "debug-validation-20260409-020000"})
            + "\n",
            encoding="utf-8",
        )
        with (
            patch("opencas.api.routes.operations.QUALIFICATION_SUMMARY_PATH", summary_path),
            patch("opencas.api.routes.operations.VALIDATION_RUNS_DIR", runs_dir),
            patch("opencas.api.routes.operations.QUALIFICATION_RERUN_HISTORY_PATH", history_path),
        ):
            resp = client.get("/api/operations/qualification/labels/integrated_operator_workflow")

    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["label"] == "integrated_operator_workflow"
    assert data["detail"]["stats"]["success_rate"] == 0.5
    assert data["detail"]["comparison"]["trend"] == "improved"
    assert data["detail"]["rate_window"]["recent_success_rate"] == 0.5
    assert data["detail"]["recommendation"]["label"] == "integrated_operator_workflow"
    assert data["detail"]["recent_runs"][0]["run_id"] == "debug-validation-20260409-020000"
    assert data["detail"]["recent_rerun_history"][0]["event"] == "completed"
    assert data["detail"]["active_reruns"][0]["process_id"] == "proc-001"


def test_get_qualification_rerun_detail() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    latest_payload = {
        "run_id": "debug-validation-20260409-020000",
        "started_at": "2026-04-09T02:00:00+00:00",
        "finished_at": "2026-04-09T02:01:30+00:00",
        "model": "kimi-coding/k2p5",
        "embedding_model": "google/gemini-embedding-2-preview",
        "agent_checks": [
            {"label": "integrated_operator_workflow", "material_success": True, "outcome": "artifact_verified", "response": "ok"},
        ],
    }
    earlier_payload = {
        "run_id": "debug-validation-20260409-015500",
        "started_at": "2026-04-09T01:55:00+00:00",
        "finished_at": "2026-04-09T01:56:30+00:00",
        "model": "kimi-coding/k2p5",
        "embedding_model": "google/gemini-embedding-2-preview",
        "agent_checks": [
            {"label": "integrated_operator_workflow", "material_success": False, "outcome": "failed", "response": "not ok"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        runs_dir = Path(tmpdir) / "runs"
        earlier_run_dir = runs_dir / "debug-validation-20260409-015500"
        earlier_run_dir.mkdir(parents=True, exist_ok=True)
        (earlier_run_dir / "live_debug_validation_report.json").write_text(json.dumps(earlier_payload), encoding="utf-8")
        run_dir = runs_dir / "debug-validation-20260409-020000"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "live_debug_validation_report.json").write_text(json.dumps(latest_payload), encoding="utf-8")
        history_path = Path(tmpdir) / "qualification_rerun_history.jsonl"
        history_path.write_text(
            json.dumps(
                {
                    "event": "requested",
                    "request_id": "req-xyz",
                    "process_id": "proc-001",
                    "label": "integrated_operator_workflow",
                    "source_note": "Treat this as a coordination-budget issue first.",
                    "requested_at": 1700000001.0,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "event": "completed",
                    "request_id": "req-xyz",
                    "labels": ["integrated_operator_workflow"],
                    "completed_at": 1700000100.0,
                    "returncode": 0,
                    "generated_run_ids": ["debug-validation-20260409-015500", "debug-validation-20260409-020000"],
                    "latest_run_id": "debug-validation-20260409-020000",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runtime.process_supervisor.snapshot.return_value["entries"][0]["metadata"]["request_id"] = "req-xyz"
        with (
            patch("opencas.api.routes.operations.VALIDATION_RUNS_DIR", runs_dir),
            patch("opencas.api.routes.operations.QUALIFICATION_RERUN_HISTORY_PATH", history_path),
        ):
            resp = client.get("/api/operations/qualification/reruns/req-xyz")

    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["request_id"] == "req-xyz"
    assert data["detail"]["request"]["process_id"] == "proc-001"
    assert data["detail"]["completion"]["latest_run_id"] == "debug-validation-20260409-020000"
    assert data["detail"]["labels"] == ["integrated_operator_workflow"]
    assert data["detail"]["latest_run_detail"]["run_id"] == "debug-validation-20260409-020000"
    assert [item["run_id"] for item in data["detail"]["generated_runs"]] == [
        "debug-validation-20260409-015500",
        "debug-validation-20260409-020000",
    ]
    assert data["detail"]["label_outcomes"][0]["label"] == "integrated_operator_workflow"
    assert data["detail"]["label_outcomes"][0]["latest_success"] is True
    assert data["detail"]["label_outcomes"][0]["comparison"]["trend"] == "improved"
    assert data["detail"]["request_progress"][0]["label"] == "integrated_operator_workflow"
    assert data["detail"]["request_progress"][0]["first_success"] is False
    assert data["detail"]["request_progress"][0]["latest_success"] is True
    assert data["detail"]["request_progress"][0]["trend"] == "improved"
    assert data["detail"]["active_processes"][0]["process_id"] == "proc-001"


def test_get_qualification_summary_missing() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    missing_path = QUALIFICATION_SUMMARY_PATH.parent / "does-not-exist.json"
    with patch("opencas.api.routes.operations.QUALIFICATION_SUMMARY_PATH", missing_path):
        resp = client.get("/api/operations/qualification")

    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is False
    assert data["path"].endswith("does-not-exist.json")


def test_list_validation_runs() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    payload = {
        "run_id": "debug-validation-20260409-010000",
        "started_at": "2026-04-09T01:00:00+00:00",
        "finished_at": "2026-04-09T01:01:30+00:00",
        "model": "kimi-coding/k2p5",
        "direct_checks": {
            "runtime_status": {"success": True},
            "browser_probe": {"success": False},
        },
        "agent_checks": [
            {"label": "writing_workflow", "material_success": True},
            {"label": "integrated_operator_workflow", "material_success": False},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "debug-validation-20260409-010000"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "live_debug_validation_report.json").write_text(json.dumps(payload), encoding="utf-8")
        with patch("opencas.api.routes.operations.VALIDATION_RUNS_DIR", Path(tmpdir)):
            resp = client.get("/api/operations/validation-runs?limit=5")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["label_filter"] is None
    assert data["items"][0]["run_id"] == "debug-validation-20260409-010000"
    assert data["items"][0]["direct_successes"] == 1
    assert data["items"][0]["agent_successes"] == 1
    assert data["items"][0]["failed_labels"] == ["integrated_operator_workflow"]
    assert data["items"][0]["aborted"] is False


def test_list_validation_runs_with_label_filter() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    payload_one = {
        "run_id": "debug-validation-20260409-010000",
        "started_at": "2026-04-09T01:00:00+00:00",
        "finished_at": "2026-04-09T01:01:30+00:00",
        "model": "kimi-coding/k2p5",
        "direct_checks": {},
        "agent_checks": [
            {"label": "writing_workflow", "material_success": True},
        ],
    }
    payload_two = {
        "run_id": "debug-validation-20260409-020000",
        "started_at": "2026-04-09T02:00:00+00:00",
        "finished_at": "2026-04-09T02:01:30+00:00",
        "model": "kimi-coding/k2p5",
        "direct_checks": {},
        "agent_checks": [
            {"label": "integrated_operator_workflow", "material_success": False},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        run_one = Path(tmpdir) / "debug-validation-20260409-010000"
        run_one.mkdir(parents=True, exist_ok=True)
        (run_one / "live_debug_validation_report.json").write_text(json.dumps(payload_one), encoding="utf-8")
        run_two = Path(tmpdir) / "debug-validation-20260409-020000"
        run_two.mkdir(parents=True, exist_ok=True)
        (run_two / "live_debug_validation_report.json").write_text(json.dumps(payload_two), encoding="utf-8")
        with patch("opencas.api.routes.operations.VALIDATION_RUNS_DIR", Path(tmpdir)):
            resp = client.get("/api/operations/validation-runs?limit=10&label=integrated_operator_workflow")

    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["label_filter"] == "integrated_operator_workflow"
    assert data["items"][0]["run_id"] == "debug-validation-20260409-020000"
    assert data["items"][0]["failed_labels"] == ["integrated_operator_workflow"]


def test_get_validation_run_detail() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    payload = {
        "run_id": "debug-validation-20260409-010000",
        "started_at": "2026-04-09T01:00:00+00:00",
        "finished_at": "2026-04-09T01:01:30+00:00",
        "model": "kimi-coding/k2p5",
        "embedding_model": "google/gemini-embedding-2-preview",
        "direct_checks": {
            "runtime_status": {"success": True},
        },
        "agent_checks": [
            {
                "label": "integrated_operator_workflow",
                "material_success": False,
                "outcome": "artifact_missing",
                "response": "artifact not created",
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "debug-validation-20260409-010000"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "live_debug_validation_report.json").write_text(json.dumps(payload), encoding="utf-8")
        with patch("opencas.api.routes.operations.VALIDATION_RUNS_DIR", Path(tmpdir)):
            resp = client.get("/api/operations/validation-runs/debug-validation-20260409-010000")

    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["run"]["run_id"] == "debug-validation-20260409-010000"
    assert data["run"]["focus_label"] is None
    assert data["run"]["failed_agent_checks"][0]["label"] == "integrated_operator_workflow"


def test_get_validation_run_detail_with_label_focus() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    payload = {
        "run_id": "debug-validation-20260409-010000",
        "started_at": "2026-04-09T01:00:00+00:00",
        "finished_at": "2026-04-09T01:01:30+00:00",
        "model": "kimi-coding/k2p5",
        "embedding_model": "google/gemini-embedding-2-preview",
        "direct_checks": {"runtime_status": {"success": True}},
        "agent_checks": [
            {
                "label": "integrated_operator_workflow",
                "material_success": False,
                "outcome": "artifact_missing",
                "response": "artifact not created",
            },
            {
                "label": "writing_workflow",
                "material_success": True,
                "outcome": "artifact_verified",
                "response": "draft complete",
            },
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "debug-validation-20260409-010000"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "live_debug_validation_report.json").write_text(json.dumps(payload), encoding="utf-8")
        history_path = Path(tmpdir) / "empty-history.jsonl"
        with (
            patch("opencas.api.routes.operations.VALIDATION_RUNS_DIR", Path(tmpdir)),
            patch("opencas.api.routes.operations.QUALIFICATION_RERUN_HISTORY_PATH", history_path),
        ):
            resp = client.get("/api/operations/validation-runs/debug-validation-20260409-010000?label=integrated_operator_workflow")

    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["run"]["focus_label"] == "integrated_operator_workflow"
    assert len(data["run"]["matching_agent_checks"]) == 1
    assert data["run"]["matching_agent_checks"][0]["label"] == "integrated_operator_workflow"
    assert data["run"]["rerun_request"] is None
    assert data["run"]["rerun_completion"] is None


def test_get_validation_run_detail_includes_rerun_provenance() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    payload = {
        "run_id": "debug-validation-20260409-020000",
        "started_at": "2026-04-09T02:00:00+00:00",
        "finished_at": "2026-04-09T02:01:30+00:00",
        "model": "kimi-coding/k2p5",
        "direct_checks": {},
        "agent_checks": [
            {"label": "integrated_operator_workflow", "material_success": True, "outcome": "artifact_verified"},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "debug-validation-20260409-020000"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "live_debug_validation_report.json").write_text(json.dumps(payload), encoding="utf-8")
        history_path = Path(tmpdir) / "qualification_rerun_history.jsonl"
        history_path.write_text(
            json.dumps(
                {
                    "event": "requested",
                    "request_id": "req-777",
                    "process_id": "proc-777",
                    "label": "integrated_operator_workflow",
                    "requested_at": 1700000200.0,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "event": "completed",
                    "request_id": "req-777",
                    "labels": ["integrated_operator_workflow"],
                    "completed_at": 1700000300.0,
                    "returncode": 0,
                    "generated_run_ids": ["debug-validation-20260409-020000"],
                    "latest_run_id": "debug-validation-20260409-020000",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with (
            patch("opencas.api.routes.operations.VALIDATION_RUNS_DIR", Path(tmpdir)),
            patch("opencas.api.routes.operations.QUALIFICATION_RERUN_HISTORY_PATH", history_path),
        ):
            resp = client.get("/api/operations/validation-runs/debug-validation-20260409-020000?label=integrated_operator_workflow")

    assert resp.status_code == 200
    data = resp.json()
    assert data["run"]["rerun_request"]["request_id"] == "req-777"
    assert data["run"]["rerun_completion"]["latest_run_id"] == "debug-validation-20260409-020000"


def test_start_qualification_rerun() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "qualification_rerun_history.jsonl"
        with patch("opencas.api.routes.operations.QUALIFICATION_RERUN_HISTORY_PATH", history_path):
            resp = client.post(
                "/api/operations/qualification/reruns",
                json={
                    "label": "integrated_operator_workflow",
                    "iterations": 2,
                    "include_direct_checks": False,
                    "source_note": "Treat this as a coordination-budget issue first.",
                },
            )
            assert history_path.exists() is True
            recorded = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["process_id"] == "proc-001"
    runtime.process_supervisor.start.assert_called_once()
    called_scope, called_command = runtime.process_supervisor.start.call_args[0][:2]
    assert called_scope == "qualification"
    assert "run_qualification_cycle.py" in called_command
    assert "integrated_operator_workflow" in called_command
    assert runtime.process_supervisor.start.call_args.kwargs["metadata"]["source_label"] == "integrated_operator_workflow"
    assert runtime.process_supervisor.start.call_args.kwargs["metadata"]["source_note"] == "Treat this as a coordination-budget issue first."
    assert runtime.process_supervisor.start.call_args.kwargs["metadata"]["request_id"]
    assert data["metadata"]["source_label"] == "integrated_operator_workflow"
    assert data["metadata"]["source_note"] == "Treat this as a coordination-budget issue first."
    assert data["metadata"]["request_id"]
    assert data["history_entry"]["process_id"] == "proc-001"
    assert data["history_entry"]["event"] == "requested"
    assert data["history_entry"]["request_id"] == data["metadata"]["request_id"]
    assert recorded[0]["process_id"] == "proc-001"
    assert recorded[0]["label"] == "integrated_operator_workflow"


def test_list_sessions_with_scope_filter() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/sessions?scope_key=test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["current_scope"] == "test"
    runtime.process_supervisor.snapshot.assert_called_with(scope_key="test")
    runtime.pty_supervisor.snapshot.assert_called_with(scope_key="test")


def test_get_process_session_detail() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "empty-history.jsonl"
        with patch("opencas.api.routes.operations.QUALIFICATION_RERUN_HISTORY_PATH", history_path):
            resp = client.get("/api/operations/sessions/process/proc-001?scope_key=qualification")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["process"]["process_id"] == "proc-001"
    assert data["process"]["stdout_preview"] == "cycle output"
    assert data["process"]["metadata"]["source_label"] == "integrated_operator_workflow"
    assert data["process"]["rerun_request"] is None
    assert data["process"]["rerun_completion"] is None


def test_get_process_session_detail_includes_rerun_provenance() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = Path(tmpdir) / "qualification-process-history.jsonl"
        history_path.write_text(
            json.dumps(
                {
                    "event": "requested",
                    "request_id": "req-abc",
                    "process_id": "proc-001",
                    "label": "integrated_operator_workflow",
                    "requested_at": 1700000001.0,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "event": "completed",
                    "request_id": "req-abc",
                    "labels": ["integrated_operator_workflow"],
                    "completed_at": 1700000101.0,
                    "returncode": 0,
                    "latest_run_id": "debug-validation-20260409-020000",
                    "generated_run_ids": ["debug-validation-20260409-020000"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        runtime.process_supervisor.snapshot.return_value["entries"][0]["metadata"]["request_id"] = "req-abc"
        runtime.process_supervisor.poll.return_value["metadata"]["request_id"] = "req-abc"

        with patch("opencas.api.routes.operations.QUALIFICATION_RERUN_HISTORY_PATH", history_path):
            resp = client.get("/api/operations/sessions/process/proc-001?scope_key=qualification")

    assert resp.status_code == 200
    data = resp.json()
    assert data["process"]["rerun_request"]["request_id"] == "req-abc"
    assert data["process"]["rerun_completion"]["latest_run_id"] == "debug-validation-20260409-020000"
    assert data["process"]["polled"]["stdout"] == "cycle output"


def test_get_process_session_detail_refresh_overrides_running_state() -> None:
    runtime = _make_mock_runtime()
    runtime.process_supervisor.poll.return_value = {
        "found": True,
        "pid": 4242,
        "command": "python scripts/run_qualification_cycle.py --agent-check-label integrated_operator_workflow",
        "running": False,
        "returncode": 0,
        "stdout": "done",
        "stderr": "",
    }
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/sessions/process/proc-001?scope_key=qualification")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["process"]["running"] is False
    assert data["process"]["returncode"] == 0
    assert data["process"]["polled"]["stdout"] == "done"


def test_kill_process_session() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.delete("/api/operations/sessions/process/proc-001?scope_key=qualification")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    runtime.process_supervisor.kill.assert_called_once_with("qualification", "proc-001")
    runtime.process_supervisor.remove.assert_called_once_with("qualification", "proc-001")


def test_clear_process_sessions() -> None:
    runtime = _make_mock_runtime()
    runtime.process_supervisor.clear.return_value = 1
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.delete("/api/operations/sessions/process?scope_key=qualification")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["removed"] == 1
    assert data["scope_key"] == "qualification"
    runtime.process_supervisor.clear.assert_called_once_with("qualification")


def test_kill_pty_session() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.delete("/api/operations/sessions/pty/pty-001?scope_key=test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    runtime.pty_supervisor.kill.assert_called_once_with("test", "pty-001")
    runtime.pty_supervisor.remove.assert_called_once_with("test", "pty-001")


def test_clear_pty_sessions() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.delete("/api/operations/sessions/pty?scope_key=test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["removed"] == 1
    runtime.pty_supervisor.clear.assert_called_once_with("test")


def test_get_pty_session_detail() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/sessions/pty/pty-001?scope_key=test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["session"]["session_id"] == "pty-001"
    assert data["session"]["last_screen_state"]["app"] == "claude"
    assert data["observed"] is None


def test_get_pty_session_detail_with_refresh() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/sessions/pty/pty-001?scope_key=test&refresh=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["observed"]["cleaned_combined_output"] == "Refreshed output"
    assert data["session"]["last_cleaned_output"] == "Refreshed output"
    runtime.pty_supervisor.observe_until_quiet.assert_called_once()


def test_send_pty_input() -> None:
    runtime = _make_mock_runtime()
    runtime.pty_supervisor.write.return_value = True
    app = _make_test_app(runtime)
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmpdir:
        runtime.ctx.config.state_dir = tmpdir
        resp = client.post(
            "/api/operations/sessions/pty/pty-001/input?scope_key=test",
            json={"input": "hello\r", "observe": True, "idle_seconds": 0.2, "max_wait_seconds": 1.0},
        )
        detail = client.get("/api/operations/sessions/pty/pty-001?scope_key=test").json()
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["ok"] is True
    assert data["observed"]["cleaned_combined_output"] == "Refreshed output"
    assert data["recent_operator_actions"][0]["action"] == "pty_input"
    assert data["recent_operator_actions"][0]["input_preview"] == "hello\r"
    runtime.pty_supervisor.write.assert_called_once_with("test", "pty-001", "hello\r")
    runtime.pty_supervisor.observe_until_quiet.assert_called_once()
    assert detail["recent_operator_actions"][0]["action"] == "pty_input"
    assert detail["recent_operator_actions"][0]["target_id"] == "pty-001"


def test_get_browser_session_detail() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/sessions/browser/browser-001?scope_key=test-browser")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["session"]["session_id"] == "browser-001"
    assert data["session"]["url"] == "https://example.com"
    assert data["observed"] is None


def test_get_browser_session_detail_with_refresh() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/sessions/browser/browser-001?scope_key=test-browser&refresh=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["observed"]["title"] == "Example Domain"
    assert data["session"]["last_snapshot_text"] == "Example Domain body"
    runtime.browser_supervisor.snapshot_page.assert_awaited_once()


def test_navigate_browser_session() -> None:
    runtime = _make_mock_runtime()
    runtime.browser_supervisor.snapshot.side_effect = [
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com/next",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com/next",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
    ]
    app = _make_test_app(runtime)
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmpdir:
        runtime.ctx.config.state_dir = tmpdir
        resp = client.post(
            "/api/operations/sessions/browser/browser-001/navigate?scope_key=test-browser",
            json={"url": "https://example.com/next", "refresh": True},
        )
        detail = client.get("/api/operations/sessions/browser/browser-001?scope_key=test-browser").json()
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["navigate"]["status"] == 200
    assert data["session"]["url"] == "https://example.com/next"
    assert data["recent_operator_actions"][0]["action"] == "browser_navigate"
    assert data["recent_operator_actions"][0]["url"] == "https://example.com/next"
    runtime.browser_supervisor.navigate.assert_awaited_once()
    runtime.browser_supervisor.snapshot_page.assert_awaited()
    assert detail["recent_operator_actions"][0]["action"] == "browser_navigate"
    assert detail["recent_operator_actions"][0]["target_id"] == "browser-001"


def test_click_browser_session() -> None:
    runtime = _make_mock_runtime()
    runtime.browser_supervisor.snapshot.side_effect = [
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com/next",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
    ]
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.post(
        "/api/operations/sessions/browser/browser-001/click?scope_key=test-browser",
        json={"selector": "#go", "refresh": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["session"]["url"] == "https://example.com/next"
    runtime.browser_supervisor.click.assert_awaited_once()
    runtime.browser_supervisor.snapshot_page.assert_awaited()


def test_type_browser_session() -> None:
    runtime = _make_mock_runtime()
    runtime.browser_supervisor.snapshot.side_effect = [
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com/form",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com/form",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
    ]
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.post(
        "/api/operations/sessions/browser/browser-001/type?scope_key=test-browser",
        json={"selector": "#name", "text": "OpenCAS", "refresh": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["session"]["url"] == "https://example.com/form"
    runtime.browser_supervisor.type_text.assert_awaited_once()
    runtime.browser_supervisor.snapshot_page.assert_awaited()


def test_press_browser_session() -> None:
    runtime = _make_mock_runtime()
    runtime.browser_supervisor.snapshot.side_effect = [
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com/form",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com/form",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
    ]
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.post(
        "/api/operations/sessions/browser/browser-001/press?scope_key=test-browser",
        json={"key": "Enter", "refresh": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["session"]["url"] == "https://example.com/form"
    runtime.browser_supervisor.press.assert_awaited_once()
    runtime.browser_supervisor.snapshot_page.assert_awaited()


def test_wait_browser_session() -> None:
    runtime = _make_mock_runtime()
    runtime.browser_supervisor.snapshot.side_effect = [
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com/form",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com/form",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
    ]
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.post(
        "/api/operations/sessions/browser/browser-001/wait?scope_key=test-browser",
        json={"timeout_ms": 2500, "load_state": "domcontentloaded", "refresh": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["session"]["url"] == "https://example.com/form"
    runtime.browser_supervisor.wait.assert_awaited_once()
    runtime.browser_supervisor.snapshot_page.assert_awaited()


def test_get_work_item_detail() -> None:
    runtime = _make_mock_runtime()
    work_id = UUID("00000000-0000-0000-0000-000000000001")
    runtime.ctx.work_store.get = AsyncMock(return_value=WorkObject(
        work_id=work_id,
        content="Investigate dashboard state",
        stage=WorkStage.NOTE,
        project_id="proj-1",
        blocked_by=["dep-1"],
        meta={"owner": "ops"},
    ))
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/work/work-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["item"]["work_id"] == str(work_id)
    assert data["item"]["stage"] == "note"
    assert data["item"]["blocked_by"] == ["dep-1"]


def test_update_work_item() -> None:
    runtime = _make_mock_runtime()
    work_id = UUID("00000000-0000-0000-0000-000000000002")
    work = WorkObject(
        work_id=work_id,
        content="Old content",
        stage=WorkStage.NOTE,
        blocked_by=["dep-1"],
    )
    runtime.ctx.work_store.get = AsyncMock(return_value=work)
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.patch(
        "/api/operations/work/work-001",
        json={"stage": "project", "content": "New content", "blocked_by": []},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["item"]["stage"] == "project"
    assert data["item"]["content"] == "New content"
    assert data["item"]["blocked_by"] == []
    runtime.ctx.work_store.save.assert_awaited_once()


def test_get_commitment_detail() -> None:
    runtime = _make_mock_runtime()
    commitment_id = UUID("00000000-0000-0000-0000-000000000011")
    runtime.commitment_store.get = AsyncMock(return_value=Commitment(
        commitment_id=commitment_id,
        content="Stabilize operations dashboard",
        status=CommitmentStatus.ACTIVE,
        tags=["ops", "ui"],
    ))
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/commitments/commit-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["commitment"]["commitment_id"] == str(commitment_id)
    assert data["commitment"]["status"] == "active"
    assert data["commitment"]["tags"] == ["ops", "ui"]


def test_update_commitment() -> None:
    runtime = _make_mock_runtime()
    commitment_id = UUID("00000000-0000-0000-0000-000000000012")
    commitment = Commitment(
        commitment_id=commitment_id,
        content="Old commitment",
        status=CommitmentStatus.ACTIVE,
        priority=5.0,
        tags=["ops"],
    )
    runtime.commitment_store.get = AsyncMock(return_value=commitment)
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.patch(
        "/api/operations/commitments/commit-001",
        json={"status": "completed", "content": "Updated commitment", "priority": 8.0, "tags": ["done"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["commitment"]["status"] == "completed"
    assert data["commitment"]["content"] == "Updated commitment"
    assert data["commitment"]["priority"] == 8.0
    assert data["commitment"]["tags"] == ["done"]
    runtime.commitment_store.save.assert_awaited_once()


def test_update_plan() -> None:
    runtime = _make_mock_runtime()
    updated_plan = PlanEntry(
        plan_id="plan-001",
        status="completed",
        content="Updated plan body",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        project_id="proj-1",
        task_id="task-1",
    )
    runtime.ctx.plan_store.get_plan = AsyncMock(side_effect=[
        PlanEntry(
            plan_id="plan-001",
            status="active",
            content="Old plan body",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            project_id="proj-1",
            task_id="task-1",
        ),
        updated_plan,
    ])
    runtime.ctx.plan_store.get_actions = AsyncMock(return_value=[
        PlanAction(
            action_id="action-1",
            plan_id="plan-001",
            tool_name="workflow_create_plan",
            result_summary="created",
            success=True,
            timestamp=datetime.now(timezone.utc),
        )
    ])
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.patch(
        "/api/operations/plans/plan-001",
        json={"status": "completed", "content": "Updated plan body"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["plan"]["status"] == "completed"
    assert data["plan"]["content"] == "Updated plan body"
    assert data["actions"][0]["tool_name"] == "workflow_create_plan"
    assert data["actions"][0]["created_at"] is not None
    runtime.ctx.plan_store.update_content.assert_awaited_once_with("plan-001", "Updated plan body")
    runtime.ctx.plan_store.set_status.assert_awaited_once_with("plan-001", "completed")


def test_update_work_item_invalid_stage_returns_422() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.patch("/api/operations/work/work-001", json={"stage": "not-a-stage"})
    assert resp.status_code == 422


def test_update_commitment_invalid_status_returns_422() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.patch(
        "/api/operations/commitments/commit-001",
        json={"status": "not-a-status"},
    )
    assert resp.status_code == 422


def test_update_plan_invalid_status_returns_422() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.patch(
        "/api/operations/plans/plan-001",
        json={"status": "not-a-status"},
    )
    assert resp.status_code == 422


def test_capture_browser_session() -> None:
    runtime = _make_mock_runtime()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        screenshot_path = Path(tmp.name)
        tmp.write(b"\x89PNG\r\n\x1a\nfake")
    runtime.browser_supervisor.snapshot_page = AsyncMock(return_value={
        "found": True,
        "url": "https://example.com",
        "title": "Example Domain",
        "text": "Example Domain body",
        "links": [{"text": "More information", "href": "https://iana.org"}],
        "screenshot_path": str(screenshot_path),
    })
    runtime.browser_supervisor.snapshot.side_effect = [
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com",
                    "title": "Example Domain",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                }
            ],
        },
        {
            "total_count": 1,
            "entries": [
                {
                    "session_id": "browser-001",
                    "scope_key": "test-browser",
                    "url": "https://example.com",
                    "title": "Example Domain",
                    "headless": True,
                    "viewport": "1280x900",
                    "created_at": 1700000200.0,
                    "last_snapshot_screenshot": str(screenshot_path),
                }
            ],
        },
    ]
    app = _make_test_app(runtime)
    client = TestClient(app)

    try:
        resp = client.post(
            "/api/operations/sessions/browser/browser-001/capture?scope_key=test-browser",
            json={"full_page": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["capture"]["screenshot_path"] == str(screenshot_path)
        runtime.browser_supervisor.snapshot_page.assert_awaited_once()
    finally:
        screenshot_path.unlink(missing_ok=True)


def test_get_browser_session_screenshot() -> None:
    runtime = _make_mock_runtime()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        screenshot_path = Path(tmp.name)
        tmp.write(b"\x89PNG\r\n\x1a\nfake")
    runtime.browser_supervisor.snapshot.return_value = {
        "total_count": 1,
        "entries": [
            {
                "session_id": "browser-001",
                "scope_key": "test-browser",
                "url": "https://example.com",
                "title": "Example Domain",
                "headless": True,
                "viewport": "1280x900",
                "created_at": 1700000200.0,
                "last_snapshot_screenshot": str(screenshot_path),
            }
        ],
    }
    app = _make_test_app(runtime)
    client = TestClient(app)

    try:
        resp = client.get(
            "/api/operations/sessions/browser/browser-001/screenshot?scope_key=test-browser"
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content.startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        screenshot_path.unlink(missing_ok=True)


def test_close_browser_session() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.delete("/api/operations/sessions/browser/browser-001?scope_key=test-browser")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    runtime.browser_supervisor.close.assert_awaited_once_with(
        scope_key="test-browser",
        session_id="browser-001",
    )


def test_clear_browser_sessions() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.delete("/api/operations/sessions/browser?scope_key=test-browser")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["removed"] == 1
    runtime.browser_supervisor.clear.assert_awaited_once_with(scope_key="test-browser")


def test_list_receipts_empty() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/receipts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["items"] == []


def test_get_receipt_not_found() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/receipts/nonexistent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is False


def test_list_work_empty() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/work")
    assert resp.status_code == 200
    data = resp.json()
    assert data["counts"]["total"] == 0
    assert data["items"] == []


def test_list_commitments_empty() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/commitments")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0


def test_list_commitments_with_data() -> None:
    runtime = _make_mock_runtime()
    c = Commitment(content="Ship v1", priority=8.0, tags=["release"])
    runtime.commitment_store.list_by_status = AsyncMock(return_value=[c])
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/commitments?status=active")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["items"][0]["content"] == "Ship v1"
    assert data["items"][0]["priority"] == 8.0


def test_list_plans_empty() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/plans")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0


def test_get_plan_not_found() -> None:
    runtime = _make_mock_runtime()
    app = _make_test_app(runtime)
    client = TestClient(app)

    resp = client.get("/api/operations/plans/nonexistent")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is False
