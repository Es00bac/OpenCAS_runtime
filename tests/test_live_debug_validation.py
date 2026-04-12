"""Tests for the live debug validation harness helpers."""

from unittest.mock import AsyncMock, MagicMock

from pathlib import Path

import pytest

from scripts.run_live_debug_validation import (
    _build_arg_parser,
    _cleanup_runtime_sessions,
    _collect_expected_artifact,
    _finalize_agent_check_record,
    _render_markdown_report,
)


def test_collect_expected_artifact_reads_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "note.md"
    target.write_text("hello\n", encoding="utf-8")

    payload = _collect_expected_artifact(target)

    assert payload["expected_file"] == str(target)
    assert payload["expected_file_exists"] is True
    assert payload["expected_file_content"] == "hello\n"


def test_finalize_agent_check_record_marks_artifact_verified_after_timeout() -> None:
    record = {
        "label": "kilocode_supervised_work",
        "timed_out": True,
        "response": "",
        "expected_file": "/tmp/out.md",
        "expected_file_exists": True,
    }

    _finalize_agent_check_record(record, prompt_timeout_seconds=180.0)

    assert record["outcome"] == "artifact_verified_after_timeout"
    assert record["material_success"] is True
    assert "materially succeeded" in record["response"]


def test_finalize_agent_check_record_marks_missing_artifact_on_timeout() -> None:
    record = {
        "label": "kilocode_supervised_work",
        "timed_out": True,
        "response": "",
        "expected_file": "/tmp/out.md",
        "expected_file_exists": False,
    }

    _finalize_agent_check_record(record, prompt_timeout_seconds=180.0)

    assert record["outcome"] == "artifact_missing_after_timeout"
    assert record["material_success"] is False
    assert "was not created" in record["response"]


def test_finalize_agent_check_record_marks_non_artifact_timeout() -> None:
    record = {
        "label": "browser_probe",
        "timed_out": True,
        "response": "",
    }

    _finalize_agent_check_record(record, prompt_timeout_seconds=45.0)

    assert record["outcome"] == "timed_out"
    assert record["material_success"] is False
    assert "Inspect telemetry/context" in record["response"]


def test_render_markdown_report_supports_in_progress_reports() -> None:
    report = {
        "run_id": "debug-validation-test",
        "session_id": "debug-validation-test",
        "state_dir": "/tmp/debug-validation-test",
        "workspace_root": "(workspace_root)",
        "model": "kimi-coding/k2p5",
        "embedding_model": "google/gemini-embedding-2-preview",
        "started_at": "2026-04-08T00:00:00+00:00",
        "direct_checks": {},
        "agent_checks": [],
    }

    rendered = _render_markdown_report(report)

    assert "- Finished: `in_progress`" in rendered


def test_arg_parser_accepts_qualification_provenance_flags() -> None:
    parser = _build_arg_parser()

    args = parser.parse_args([
        "--agent-check-label",
        "integrated_operator_workflow",
        "--request-id",
        "req-123",
        "--rerun-history-path",
        "/tmp/rerun-history.jsonl",
    ])

    assert args.request_id == "req-123"
    assert args.rerun_history_path == "/tmp/rerun-history.jsonl"
    assert args.agent_check_label == ["integrated_operator_workflow"]


def test_render_markdown_report_includes_request_id() -> None:
    report = {
        "run_id": "debug-validation-test",
        "session_id": "debug-validation-test",
        "state_dir": "/tmp/debug-validation-test",
        "workspace_root": "(workspace_root)",
        "model": "kimi-coding/k2p5",
        "embedding_model": "google/gemini-embedding-2-preview",
        "request_id": "req-123",
        "started_at": "2026-04-08T00:00:00+00:00",
        "finished_at": "2026-04-08T00:00:10+00:00",
        "direct_checks": {},
        "agent_checks": [],
    }

    rendered = _render_markdown_report(report)

    assert "- Request ID: `req-123`" in rendered


@pytest.mark.asyncio
async def test_cleanup_runtime_sessions_sweeps_all_supervisors() -> None:
    runtime = MagicMock()
    runtime.process_supervisor = MagicMock()
    runtime.process_supervisor.clear_all.return_value = 2
    runtime.pty_supervisor = MagicMock()
    runtime.pty_supervisor.clear_all.return_value = 3
    runtime.browser_supervisor = MagicMock()
    runtime.browser_supervisor.clear_all = AsyncMock(return_value=1)

    cleaned = await _cleanup_runtime_sessions(runtime)

    assert cleaned == {"processes": 2, "pty": 3, "browser": 1}
    runtime.process_supervisor.clear_all.assert_called_once_with()
    runtime.pty_supervisor.clear_all.assert_called_once_with()
    runtime.browser_supervisor.clear_all.assert_awaited_once_with()
