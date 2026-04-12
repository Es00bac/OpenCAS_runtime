"""Tests for repeated qualification cycle tooling."""

from __future__ import annotations

import argparse
import subprocess
import json
from pathlib import Path

from scripts.run_qualification_cycle import (
    build_summary_command,
    build_validation_command,
    main,
    render_results,
    resolve_python_executable,
    run_cycle,
)


def test_build_validation_command_includes_labels_and_skip_direct() -> None:
    command = build_validation_command(
        workspace_root=Path("/tmp/workspace"),
        labels=["writing_workflow", "integrated_operator_workflow"],
        skip_direct_checks=True,
        prompt_timeout_seconds=90.0,
        run_timeout_seconds=240.0,
        model="kimi-coding/k2p5",
        embedding_model="google/gemini-embedding-2-preview",
        request_id="req-123",
        rerun_history_path=Path("/tmp/rerun-history.jsonl"),
    )
    assert "--workspace-root" in command
    assert "/tmp/workspace" in command
    assert "--skip-direct-checks" in command
    assert "--request-id" in command
    assert "req-123" in command
    assert "--rerun-history-path" in command
    assert "/tmp/rerun-history.jsonl" in command
    assert command.count("--agent-check-label") == 2
    assert "writing_workflow" in command
    assert "integrated_operator_workflow" in command


def test_resolve_python_executable_prefers_repo_venv(monkeypatch) -> None:
    monkeypatch.setattr("scripts.run_qualification_cycle.DEFAULT_VENV_PYTHON", Path("/tmp/opencas/.venv/bin/python"))
    assert resolve_python_executable().endswith("/.venv/bin/python")


def test_resolve_python_executable_falls_back_to_current(monkeypatch) -> None:
    monkeypatch.setattr("scripts.run_qualification_cycle.DEFAULT_VENV_PYTHON", Path("/tmp/opencas/.venv/bin/missing-python"))
    assert resolve_python_executable() == __import__("sys").executable


def test_build_summary_command_targets_runs_and_output_dirs() -> None:
    command = build_summary_command(
        runs_dir=Path("/tmp/runs"),
        output_dir=Path("/tmp/out"),
    )
    assert command[-4:] == ["--runs-dir", "/tmp/runs", "--output-dir", "/tmp/out"]


def test_run_cycle_runs_summary_after_iterations(monkeypatch) -> None:
    calls: list[list[str]] = []
    run_ids = [
        set(),
        {"debug-validation-1"},
        {"debug-validation-1"},
        {"debug-validation-1", "debug-validation-2"},
    ]

    def _fake_run(command, check=False):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("scripts.run_qualification_cycle.subprocess.run", _fake_run)
    monkeypatch.setattr("scripts.run_qualification_cycle._list_run_ids", lambda _runs_dir: run_ids.pop(0))

    validation = ["python", "run_live_debug_validation.py", "--agent-check-label", "writing_workflow"]
    summary = ["python", "summarize_live_validations.py", "--runs-dir", "/tmp/runs", "--output-dir", "/tmp/out"]
    history_path = Path("/tmp/qualification-cycle-history.jsonl")
    if history_path.exists():
        history_path.unlink()
    results = run_cycle(
        iterations=2,
        delay_seconds=0.0,
        validation_command=validation,
        summary_command=summary,
        runs_dir=Path("/tmp/runs"),
        labels=["writing_workflow"],
        request_id="req-123",
        rerun_history_path=history_path,
        continue_on_failure=False,
        dry_run=False,
    )

    assert [result.returncode for result in results] == [0, 0]
    assert results[0].run_ids == ["debug-validation-1"]
    assert results[1].run_ids == ["debug-validation-2"]
    assert calls == [validation, validation, summary]
    recorded = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert recorded[-1]["event"] == "completed"
    assert recorded[-1]["request_id"] == "req-123"
    assert recorded[-1]["latest_run_id"] == "debug-validation-2"
    history_path.unlink()


def test_run_cycle_stops_on_failure_without_continue(monkeypatch) -> None:
    calls: list[list[str]] = []

    def _fake_run(command, check=False):
        calls.append(list(command))
        if command[1] == "run_live_debug_validation.py":
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("scripts.run_qualification_cycle.subprocess.run", _fake_run)

    validation = ["python", "run_live_debug_validation.py"]
    summary = ["python", "summarize_live_validations.py"]
    results = run_cycle(
        iterations=3,
        delay_seconds=0.0,
        validation_command=validation,
        summary_command=summary,
        runs_dir=Path("/tmp/runs"),
        labels=["writing_workflow"],
        request_id=None,
        rerun_history_path=None,
        continue_on_failure=False,
        dry_run=False,
    )

    assert len(results) == 1
    assert results[0].returncode == 1
    assert calls == [validation, summary]


def test_render_results_mentions_iterations() -> None:
    rendered = render_results([], dry_run=True)
    assert "Dry run" in rendered
    assert "No iterations executed." in rendered


def test_main_records_manual_request_event(monkeypatch, tmp_path, capsys) -> None:
    captured: dict[str, object] = {}
    history_path = tmp_path / "qualification_rerun_history.jsonl"

    monkeypatch.setattr(
        "scripts.run_qualification_cycle.parse_args",
        lambda: argparse.Namespace(
            workspace_root=str(tmp_path),
            runs_dir=str(tmp_path / "runs"),
            summary_output_dir=str(tmp_path / "out"),
            agent_check_label=["kilocode_supervised_work"],
            iterations=1,
            delay_seconds=0.0,
            prompt_timeout_seconds=180.0,
            run_timeout_seconds=420.0,
            model="kimi-coding/k2p5",
            embedding_model="google/gemini-embedding-2-preview",
            include_direct_checks=False,
            continue_on_failure=False,
            dry_run=False,
            request_id=None,
            rerun_history_path=str(history_path),
        ),
    )
    def _fake_build_validation_command(**kwargs):
        captured["validation_kwargs"] = kwargs
        return ["python", "validate.py"]

    monkeypatch.setattr(
        "scripts.run_qualification_cycle.build_summary_command",
        lambda **kwargs: ["python", "summary.py"],
    )
    monkeypatch.setattr("scripts.run_qualification_cycle.build_validation_command", _fake_build_validation_command)
    monkeypatch.setattr(
        "scripts.run_qualification_cycle._append_request_event",
        lambda path, payload: captured.setdefault("request_event", (path, payload)),
    )

    def _fake_run_cycle(**kwargs):
        captured["run_cycle_kwargs"] = kwargs
        return []

    monkeypatch.setattr("scripts.run_qualification_cycle.run_cycle", _fake_run_cycle)

    assert main() == 0
    out = capsys.readouterr().out
    assert "No iterations executed." in out
    validation_kwargs = captured["validation_kwargs"]
    run_cycle_kwargs = captured["run_cycle_kwargs"]
    request_path, request_payload = captured["request_event"]
    assert request_path == history_path.resolve()
    assert validation_kwargs["request_id"]
    assert run_cycle_kwargs["request_id"] == validation_kwargs["request_id"]
    assert request_payload["request_id"] == validation_kwargs["request_id"]
    assert request_payload["label"] == "kilocode_supervised_work"
    assert request_payload["source_note"] == "manual_cli_rerun"
