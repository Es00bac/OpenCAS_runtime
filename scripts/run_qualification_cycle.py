"""Run repeated focused live validation cycles and refresh the qualification summary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_VALIDATION_SCRIPT = REPO_ROOT / "scripts" / "run_live_debug_validation.py"
SUMMARIZE_SCRIPT = REPO_ROOT / "scripts" / "summarize_live_validations.py"
DEFAULT_RUNS_DIR = REPO_ROOT / ".opencas_live_test_state"
DEFAULT_RERUN_HISTORY_PATH = DEFAULT_RUNS_DIR / "qualification_rerun_history.jsonl"
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


@dataclass
class QualificationCycleResult:
    iteration: int
    returncode: int
    command: List[str]
    run_ids: List[str] = field(default_factory=list)


def resolve_python_executable() -> str:
    """Prefer the repo-local venv interpreter when available."""
    if DEFAULT_VENV_PYTHON.exists():
        return str(DEFAULT_VENV_PYTHON)
    return sys.executable


def build_validation_command(
    *,
    workspace_root: Path,
    labels: Sequence[str],
    skip_direct_checks: bool,
    prompt_timeout_seconds: float,
    run_timeout_seconds: float,
    model: str,
    embedding_model: str,
    request_id: str | None = None,
    rerun_history_path: Path | None = None,
) -> List[str]:
    command = [
        resolve_python_executable(),
        str(RUN_VALIDATION_SCRIPT),
        "--workspace-root",
        str(workspace_root),
        "--prompt-timeout-seconds",
        str(prompt_timeout_seconds),
        "--run-timeout-seconds",
        str(run_timeout_seconds),
        "--model",
        model,
        "--embedding-model",
        embedding_model,
    ]
    if skip_direct_checks:
        command.append("--skip-direct-checks")
    if request_id:
        command.extend(["--request-id", request_id])
    if rerun_history_path:
        command.extend(["--rerun-history-path", str(rerun_history_path)])
    for label in labels:
        command.extend(["--agent-check-label", label])
    return command


def build_summary_command(*, runs_dir: Path, output_dir: Path) -> List[str]:
    return [
        resolve_python_executable(),
        str(SUMMARIZE_SCRIPT),
        "--runs-dir",
        str(runs_dir),
        "--output-dir",
        str(output_dir),
    ]


def _list_run_ids(runs_dir: Path) -> set[str]:
    return {
        path.parent.name
        for path in runs_dir.glob("*/live_debug_validation_report.json")
    }


def _append_completion_event(history_path: Path, payload: dict[str, object]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _append_request_event(history_path: Path, payload: dict[str, object]) -> None:
    _append_completion_event(history_path, payload)


def run_cycle(
    *,
    iterations: int,
    delay_seconds: float,
    validation_command: Sequence[str],
    summary_command: Sequence[str],
    runs_dir: Path,
    labels: Sequence[str],
    request_id: str | None,
    rerun_history_path: Path | None,
    continue_on_failure: bool,
    dry_run: bool,
) -> List[QualificationCycleResult]:
    results: List[QualificationCycleResult] = []
    for iteration in range(1, iterations + 1):
        if dry_run:
            returncode = 0
            run_ids: List[str] = []
        else:
            before_run_ids = _list_run_ids(runs_dir)
            completed = subprocess.run(list(validation_command), check=False)
            returncode = completed.returncode
            after_run_ids = _list_run_ids(runs_dir)
            run_ids = sorted(after_run_ids - before_run_ids)
        results.append(
            QualificationCycleResult(
                iteration=iteration,
                returncode=returncode,
                command=list(validation_command),
                run_ids=run_ids,
            )
        )
        if returncode != 0 and not continue_on_failure:
            break
        if iteration < iterations and delay_seconds > 0:
            time.sleep(delay_seconds)

    if not dry_run:
        subprocess.run(list(summary_command), check=True)
        if request_id and rerun_history_path:
            all_run_ids = [run_id for result in results for run_id in result.run_ids]
            payload = {
                "event": "completed",
                "request_id": request_id,
                "labels": list(labels),
                "completed_at": time.time(),
                "returncode": 0 if all(result.returncode == 0 for result in results) else 1,
                "iterations_executed": len(results),
                "generated_run_ids": all_run_ids,
                "latest_run_id": all_run_ids[-1] if all_run_ids else None,
            }
            _append_completion_event(rerun_history_path, payload)

    return results


def render_results(results: Iterable[QualificationCycleResult], *, dry_run: bool) -> str:
    lines = [
        "# Qualification Cycle",
        "",
        f"- Dry run: `{str(dry_run).lower()}`",
    ]
    for result in results:
        lines.append(
            f"- Iteration `{result.iteration}` returncode `{result.returncode}` command "
            f"`{' '.join(result.command)}`"
        )
    if not results:
        lines.append("- No iterations executed.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated focused OpenCAS live validation cycles and refresh the qualification summary."
    )
    parser.add_argument(
        "--workspace-root",
        default=str(REPO_ROOT),
        help="Workspace root exposed to the validation agent.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Directory containing live validation run state.",
    )
    parser.add_argument(
        "--summary-output-dir",
        default=str(REPO_ROOT / "docs" / "qualification"),
        help="Output directory for the refreshed qualification summary.",
    )
    parser.add_argument(
        "--agent-check-label",
        action="append",
        default=[],
        help="Focused agent-check label to run. Repeat to select multiple labels.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="How many focused live validation runs to execute.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay between iterations.",
    )
    parser.add_argument(
        "--prompt-timeout-seconds",
        type=float,
        default=180.0,
        help="Prompt timeout passed through to the live validation harness.",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=420.0,
        help="Run timeout passed through to the live validation harness.",
    )
    parser.add_argument(
        "--model",
        default="kimi-coding/k2p5",
        help="Conversation model passed through to the live validation harness.",
    )
    parser.add_argument(
        "--embedding-model",
        default="google/gemini-embedding-2-preview",
        help="Embedding model passed through to the live validation harness.",
    )
    parser.add_argument(
        "--include-direct-checks",
        action="store_true",
        help="Include direct checks instead of focused agent-only validation.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running later iterations after a failed validation run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned qualification cycle without executing it.",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Optional request identifier used to correlate rerun launch and completion provenance.",
    )
    parser.add_argument(
        "--rerun-history-path",
        default=str(DEFAULT_RERUN_HISTORY_PATH),
        help="Path to the local rerun provenance JSONL file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request_id = args.request_id or uuid4().hex
    rerun_history_path = Path(args.rerun_history_path).expanduser().resolve()
    validation_command = build_validation_command(
        workspace_root=Path(args.workspace_root).expanduser().resolve(),
        labels=args.agent_check_label,
        skip_direct_checks=not args.include_direct_checks,
        prompt_timeout_seconds=args.prompt_timeout_seconds,
        run_timeout_seconds=args.run_timeout_seconds,
        model=args.model,
        embedding_model=args.embedding_model,
        request_id=request_id,
        rerun_history_path=rerun_history_path,
    )
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    summary_command = build_summary_command(
        runs_dir=runs_dir,
        output_dir=Path(args.summary_output_dir).expanduser().resolve(),
    )
    if not args.dry_run:
        labels = list(args.agent_check_label)
        primary_label = labels[0] if labels else ""
        _append_request_event(
            rerun_history_path,
            {
                "event": "requested",
                "request_id": request_id,
                "process_id": f"manual-{os.getpid()}",
                "label": primary_label,
                "labels": labels,
                "source_label": primary_label,
                "source_note": "manual_cli_rerun",
                "requested_at": time.time(),
                "command": " ".join(validation_command),
                "iterations": max(1, args.iterations),
                "include_direct_checks": bool(args.include_direct_checks),
            },
        )
    results = run_cycle(
        iterations=max(1, args.iterations),
        delay_seconds=max(0.0, args.delay_seconds),
        validation_command=validation_command,
        summary_command=summary_command,
        runs_dir=runs_dir,
        labels=args.agent_check_label,
        request_id=request_id,
        rerun_history_path=rerun_history_path,
        continue_on_failure=args.continue_on_failure,
        dry_run=args.dry_run,
    )
    print(render_results(results, dry_run=args.dry_run))
    if any(result.returncode != 0 for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
