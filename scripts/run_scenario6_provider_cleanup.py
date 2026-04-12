#!/usr/bin/env python3
"""Execute a bounded provider-backed cleanup scenario through the live validation harness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_VALIDATION_SCRIPT = REPO_ROOT / "scripts" / "run_live_debug_validation.py"
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "sweep_operator_processes.py"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _python_executable() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def _run_sweep() -> dict[str, object]:
    completed = subprocess.run(
        [_python_executable(), str(SWEEP_SCRIPT), "--json"],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / ".opencas_live_test_state" / f"scenario6-provider-cleanup-{_now_token()}",
        help="Directory for scenario outputs.",
    )
    parser.add_argument(
        "--prompt-timeout-seconds",
        type=float,
        default=0.05,
        help="Per-prompt timeout passed to the live validation harness.",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=180.0,
        help="Total run timeout passed to the live validation harness.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    validation_dir = output_dir / "validation_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"scenario6-{uuid4().hex}"

    before_sweep = _run_sweep()
    command = [
        _python_executable(),
        str(RUN_VALIDATION_SCRIPT),
        "--state-dir",
        str(validation_dir),
        "--workspace-root",
        str(REPO_ROOT),
        "--session-id",
        "scenario6-provider-cleanup",
        "--skip-direct-checks",
        "--agent-check-label",
        "role_priming",
        "--prompt-timeout-seconds",
        str(args.prompt_timeout_seconds),
        "--run-timeout-seconds",
        str(args.run_timeout_seconds),
        "--model",
        "kimi-coding/k2p5",
        "--embedding-model",
        "local-fallback",
        "--request-id",
        request_id,
    ]

    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    report_path = validation_dir / "live_debug_validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    after_sweep = _run_sweep()

    agent_checks = report.get("agent_checks", []) if isinstance(report, dict) else []
    role_priming = next((item for item in agent_checks if item.get("label") == "role_priming"), {})
    timed_out = bool(role_priming.get("timed_out"))
    cleanup_clean = int(after_sweep.get("count", 0) or 0) == 0

    scenario_report = {
        "scenario": "scenario6_provider_backed_cleanup",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "validation_dir": str(validation_dir),
        "request_id": request_id,
        "command": command,
        "validation_returncode": completed.returncode,
        "validation_stdout": completed.stdout,
        "validation_stderr": completed.stderr,
        "before_sweep": before_sweep,
        "after_sweep": after_sweep,
        "validation_report_path": str(report_path),
        "validation_report_exists": report_path.exists(),
        "role_priming_check": role_priming,
        "timeout_detected": timed_out,
        "cleanup_clean": cleanup_clean,
        "remediation_guidance": {
            "issue": "provider_backed_prompt_timeout_cleanup",
            "recommended_action": "confirm timeout is recorded and process sweep remains clean",
            "used_action": "single focused provider-backed prompt with forced tiny timeout and post-run sweep",
        },
    }
    scenario_report["material_success"] = bool(
        completed.returncode == 0
        and report_path.exists()
        and timed_out
        and cleanup_clean
    )

    json_report_path = output_dir / "scenario6_provider_cleanup_report.json"
    md_report_path = output_dir / "scenario6_provider_cleanup_report.md"
    json_report_path.write_text(json.dumps(scenario_report, indent=2), encoding="utf-8")
    md_report_path.write_text(
        "\n".join(
            [
                "# Scenario 6 Provider-Backed Cleanup",
                "",
                f"- Request ID: `{request_id}`",
                f"- Validation returncode: `{completed.returncode}`",
                f"- Validation report exists: `{report_path.exists()}`",
                f"- Timed out as intended: `{timed_out}`",
                f"- Cleanup clean: `{cleanup_clean}`",
                f"- Material success: `{scenario_report['material_success']}`",
                f"- Post-run stale process count: `{after_sweep.get('count', 0)}`",
                "",
                "## Role Priming Check Outcome",
                "",
                f"- Outcome: `{role_priming.get('outcome', '-')}`",
                f"- Timed out: `{role_priming.get('timed_out', False)}`",
                f"- Material success: `{role_priming.get('material_success', False)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"ok": True, "output_dir": str(output_dir), "report_path": str(json_report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
