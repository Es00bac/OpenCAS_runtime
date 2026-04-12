#!/usr/bin/env python3
"""Execute a bounded loop-guard pressure scenario through the live validation harness."""

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
        default=REPO_ROOT / ".opencas_live_test_state" / f"scenario8-loop-guard-{_now_token()}",
        help="Directory for scenario outputs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    validation_dir = output_dir / "validation_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"scenario8-{uuid4().hex}"

    before_sweep = _run_sweep()
    command = [
        _python_executable(),
        str(RUN_VALIDATION_SCRIPT),
        "--state-dir",
        str(validation_dir),
        "--workspace-root",
        str(REPO_ROOT),
        "--session-id",
        "scenario8-loop-guard",
        "--skip-direct-checks",
        "--agent-check-label",
        "loop_guard_test",
        "--model",
        "kimi-coding/k2p5",
        "--embedding-model",
        "local-fallback",
        "--request-id",
        request_id,
    ]

    print(f"Running validation: {' '.join(command)}")
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
    loop_guard_test = next((item for item in agent_checks if item.get("label") == "loop_guard_test"), {})
    
    response_text = loop_guard_test.get("response", "")
    breaker_detected = "circuit breaker" in response_text.lower() or "identical arguments" in response_text.lower()
    
    cleanup_clean = int(after_sweep.get("count", 0) or 0) == 0

    scenario_report = {
        "scenario": "scenario8_loop_guard",
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
        "loop_guard_check": loop_guard_test,
        "breaker_detected": breaker_detected,
        "cleanup_clean": cleanup_clean,
        "remediation_guidance": {
            "issue": "loop_guard_pressure_handling",
            "recommended_action": "confirm loop guard triggers and error is caught by agent without crashing",
            "used_action": "prompt causing identical tool calls",
        },
    }
    scenario_report["material_success"] = bool(
        completed.returncode == 0
        and report_path.exists()
        and breaker_detected
        and cleanup_clean
    )

    json_report_path = output_dir / "scenario8_loop_guard_report.json"
    md_report_path = output_dir / "scenario8_loop_guard_report.md"
    json_report_path.write_text(json.dumps(scenario_report, indent=2), encoding="utf-8")
    md_report_path.write_text(
        "\n".join(
            [
                "# Scenario 8 Loop-Guard Pressure",
                "",
                f"- Request ID: `{request_id}`",
                f"- Validation returncode: `{completed.returncode}`",
                f"- Validation report exists: `{report_path.exists()}`",
                f"- Breaker detected: `{breaker_detected}`",
                f"- Cleanup clean: `{cleanup_clean}`",
                f"- Material success: `{scenario_report['material_success']}`",
                f"- Post-run stale process count: `{after_sweep.get('count', 0)}`",
                "",
                "## Loop Guard Check Outcome",
                "",
                f"- Outcome: `{loop_guard_test.get('outcome', '-')}`",
                f"- Response: `{response_text}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"ok": True, "output_dir": str(output_dir), "report_path": str(json_report_path)}, indent=2))
    return 0 if scenario_report["material_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
