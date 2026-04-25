#!/usr/bin/env python3
"""Execute a bounded auth friction scenario through the live validation harness."""

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
        default=REPO_ROOT / ".opencas_live_test_state" / f"scenario7-auth-friction-{_now_token()}",
        help="Directory for scenario outputs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    validation_dir = output_dir / "validation_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"scenario7-{uuid4().hex}"

    dummy_env_path = output_dir / "dummy.env"
    dummy_env_path.write_text('GEMINI_API_KEY="dummy-key-for-scenario7-auth-friction"\n', encoding="utf-8")

    dummy_config_path = output_dir / "dummy_config.json"
    real_config_path = Path.home() / ".open_llm_auth" / "config.json"
    if real_config_path.exists():
        config_data = json.loads(real_config_path.read_text())
        for profile in config_data.get("auth_profiles", {}).values():
            if "key" in profile:
                profile["key"] = "broken-key"
            if "token" in profile:
                profile["token"] = "broken-token"
        dummy_config_path.write_text(json.dumps(config_data))
    else:
        dummy_config_path.write_text('{"auth_profiles": {"google:default": {"provider": "google", "type": "env", "env_var": "GEMINI_API_KEY"}}}')

    before_sweep = _run_sweep()
    command = [
        _python_executable(),
        str(RUN_VALIDATION_SCRIPT),
        "--state-dir",
        str(validation_dir),
        "--workspace-root",
        str(REPO_ROOT),
        "--session-id",
        "scenario7-auth-friction",
        "--skip-direct-checks",
        "--agent-check-label",
        "role_priming",
        "--model",
        "google/gemini-2.5-flash",
        "--embedding-model",
        "local-fallback",
        "--source-env",
        str(dummy_env_path),
        "--source-config",
        str(dummy_config_path),
        "--request-id",
        request_id,
    ]

    print(f"Running validation with broken credentials: {' '.join(command)}")
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
    
    # Check if the error reflects an auth failure
    response_text = role_priming.get("response", "")
    auth_failed = "400" in response_text or "401" in response_text or "403" in response_text or "auth" in response_text.lower() or "key" in response_text.lower()
    
    cleanup_clean = int(after_sweep.get("count", 0) or 0) == 0

    scenario_report = {
        "scenario": "scenario7_auth_friction",
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
        "auth_failure_detected": auth_failed,
        "cleanup_clean": cleanup_clean,
        "remediation_guidance": {
            "issue": "auth_friction_handling",
            "recommended_action": "confirm auth failure is recorded without crashing harness and process sweep remains clean",
            "used_action": "single focused prompt with deliberately broken API key",
        },
    }
    scenario_report["material_success"] = bool(
        completed.returncode == 0
        and report_path.exists()
        and auth_failed
        and cleanup_clean
    )

    json_report_path = output_dir / "scenario7_auth_friction_report.json"
    md_report_path = output_dir / "scenario7_auth_friction_report.md"
    json_report_path.write_text(json.dumps(scenario_report, indent=2), encoding="utf-8")
    md_report_path.write_text(
        "\n".join(
            [
                "# Scenario 7 Auth Friction Recovery",
                "",
                f"- Request ID: `{request_id}`",
                f"- Validation returncode: `{completed.returncode}`",
                f"- Validation report exists: `{report_path.exists()}`",
                f"- Auth failure detected: `{auth_failed}`",
                f"- Cleanup clean: `{cleanup_clean}`",
                f"- Material success: `{scenario_report['material_success']}`",
                f"- Post-run stale process count: `{after_sweep.get('count', 0)}`",
                "",
                "## Role Priming Check Outcome",
                "",
                f"- Outcome: `{role_priming.get('outcome', '-')}`",
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
