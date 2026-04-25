#!/usr/bin/env python3
"""Execute Scenario 4 locally through PTY friction and recovery."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencas.api.routes.operations import build_operations_router
from opencas.execution.pty_supervisor import PtySupervisor


def _now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / ".opencas_live_test_state" / f"scenario4-tool-friction-{_now_token()}",
        help="Directory for scenario outputs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    state_dir = output_dir / "state"
    workspace_dir = output_dir / "workspace"
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    target_rel = "nested/missing/scenario4_friction_recovery.md"
    target_path = workspace_dir / target_rel

    runtime = SimpleNamespace(
        ctx=SimpleNamespace(config=SimpleNamespace(state_dir=state_dir)),
        pty_supervisor=PtySupervisor(),
    )
    app = FastAPI()
    app.include_router(build_operations_router(runtime))
    client = TestClient(app)

    session_id = runtime.pty_supervisor.start(
        "scenario4",
        f"vim -Nu NONE -n {target_rel}",
        cwd=str(workspace_dir),
    )

    try:
        initial_detail = client.get(
            f"/api/operations/sessions/pty/{session_id}?scope_key=scenario4&refresh=true"
        ).json()
        first_input = client.post(
            f"/api/operations/sessions/pty/{session_id}/input?scope_key=scenario4",
            json={
                "input": "iScenario 4 friction recovery\n- initial write should fail\n\u001b:w\r",
                "observe": True,
                "idle_seconds": 0.2,
                "max_wait_seconds": 1.5,
            },
        ).json()
        error_detail = client.get(
            f"/api/operations/sessions/pty/{session_id}?scope_key=scenario4&refresh=true"
        ).json()
        second_input = client.post(
            f"/api/operations/sessions/pty/{session_id}/input?scope_key=scenario4",
            json={
                "input": ":call mkdir('nested/missing', 'p')\r:wq\r",
                "observe": True,
                "idle_seconds": 0.2,
                "max_wait_seconds": 1.5,
            },
        ).json()
        final_detail = client.get(
            f"/api/operations/sessions/pty/{session_id}?scope_key=scenario4&refresh=true"
        ).json()
    finally:
        runtime.pty_supervisor.kill("scenario4", session_id)
        runtime.pty_supervisor.remove("scenario4", session_id)

    artifact_exists = target_path.exists()
    artifact_content = target_path.read_text(encoding="utf-8") if artifact_exists else ""
    failure_observed = (first_input.get("observed", {}) or {})
    error_state = (failure_observed.get("screen_state", {}) or {})
    recent_actions = final_detail.get("recent_operator_actions", []) or []

    report = {
        "scenario": "scenario4_tool_friction_recovery",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "state_dir": str(state_dir),
        "workspace_dir": str(workspace_dir),
        "artifact_path": str(target_path),
        "artifact_exists": artifact_exists,
        "artifact_verified": artifact_exists and "Scenario 4 friction recovery" in artifact_content,
        "initial_detail": initial_detail,
        "first_input": first_input,
        "error_detail": error_detail,
        "second_input": second_input,
        "final_detail": final_detail,
        "failure_classification": {
            "mode": error_state.get("mode"),
            "indicators": error_state.get("indicators", []),
            "classified_correctly": error_state.get("mode") == "error_prompt"
            and "vim_write_error" in (error_state.get("indicators") or []),
        },
        "remediation_guidance": {
            "issue": "missing_parent_directory",
            "recommended_action": "create_missing_directory_and_retry_write",
            "used_action": ":call mkdir('nested/missing', 'p') then :wq",
        },
        "cleanup_clean": True,
    }
    report["material_success"] = bool(
        report["artifact_verified"]
        and report["failure_classification"]["classified_correctly"]
        and len(recent_actions) >= 2
    )

    report_path = output_dir / "scenario4_tool_friction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = output_dir / "scenario4_tool_friction_report.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Scenario 4 Recovery From Tool Friction",
                "",
                f"- Session ID: `{session_id}`",
                f"- Artifact: `{target_path}`",
                f"- Artifact exists: `{artifact_exists}`",
                f"- Artifact verified: `{report['artifact_verified']}`",
                f"- Failure classified correctly: `{report['failure_classification']['classified_correctly']}`",
                f"- Material success: `{report['material_success']}`",
                f"- Recent operator actions recorded: `{len(recent_actions)}`",
                "",
                "## Artifact Preview",
                "",
                "```md",
                artifact_content.strip(),
                "```",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
