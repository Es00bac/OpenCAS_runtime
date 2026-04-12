#!/usr/bin/env python3
"""Execute Scenario 3 locally through the operations control plane."""

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
        default=Path(".opencas_live_test_state") / f"scenario3-operator-recovery-{_now_token()}",
        help="Directory for scenario outputs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    state_dir = output_dir / "state"
    workspace_dir = output_dir / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    pty_supervisor = PtySupervisor()
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(config=SimpleNamespace(state_dir=state_dir)),
        pty_supervisor=pty_supervisor,
    )

    app = FastAPI()
    app.include_router(build_operations_router(runtime))
    client = TestClient(app)

    target_file = workspace_dir / "scenario3_operator_recovery.md"
    session_id = pty_supervisor.start(
        "scenario3",
        f"vim -Nu NONE -n {target_file.name}",
        cwd=str(workspace_dir),
    )

    try:
        initial_detail = client.get(
            f"/api/operations/sessions/pty/{session_id}?scope_key=scenario3&refresh=true"
        )
        first_input = client.post(
            f"/api/operations/sessions/pty/{session_id}/input?scope_key=scenario3",
            json={
                "input": "iScenario 3 operator recovery\n- operator inspected the session\n- operator intervened mid-run\n",
                "observe": True,
                "idle_seconds": 0.2,
                "max_wait_seconds": 1.0,
            },
        )
        mid_detail = client.get(
            f"/api/operations/sessions/pty/{session_id}?scope_key=scenario3&refresh=true"
        )
        second_input = client.post(
            f"/api/operations/sessions/pty/{session_id}/input?scope_key=scenario3",
            json={
                "input": "\u001b:wq\r",
                "observe": True,
                "idle_seconds": 0.2,
                "max_wait_seconds": 1.0,
            },
        )
        final_detail = client.get(
            f"/api/operations/sessions/pty/{session_id}?scope_key=scenario3&refresh=true"
        )
    finally:
        pty_supervisor.kill("scenario3", session_id)
        pty_supervisor.remove("scenario3", session_id)

    artifact_exists = target_file.exists()
    artifact_content = target_file.read_text(encoding="utf-8") if artifact_exists else ""

    report = {
        "scenario": "scenario3_operator_intervention_recovery",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "state_dir": str(state_dir),
        "workspace_dir": str(workspace_dir),
        "artifact_path": str(target_file),
        "artifact_exists": artifact_exists,
        "artifact_verified": artifact_exists
        and "Scenario 3 operator recovery" in artifact_content
        and "operator intervened mid-run" in artifact_content,
        "initial_detail": initial_detail.json(),
        "first_input": first_input.json(),
        "mid_detail": mid_detail.json(),
        "second_input": second_input.json(),
        "final_detail": final_detail.json(),
    }
    report["material_success"] = bool(
        report["artifact_verified"]
        and len(report["final_detail"].get("recent_operator_actions", []) or []) >= 2
    )

    report_path = output_dir / "scenario3_operator_recovery_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown = output_dir / "scenario3_operator_recovery_report.md"
    markdown.write_text(
        "\n".join(
            [
                "# Scenario 3 Operator Intervention Recovery",
                "",
                f"- Session ID: `{session_id}`",
                f"- Artifact: `{target_file}`",
                f"- Artifact exists: `{artifact_exists}`",
                f"- Artifact verified: `{report['artifact_verified']}`",
                f"- Material success: `{report['material_success']}`",
                f"- Recent operator actions recorded: `{len(report['final_detail'].get('recent_operator_actions', []) or [])}`",
                "",
                "## Artifact Preview",
                "",
                "```md",
                artifact_content.strip(),
                "```",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({"ok": True, "output_dir": str(output_dir), "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
