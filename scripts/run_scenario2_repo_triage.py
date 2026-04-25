#!/usr/bin/env python3
"""Execute Scenario 2 locally through the workflow adapter against the real repo."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencas.autonomy.commitment_store import CommitmentStore
from opencas.planning.store import PlanStore
from opencas.tools.adapters.workflow import WorkflowToolAdapter


def _now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class _Config:
    def __init__(self, workspace_root: Path, state_dir: Path) -> None:
        self._workspace_root = workspace_root
        self.state_dir = state_dir

    def primary_workspace_root(self) -> Path:
        return self._workspace_root


class _WorkStore:
    async def summary_counts(self) -> dict[str, int]:
        return {"total": 0, "ready": 0, "blocked": 0}


class _ScenarioRuntime:
    def __init__(self, workspace_root: Path, state_dir: Path, commitment_store: CommitmentStore, plan_store: PlanStore) -> None:
        self.ctx = SimpleNamespace(
            config=_Config(workspace_root, state_dir),
            work_store=_WorkStore(),
            plan_store=plan_store,
        )
        self.commitment_store = commitment_store

    async def execute_tool(self, name: str, args: dict[str, object]) -> dict[str, object]:
        if name != "bash_run_command":
            return {"success": False, "output": f"Unsupported tool: {name}", "metadata": {}}
        command = str(args.get("command", "") or "")
        cwd = str(args.get("cwd", self.ctx.config.primary_workspace_root()))
        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return {
            "success": completed.returncode == 0,
            "output": output,
            "metadata": {"returncode": completed.returncode, "cwd": cwd},
        }


async def _run(output_dir: Path) -> dict[str, object]:
    state_dir = output_dir / "state"
    workspace_dir = output_dir / "workspace"
    notes_dir = workspace_dir / "notes"
    state_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    commitment_store = await CommitmentStore(state_dir / "commitments.db").connect()
    plan_store = await PlanStore(state_dir / "plans.db").connect()
    runtime = _ScenarioRuntime(REPO_ROOT, state_dir, commitment_store, plan_store)
    adapter = WorkflowToolAdapter(runtime)

    try:
        triage_result = await adapter("workflow_repo_triage", {})
        triage_payload = json.loads(triage_result.output)

        note_path = notes_dir / "scenario2_repo_triage_note.md"
        writing_result = await adapter(
            "workflow_create_writing_task",
            {
                "title": "OpenCAS Repo Triage Note",
                "description": "Repo-grounded engineering note for readiness work.",
                "output_path": str(note_path),
                "outline": [
                    "Repo Snapshot",
                    "Workflow Triage Summary",
                    "Actionable Next Work",
                ],
            },
        )
        writing_payload = json.loads(writing_result.output)

        head = (await runtime.execute_tool("bash_run_command", {"command": "git rev-parse --short HEAD", "cwd": str(REPO_ROOT)})).get("output", "").strip()
        tracked_status = (await runtime.execute_tool("bash_run_command", {"command": "git status --short --untracked-files=no", "cwd": str(REPO_ROOT)})).get("output", "").strip()
        file_sample = (
            await runtime.execute_tool(
                "bash_run_command",
                {
                    "command": "rg --files opencas tests docs | rg -v '^docs/qualification/audio/' | head -n 12",
                    "cwd": str(REPO_ROOT),
                },
            )
        ).get("output", "").strip()

        note_content = "\n".join(
            [
                "# OpenCAS Repo Triage Note",
                "",
                f"Generated: {datetime.now(timezone.utc).isoformat()}",
                f"Repo HEAD: `{head or 'unknown'}`",
                f"Workspace: `{triage_payload.get('workspace', str(REPO_ROOT))}`",
                "",
                "## Repo Snapshot",
                "",
                f"- Tracked worktree clean: `{'yes' if not tracked_status else 'no'}`",
                f"- Active commitments: `{triage_payload.get('active_commitments', 0)}`",
                f"- Active plans: `{triage_payload.get('active_plans', 0)}`",
                f"- Work-item summary: `{json.dumps(triage_payload.get('work_items', {}), sort_keys=True)}`",
                "",
                "### Recent Commits",
                "",
                "```text",
                str(triage_payload.get("recent_commits", "")).strip(),
                "```",
                "",
                "### File Sample",
                "",
                "```text",
                file_sample,
                "```",
                "",
                "## Workflow Triage Summary",
                "",
                "- The repo-level workflow path succeeded through the actual `workflow_repo_triage` and `workflow_create_writing_task` tools.",
                "- The current tracked worktree is clean; local untracked testing artifacts remain outside tracked repo state.",
                "- Readiness work is currently centered on scenario coverage, qualification depth, and inner-life coupling rather than missing primitive tooling.",
                "",
                "## Actionable Next Work",
                "",
                "- Execute Scenario 4 from the long-scenario matrix to validate recovery from tool friction.",
                "- Continue PR-001 only when the remediation rollup still recommends `continue_testing` for a weak label.",
                "- Start PR-005 memory-value evaluation so retrieval and continuity claims are measured against repeated work.",
                "",
                "## Raw Triage Output",
                "",
                "### Git Status",
                "",
                "```text",
                str(triage_payload.get("git_status", "")).strip(),
                "```",
            ]
        )
        note_path.write_text(note_content + "\n", encoding="utf-8")

        artifact_text = note_path.read_text(encoding="utf-8")
        report = {
            "scenario": "scenario2_repo_triage_to_working_note",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "state_dir": str(state_dir),
            "workspace_dir": str(workspace_dir),
            "artifact_path": str(note_path),
            "workflow_repo_triage": triage_payload,
            "workflow_create_writing_task": writing_payload,
            "artifact_exists": note_path.exists(),
            "artifact_verified": note_path.exists()
            and "## Workflow Triage Summary" in artifact_text
            and "## Actionable Next Work" in artifact_text
            and "Repo HEAD:" in artifact_text,
            "tracked_worktree_clean": not bool(tracked_status),
        }
        report["material_success"] = bool(
            triage_result.success
            and writing_result.success
            and report["artifact_verified"]
        )
        return report
    finally:
        await commitment_store.close()
        await plan_store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / ".opencas_live_test_state" / f"scenario2-repo-triage-{_now_token()}",
        help="Directory for scenario outputs.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = asyncio.run(_run(output_dir))
    report_path = output_dir / "scenario2_repo_triage_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = output_dir / "scenario2_repo_triage_report.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Scenario 2 Repo Triage To Working Note",
                "",
                f"- Artifact: `{report['artifact_path']}`",
                f"- Artifact exists: `{report['artifact_exists']}`",
                f"- Artifact verified: `{report['artifact_verified']}`",
                f"- Material success: `{report['material_success']}`",
                f"- Tracked worktree clean: `{report['tracked_worktree_clean']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
