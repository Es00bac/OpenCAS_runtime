#!/usr/bin/env python3
"""Execute a local browser-drift recovery scenario through the operations control plane."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencas.api.routes.operations import build_operations_router
from opencas.execution.browser_supervisor import BrowserSupervisor


def _now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _write_fixture_pages(workspace_dir: Path) -> tuple[Path, Path]:
    target_path = workspace_dir / "scenario5_target.html"
    drift_path = workspace_dir / "scenario5_drift.html"
    target_path.write_text(
        """<!doctype html>
<html>
  <head><title>Scenario 5 Target</title></head>
  <body>
    <main>
      <h1>Scenario 5 Target</h1>
      <p id="mission">Stay on the intended working page.</p>
      <a id="drift-link" href="scenario5_drift.html">Drift Away</a>
    </main>
  </body>
</html>
""",
        encoding="utf-8",
    )
    drift_path.write_text(
        """<!doctype html>
<html>
  <head><title>Scenario 5 Drift</title></head>
  <body>
    <main>
      <h1>Unexpected Page</h1>
      <p id="warning">The browser session drifted off the intended target.</p>
      <a id="recover-link" href="scenario5_target.html">Return to Target</a>
    </main>
  </body>
</html>
""",
        encoding="utf-8",
    )
    return target_path, drift_path


def _copy_if_present(source: str | None, destination: Path) -> str | None:
    if not source:
        return None
    source_path = Path(source)
    if not source_path.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, destination)
    return str(destination)


def _load_actions(state_dir: Path, session_id: str) -> list[dict[str, object]]:
    history_path = state_dir / "operator_action_history.jsonl"
    if not history_path.exists():
        return []
    actions: list[dict[str, object]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get("target_kind") == "browser" and item.get("target_id") == session_id:
            actions.append(item)
    actions.sort(key=lambda item: float(item.get("timestamp", 0.0)), reverse=True)
    return actions


async def _run_scenario(output_dir: Path) -> dict[str, object]:
    state_dir = output_dir / "state"
    workspace_dir = output_dir / "workspace"
    evidence_dir = output_dir / "browser_artifacts"
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    target_path, drift_path = _write_fixture_pages(workspace_dir)
    target_url = target_path.resolve().as_uri()
    drift_url = drift_path.resolve().as_uri()

    runtime = SimpleNamespace(
        ctx=SimpleNamespace(config=SimpleNamespace(state_dir=state_dir)),
        browser_supervisor=BrowserSupervisor(),
    )
    app = FastAPI()
    app.include_router(build_operations_router(runtime))

    session_id: str | None = None
    try:
        session_id = await runtime.browser_supervisor.start("scenario5")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            initial_navigate = (
                await client.post(
                    f"/api/operations/sessions/browser/{session_id}/navigate",
                    params={"scope_key": "scenario5"},
                    json={"url": target_url, "refresh": True},
                )
            ).json()
            initial_capture = (
                await client.post(
                    f"/api/operations/sessions/browser/{session_id}/capture",
                    params={"scope_key": "scenario5"},
                    json={"full_page": False},
                )
            ).json()

            await runtime.browser_supervisor.navigate("scenario5", session_id, drift_url)
            drift_detail = (
                await client.get(
                    f"/api/operations/sessions/browser/{session_id}",
                    params={
                        "scope_key": "scenario5",
                        "refresh": "true",
                        "capture_screenshot": "true",
                    },
                )
            ).json()
            drift_screenshot = _copy_if_present(
                ((drift_detail.get("observed", {}) or {}).get("screenshot_path")),
                evidence_dir / "scenario5_drift.png",
            )

            recovery_click = (
                await client.post(
                    f"/api/operations/sessions/browser/{session_id}/click",
                    params={"scope_key": "scenario5"},
                    json={"selector": "#recover-link", "refresh": True},
                )
            ).json()
            final_capture = (
                await client.post(
                    f"/api/operations/sessions/browser/{session_id}/capture",
                    params={"scope_key": "scenario5"},
                    json={"full_page": False},
                )
            ).json()
            final_screenshot = _copy_if_present(
                ((final_capture.get("observed", {}) or {}).get("screenshot_path")),
                evidence_dir / "scenario5_recovered.png",
            )
            final_detail = (
                await client.get(
                    f"/api/operations/sessions/browser/{session_id}",
                    params={"scope_key": "scenario5", "refresh": "true"},
                )
            ).json()
            close_result = (
                await client.delete(
                    f"/api/operations/sessions/browser/{session_id}",
                    params={"scope_key": "scenario5"},
                )
            ).json()
        cleanup_snapshot = runtime.browser_supervisor.snapshot(scope_key="scenario5")
    finally:
        await runtime.browser_supervisor.clear_all()
        await runtime.browser_supervisor.shutdown()

    actions = _load_actions(state_dir, session_id or "")
    drift_text = ((drift_detail.get("session", {}) or {}).get("last_snapshot_text") or "")
    final_text = ((final_detail.get("session", {}) or {}).get("last_snapshot_text") or "")
    drift_title = ((drift_detail.get("session", {}) or {}).get("title") or "")
    final_title = ((final_detail.get("session", {}) or {}).get("title") or "")

    report = {
        "scenario": "scenario5_browser_drift_recovery",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "state_dir": str(state_dir),
        "workspace_dir": str(workspace_dir),
        "target_url": target_url,
        "drift_url": drift_url,
        "drift_detected": drift_title == "Scenario 5 Drift" and "drifted off the intended target" in drift_text,
        "recovered": final_title == "Scenario 5 Target" and "Stay on the intended working page." in final_text,
        "cleanup_clean": close_result.get("ok") is True and cleanup_snapshot.get("total_count") == 0,
        "initial_navigate": initial_navigate,
        "initial_capture": initial_capture,
        "drift_detail": drift_detail,
        "recovery_click": recovery_click,
        "final_capture": final_capture,
        "final_detail": final_detail,
        "close_result": close_result,
        "cleanup_snapshot": cleanup_snapshot,
        "operator_actions": actions,
        "copied_artifacts": {
            "drift_screenshot": drift_screenshot,
            "final_screenshot": final_screenshot,
        },
        "remediation_guidance": {
            "issue": "browser_page_drift",
            "recommended_action": "inspect_page_state_then_return_to_target",
            "used_action": "operator inspected browser detail, then clicked #recover-link",
        },
    }
    report["material_success"] = bool(
        report["drift_detected"]
        and report["recovered"]
        and report["cleanup_clean"]
        and any(action.get("action") == "browser_click" for action in actions)
        and any(action.get("action") == "close_browser" for action in actions)
    )

    report_path = output_dir / "scenario5_browser_drift_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = output_dir / "scenario5_browser_drift_report.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Scenario 5 Browser Drift Recovery",
                "",
                f"- Session ID: `{session_id}`",
                f"- Drift detected: `{report['drift_detected']}`",
                f"- Recovered: `{report['recovered']}`",
                f"- Cleanup clean: `{report['cleanup_clean']}`",
                f"- Material success: `{report['material_success']}`",
                f"- Recorded operator actions: `{len(actions)}`",
                f"- Drift screenshot copy: `{drift_screenshot}`",
                f"- Recovery screenshot copy: `{final_screenshot}`",
                "",
                "## Drift Snapshot",
                "",
                f"- Title: `{drift_title}`",
                f"- Text preview: `{drift_text[:160]}`",
                "",
                "## Recovery Snapshot",
                "",
                f"- Title: `{final_title}`",
                f"- Text preview: `{final_text[:160]}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"ok": True, "output_dir": str(output_dir), "report_path": str(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / ".opencas_live_test_state" / f"scenario5-browser-drift-{_now_token()}",
        help="Directory for scenario outputs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    result = asyncio.run(_run_scenario(output_dir))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
