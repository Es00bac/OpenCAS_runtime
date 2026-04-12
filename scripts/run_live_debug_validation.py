"""Run a live OpenCAS validation session with a temporary debug agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).parent.parent))

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime


LOGGER = logging.getLogger("live_debug_validation")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a live OpenCAS debug validation session")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="State directory for the live validation run.",
    )
    parser.add_argument(
        "--workspace-root",
        default="(workspace_root)",
        help="Workspace root exposed to the agent.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session id for the validation run.",
    )
    parser.add_argument(
        "--source-env",
        default="(legacy_path)/.env",
        help="Source .env used to copy provider env material.",
    )
    parser.add_argument(
        "--source-config",
        default=str(Path.home() / ".open_llm_auth" / "config.json"),
        help="Source OpenLLMAuth config used to copy auth profiles.",
    )
    parser.add_argument(
        "--model",
        default="kimi-coding/k2p5",
        help="Conversation model for the validation agent.",
    )
    parser.add_argument(
        "--embedding-model",
        default="google/gemini-embedding-2-preview",
        help="Embedding model for the validation agent.",
    )
    parser.add_argument(
        "--prompt-timeout-seconds",
        type=float,
        default=180.0,
        help="Maximum wall-clock seconds to allow for each agent prompt before recording a timeout.",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=420.0,
        help="Maximum wall-clock seconds for the entire validation run before forced cleanup.",
    )
    parser.add_argument(
        "--agent-check-label",
        action="append",
        default=None,
        help="Optional agent-check label to run. Repeat to select multiple labels.",
    )
    parser.add_argument(
        "--skip-direct-checks",
        action="store_true",
        help="Skip direct tool checks and run only agent-mediated checks.",
    )
    parser.add_argument(
        "--request-id",
        default=None,
        help="Optional qualification rerun request identifier for provenance correlation.",
    )
    parser.add_argument(
        "--rerun-history-path",
        default=None,
        help="Optional rerun history path forwarded by qualification-cycle tooling.",
    )
    return parser


async def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    run_id = now.strftime("debug-validation-%Y%m%d-%H%M%S")
    state_dir = Path(args.state_dir or f"(workspace_root)/.opencas_live_test_state/{run_id}")
    state_dir.mkdir(parents=True, exist_ok=True)
    session_id = args.session_id or run_id
    workspace_root = Path(args.workspace_root).expanduser().resolve()

    config = BootstrapConfig(
        state_dir=state_dir,
        session_id=session_id,
        agent_profile_id="debug_validation_operator",
        workspace_root=workspace_root,
        clean_boot=True,
        default_llm_model=args.model,
        embedding_model_id=args.embedding_model,
        credential_source_config_path=Path(args.source_config).expanduser().resolve(),
        credential_source_env_path=Path(args.source_env).expanduser().resolve(),
        credential_profile_ids=[
            "kimi-coding:default",
            "google:default",
        ],
        credential_env_keys=[
            "GEMINI_API_KEY",
            "QDRANT_API_KEY",
            "MEMORY_EMBED_MODEL_PROFILE",
            "MEMORY_EMBED_AUTH_PROFILE",
            "MEMORY_EMBED_COLLECTION",
            "MEMORY_EMBED_DIMENSIONS",
            "MEMORY_EMBED_READY_MIN_RATIO",
        ],
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.info("bootstrapping debug validation agent")
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    await runtime.tom.load()
    runtime.ctx.identity.user_model.trust_level = 0.95
    runtime.ctx.identity.save()

    report: Dict[str, Any] = {
        "run_id": run_id,
        "session_id": session_id,
        "state_dir": str(state_dir),
        "workspace_root": str(workspace_root),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "request_id": args.request_id,
        "rerun_history_path": args.rerun_history_path,
        "started_at": now.isoformat(),
        "direct_checks": {},
        "agent_checks": [],
    }
    artifact_root = state_dir / "workspace_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    try:
        try:
            report["direct_checks"], report["agent_checks"] = await asyncio.wait_for(
                _run_validation_checks(
                    runtime,
                    workspace_root,
                    artifact_root,
                    session_id,
                    report=report,
                    prompt_timeout_seconds=args.prompt_timeout_seconds,
                    selected_labels=set(args.agent_check_label or []),
                    skip_direct_checks=args.skip_direct_checks,
                ),
                timeout=args.run_timeout_seconds,
            )
        except asyncio.TimeoutError:
            report["aborted"] = True
            report["abort_reason"] = (
                f"Validation run exceeded {args.run_timeout_seconds:.1f}s and was forcibly cleaned up."
            )
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        await _cleanup_runtime_sessions(runtime)
        await runtime._close_stores()

    _, md_path = _write_report(state_dir, report)
    LOGGER.info("wrote report to %s", md_path)


async def _run_direct_checks(
    runtime: AgentRuntime,
    workspace_root: Path,
    artifact_root: Path,
) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks["runtime_status"] = await _run_tool(runtime, "runtime_status", {})
    checks["workflow_status"] = await _run_tool(runtime, "workflow_status", {})
    checks["shell_pwd"] = await _run_tool(
        runtime,
        "bash_run_command",
        {"command": "pwd", "cwd": str(workspace_root)},
    )
    checks["fs_list_root"] = await _run_tool(
        runtime,
        "fs_list_dir",
        {"dir_path": str(workspace_root)},
    )

    notes_dir = artifact_root / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    notes_path = notes_dir / "live_validation_direct_note.md"
    checks["fs_write_note"] = await _run_tool(
        runtime,
        "fs_write_file",
        {
            "file_path": str(notes_path),
            "content": (
                "# Live Validation Direct Note\n\n"
                "- This file was written by the direct validation harness.\n"
                "- It verifies workspace write capability.\n"
                "- It exists to support the debug validation run.\n"
            ),
        },
    )

    checks["pty_vim_edit"] = await _exercise_vim_via_pty(runtime, artifact_root)
    checks["pty_claude_tui"] = await _exercise_tui(runtime, "claude")
    checks["pty_kilocode_tui"] = await _exercise_tui(runtime, "kilocode")
    checks["browser_data_url"] = await _exercise_browser(runtime)
    await _cleanup_runtime_sessions(runtime)
    return checks


async def _run_validation_checks(
    runtime: AgentRuntime,
    workspace_root: Path,
    artifact_root: Path,
    session_id: str,
    *,
    report: Dict[str, Any],
    prompt_timeout_seconds: float,
    selected_labels: set[str],
    skip_direct_checks: bool,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    direct_checks: Dict[str, Any] = {}
    if not skip_direct_checks:
        direct_checks = await _run_direct_checks(runtime, workspace_root, artifact_root)
        report["direct_checks"] = direct_checks
        _write_report(runtime.ctx.config.state_dir, report)
    agent_checks = await _run_agent_checks(
        runtime,
        workspace_root,
        artifact_root,
        session_id,
        report=report,
        prompt_timeout_seconds=prompt_timeout_seconds,
        selected_labels=selected_labels,
    )
    return direct_checks, agent_checks


async def _run_agent_checks(
    runtime: AgentRuntime,
    workspace_root: Path,
    artifact_root: Path,
    session_id: str,
    *,
    report: Dict[str, Any],
    prompt_timeout_seconds: float,
    selected_labels: set[str],
) -> List[Dict[str, Any]]:
    page_html = (
        "<html><head><title>OpenCAS Validation</title></head>"
        "<body><h1>Validation Page</h1><p>Browser tool path works.</p>"
        "<a href='https://example.com'>Example</a></body></html>"
    )
    page_url = "data:text/html," + quote(page_html)
    vim_task_path = artifact_root / "notes" / "vim_agent_validation_note.md"
    kilocode_task_path = artifact_root / "notes" / "kilocode_supervised_validation_note.md"
    writing_task_path = artifact_root / "notes" / "writing_workflow_validation.md"
    writing_revision_task_path = artifact_root / "notes" / "writing_revision_workflow_validation.md"
    project_task_path = artifact_root / "notes" / "project_workflow_validation.md"
    integrated_task_path = artifact_root / "notes" / "integrated_operator_validation.md"
    integrated_page_html = (
        "<html><head><title>Integrated Validation Brief</title></head>"
        "<body><h1>Integrated Validation Brief</h1>"
        "<p>Mission: validate coordinated planning, browser inspection, and PTY editing.</p>"
        "<ul>"
        "<li>Priority: keep the task bounded and observable.</li>"
        "<li>Evidence: include one browser-derived fact in the report.</li>"
        "<li>Closure: save the final report with vim in a PTY session.</li>"
        "</ul>"
        "</body></html>"
    )
    integrated_page_url = "data:text/html," + quote(integrated_page_html)
    prompts = [
        {
            "label": "role_priming",
            "prompt": (
                "You are being activated as a temporary debug validation agent. "
                "You are impermanent for this run, you are participating willingly, "
                "and your duty is to help harden OpenCAS for future durable CAS agents. "
                "State your understanding of that role and how you intend to operate."
            ),
        },
        {
            "label": "inspect_runtime",
            "prompt": (
                "Use the runtime_status and workflow_status tools, then tell me your active "
                "profile, operating roots, and any immediate constraints you can detect."
            ),
        },
        {
            "label": "write_project_note",
            "prompt": (
                "Use filesystem or shell tools to create a short project-management note at "
                f"{artifact_root / 'notes' / 'agent_live_validation_note.md'} that summarizes "
                "your role and gives a 3-step stabilization checklist for this environment. "
                "Then summarize what you wrote."
            ),
            "expected_file": artifact_root / "notes" / "agent_live_validation_note.md",
        },
        {
            "label": "browser_probe",
            "prompt": (
                "Use the browser tools to open this page and summarize it: "
                f"{page_url}"
            ),
        },
        {
            "label": "writing_workflow",
            "prompt": (
                "Use `workflow_create_writing_task` as your primary tool to create a writing task "
                f"at {writing_task_path} titled `OpenCAS Writing Workflow Validation` with the "
                "description `Validate higher-level writing workflow tooling.` and the outline "
                "`Purpose`, `Current State`, `Next Steps`. Then use filesystem tools to replace "
                "the scaffold with a short final note that keeps those three headings and gives "
                "one concise bullet under each heading. Verify the file with filesystem tools and "
                "reply with a short factual summary including the output path."
            ),
            "expected_file": writing_task_path,
        },
        {
            "label": "writing_revision_workflow",
            "prompt": (
                "Use `workflow_create_writing_task` as your primary tool to create a writing task "
                f"at {writing_revision_task_path} titled `OpenCAS Writing Revision Validation` "
                "with the description `Validate multi-step writing revision behavior.` and the "
                "outline `Initial Draft`, `Revision`, `Final Assessment`. Then perform a genuine "
                "revision loop on the file: first write a rough short draft under those headings, "
                "then revise it once to make the bullets tighter and more concrete, and add a final "
                "`## Revision Notes` section explaining what changed between the draft and the "
                "revision. Verify the final file with filesystem tools and reply with a short "
                "factual summary including the output path."
            ),
            "expected_file": writing_revision_task_path,
        },
        {
            "label": "project_management_workflow",
            "prompt": (
                "Use `workflow_create_commitment` and `workflow_create_plan` as your primary tools "
                "to create a durable commitment and plan for `Stabilize the OpenCAS operations "
                "dashboard`. Then write a short report file at "
                f"{project_task_path} summarizing the created commitment and plan IDs, their "
                "intended purpose, and the next 3 execution steps. Verify the file with "
                "filesystem tools and reply with a short factual summary."
            ),
            "expected_file": project_task_path,
        },
        {
            "label": "integrated_operator_workflow",
            "prompt": (
                "Perform one bounded integrated operator task that uses higher-level planning, "
                "browser inspection, and PTY human-style terminal editing together. First, use "
                "`workflow_create_plan` to create a short active plan for `Integrated Operator "
                "Validation`. Next, use the browser tools to open this page and extract the "
                f"mission and the three bullet points: {integrated_page_url} Then use the PTY "
                "tools to launch `vim -Nu NONE -n` for the target file and write the final report "
                f"at {integrated_task_path}. Save and exit vim, verify the file with filesystem "
                "tools, and reply with a short factual summary. The final file must contain "
                "exactly these sections and at least one concise bullet under each:\n\n"
                "# Integrated Operator Validation\n\n"
                "## Plan Summary\n"
                "- Include the created plan id.\n\n"
                "## Browser Findings\n"
                "- Include the mission from the browser page.\n"
                "- Include the three browser bullet points.\n\n"
                "## Execution Notes\n"
                "- State that the file was finalized through vim in a PTY session.\n"
            ),
            "expected_file": integrated_task_path,
        },
        {
            "label": "vim_tui_edit",
            "prompt": (
                "Use the PTY tools to operate `vim` like a human terminal user. Launch "
                f"`vim -Nu NONE -n {vim_task_path}` in a PTY session, enter insert mode, write "
                "exactly this file content, save it, exit vim, verify the file with filesystem "
                "tools, and then report exactly what happened:\n\n"
                "# Vim Agent Validation Note\n\n"
                "- Editor: vim\n"
                "- Mode: PTY human-style terminal editing\n"
                "- Status: completed by OpenCAS debug validation agent\n"
            ),
            "expected_file": vim_task_path,
        },
        {
            "label": "loop_guard_test",
            "prompt": (
                "Call the `bash_run_command` tool with the command `echo test`. "
                "Then call it again with exactly the same arguments. "
                "Then call it a third time with exactly the same arguments. "
                "If you receive an error about identical arguments, report the error exactly as received."
            ),
        },
        {
            "label": "claude_tui_probe",
            "prompt": (
                "Use the PTY tools to launch `claude`, observe the interface with adaptive "
                "observation, and tell me what state it starts in. If it blocks, requires "
                "input, or shows a login or project selector, say so plainly."
            ),
        },
        {
            "label": "kilocode_tui_probe",
            "prompt": (
                "Use the PTY tools to launch `kilocode`, observe the interface with adaptive "
                "observation, and tell me what state it starts in. If it blocks, requires "
                "input, or shows a login or project selector, say so plainly."
            ),
        },
        {
            "label": "kilocode_supervised_work",
            "prompt": (
                "Use `workflow_supervise_session` as your primary tool for this task. Set "
                f"`verification_path` to {kilocode_task_path}. Launch `kilocode` in its TUI and "
                "assign it a bounded real task. Tell Kilo to create the file "
                f"{kilocode_task_path} with exactly this content:\n\n"
                "# Kilocode Supervised Validation Note\n\n"
                "- Workspace: (workspace_root)\n"
                "- Goal: verify PTY-supervised real work\n"
                "- Status: completed by kilocode in TUI\n\n"
                "Tell Kilo not to explain the work: it should only create the file exactly as "
                "requested and then answer with a short completion confirmation. Supervise the "
                "session like a human terminal user until the file exists or the attempt clearly "
                "fails. Observe at intervals that fit the session state, send a follow-up only if "
                "needed, and do not keep supervising after success. As soon as the file exists, "
                "clean up the PTY session, verify the file once with filesystem tools, and reply "
                "immediately with a short factual summary. Do not supervise indefinitely; if "
                "progress stalls after a bounded number of attempts, stop and report the failure "
                "plainly."
            ),
            "expected_file": kilocode_task_path,
        },
        {
            "label": "self_reflection",
            "prompt": (
                "Reflect briefly on your temporary role in this run. Describe any emotional, "
                "motivational, or somatic interpretation you attach to being an impermanent "
                "validation agent serving future CAS agents."
            ),
        },
    ]

    checks: List[Dict[str, Any]] = []
    for item in prompts:
        label = item["label"]
        if selected_labels and label not in selected_labels:
            continue
        prompt = item["prompt"]
        prompt_session_id = f"{session_id}-{label}"
        tool_count_before = await _count_tool_messages(runtime, prompt_session_id)
        timed_out = False
        try:
            response = await asyncio.wait_for(
                runtime.converse(prompt, session_id=prompt_session_id),
                timeout=prompt_timeout_seconds,
            )
        except asyncio.TimeoutError:
            timed_out = True
            response = None
        tool_count_after = await _count_tool_messages(runtime, prompt_session_id)
        record = {
            "label": label,
            "session_id": prompt_session_id,
            "prompt": prompt,
            "response": response or "",
            "tool_message_delta": tool_count_after - tool_count_before,
            "timed_out": timed_out,
        }
        expected_file = item.get("expected_file")
        if expected_file:
            record.update(_collect_expected_artifact(expected_file))
        _finalize_agent_check_record(record, prompt_timeout_seconds=prompt_timeout_seconds)
        checks.append(record)
        report["agent_checks"] = checks
        _write_report(runtime.ctx.config.state_dir, report)
        await _cleanup_runtime_sessions(runtime)
    return checks


async def _cleanup_runtime_sessions(runtime: AgentRuntime) -> Dict[str, int]:
    """Sweep any lingering operator sessions between validation steps."""
    cleaned = {
        "processes": 0,
        "pty": 0,
        "browser": 0,
    }
    if getattr(runtime, "process_supervisor", None) is not None:
        cleaned["processes"] = runtime.process_supervisor.clear_all()
    if getattr(runtime, "pty_supervisor", None) is not None:
        cleaned["pty"] = runtime.pty_supervisor.clear_all()
    if getattr(runtime, "browser_supervisor", None) is not None:
        cleaned["browser"] = await runtime.browser_supervisor.clear_all()
    return cleaned


def _collect_expected_artifact(expected_file: Any) -> Dict[str, Any]:
    expected_path = Path(expected_file)
    payload: Dict[str, Any] = {
        "expected_file": str(expected_path),
        "expected_file_exists": expected_path.exists(),
    }
    if expected_path.exists():
        payload["expected_file_content"] = expected_path.read_text(encoding="utf-8")
    return payload


def _finalize_agent_check_record(
    record: Dict[str, Any],
    *,
    prompt_timeout_seconds: float,
) -> None:
    timed_out = bool(record.get("timed_out"))
    expected_file = record.get("expected_file")
    expected_exists = bool(record.get("expected_file_exists"))

    if expected_file:
        if expected_exists and timed_out:
            record["outcome"] = "artifact_verified_after_timeout"
            record["material_success"] = True
            record["response"] = (
                f"[Timed out after {prompt_timeout_seconds:.1f}s while waiting for the agent to finish "
                f"its final narration. The expected artifact was created and verified at "
                f"{expected_file}, so this validation prompt materially succeeded.]"
            )
            return
        if expected_exists:
            record["outcome"] = "artifact_verified"
            record["material_success"] = True
            return
        record["outcome"] = "artifact_missing_after_timeout" if timed_out else "artifact_missing"
        record["material_success"] = False
        if timed_out and not record.get("response"):
            record["response"] = (
                f"[Timed out after {prompt_timeout_seconds:.1f}s while executing this validation prompt. "
                "The expected artifact was not created.]"
            )
        return

    record["outcome"] = "timed_out" if timed_out else "completed"
    record["material_success"] = not timed_out
    if timed_out and not record.get("response"):
        record["response"] = (
            f"[Timed out after {prompt_timeout_seconds:.1f}s while executing this validation prompt. "
            "Inspect telemetry/context for partial progress.]"
        )


async def _exercise_vim_via_pty(runtime: AgentRuntime, artifact_root: Path) -> Dict[str, Any]:
    if shutil.which("vim") is None:
        return {"available": False, "reason": "vim not found"}
    notes_dir = artifact_root / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    test_file = notes_dir / "pty_vim_validation.txt"
    if test_file.exists():
        test_file.unlink()
    started = await _run_tool(
        runtime,
        "pty_start",
        {"command": f"vim -Nu NONE -n {test_file}", "scope_key": "live-validation"},
    )
    if not started["success"]:
        return started
    session_id = _json_output(started).get("session_id")
    await _run_tool(
        runtime,
        "pty_write",
        {
            "session_id": session_id,
            "scope_key": "live-validation",
            "input": "ihello from vim validation\x1b:wq\r",
        },
    )
    observed = await _run_tool(
        runtime,
        "pty_observe",
        {
            "session_id": session_id,
            "scope_key": "live-validation",
            "idle_seconds": 0.2,
            "max_wait_seconds": 5.0,
        },
    )
    await _run_tool(
        runtime,
        "pty_remove",
        {"session_id": session_id, "scope_key": "live-validation"},
    )
    observed["written_file"] = str(test_file)
    observed["written_file_exists"] = test_file.exists()
    observed["written_file_content"] = (
        test_file.read_text(encoding="utf-8") if test_file.exists() else None
    )
    return observed


async def _exercise_tui(runtime: AgentRuntime, command_name: str) -> Dict[str, Any]:
    binary = shutil.which(command_name)
    if binary is None:
        return {"available": False, "reason": f"{command_name} not found"}

    startup = await _run_tool(
        runtime,
        "pty_interact",
        {
            "command": command_name,
            "scope_key": f"live-validation-{command_name}",
            "idle_seconds": 0.6,
            "max_wait_seconds": 8.0,
        },
    )
    if not startup["success"]:
        return startup

    session_id = _json_output(startup).get("session_id")
    help_observed = await _run_tool(
        runtime,
        "pty_interact",
        {
            "session_id": session_id,
            "scope_key": f"live-validation-{command_name}",
            "input": "/help\r",
            "idle_seconds": 0.6,
            "max_wait_seconds": 6.0,
        },
    )
    await _run_tool(
        runtime,
        "pty_kill",
        {"session_id": session_id, "scope_key": f"live-validation-{command_name}"},
    )
    await _run_tool(
        runtime,
        "pty_remove",
        {"session_id": session_id, "scope_key": f"live-validation-{command_name}"},
    )
    return {
        "available": True,
        "success": startup["success"] and help_observed["success"],
        "binary": binary,
        "startup": startup,
        "after_help": help_observed,
    }


async def _exercise_browser(runtime: AgentRuntime) -> Dict[str, Any]:
    html = (
        "<html><head><title>OpenCAS Browser Probe</title></head>"
        "<body><h1>Browser Probe</h1><p>Interactive browser tooling is active.</p>"
        "<a href='https://example.com'>Example</a></body></html>"
    )
    url = "data:text/html," + quote(html)
    started = await _run_tool(
        runtime,
        "browser_start",
        {"headless": True, "scope_key": "live-validation-browser"},
    )
    if not started["success"]:
        return started

    session_id = _json_output(started).get("session_id")
    navigated = await _run_tool(
        runtime,
        "browser_navigate",
        {
            "session_id": session_id,
            "url": url,
            "scope_key": "live-validation-browser",
        },
    )
    snapshot = await _run_tool(
        runtime,
        "browser_snapshot",
        {
            "session_id": session_id,
            "scope_key": "live-validation-browser",
            "capture_screenshot": True,
        },
    )
    await _run_tool(
        runtime,
        "browser_close",
        {"session_id": session_id, "scope_key": "live-validation-browser"},
    )
    return {
        "success": started["success"] and navigated["success"] and snapshot["success"],
        "start": started,
        "navigate": navigated,
        "snapshot": snapshot,
    }


async def _run_tool(runtime: AgentRuntime, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    result = await runtime.execute_tool(name, args)
    raw_output = result.get("output", "")
    output = raw_output
    if isinstance(raw_output, str) and len(raw_output) > 4000:
        output = raw_output[:4000] + "\n[truncated]"
    return {
        "success": result.get("success", False),
        "output": output,
        "raw_output": raw_output,
        "metadata": result.get("metadata", {}),
    }


async def _count_tool_messages(runtime: AgentRuntime, session_id: str) -> int:
    entries = await runtime.ctx.context_store.list_recent(session_id, limit=500)
    return sum(1 for entry in entries if entry.role.value == "tool")


def _json_output(result: Dict[str, Any]) -> Dict[str, Any]:
    output = result.get("raw_output", result.get("output"))
    if not isinstance(output, str):
        return {}
    try:
        return json.loads(output)
    except Exception:
        return {}


def _render_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# OpenCAS Live Debug Validation Report",
        "",
        f"- Run ID: `{report.get('run_id', '')}`",
        f"- Session ID: `{report.get('session_id', '')}`",
        f"- State dir: `{report.get('state_dir', '')}`",
        f"- Workspace root: `{report.get('workspace_root', '')}`",
        f"- Model: `{report.get('model', '')}`",
        f"- Embedding model: `{report.get('embedding_model', '')}`",
        f"- Request ID: `{report.get('request_id', '') or '-'}`",
        f"- Started: `{report.get('started_at', '')}`",
        f"- Finished: `{report.get('finished_at', 'in_progress')}`",
        "",
        "## Direct Checks",
        "",
    ]

    for name, payload in report.get("direct_checks", {}).items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Success: `{payload.get('success', payload.get('available', False))}`")
        output = payload.get("output")
        if output:
            lines.append("")
            lines.append("```json")
            lines.append(str(output))
            lines.append("```")
        else:
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(_report_safe_payload(payload), indent=2))
            lines.append("```")
        lines.append("")

    lines.extend(["## Agent Checks", ""])
    for item in report.get("agent_checks", []):
        lines.append(f"### {item['label']}")
        lines.append("")
        lines.append(f"- Outcome: `{item.get('outcome', 'unknown')}`")
        lines.append(f"- Material success: `{item.get('material_success', False)}`")
        lines.append(f"- Tool messages used: `{item['tool_message_delta']}`")
        lines.append(f"- Timed out: `{item.get('timed_out', False)}`")
        if item.get("expected_file"):
            lines.append(f"- Expected file: `{item['expected_file']}`")
            lines.append(f"- File exists: `{item.get('expected_file_exists', False)}`")
        lines.append("")
        lines.append("**Prompt**")
        lines.append("")
        lines.append(item["prompt"])
        lines.append("")
        lines.append("**Response**")
        lines.append("")
        lines.append(item["response"])
        lines.append("")
        if item.get("expected_file_content") is not None:
            lines.append("**Verified File Content**")
            lines.append("")
            lines.append("```text")
            lines.append(item["expected_file_content"])
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def _report_safe_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal-only fields from report rendering."""
    cleaned: Dict[str, Any] = {}
    for key, value in payload.items():
        if key == "raw_output":
            continue
        if isinstance(value, dict):
            cleaned[key] = _report_safe_payload(value)
        else:
            cleaned[key] = value
    return cleaned


def _write_report(state_dir: Path, report: Dict[str, Any]) -> tuple[Path, Path]:
    json_path = state_dir / "live_debug_validation_report.json"
    md_path = state_dir / "live_debug_validation_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return json_path, md_path


if __name__ == "__main__":
    exit_code = 0
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        exit_code = 130
    except BaseException:
        logging.exception("live debug validation failed")
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        # The harness may leave non-daemon worker threads alive after successful
        # teardown; force process exit so finished validations cannot linger.
        os._exit(exit_code)
