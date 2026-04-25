"""Shell, process, and PTY tool registration for AgentRuntime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from opencas.autonomy.models import ActionRiskTier
from opencas.sandbox import DockerSandbox, SandboxMode
from opencas.tools import ShellToolAdapter
from opencas.tools.adapters.process import ProcessToolAdapter
from opencas.tools.adapters.pty import PtyToolAdapter

from .tool_registration_specs import ToolRegistrationSpec, register_tool_specs


def register_foundation_process_tools(
    runtime: Any,
    *,
    roots: Sequence[str],
    default_cwd: str,
) -> None:
    docker_sandbox = None
    if runtime.ctx.sandbox.mode == SandboxMode.DOCKER:
        docker_sandbox = DockerSandbox(
            allowed_roots=runtime.ctx.sandbox.allowed_roots or [Path(roots[0])],
            timeout=30.0,
        )

    shell = ShellToolAdapter(cwd=default_cwd, timeout=30.0, docker_sandbox=docker_sandbox)
    register_tool_specs(
        runtime,
        shell,
        [
            ToolRegistrationSpec(
                name="bash_run_command",
                description="Execute a bash shell command in the project repository. Returns stdout and stderr.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute (e.g. pytest tests/)",
                        }
                    },
                    "required": ["command"],
                },
            )
        ],
    )

    process = ProcessToolAdapter(supervisor=runtime.process_supervisor, default_cwd=default_cwd)
    register_tool_specs(
        runtime,
        process,
        [
            ToolRegistrationSpec(
                name="process_start",
                description="Start a long-running background shell process.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The shell command to start."},
                        "cwd": {"type": "string", "description": "Working directory for the process."},
                        "scope_key": {"type": "string", "description": "Scope key for process isolation."},
                    },
                    "required": ["command"],
                },
            ),
            ToolRegistrationSpec(
                name="process_poll",
                description="Poll the status and recent output of a managed background process.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string", "description": "The process ID returned by process_start."},
                        "scope_key": {"type": "string", "description": "Scope key for process isolation."},
                    },
                    "required": ["process_id"],
                },
            ),
            ToolRegistrationSpec(
                name="process_write",
                description="Write input text to the stdin of a managed background process.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string", "description": "The process ID returned by process_start."},
                        "input": {"type": "string", "description": "Text to write to the process stdin."},
                        "scope_key": {"type": "string", "description": "Scope key for process isolation."},
                    },
                    "required": ["process_id", "input"],
                },
            ),
            ToolRegistrationSpec(
                name="process_send_signal",
                description="Send a POSIX signal to a managed background process (default SIGTERM).",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string", "description": "The process ID returned by process_start."},
                        "signal": {"type": "integer", "description": "Signal number to send (default 15)."},
                        "scope_key": {"type": "string", "description": "Scope key for process isolation."},
                    },
                    "required": ["process_id"],
                },
            ),
            ToolRegistrationSpec(
                name="process_kill",
                description="Forcefully kill a managed background process.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string", "description": "The process ID returned by process_start."},
                        "scope_key": {"type": "string", "description": "Scope key for process isolation."},
                    },
                    "required": ["process_id"],
                },
            ),
            ToolRegistrationSpec(
                name="process_clear",
                description="Kill and remove all managed background processes in a scope.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "scope_key": {"type": "string", "description": "Scope key for process isolation."},
                    },
                    "required": [],
                },
            ),
            ToolRegistrationSpec(
                name="process_remove",
                description="Remove a managed background process from tracking (kills if running).",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "process_id": {"type": "string", "description": "The process ID returned by process_start."},
                        "scope_key": {"type": "string", "description": "Scope key for process isolation."},
                    },
                    "required": ["process_id"],
                },
            ),
        ],
    )

    pty = PtyToolAdapter(supervisor=runtime.pty_supervisor, default_cwd=default_cwd)
    register_tool_specs(
        runtime,
        pty,
        [
            ToolRegistrationSpec(
                name="pty_start",
                description="Start an interactive PTY-backed terminal session.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to run in the PTY session."},
                        "cwd": {"type": "string", "description": "Working directory for the PTY session."},
                        "rows": {"type": "integer", "description": "Terminal rows."},
                        "cols": {"type": "integer", "description": "Terminal columns."},
                        "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                    },
                    "required": ["command"],
                },
            ),
            ToolRegistrationSpec(
                name="pty_poll",
                description="Poll the current PTY session state and read available terminal output.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                        "max_bytes": {"type": "integer", "description": "Maximum bytes of output to read."},
                        "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                    },
                    "required": ["session_id"],
                },
            ),
            ToolRegistrationSpec(
                name="pty_observe",
                description="Observe a PTY session with adaptive backoff until it goes quiet or exits.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                        "idle_seconds": {"type": "number", "description": "Return after this much PTY silence once output has started."},
                        "max_wait_seconds": {"type": "number", "description": "Maximum total time to observe before timing out."},
                        "max_bytes_per_poll": {"type": "integer", "description": "Maximum bytes to read per internal poll step."},
                        "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                    },
                    "required": ["session_id"],
                },
            ),
            ToolRegistrationSpec(
                name="pty_interact",
                description="Start or continue a PTY session, optionally send input, then observe until the terminal goes quiet. Prefer this for terminal UIs like claude, codex, vim, shells, and editors.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Existing PTY session id to continue. Omit to start a new session."},
                        "command": {"type": "string", "description": "Command to start when opening a new PTY session."},
                        "cwd": {"type": "string", "description": "Working directory for a new PTY session."},
                        "rows": {"type": "integer", "description": "Terminal rows for a new PTY session."},
                        "cols": {"type": "integer", "description": "Terminal columns for a new PTY session."},
                        "input": {"type": "string", "description": "Optional text or control sequence to write before observing."},
                        "idle_seconds": {"type": "number", "description": "Return after this much PTY silence once output has started."},
                        "max_wait_seconds": {"type": "number", "description": "Maximum total time to observe before timing out."},
                        "max_bytes_per_poll": {"type": "integer", "description": "Maximum bytes to read per internal poll step."},
                        "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                    },
                    "required": [],
                },
            ),
            ToolRegistrationSpec(
                name="pty_write",
                description="Write text or control sequences to an interactive PTY session.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                        "input": {"type": "string", "description": "Text or control sequence to write."},
                        "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                    },
                    "required": ["session_id", "input"],
                },
            ),
            ToolRegistrationSpec(
                name="pty_resize",
                description="Resize an interactive PTY session.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                        "rows": {"type": "integer", "description": "Terminal rows."},
                        "cols": {"type": "integer", "description": "Terminal columns."},
                        "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                    },
                    "required": ["session_id", "rows", "cols"],
                },
            ),
            ToolRegistrationSpec(
                name="pty_kill",
                description="Kill the process attached to a PTY session.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                        "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                    },
                    "required": ["session_id"],
                },
            ),
            ToolRegistrationSpec(
                name="pty_remove",
                description="Remove a PTY session from tracking and kill it if still running.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "PTY session id from pty_start."},
                        "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                    },
                    "required": ["session_id"],
                },
            ),
            ToolRegistrationSpec(
                name="pty_clear",
                description="Kill and remove all PTY sessions in a scope.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "scope_key": {"type": "string", "description": "Scope key for PTY isolation."},
                    },
                    "required": [],
                },
            ),
        ],
    )
