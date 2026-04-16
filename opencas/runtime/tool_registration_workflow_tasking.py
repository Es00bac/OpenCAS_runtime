"""Workflow tasking tool registration for AgentRuntime."""

from __future__ import annotations

from typing import Any

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.adapters.workflow import WorkflowToolAdapter

from .tool_registration_specs import ToolRegistrationSpec, register_tool_specs


def register_workflow_tasking_tools(runtime: Any) -> None:
    workflow = WorkflowToolAdapter(runtime=runtime)
    register_tool_specs(
        runtime,
        workflow,
        [
            ToolRegistrationSpec(
                name="workflow_create_commitment",
                description="Create a durable goal or commitment to track ongoing work.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What this commitment is about."},
                        "priority": {"type": "number", "description": "Priority from 1.0 (low) to 10.0 (critical). Default 5.0."},
                        "deadline": {"type": "string", "description": "Optional ISO-8601 deadline."},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for categorization."},
                    },
                    "required": ["content"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_update_commitment",
                description="Update a commitment's status: completed, abandoned, blocked, or active.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "commitment_id": {"type": "string", "description": "The commitment ID to update."},
                        "status": {"type": "string", "description": "New status: completed, abandoned, blocked, or active."},
                    },
                    "required": ["commitment_id", "status"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_list_commitments",
                description="List commitments filtered by status.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "description": "Filter by status: active, completed, abandoned, blocked. Default active."},
                        "limit": {"type": "integer", "description": "Maximum items to return. Default 20."},
                    },
                    "required": [],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_create_schedule",
                description="Create a scheduled task or calendar event. Use ISO-8601 start_at; supports none, interval_hours, daily, weekly, and weekdays recurrence.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["task", "event"]},
                        "action": {"type": "string", "enum": ["submit_baa", "reminder_only"]},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "objective": {"type": "string"},
                        "start_at": {"type": "string"},
                        "end_at": {"type": "string"},
                        "timezone": {"type": "string"},
                        "recurrence": {"type": "string", "enum": ["none", "interval_hours", "daily", "weekly", "weekdays"]},
                        "interval_hours": {"type": "number"},
                        "weekdays": {"type": "array", "items": {"type": "integer"}},
                        "max_occurrences": {"type": "integer"},
                        "priority": {"type": "number"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "commitment_id": {"type": "string"},
                        "plan_id": {"type": "string"},
                    },
                    "required": ["title", "start_at"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_update_schedule",
                description="Update a schedule's status, title, description, objective, priority, or tags.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "schedule_id": {"type": "string"},
                        "status": {"type": "string", "enum": ["active", "paused", "completed", "cancelled"]},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "objective": {"type": "string"},
                        "priority": {"type": "number"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["schedule_id"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_list_schedules",
                description="List scheduled tasks and events.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "kind": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": [],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_create_writing_task",
                description="Set up a writing task with commitment tracking, output path, and optional outline scaffold.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Title of the writing piece."},
                        "description": {"type": "string", "description": "Brief description of the writing task."},
                        "output_path": {"type": "string", "description": "Optional file path for the output. Auto-generated if omitted."},
                        "outline": {"description": "Optional outline: a list of section headings or a text outline."},
                        "priority": {"type": "number", "description": "Priority from 1.0 to 10.0. Default 6.0."},
                    },
                    "required": ["title"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_create_plan",
                description="Create a structured plan for a project or task.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The plan content (markdown or plain text)."},
                        "project_id": {"type": "string", "description": "Optional project or commitment ID to link this plan to."},
                        "task_id": {"type": "string", "description": "Optional task ID to link this plan to."},
                    },
                    "required": ["content"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_update_plan",
                description="Update a plan's content.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string", "description": "The plan ID to update."},
                        "content": {"type": "string", "description": "Updated plan content."},
                    },
                    "required": ["plan_id", "content"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_repo_triage",
                description="Quick repo triage: git status, recent commits, work items, commitments, and plans summary.",
                risk_tier=ActionRiskTier.READONLY,
                schema={"type": "object", "properties": {}, "required": []},
            ),
            ToolRegistrationSpec(
                name="workflow_supervise_session",
                description="Launch or resume a PTY session (claude, kilocode, codex, vim, etc.), send a task with an Enter key, and supervise the cleaned output across multiple observation rounds. Returns a screen-state summary plus a supervision advisory so you can tell whether to keep observing, send follow-up input, or resolve an auth gate. Prefer this over raw PTY choreography for external TUI work.",
                risk_tier=ActionRiskTier.SHELL_LOCAL,
                schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to start (e.g. 'claude', 'codex'). Required if no session_id."},
                        "session_id": {"type": "string", "description": "Resume an existing PTY session instead of starting new."},
                        "task": {"type": "string", "description": "Text to send as input to the session."},
                        "verification_path": {"type": "string", "description": "Optional file path to verify after each supervision round. Useful for bounded artifact-producing tasks."},
                        "scope_key": {"type": "string", "description": "Scope key for session isolation. Default: workflow-supervision."},
                        "max_wait_seconds": {"type": "number", "description": "Maximum seconds for the initial submit/observe round. Default 15."},
                        "startup_wait_seconds": {"type": "number", "description": "When starting a new TUI process, maximum seconds to wait for the UI to reach a stable ready state before task submission. Default min(max_wait_seconds, 8)."},
                        "idle_seconds": {"type": "number", "description": "Seconds of silence before considering output complete. Default 1.0."},
                        "continue_wait_seconds": {"type": "number", "description": "Maximum seconds for later observation rounds after the initial submit. Defaults to max_wait_seconds."},
                        "max_rounds": {"type": "integer", "description": "Total supervision rounds including the initial submit round. Default 3."},
                    },
                    "required": [],
                },
            ),
        ],
    )
