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
                description=(
                    "Update a commitment's status: completed, abandoned, blocked, or active. "
                    "When completing ongoing-support/build/test/verify/project-return work, include "
                    "completion_evidence with the command, receipt, artifact, or runtime proof."
                ),
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "commitment_id": {"type": "string", "description": "The commitment ID to update."},
                        "status": {"type": "string", "description": "New status: completed, abandoned, blocked, or active."},
                        "completion_evidence": {
                            "type": "string",
                            "description": "Positive proof used when marking evidence-bearing work complete.",
                        },
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
                name="workflow_get_commitment",
                description="Get full details for one durable commitment by commitment_id, including lifecycle and metadata.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "commitment_id": {
                            "type": "string",
                            "description": "Commitment ID returned by workflow_list_commitments or workflow_create_commitment.",
                        },
                    },
                    "required": ["commitment_id"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_create_schedule",
                description="Create a durable calendar item for the agent: a scheduled task, self-reminder, Gmail alert monitor, future intention, or event. This is OpenCAS's own calendar, separate from OS cron. Use a future ISO-8601 start_at; supports none, interval_hours, daily, weekly, and weekdays recurrence.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["task", "event"]},
                        "action": {
                            "type": "string",
                            "enum": ["submit_baa", "reminder_only", "gmail_alert"],
                            "description": (
                                "submit_baa starts bounded agent work; reminder_only records a reminder/event; "
                                "gmail_alert checks Gmail through GWS and sends Telegram only for new matching message IDs."
                            ),
                        },
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
                        "delay_reason": {
                            "type": "string",
                            "description": "Required when intentionally deferring an active unfinished project return by more than about two weeks.",
                        },
                        "meta": {"type": "object"},
                    },
                    "required": ["title", "start_at"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_update_schedule",
                description=(
                    "Update an OpenCAS schedule record. Use status=cancelled to unschedule a future "
                    "OpenCAS scheduled task, reminder, or recurring item. This is the right tool for "
                    "schedule IDs; workflow_cancel_task is only for BAA task IDs."
                ),
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "schedule_id": {
                            "type": "string",
                            "description": "OpenCAS schedule_id returned by workflow_list_schedules or workflow_create_schedule.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "paused", "completed", "cancelled"],
                            "description": "Set to cancelled to unschedule; paused temporarily stops; active resumes.",
                        },
                        "title": {"type": "string", "description": "Optional replacement schedule title."},
                        "description": {"type": "string", "description": "Optional replacement schedule description."},
                        "objective": {"type": "string", "description": "Optional replacement task objective."},
                        "priority": {"type": "number", "description": "Optional replacement priority."},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional replacement tag list."},
                    },
                    "required": ["schedule_id"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_cancel_schedule",
                description=(
                    "Cancel or unschedule an OpenCAS scheduled task, event, reminder, or recurring "
                    "calendar item by schedule_id. Use this for schedule IDs returned by "
                    "workflow_list_schedules; workflow_cancel_task is only for BAA task IDs."
                ),
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "schedule_id": {
                            "type": "string",
                            "description": "OpenCAS schedule_id returned by workflow_list_schedules or workflow_create_schedule.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Optional reason for cancelling the schedule.",
                        },
                    },
                    "required": ["schedule_id"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_list_schedules",
                description="List the agent's durable calendar items: scheduled tasks, reminders, future intentions, and events.",
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
                name="workflow_get_schedule",
                description="Get full details and recent run history for one OpenCAS schedule by schedule_id.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "schedule_id": {
                            "type": "string",
                            "description": "OpenCAS schedule_id returned by workflow_list_schedules or workflow_create_schedule.",
                        },
                        "run_limit": {
                            "type": "integer",
                            "description": "Maximum recent schedule runs to include. Default 10.",
                        },
                    },
                    "required": ["schedule_id"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_cancel_project",
                description=(
                    "Cancel or compost a project cleanly: move its managed workspace to _compost, "
                    "abandon linked project commitments, cancel linked schedules, cancel linked BAA tasks, "
                    "and write a receipt. Use this instead of ad hoc deletion when a project should be thrown out, "
                    "composted, or restarted."
                ),
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "project_key": {"type": "string", "description": "Stable project key, e.g. kpony."},
                        "project_title": {"type": "string", "description": "Human project title, e.g. kPony."},
                        "workspace_path": {
                            "type": "string",
                            "description": "Managed-workspace relative path to compost, e.g. kPony.",
                        },
                        "reason": {"type": "string"},
                        "hard_delete_tasks": {"type": "boolean"},
                    },
                    "required": ["reason"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_cancel_task",
                description=(
                    "Cancel or delete a BAA task. Soft cancel marks the task cancelled/failed with a reason; "
                    "hard_delete removes the task row. Does not cancel OpenCAS schedules; use "
                    "workflow_update_schedule with status=cancelled for schedule IDs."
                ),
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "BAA task_id to cancel, not a schedule_id."},
                        "reason": {"type": "string", "description": "Why the BAA task is being cancelled."},
                        "hard_delete": {
                            "type": "boolean",
                            "description": "If true, delete the BAA task row instead of soft-cancelling it.",
                        },
                    },
                    "required": ["task_id", "reason"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_list_tasks",
                description="List BAA background tasks so a task_id can be inspected or cancelled.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Maximum tasks to return. Default 20."},
                        "stage": {"type": "string", "description": "Optional raw execution stage filter."},
                        "status": {"type": "string", "description": "Optional raw task status filter."},
                        "project_id": {"type": "string", "description": "Optional project ID filter."},
                        "commitment_id": {"type": "string", "description": "Optional linked commitment ID filter."},
                    },
                    "required": [],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_get_task",
                description="Get full detail for one BAA background task by task_id, including lifecycle transitions and result.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "BAA task_id returned by workflow_list_tasks or schedule run history.",
                        },
                        "transition_limit": {
                            "type": "integer",
                            "description": "Maximum lifecycle transitions to include. Default 50.",
                        },
                        "salvage_limit": {
                            "type": "integer",
                            "description": "Maximum salvage packets to include. Default 10.",
                        },
                    },
                    "required": ["task_id"],
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
                description="Update a plan's content and/or status.",
                risk_tier=ActionRiskTier.WORKSPACE_WRITE,
                schema={
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string", "description": "The plan ID to update."},
                        "content": {"type": "string", "description": "Updated plan content."},
                        "status": {"type": "string", "description": "Optional new status: draft, active, completed, or abandoned."},
                    },
                    "required": ["plan_id"],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_list_plans",
                description="List active durable plans, optionally filtered by project_id or task_id.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string", "description": "Optional linked project or commitment ID filter."},
                        "task_id": {"type": "string", "description": "Optional linked BAA task ID filter."},
                        "limit": {"type": "integer", "description": "Maximum plans to return. Default 20."},
                    },
                    "required": [],
                },
            ),
            ToolRegistrationSpec(
                name="workflow_get_plan",
                description="Get full details and recent action history for one durable plan by plan_id.",
                risk_tier=ActionRiskTier.READONLY,
                schema={
                    "type": "object",
                    "properties": {
                        "plan_id": {
                            "type": "string",
                            "description": "Plan ID returned by workflow_list_plans, workflow_create_plan, or related workflow tools.",
                        },
                        "action_limit": {
                            "type": "integer",
                            "description": "Maximum recent plan actions to include. Default 25.",
                        },
                    },
                    "required": ["plan_id"],
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
