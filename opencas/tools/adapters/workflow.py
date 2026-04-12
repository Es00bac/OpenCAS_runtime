"""Higher-level operator workflow tools.

These composite tools let the LLM operate at task level rather than
choreographing low-level tool calls manually.  Each tool wraps underlying
stores (commitments, work objects, plans) and runtime capabilities.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from ...autonomy.commitment import Commitment, CommitmentStatus
from ..models import ToolResult


class WorkflowToolAdapter:
    """Composite workflow tools for writing, project management, and supervision."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        handler = {
            "workflow_create_commitment": self._create_commitment,
            "workflow_update_commitment": self._update_commitment,
            "workflow_list_commitments": self._list_commitments,
            "workflow_create_schedule": self._create_schedule,
            "workflow_update_schedule": self._update_schedule,
            "workflow_list_schedules": self._list_schedules,
            "workflow_create_writing_task": self._create_writing_task,
            "workflow_create_plan": self._create_plan,
            "workflow_update_plan": self._update_plan,
            "workflow_repo_triage": self._repo_triage,
            "workflow_supervise_session": self._supervise_session,
        }.get(name)
        if handler is None:
            return ToolResult(False, f"Unknown workflow tool: {name}", {})
        try:
            return await handler(args)
        except Exception as exc:
            return ToolResult(False, str(exc), {"error_type": type(exc).__name__})

    # ── Commitment / Goal Management ──────────────────────────────────

    async def _create_commitment(self, args: Dict[str, Any]) -> ToolResult:
        """Create a durable commitment with optional deadline and priority."""
        store = self.runtime.commitment_store
        if store is None:
            return ToolResult(False, "Commitment store not available", {})

        content = str(args.get("content", "")).strip()
        if not content:
            return ToolResult(False, "Missing required argument: content", {})

        priority = float(args.get("priority", 5.0))
        deadline_str = args.get("deadline")
        deadline = None
        if deadline_str:
            deadline = datetime.fromisoformat(str(deadline_str))

        tags = args.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        commitment = Commitment(
            content=content,
            priority=priority,
            deadline=deadline,
            tags=tags,
        )
        await store.save(commitment)

        return ToolResult(
            True,
            json.dumps({
                "commitment_id": str(commitment.commitment_id),
                "content": content,
                "priority": priority,
                "status": commitment.status.value,
            }),
            {"commitment_id": str(commitment.commitment_id)},
        )

    async def _update_commitment(self, args: Dict[str, Any]) -> ToolResult:
        """Update a commitment's status (complete, abandon, block, activate)."""
        store = self.runtime.commitment_store
        if store is None:
            return ToolResult(False, "Commitment store not available", {})

        commitment_id = str(args.get("commitment_id", "")).strip()
        if not commitment_id:
            return ToolResult(False, "Missing required argument: commitment_id", {})

        new_status = str(args.get("status", "")).strip().lower()
        status_map = {
            "completed": CommitmentStatus.COMPLETED,
            "complete": CommitmentStatus.COMPLETED,
            "abandoned": CommitmentStatus.ABANDONED,
            "abandon": CommitmentStatus.ABANDONED,
            "blocked": CommitmentStatus.BLOCKED,
            "block": CommitmentStatus.BLOCKED,
            "active": CommitmentStatus.ACTIVE,
            "activate": CommitmentStatus.ACTIVE,
        }
        status = status_map.get(new_status)
        if status is None:
            return ToolResult(
                False,
                f"Invalid status '{new_status}'. Use: completed, abandoned, blocked, active",
                {},
            )

        ok = await store.update_status(commitment_id, status)
        if not ok:
            return ToolResult(False, f"Commitment {commitment_id} not found", {})

        return ToolResult(
            True,
            json.dumps({"commitment_id": commitment_id, "status": status.value}),
            {},
        )

    async def _list_commitments(self, args: Dict[str, Any]) -> ToolResult:
        """List active commitments with optional filtering."""
        store = self.runtime.commitment_store
        if store is None:
            return ToolResult(False, "Commitment store not available", {})

        status_filter = str(args.get("status", "active")).strip().lower()
        limit = int(args.get("limit", 20))

        status_map = {
            "active": CommitmentStatus.ACTIVE,
            "completed": CommitmentStatus.COMPLETED,
            "abandoned": CommitmentStatus.ABANDONED,
            "blocked": CommitmentStatus.BLOCKED,
        }
        status = status_map.get(status_filter, CommitmentStatus.ACTIVE)
        items = await store.list_by_status(status, limit=limit)

        entries = []
        for item in items:
            entries.append({
                "commitment_id": str(item.commitment_id),
                "content": item.content,
                "priority": item.priority,
                "status": item.status.value,
                "tags": item.tags,
                "deadline": item.deadline.isoformat() if item.deadline else None,
                "created_at": item.created_at.isoformat(),
            })

        return ToolResult(
            True,
            json.dumps({"count": len(entries), "items": entries}),
            {},
        )

    # ── Scheduling ───────────────────────────────────────────────────

    async def _create_schedule(self, args: Dict[str, Any]) -> ToolResult:
        service = getattr(self.runtime, "schedule_service", None)
        if service is None:
            return ToolResult(False, "Schedule service not available", {})

        from opencas.scheduling import ScheduleAction, ScheduleKind

        kind = ScheduleKind(str(args.get("kind", "task")))
        action_arg = args.get("action")
        action = (
            ScheduleAction(str(action_arg))
            if action_arg
            else ScheduleAction.SUBMIT_BAA
            if kind == ScheduleKind.TASK
            else ScheduleAction.REMINDER_ONLY
        )
        payload = {
            "kind": kind,
            "action": action,
            "title": str(args.get("title", "")).strip(),
            "description": str(args.get("description", "") or ""),
            "objective": args.get("objective"),
            "start_at": datetime.fromisoformat(str(args.get("start_at"))),
            "end_at": datetime.fromisoformat(str(args["end_at"])) if args.get("end_at") else None,
            "timezone": str(args.get("timezone", "America/Denver")),
            "recurrence": str(args.get("recurrence", "none")),
            "interval_hours": args.get("interval_hours"),
            "weekdays": args.get("weekdays", []),
            "max_occurrences": args.get("max_occurrences"),
            "priority": float(args.get("priority", 5.0)),
            "tags": args.get("tags", []),
            "commitment_id": args.get("commitment_id"),
            "plan_id": args.get("plan_id"),
            "meta": args.get("meta", {}),
        }
        if not payload["title"] or not args.get("start_at"):
            return ToolResult(False, "Missing required arguments: title, start_at", {})
        item = await service.create_schedule(**payload)
        return ToolResult(
            True,
            json.dumps({
                "schedule_id": str(item.schedule_id),
                "title": item.title,
                "kind": item.kind.value,
                "action": item.action.value,
                "next_run_at": item.next_run_at.isoformat() if item.next_run_at else None,
            }),
            {"schedule_id": str(item.schedule_id)},
        )

    async def _update_schedule(self, args: Dict[str, Any]) -> ToolResult:
        store = getattr(self.runtime.ctx, "schedule_store", None)
        if store is None:
            return ToolResult(False, "Schedule store not available", {})
        schedule_id = str(args.get("schedule_id", "")).strip()
        if not schedule_id:
            return ToolResult(False, "Missing required argument: schedule_id", {})
        item = await store.get(schedule_id)
        if item is None:
            return ToolResult(False, f"Schedule {schedule_id} not found", {})
        if "status" in args:
            from opencas.scheduling import ScheduleStatus
            item.status = ScheduleStatus(str(args["status"]))
            if item.status in (ScheduleStatus.CANCELLED, ScheduleStatus.COMPLETED, ScheduleStatus.PAUSED):
                item.next_run_at = None
        for key in ("title", "description", "objective", "priority", "tags"):
            if key in args:
                setattr(item, key, args[key])
        await store.save(item)
        return ToolResult(True, json.dumps({"schedule_id": schedule_id, "updated": True}), {})

    async def _list_schedules(self, args: Dict[str, Any]) -> ToolResult:
        store = getattr(self.runtime.ctx, "schedule_store", None)
        if store is None:
            return ToolResult(False, "Schedule store not available", {})
        from opencas.scheduling import ScheduleKind, ScheduleStatus

        status = args.get("status", "active")
        kind = args.get("kind")
        items = await store.list_items(
            status=ScheduleStatus(str(status)) if status else None,
            kind=ScheduleKind(str(kind)) if kind else None,
            limit=int(args.get("limit", 20)),
        )
        payload = [
            {
                "schedule_id": str(item.schedule_id),
                "title": item.title,
                "kind": item.kind.value,
                "action": item.action.value,
                "status": item.status.value,
                "next_run_at": item.next_run_at.isoformat() if item.next_run_at else None,
                "recurrence": item.recurrence.value,
            }
            for item in items
        ]
        return ToolResult(True, json.dumps({"count": len(payload), "items": payload}), {})

    # ── Writing Workflow ──────────────────────────────────────────────

    async def _create_writing_task(self, args: Dict[str, Any]) -> ToolResult:
        """Create a writing task: sets up a commitment, output path, and optional outline plan."""
        store = self.runtime.commitment_store
        work_store = getattr(self.runtime.ctx, "work_store", None)

        title = str(args.get("title", "")).strip()
        if not title:
            return ToolResult(False, "Missing required argument: title", {})

        description = str(args.get("description", "")).strip()
        output_path = str(args.get("output_path", "")).strip()
        outline = args.get("outline")

        # Resolve output path
        if not output_path:
            workspace = str(self.runtime.ctx.config.primary_workspace_root())
            safe_name = title.lower().replace(" ", "_")[:40]
            output_path = str(Path(workspace) / "notes" / f"{safe_name}.md")

        # Create commitment
        commitment_id = None
        if store is not None:
            commitment = Commitment(
                content=f"Write: {title}",
                priority=float(args.get("priority", 6.0)),
                tags=["writing"],
            )
            await store.save(commitment)
            commitment_id = str(commitment.commitment_id)

        # Create plan if outline provided
        plan_id = None
        plan_store = getattr(self.runtime.ctx, "plan_store", None)
        if plan_store is not None and outline:
            outline_text = outline if isinstance(outline, str) else json.dumps(outline)
            plan_id = f"plan-{uuid4().hex[:8]}"
            await plan_store.create_plan(
                plan_id,
                content=f"Writing plan for: {title}\n\n{outline_text}",
                project_id=commitment_id,
            )
            await plan_store.set_status(plan_id, "active")

        # Write initial scaffold
        scaffold = f"# {title}\n\n"
        if description:
            scaffold += f"> {description}\n\n"
        if outline:
            if isinstance(outline, list):
                for section in outline:
                    scaffold += f"## {section}\n\n"
            elif isinstance(outline, str):
                scaffold += outline + "\n\n"
        scaffold += "<!-- Created by OpenCAS writing workflow -->\n"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(scaffold, encoding="utf-8")

        return ToolResult(
            True,
            json.dumps({
                "title": title,
                "output_path": output_path,
                "commitment_id": commitment_id,
                "plan_id": plan_id,
                "scaffold_written": True,
            }),
            {"output_path": output_path},
        )

    # ── Planning ──────────────────────────────────────────────────────

    async def _create_plan(self, args: Dict[str, Any]) -> ToolResult:
        """Create a structured plan with steps."""
        plan_store = getattr(self.runtime.ctx, "plan_store", None)
        if plan_store is None:
            return ToolResult(False, "Plan store not available", {})

        content = str(args.get("content", "")).strip()
        if not content:
            return ToolResult(False, "Missing required argument: content", {})

        project_id = args.get("project_id")
        task_id = args.get("task_id")

        plan_id = f"plan-{uuid4().hex[:8]}"
        await plan_store.create_plan(
            plan_id,
            content=content,
            project_id=str(project_id) if project_id else None,
            task_id=str(task_id) if task_id else None,
        )
        await plan_store.set_status(plan_id, "active")

        return ToolResult(
            True,
            json.dumps({
                "plan_id": plan_id,
                "content_preview": content[:200],
                "status": "active",
            }),
            {"plan_id": plan_id},
        )

    async def _update_plan(self, args: Dict[str, Any]) -> ToolResult:
        """Update a plan's content."""
        plan_store = getattr(self.runtime.ctx, "plan_store", None)
        if plan_store is None:
            return ToolResult(False, "Plan store not available", {})

        plan_id = str(args.get("plan_id", "")).strip()
        content = str(args.get("content", "")).strip()
        if not plan_id or not content:
            return ToolResult(False, "Missing required arguments: plan_id, content", {})

        ok = await plan_store.update_content(plan_id, content)
        if not ok:
            return ToolResult(False, f"Plan {plan_id} not found", {})

        return ToolResult(
            True,
            json.dumps({"plan_id": plan_id, "updated": True}),
            {},
        )

    # ── Repo Triage ───────────────────────────────────────────────────

    async def _repo_triage(self, args: Dict[str, Any]) -> ToolResult:
        """Quick repo triage: git status, recent commits, test health, open work items."""
        workspace = str(self.runtime.ctx.config.primary_workspace_root())

        # Git status
        git_status = await self.runtime.execute_tool(
            "bash_run_command",
            {"command": "git status --short 2>/dev/null || echo 'not a git repo'", "cwd": workspace},
        )

        # Recent commits
        git_log = await self.runtime.execute_tool(
            "bash_run_command",
            {"command": "git log --oneline -10 2>/dev/null || echo 'no git history'", "cwd": workspace},
        )

        # Work item summary
        work_summary = {"total": 0, "ready": 0, "blocked": 0}
        if getattr(self.runtime.ctx, "work_store", None) is not None:
            work_summary = await self.runtime.ctx.work_store.summary_counts()

        # Active commitments count
        commitment_count = 0
        if self.runtime.commitment_store is not None:
            commitment_count = await self.runtime.commitment_store.count_by_status(
                CommitmentStatus.ACTIVE
            )

        # Active plans count
        plan_count = 0
        if getattr(self.runtime.ctx, "plan_store", None) is not None:
            plan_count = await self.runtime.ctx.plan_store.count_active()

        result = {
            "workspace": workspace,
            "git_status": git_status.get("output", ""),
            "recent_commits": git_log.get("output", ""),
            "work_items": work_summary,
            "active_commitments": commitment_count,
            "active_plans": plan_count,
        }

        return ToolResult(True, json.dumps(result), {"workspace": workspace})

    # ── External Tool Supervision ─────────────────────────────────────

    @staticmethod
    def _supervision_advisory(
        pty_data: Dict[str, Any],
        *,
        verification_exists: bool,
    ) -> Dict[str, Any]:
        screen_state = pty_data.get("screen_state", {}) or {}
        mode = str(screen_state.get("mode", "idle"))
        ready_for_input = bool(screen_state.get("ready_for_input", False))
        blocked = bool(screen_state.get("blocked", False))
        running = pty_data.get("running")

        if verification_exists:
            return {
                "action": "stop",
                "reason": "verification_satisfied",
                "next_step": "cleanup_and_verify",
                "observe_idle_seconds": 0.2,
                "observe_max_wait_seconds": 1.0,
            }
        if running is False:
            return {
                "action": "stop",
                "reason": "process_exited",
                "next_step": "inspect_exit_state",
                "observe_idle_seconds": 0.2,
                "observe_max_wait_seconds": 1.0,
            }
        if blocked or mode == "auth_required":
            return {
                "action": "stop",
                "reason": "auth_or_gate_blocked",
                "next_step": "resolve_auth_or_gate",
                "observe_idle_seconds": 0.2,
                "observe_max_wait_seconds": 1.0,
            }
        if mode in {"shell_prompt", "error_prompt"}:
            return {
                "action": "stop",
                "reason": "awaiting_operator_input",
                "next_step": "send_follow_up_input",
                "observe_idle_seconds": 0.2,
                "observe_max_wait_seconds": 1.5,
            }
        if ready_for_input:
            return {
                "action": "observe_briefly",
                "reason": "interactive_idle_ready_for_input",
                "next_step": "send_follow_up_if_stalled",
                "observe_idle_seconds": 0.35,
                "observe_max_wait_seconds": 2.5,
            }
        return {
            "action": "observe",
            "reason": "still_running",
            "next_step": "continue_observing",
            "observe_idle_seconds": 1.0,
            "observe_max_wait_seconds": 15.0,
        }

    async def _supervise_session(self, args: Dict[str, Any]) -> ToolResult:
        """Launch or resume a PTY tool session, send a task, and observe the result.

        This is the key operator workflow for supervising claude/codex/other TUI
        tools — it handles the start-send-observe choreography in a single call.
        """
        command = str(args.get("command", "")).strip()
        session_id = args.get("session_id")
        task_input = str(args.get("task", "")).strip()
        scope_key = str(args.get("scope_key", "workflow-supervision"))
        max_wait = float(args.get("max_wait_seconds", 15.0))
        idle_seconds = float(args.get("idle_seconds", 1.0))
        startup_wait = float(args.get("startup_wait_seconds", min(max_wait, 8.0)))
        continue_wait = float(args.get("continue_wait_seconds", max_wait))
        max_rounds = max(1, int(args.get("max_rounds", 3)))
        verification_path_arg = str(args.get("verification_path", "")).strip()
        verification_path = Path(verification_path_arg) if verification_path_arg else None
        advisory: Dict[str, Any] = {}
        rounds_used = 0

        if not command and not session_id:
            return ToolResult(
                False,
                "Either command (to start new session) or session_id (to resume) is required",
                {},
            )

        # Full-screen TUIs can still be booting when the session first opens. If
        # we send the task in the same PTY interaction that starts the process,
        # the text may land before the UI is ready and remain stuck in the composer.
        # Stage new sessions into: start -> observe readiness -> submit task.
        start_args: Dict[str, Any] = {
            "scope_key": scope_key,
            "idle_seconds": idle_seconds,
            "max_wait_seconds": max_wait,
        }
        if session_id:
            start_args["session_id"] = str(session_id)
        if command:
            start_args["command"] = command

        active_session_id: Optional[str] = str(session_id).strip() if session_id else None
        if command and not session_id:
            start_args["max_wait_seconds"] = startup_wait
            start_result = await self.runtime.execute_tool("pty_interact", start_args)
            start_raw = start_result.get("output", "")
            try:
                start_data = json.loads(start_raw) if isinstance(start_raw, str) else start_raw
            except (json.JSONDecodeError, TypeError):
                start_data = {"raw": start_raw}
            active_session_id = str(start_data.get("session_id") or "").strip() or None
            rounds_used += 1
            if not task_input:
                pty_data = start_data
            elif not active_session_id:
                return ToolResult(
                    False,
                    "PTY session did not return a usable session_id during startup",
                    {"scope_key": scope_key},
                )
            else:
                startup_advisory = self._supervision_advisory(
                    start_data,
                    verification_exists=verification_path.exists() if verification_path else False,
                )
                if startup_advisory["action"] == "stop":
                    pty_data = start_data
                    advisory = startup_advisory
                else:
                    submit_args: Dict[str, Any] = {
                        "scope_key": scope_key,
                        "session_id": active_session_id,
                        "idle_seconds": idle_seconds,
                        "max_wait_seconds": max_wait,
                        "input": task_input + "\r",
                    }
                    result = await self.runtime.execute_tool("pty_interact", submit_args)
                    raw = result.get("output", "")
                    try:
                        pty_data = json.loads(raw) if isinstance(raw, str) else raw
                    except (json.JSONDecodeError, TypeError):
                        pty_data = {"raw": raw}
                    pty_data["session_id"] = active_session_id
                    rounds_used += 1
        else:
            if task_input:
                # Full-screen TUIs generally expect Enter as carriage return rather than
                # a plain newline byte. Sending "\n" leaves the prompt in the input box
                # for tools like Kilo Code instead of submitting it.
                start_args["input"] = task_input + "\r"
            result = await self.runtime.execute_tool("pty_interact", start_args)
            raw = result.get("output", "")
            try:
                pty_data = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                pty_data = {"raw": raw}
            rounds_used += 1

        active_session_id = str(pty_data.get("session_id") or active_session_id or "").strip() or None
        verification_exists = verification_path.exists() if verification_path else False
        if not advisory:
            advisory = self._supervision_advisory(
                pty_data,
                verification_exists=verification_exists,
            )

        for _ in range(max_rounds - 1):
            if verification_exists:
                break
            if not active_session_id:
                break
            if pty_data.get("running") is False:
                break
            if advisory.get("action") == "stop":
                break

            observe_idle_seconds = min(
                idle_seconds,
                float(advisory.get("observe_idle_seconds", idle_seconds)),
            )
            observe_max_wait_seconds = min(
                continue_wait,
                float(advisory.get("observe_max_wait_seconds", continue_wait)),
            )
            observe = await self.runtime.execute_tool(
                "pty_observe",
                {
                    "session_id": active_session_id,
                    "scope_key": scope_key,
                    "idle_seconds": observe_idle_seconds,
                    "max_wait_seconds": observe_max_wait_seconds,
                },
            )
            observe_raw = observe.get("output", "")
            try:
                pty_data = json.loads(observe_raw) if isinstance(observe_raw, str) else observe_raw
            except (json.JSONDecodeError, TypeError):
                pty_data = {"raw": observe_raw}
            pty_data["session_id"] = active_session_id
            rounds_used += 1
            verification_exists = verification_path.exists() if verification_path else False
            advisory = self._supervision_advisory(
                pty_data,
                verification_exists=verification_exists,
            )

        return ToolResult(
            True,
            json.dumps({
                "session_id": pty_data.get("session_id"),
                "running": pty_data.get("running"),
                "cleaned_output": pty_data.get("cleaned_combined_output", ""),
                "screen_state": pty_data.get("screen_state", {}),
                "supervision_advisory": advisory,
                "idle_reached": pty_data.get("idle_reached", False),
                "timed_out": pty_data.get("timed_out", False),
                "elapsed_ms": pty_data.get("elapsed_ms"),
                "verification_path": str(verification_path) if verification_path else None,
                "verification_exists": verification_exists,
                "rounds": max_rounds,
                "rounds_used": rounds_used,
            }),
            {"scope_key": scope_key},
        )
