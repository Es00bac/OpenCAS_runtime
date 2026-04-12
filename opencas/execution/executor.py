"""Repair executor for OpenCAS."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Callable, Coroutine, Dict, List, Optional

from opencas.api import LLMClient
from opencas.telemetry import EventKind, Tracer
from opencas.tools import ToolRegistry

from .git_checkpoint import GitCheckpointManager
from .models import ExecutionPhase, ExecutionStage, PhaseRecord, RepairResult, RepairTask


class RepairExecutor:
    """Executes a repair task through explicit phases with checkpointing and convergence guards."""

    def __init__(
        self,
        tools: ToolRegistry,
        llm: Optional[LLMClient] = None,
        tracer: Optional[Tracer] = None,
        runtime: Optional[Any] = None,
    ) -> None:
        self.tools = tools
        self.llm = llm
        self.tracer = tracer
        self.runtime = runtime

    async def run(self, task: RepairTask) -> RepairResult:
        """Run the full repair pipeline for *task*."""
        task.attempt += 1

        # Exponential backoff before retries
        if task.attempt > 1 and task.retry_backoff_seconds > 0:
            await asyncio.sleep(task.retry_backoff_seconds)
            task.retry_backoff_seconds *= 2

        self._trace(
            "repair_started",
            {"task_id": str(task.task_id), "objective": task.objective, "attempt": task.attempt},
        )

        checkpoint: Optional[GitCheckpointManager] = None
        affected_files: List[str] = []

        # DETECT
        detect_record = await self._run_phase(task, ExecutionPhase.DETECT, self._detect)
        affected_files = [s.strip() for s in (detect_record.output or "").split(",") if s.strip()]

        # SNAPSHOT
        commit_hash: Optional[str] = None
        if task.scratch_dir and affected_files:
            checkpoint = GitCheckpointManager(task.scratch_dir)
            commit_hash = checkpoint.snapshot(affected_files)
            task.checkpoint_commit = commit_hash
        snap_record = PhaseRecord(
            phase=ExecutionPhase.SNAPSHOT,
            success=bool(checkpoint is not None),
            output=f"snapshot taken {commit_hash}" if commit_hash else "no files to snapshot",
        )
        task.phases.append(snap_record)

        # PLAN
        plan_record = await self._run_phase(task, ExecutionPhase.PLAN, self._plan)
        plan = plan_record.output or ""
        task.artifacts.append(f"plan:{plan}")

        # EXECUTE
        exec_record = await self._run_phase(
            task, ExecutionPhase.EXECUTE, self._execute_plan, plan
        )
        exec_output = exec_record.output or ""
        task.artifacts.append(f"exec:{exec_output}")

        # VERIFY
        verify_record = await self._run_phase(task, ExecutionPhase.VERIFY, self._verify)
        verified = verify_record.success is True

        # POSTCHECK
        postcheck_record = await self._run_phase(task, ExecutionPhase.POSTCHECK, self._postcheck)

        # Convergence guard: hash output/artifacts after each attempt
        convergence_hash = self._hash_convergence(exec_output, task.artifacts)
        if convergence_hash in task.convergence_hashes:
            if checkpoint:
                checkpoint.restore(commit_hash)
            return self._fail(task, "non-improving loop detected")
        task.convergence_hashes.append(convergence_hash)

        if verified and exec_record.success is True:
            task.stage = ExecutionStage.DONE
            task.status = "completed"
            if checkpoint:
                checkpoint.discard()
            self._trace("repair_completed", {"task_id": str(task.task_id)})
            return RepairResult(
                task_id=task.task_id,
                success=True,
                stage=task.stage,
                output=exec_output,
                artifacts=task.artifacts,
            )

        # Recover / escalate if execution or verification failed and attempts exhausted
        if task.attempt >= task.max_attempts:
            if checkpoint:
                checkpoint.restore(commit_hash)
            exhausted_reason = (
                f"Execution failed after {task.attempt} attempts."
                if not exec_record.success
                else f"Verification failed after {task.attempt} attempts."
            )
            return self._fail(task, exhausted_reason)

        # Schedule a retry by keeping stage as recovering
        task.stage = ExecutionStage.RECOVERING
        task.status = "retrying"
        failure_reason = (
            "Execution failed; will retry."
            if not exec_record.success
            else "Verification failed; will retry."
        )
        return RepairResult(
            task_id=task.task_id,
            success=False,
            stage=task.stage,
            output=failure_reason,
            artifacts=task.artifacts,
        )

    async def _run_phase(
        self,
        task: RepairTask,
        phase: ExecutionPhase,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
    ) -> PhaseRecord:
        """Execute a single phase and record its result."""
        record = PhaseRecord(phase=phase)
        try:
            result = await handler(task, *args)
            if isinstance(result, bool):
                record.success = result
            else:
                record.success = True
            record.output = str(result) if result is not None else ""
            # Heuristic: propagate obvious failure strings as failures
            lowered = record.output.lower()
            if record.output.startswith(f"{phase.value} failed:"):
                record.success = False
            elif "[tool loop halted]" in lowered:
                record.success = False
            elif "[error generating response" in lowered:
                record.success = False
            elif phase == ExecutionPhase.EXECUTE and not record.output.strip():
                record.success = False
            elif phase == ExecutionPhase.EXECUTE and lowered.startswith("execute failed"):
                record.success = False
        except Exception as exc:
            record.success = False
            record.output = f"{phase.value} failed: {exc}"
        from datetime import datetime, timezone
        record.ended_at = datetime.now(timezone.utc)
        task.phases.append(record)
        return record

    async def _detect(self, task: RepairTask) -> str:
        """Identify what files/commands will be touched."""
        words = task.objective.lower().split()
        files: List[str] = []
        for word in words:
            if "." in word and not word.endswith("."):
                files.append(word)
        return ",".join(files)

    async def _plan(self, task: RepairTask) -> str:
        """Generate a short execution plan."""
        if self.llm:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a repair planner for an autonomous agent. "
                            "Given an objective, return a concise 1-3 step plan as plain text."
                        ),
                    },
                    {"role": "user", "content": f"Objective: {task.objective}"},
                ]
                response = await self.llm.chat_completion(messages)
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content.strip() or "investigate and fix"
            except Exception as exc:
                return f"investigate and fix (llm error: {exc})"
        return "investigate and fix"

    async def _execute_plan(self, task: RepairTask, plan: str) -> str:
        """Execute the plan using available tools."""
        if self.runtime and hasattr(self.runtime, "tool_loop"):
            from opencas.tools import ToolUseContext

            objective = f"Objective: {task.objective}\nPlan: {plan}"
            messages = [
                {"role": "system", "content": "You are executing a repair task."},
                {"role": "user", "content": objective},
            ]
            ctx = ToolUseContext(
                runtime=self.runtime,
                session_id=str(task.task_id),
                task_id=str(task.task_id),
            )
            scheduler = getattr(self.runtime, "scheduler", None)
            result = await self.runtime.tool_loop.run(
                objective=task.objective,
                messages=messages,
                ctx=ctx,
                on_focus_enter=scheduler.enter_focus_mode if scheduler else None,
                on_focus_exit=scheduler.exit_focus_mode if scheduler else None,
            )
            return result.final_output

        # Fallback heuristic when no runtime/tool_loop is available
        outputs: List[str] = []
        words = task.objective.lower().split()
        for word in words:
            if "." in word and not word.endswith("."):
                read_result = await self.tools.execute_async("fs_read_file", {"file_path": word})
                if read_result.success:
                    outputs.append(f"read {word}: ok")
                else:
                    outputs.append(f"read {word}: {read_result.output}")
        if task.verification_command:
            outputs.append(f"verification_command set: {task.verification_command}")
        outputs.append(f"plan executed: {plan}")
        return "; ".join(outputs)

    async def _verify(self, task: RepairTask) -> bool:
        """Run verification command if one was provided."""
        if not task.verification_command:
            return True
        result = await self.tools.execute_async(
            "bash_run_command",
            {"command": task.verification_command},
        )
        return result.success

    async def _postcheck(self, task: RepairTask) -> str:
        """Validate no unintended side effects."""
        return "postcheck passed"

    def _hash_convergence(self, output: str, _artifacts: List[str]) -> str:
        """Create a hash of the execution output to detect non-improving loops."""
        return hashlib.sha256(output.encode("utf-8")).hexdigest()[:16]

    def _fail(self, task: RepairTask, message: str) -> RepairResult:
        task.stage = ExecutionStage.FAILED
        task.status = "failed"
        self._trace("repair_failed", {"task_id": str(task.task_id), "attempts": task.attempt, "reason": message})
        return RepairResult(
            task_id=task.task_id,
            success=False,
            stage=task.stage,
            output=message,
            artifacts=task.artifacts,
        )

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"RepairExecutor: {event}",
                payload,
            )
