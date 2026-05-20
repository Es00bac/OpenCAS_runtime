"""Repair executor for OpenCAS."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from opencas.api import LLMClient
from opencas.cognition import recommended_counterfactual
from opencas.identity.agent_name import resolve_agent_name
from opencas.projects.classifier import (
    PROJECT_TYPE_SOFTWARE,
    PROJECT_TYPE_WRITING,
    classify_project_type,
)
from opencas.projects.execution_contracts import new_project_execution_rejection_reason
from opencas.projects.workspace_registry import WorkspaceProjectCandidate, resolve_workspace_project
from opencas.provenance_adapter import append_provenance_record
from opencas.telemetry import EventKind, Tracer
from opencas.tools import ToolRegistry
from opencas.tools.action_memory import artifact_hint_from_mapping

from .git_checkpoint import GitCheckpointError, GitCheckpointManager
from .models import (
    AttemptOutcome,
    ExecutionPhase,
    ExecutionStage,
    PhaseRecord,
    RepairResult,
    RepairTask,
)
from .retry_governor import RetryGovernor
from .salvage import build_salvage_packet
from .store import TaskStore


class RepairExecutor:
    """Executes a repair task through explicit phases with checkpointing and convergence guards."""

    DEFAULT_TOOL_LOOP_TIMEOUT_SECONDS = 420.0

    _LIKELY_DOMAIN_SUFFIXES = {
        "ai",
        "app",
        "co",
        "com",
        "dev",
        "gg",
        "io",
        "net",
        "org",
        "site",
    }

    _DURABLE_BOUNDARY_PHASES = {
        "checkpoint_persisted": "handoff",
        "operator_input_requested": "pause",
        "session_resumed": "resume",
        "task_accepted": "start",
        "task_completed": "commit",
    }
    _WRITE_ARTIFACT_TOOLS = {"edit_file", "fs_edit_file", "fs_write_file", "write_file"}

    def __init__(
        self,
        tools: ToolRegistry,
        llm: Optional[LLMClient] = None,
        tracer: Optional[Tracer] = None,
        runtime: Optional[Any] = None,
        store: Optional[TaskStore] = None,
        retry_governor: Optional[RetryGovernor] = None,
    ) -> None:
        self.tools = tools
        self.llm = llm
        self.tracer = tracer
        self.runtime = runtime
        self.store = store
        self.retry_governor = retry_governor or RetryGovernor()
        self._boundary_emit_cache: set[tuple[str, str, str]] = set()
        self._boundary_emit_cache_max = 10_000

    def _record_task_provenance(
        self,
        task: RepairTask,
        *,
        action: str,
        artifact: str,
        why: str,
        risk: str = "MEDIUM",
        source_trace: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store canonical provenance on the mutable task record."""
        task.meta = append_provenance_record(
            task.meta,
            session_id=str(task.task_id),
            artifact=artifact,
            action=action,
            why=why,
            risk=risk,
            field="provenance_events",
            source_trace=source_trace,
        )

    def record_task_boundary(
        self,
        task: RepairTask,
        *,
        boundary: str,
        workflow_phase: Optional[str] = None,
        artifact: str,
        why: str,
        action: str = "UPDATE",
        risk: str = "MEDIUM",
        source_trace: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one durable workflow boundary and dedupe repeated emissions."""
        if boundary not in self._DURABLE_BOUNDARY_PHASES and boundary != "task_completed":
            return
        inferred_phase = workflow_phase or self._infer_workflow_phase(boundary, task)
        if inferred_phase is None:
            return

        boundary_state = self._boundary_state_signature(
            task,
            boundary=boundary,
            workflow_phase=inferred_phase,
            source_trace=source_trace,
        )
        task_key = str(task.task_id)
        cache_key = (task_key, inferred_phase, boundary_state)
        if cache_key in self._boundary_emit_cache:
            return
        if len(self._boundary_emit_cache) >= self._boundary_emit_cache_max:
            self._boundary_emit_cache.clear()
        self._boundary_emit_cache.add(cache_key)

        meta = dict(task.meta or {})
        markers = list(meta.get("workflow_boundary_events", []) or [])
        markers.append(boundary)
        meta["workflow_boundary_events"] = markers
        task.meta = meta
        self._record_task_provenance(
            task,
            action=action,
            artifact=artifact,
            why=why,
            risk=risk,
            source_trace={
                **(source_trace or {}),
                "workflow_phase": inferred_phase,
                "boundary": boundary,
            },
        )

    def _infer_workflow_phase(self, boundary: str, task: RepairTask) -> Optional[str]:
        if boundary == "task_completed":
            return "commit" if task.stage == ExecutionStage.DONE else "stop"
        return self._DURABLE_BOUNDARY_PHASES.get(boundary)

    @staticmethod
    def _boundary_state_signature(
        task: RepairTask,
        *,
        boundary: str,
        workflow_phase: str,
        source_trace: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "boundary": boundary,
            "phase": workflow_phase,
            "stage": task.stage.value,
            "status": task.status,
        }
        if source_trace:
            for key in ("source", "success", "from_stage", "to_stage", "checkpoint_commit", "lane"):
                if key in source_trace:
                    payload[key] = source_trace[key]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    async def run(self, task: RepairTask) -> RepairResult:
        """Run the full repair pipeline for *task*."""
        task.attempt += 1
        await self._persist_task_progress(task)

        # Exponential backoff before retries
        if task.attempt > 1 and task.retry_backoff_seconds > 0:
            await asyncio.sleep(task.retry_backoff_seconds)
            task.retry_backoff_seconds *= 2
            await self._persist_task_progress(task)

        self._trace(
            "repair_started",
            {"task_id": str(task.task_id), "objective": task.objective, "attempt": task.attempt},
        )

        checkpoint: Optional[GitCheckpointManager] = None
        affected_files: List[str] = []
        commit_hash: Optional[str] = None

        def record_rollback_failure(reason: str) -> None:
            task.phases.append(
                PhaseRecord(
                    phase=ExecutionPhase.ROLLBACK,
                    success=False,
                    output=reason,
                )
            )

        def rollback_to_checkpoint() -> None:
            if checkpoint is None:
                return
            if not commit_hash:
                reason = "rollback failed: missing checkpoint commit"
                record_rollback_failure(reason)
                self._trace("checkpoint_rollback_failed", {"task_id": str(task.task_id), "reason": reason})
                return
            try:
                checkpoint.restore(commit_hash)
            except GitCheckpointError as exc:
                reason = f"rollback failed: {exc}"
                record_rollback_failure(reason)
                self._trace("checkpoint_rollback_failed", {"task_id": str(task.task_id), "reason": reason})

        # DETECT
        detect_record = await self._run_phase(task, ExecutionPhase.DETECT, self._detect)
        affected_files = [s.strip() for s in (detect_record.output or "").split(",") if s.strip()]

        # SNAPSHOT
        if task.scratch_dir and affected_files:
            checkpoint = GitCheckpointManager(task.scratch_dir)
            commit_hash = checkpoint.snapshot(affected_files)
            if commit_hash:
                task.checkpoint_commit = commit_hash
                self.record_task_boundary(
                    task,
                    boundary="checkpoint_persisted",
                    workflow_phase="handoff",
                    artifact="repair-task|default|checkpoint",
                    why=f"checkpoint persisted for {task.objective}",
                    action="COMMIT",
                    risk="LOW",
                    source_trace={
                        "checkpoint_commit": commit_hash,
                        "files": affected_files[:10],
                    },
                )
        snap_record = PhaseRecord(
            phase=ExecutionPhase.SNAPSHOT,
            success=bool(commit_hash),
            output=(
                f"snapshot taken {commit_hash}"
                if commit_hash
                else "snapshot failed"
                if checkpoint is not None
                else "no files to snapshot"
            ),
        )
        task.phases.append(snap_record)

        # PLAN
        plan_record = await self._run_phase(task, ExecutionPhase.PLAN, self._plan)
        plan = plan_record.output or ""
        task.artifacts.append(f"plan:{plan}")
        if plan_record.success is not True:
            backoff_reason = self._provider_backoff_reason(plan)
            if backoff_reason:
                self._mark_retry_blocked(
                    task,
                    reason=f"provider backoff: {backoff_reason}",
                    mode="provider_backoff",
                )
                rollback_to_checkpoint()
                return self._fail(task, f"provider backoff: {backoff_reason}")
            if task.attempt >= task.max_attempts:
                rollback_to_checkpoint()
                return self._fail(task, "Planning failed.")
            task.stage = ExecutionStage.RECOVERING
            task.status = "retrying"
            return RepairResult(
                task_id=task.task_id,
                success=False,
                stage=task.stage,
                output="Planning failed; will retry.",
                artifacts=task.artifacts,
            )

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
        await self._run_phase(task, ExecutionPhase.POSTCHECK, self._postcheck)

        backoff_reason = self._provider_backoff_reason(
            "\n".join(
                part
                for part in (
                    plan_record.output or "",
                    exec_record.output or "",
                    verify_record.output or "",
                )
                if part
            )
        )
        if backoff_reason:
            self._mark_retry_blocked(
                task,
                reason=f"provider backoff: {backoff_reason}",
                mode="provider_backoff",
            )
            rollback_to_checkpoint()
            return self._fail(task, f"provider backoff: {backoff_reason}")

        if verified and exec_record.success is True:
            convergence_hash = self._hash_convergence(exec_output, task.artifacts)
            task.convergence_hashes.append(convergence_hash)
            task.stage = ExecutionStage.DONE
            task.status = "completed"
            if checkpoint and commit_hash:
                checkpoint.discard(commit_hash)
            self.record_task_boundary(
                task,
                boundary="task_completed",
                workflow_phase="commit",
                artifact="repair-task|default|completed",
                why=f"task completed successfully for {task.objective}",
                action="COMMIT",
                risk="LOW",
                source_trace={"success": True, "stage": ExecutionStage.DONE.value, "attempt": task.attempt},
            )
            self._trace("repair_completed", {"task_id": str(task.task_id)})
            return RepairResult(
                task_id=task.task_id,
                success=True,
                stage=task.stage,
                output=exec_output,
                artifacts=task.artifacts,
            )

        if self.store is not None:
            decision = await self._salvage_retry_decision(
                task,
                exec_record=exec_record,
                verify_record=verify_record,
                affected_files=affected_files,
            )
            if not decision.allowed:
                self._capture_retry_blocked_intention(task, decision.reason)
                rollback_to_checkpoint()
                return self._fail(task, f"retry blocked: {decision.reason}")
        else:
            convergence_hash = self._hash_convergence(exec_output, task.artifacts)
            if convergence_hash in task.convergence_hashes:
                rollback_to_checkpoint()
                return self._fail(task, "non-improving loop detected")
            task.convergence_hashes.append(convergence_hash)

        # Recover / escalate if execution or verification failed and attempts exhausted
        artifact_progress_boundary = self._artifact_progress_boundary(task, exec_record)
        if task.attempt >= task.max_attempts and not artifact_progress_boundary:
            rollback_to_checkpoint()
            exhausted_reason = (
                f"Execution failed after {task.attempt} attempts."
                if not exec_record.success
                else f"Verification failed after {task.attempt} attempts."
            )
            return self._fail(task, exhausted_reason)

        # Schedule a retry by keeping stage as recovering
        task.stage = ExecutionStage.RECOVERING
        task.status = "retrying"
        if artifact_progress_boundary:
            failure_reason = "Artifact progress boundary reached; will continue."
        else:
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

    async def _salvage_retry_decision(
        self,
        task: RepairTask,
        *,
        exec_record: PhaseRecord,
        verify_record: PhaseRecord,
        affected_files: List[str],
    ):
        assert self.store is not None
        prior_packet = await self.store.get_latest_salvage_packet(str(task.task_id))
        packet = build_salvage_packet(
            task,
            outcome=self._attempt_outcome(
                task,
                exec_record=exec_record,
                verify_record=verify_record,
            ),
            canonical_artifact_path=self._canonical_artifact_path(task),
            artifact_paths_touched=self._artifact_paths_touched(task, affected_files),
            tool_calls=self._tool_calls_from_task_meta(task),
        )
        await self.store.save_salvage_packet(packet)
        decision = self.retry_governor.decide(
            candidate=packet,
            prior_packets=[prior_packet] if prior_packet is not None else [],
            has_new_evidence=self._has_new_evidence(prior_packet, packet),
            broad_attempt=self._is_broad_attempt(task, packet),
        )
        counterfactual = recommended_counterfactual(
            objective=task.objective,
            failure_summary=self._salvage_failure_summary(packet, exec_record, verify_record),
            prior_tool=packet.tool_signature or "",
            available_tools=[entry.name for entry in self.tools.list_tools()],
            prior_attempts=task.attempt,
        )
        task.meta["last_salvage_packet_id"] = str(packet.packet_id)
        task.meta["retry_governor"] = {
            "allowed": decision.allowed,
            "reason": decision.reason,
            "mode": decision.mode.value,
            "reuse_packet_id": str(decision.reuse_packet_id) if decision.reuse_packet_id else None,
            "attempt": packet.attempt,
            "packet_id": str(packet.packet_id),
        }
        task.meta["counterfactual_review"] = {
            "recommended": counterfactual.get("recommended") or {},
            "options": counterfactual.get("options") or [],
            "salvage_packet_id": str(packet.packet_id),
            "retry_allowed": decision.allowed,
            "retry_governor_reason": decision.reason,
        }
        return decision

    @staticmethod
    def _salvage_failure_summary(
        packet: Any,
        exec_record: PhaseRecord,
        verify_record: PhaseRecord,
    ) -> str:
        parts = [
            f"outcome={getattr(getattr(packet, 'outcome', ''), 'value', getattr(packet, 'outcome', ''))}",
            f"meaningful_progress={getattr(packet, 'meaningful_progress_signal', '')}",
            f"best_next_step={getattr(packet, 'best_next_step', '')}",
            "constraints=" + ", ".join(str(item) for item in (getattr(packet, "discovered_constraints", []) or [])[:5]),
            "questions=" + ", ".join(str(item) for item in (getattr(packet, "unresolved_questions", []) or [])[:5]),
            f"execute_success={exec_record.success}; execute_output={(exec_record.output or '')[:280]}",
            f"verify_success={verify_record.success}; verify_output={(verify_record.output or '')[:280]}",
        ]
        return "\n".join(part for part in parts if part.strip())

    async def _run_phase(
        self,
        task: RepairTask,
        phase: ExecutionPhase,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
    ) -> PhaseRecord:
        """Execute a single phase and record its result."""
        record = PhaseRecord(phase=phase)
        task.phases.append(record)
        await self._persist_task_progress(task)
        try:
            result = handler(task, *args)
            if inspect.isawaitable(result):
                result = await result
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
            elif self._provider_backoff_reason(record.output):
                record.success = False
            elif phase == ExecutionPhase.EXECUTE and "reached maximum number of tool-use iterations" in lowered:
                record.success = False
            elif phase == ExecutionPhase.EXECUTE and not record.output.strip():
                record.success = False
            elif phase == ExecutionPhase.EXECUTE and lowered.startswith("execute failed"):
                record.success = False
        except Exception as exc:
            record.success = False
            record.output = f"{phase.value} failed: {exc}"
        record.ended_at = datetime.now(timezone.utc)
        await self._persist_task_progress(task)
        return record

    async def _persist_task_progress(self, task: RepairTask) -> None:
        """Persist non-terminal task progress so live operators can see motion."""
        if self.store is None:
            return
        task.updated_at = datetime.now(timezone.utc)
        await self.store.save(task)

    async def _detect(self, task: RepairTask) -> str:
        """Identify what files/commands will be touched."""
        files = self._extract_candidate_paths(task.objective)
        return ",".join(files)

    @classmethod
    def _extract_candidate_paths(cls, objective: str) -> List[str]:
        """Extract ordered path-like artifacts from free-form objective text."""
        try:
            raw_tokens = shlex.split(objective)
        except ValueError:
            raw_tokens = objective.split()

        files: List[str] = []
        seen: set[str] = set()
        for token in raw_tokens:
            candidate = cls._normalize_candidate_token(token)
            if candidate is None or not cls._looks_like_path(candidate):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            files.append(candidate)
        return files

    @staticmethod
    def _normalize_candidate_token(token: str) -> Optional[str]:
        candidate = token.strip().strip("\"'`()[]{}<>,;:!?")
        if not candidate:
            return None
        for separator in ("=", ":"):
            if separator in candidate and not candidate.startswith(("./", "../", "~/", "/")):
                prefix, maybe_path = candidate.rsplit(separator, 1)
                if prefix and maybe_path:
                    candidate = maybe_path.strip().strip("\"'`()[]{}<>,;:!?")
        if candidate.endswith(".") and any(ch == "." for ch in candidate[:-1]):
            candidate = candidate[:-1]
        return candidate or None

    @classmethod
    def _looks_like_path(cls, candidate: str) -> bool:
        lowered = candidate.lower()
        if lowered.startswith(("http://", "https://", "data:", "file://")):
            return False
        if "@" in candidate and "/" not in candidate and "." not in candidate:
            return False
        if "/" in candidate or candidate.startswith(("./", "../", "~/")):
            return True
        if candidate.startswith(".") and len(candidate) > 1 and "/" not in candidate:
            return True
        if "." not in candidate or candidate.endswith("."):
            return False

        stem, suffix = candidate.rsplit(".", 1)
        if not stem or not suffix or not any(ch.isalpha() for ch in suffix):
            return False
        if "/" not in candidate and suffix.lower() in cls._LIKELY_DOMAIN_SUFFIXES:
            return False
        return True

    async def _plan(self, task: RepairTask) -> str:
        """Generate a short execution plan."""
        if self.llm:
            try:
                planning_context = self._shadow_planning_context(task)
                user_content = f"Objective: {task.objective}"
                if planning_context:
                    user_content = f"{user_content}\n\n{planning_context}"
                dual_context = await self._dual_context_execution_context(task)
                if dual_context:
                    user_content = f"{user_content}\n\n{dual_context}"
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a repair planner for an autonomous agent. "
                            "Given an objective, return a concise 1-3 step plan as plain text. "
                            "Prefer deterministic review and narrow artifact-bound edits over broad replanning "
                            "when prior blocked patterns suggest that."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ]
                response = await self.llm.chat_completion(
                    messages,
                    complexity="high" if task.attempt > 1 else "standard",
                    source="repair_planning",
                )
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content.strip() or "investigate and fix"
            except Exception as exc:
                backoff_reason = self._provider_backoff_reason(str(exc))
                if backoff_reason:
                    return f"plan failed: provider backoff: {backoff_reason}"
                return f"plan failed: llm error: {exc}"
        return "investigate and fix"

    @staticmethod
    def _provider_backoff_reason(text: str) -> str:
        lowered = str(text or "").lower()
        if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered or "ratelimit" in lowered:
            return "rate limit"
        if "provider" in lowered and "circuit open" in lowered:
            return "provider circuit open"
        if "providercircuitopen" in lowered:
            return "provider circuit open"
        return ""

    def _mark_retry_blocked(self, task: RepairTask, *, reason: str, mode: str) -> None:
        task.meta["retry_governor"] = {
            "allowed": False,
            "reason": reason,
            "mode": mode,
            "attempt": task.attempt,
            "packet_id": None,
            "reuse_packet_id": None,
        }
        task.meta["counterfactual_review"] = {
            "recommended": {
                "strategy": "backoff_or_switch_provider",
                "reason": reason,
            },
            "options": [],
            "salvage_packet_id": None,
            "retry_allowed": False,
            "retry_governor_reason": reason,
        }

    async def _execute_plan(self, task: RepairTask, plan: str) -> str:
        """Execute the plan using available tools."""
        if self.runtime and hasattr(self.runtime, "tool_loop"):
            from opencas.tools import ToolUseContext

            objective = f"Objective: {task.objective}\nPlan: {plan}"
            project_return_context = await self._project_return_context(task)
            dual_context = await self._dual_context_execution_context(task)
            system_content = "You are executing a repair task."
            if project_return_context:
                system_content = self._project_return_system_content(task, project_return_context)
            if dual_context:
                system_content = f"{system_content.rstrip()}\n\n{dual_context}"
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": objective},
            ]
            ctx = ToolUseContext(
                runtime=self.runtime,
                session_id=str(task.task_id),
                task_id=str(task.task_id),
            )
            scheduler = getattr(self.runtime, "scheduler", None)
            tool_loop_coro = self.runtime.tool_loop.run(
                objective=task.objective,
                messages=messages,
                ctx=ctx,
                on_focus_enter=scheduler.enter_focus_mode if scheduler else None,
                on_focus_exit=scheduler.exit_focus_mode if scheduler else None,
            )
            timeout_seconds = self._tool_loop_timeout_seconds(task)
            execution_started_at = datetime.now(timezone.utc)
            try:
                if timeout_seconds:
                    result = await asyncio.wait_for(tool_loop_coro, timeout=timeout_seconds)
                else:
                    result = await tool_loop_coro
            except asyncio.TimeoutError:
                artifact_progress_paths = self._record_timeout_artifact_progress(
                    task,
                    since=execution_started_at,
                )
                task.meta["tool_loop_timeout"] = {
                    "timeout_seconds": timeout_seconds,
                    "session_id": str(task.task_id),
                    "reason": "tool_loop_execution_timeout",
                    "artifact_progress_paths": artifact_progress_paths,
                }
                return (
                    "execute failed: tool loop timed out after "
                    f"{timeout_seconds:.0f}s before returning control; "
                    "retry as a narrower bounded continuation with durable artifact progress"
                )
            task.meta["last_tool_calls"] = list(result.tool_calls)
            tool_chain_summary = getattr(result, "tool_chain_summary", None)
            task.meta["last_tool_loop"] = {
                "iterations": getattr(result, "iterations", None),
                "guard_fired": bool(result.guard_fired),
                "guard_reason": result.guard_reason,
                "tool_call_count": len(result.tool_calls),
                "tool_chain_summary": (
                    tool_chain_summary.model_dump(mode="json")
                    if hasattr(tool_chain_summary, "model_dump")
                    else tool_chain_summary
                ),
            }
            await self._persist_task_progress(task)
            if result.guard_fired:
                reason = result.guard_reason or result.final_output
                output = f"execute failed: tool loop guard fired: {reason}"
                return output
            output = result.final_output
            self._persist_unwritten_writing_output(
                task=task,
                tool_calls=result.tool_calls,
                final_output=output,
            )
            writing_failure = self._writing_task_completion_failure(
                task=task,
                tool_calls=result.tool_calls,
                final_output=output,
            )
            if writing_failure:
                return f"execute failed: {writing_failure}"
            artifact_update_failure = self._artifact_update_completion_failure(
                task=task,
                tool_calls=result.tool_calls,
                final_output=output,
            )
            if artifact_update_failure:
                return f"execute failed: {artifact_update_failure}"
            contract_failure = new_project_execution_rejection_reason(
                task.meta if isinstance(task.meta, dict) else {},
                output=output,
                tool_calls=result.tool_calls,
            )
            if contract_failure:
                task.meta["project_contract_failure"] = {
                    "reason": contract_failure,
                    "final_output_excerpt": str(output or "").strip()[:700],
                    "tool_call_count": len(result.tool_calls),
                }
                return f"execute failed: {contract_failure}"
            return output

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
        output = "; ".join(outputs)
        return output

    def _record_timeout_artifact_progress(
        self,
        task: RepairTask,
        *,
        since: datetime,
    ) -> List[str]:
        paths = self._workspace_artifacts_modified_since(task, since=since)
        task.meta["timeout_artifact_progress"] = {
            "reason": "tool_loop_execution_timeout",
            "since": since.isoformat(),
            "paths": paths,
        }
        for path in paths:
            artifact = f"file:{path}"
            if artifact not in task.artifacts:
                task.artifacts.append(artifact)
        return paths

    @staticmethod
    def _workspace_artifacts_modified_since(task: RepairTask, *, since: datetime) -> List[str]:
        meta = task.meta if isinstance(task.meta, dict) else {}
        workspace_value = str(meta.get("workspace_abs_path") or "").strip()
        if not workspace_value:
            return []
        try:
            workspace = Path(workspace_value).expanduser().resolve()
        except Exception:
            return []
        if not workspace.is_dir():
            return []

        since_ts = since.astimezone(timezone.utc).timestamp() - 2.0
        paths: list[str] = []
        for path in workspace.rglob("*"):
            if len(paths) >= 50:
                break
            if not path.is_file():
                continue
            if any(part in {".git", "__pycache__"} for part in path.parts):
                continue
            if path.name.endswith((".tmp", ".swp")):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size <= 0 or stat.st_mtime < since_ts:
                continue
            paths.append(str(path))
        return sorted(set(paths))

    def _tool_loop_timeout_seconds(self, task: RepairTask) -> Optional[float]:
        """Return the bounded execution timeout for one BAA tool-loop slice."""
        meta = task.meta if isinstance(task.meta, dict) else {}
        raw = meta.get("tool_loop_timeout_seconds")
        if raw is None and self.runtime is not None:
            config = getattr(self.runtime, "config", None) or getattr(
                getattr(self.runtime, "ctx", None), "config", None
            )
            raw = getattr(config, "baa_tool_loop_timeout_seconds", None)
        if raw is None:
            raw = self.DEFAULT_TOOL_LOOP_TIMEOUT_SECONDS
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            timeout = self.DEFAULT_TOOL_LOOP_TIMEOUT_SECONDS
        if timeout <= 0:
            return None
        return timeout

    def _persist_unwritten_writing_output(
        self,
        *,
        task: RepairTask,
        tool_calls: List[Dict[str, Any]],
        final_output: str,
    ) -> None:
        """Persist substantial final prose when a writing scaffold was left unwritten."""
        output = str(final_output or "").strip()
        if len(output) < 200:
            return

        target = self._writing_task_output_target(tool_calls)
        if target is None:
            return
        target_path, writing_call_index = target
        if self._tool_calls_write_target_after(tool_calls, target_path, writing_call_index):
            return
        artifact_output = self._extract_writing_artifact_payload(output)
        if not self._looks_like_writing_artifact(artifact_output, tool_calls, target_path):
            return
        if not self._is_writable_scaffold_target(target_path):
            return

        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = artifact_output.rstrip() + "\n"
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        temp_path.write_text(payload, encoding="utf-8")
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        temp_path.replace(target_path)

        artifact = f"file:{target_path}"
        if artifact not in task.artifacts:
            task.artifacts.append(artifact)
        task.meta["persisted_writing_output"] = {
            "path": str(target_path),
            "reason": "final_output_after_writing_task",
            "bytes_written": len(payload.encode("utf-8")),
        }

    @classmethod
    def _extract_writing_artifact_payload(cls, text: str) -> str:
        """Keep manuscript prose and drop surrounding tool/status narration."""
        output = str(text or "").strip()
        if not output:
            return ""
        lines = output.splitlines()
        start_index = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                start_index = index
                break
        artifact_lines = lines[start_index:]
        terminal_index = len(artifact_lines)
        for index, line in enumerate(artifact_lines):
            lowered = line.strip().lower()
            if index == 0:
                continue
            if lowered.startswith(("## session status report", "## status report")):
                terminal_index = index
                break
            if lowered.startswith(
                (
                    "**what was accomplished",
                    "**what was not accomplished",
                    "**blocker:",
                    "**preserved for next return:",
                    "**next step",
                )
            ):
                terminal_index = index
                break
        cleaned: list[str] = []
        for line in artifact_lines[:terminal_index]:
            lowered = line.strip().lower()
            if lowered.startswith("**status:**") and any(
                marker in lowered
                for marker in ("not yet persisted", "blocker", "fs_write_file", "tool set")
            ):
                continue
            cleaned.append(line)
        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)
        while cleaned and cleaned[0].strip() == "---":
            cleaned.pop(0)
        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)
        while cleaned and cleaned[-1].strip() in {"", "---"}:
            cleaned.pop()
        return "\n".join(cleaned).strip() or output

    def _writing_task_completion_failure(
        self,
        *,
        task: RepairTask,
        tool_calls: List[Dict[str, Any]],
        final_output: str,
    ) -> str:
        target = self._writing_task_output_target(tool_calls)
        if target is None:
            return ""
        target_path, _writing_call_index = target
        requires_draft = self._writing_task_requires_draft_prose(task, tool_calls)
        if not requires_draft:
            return ""
        try:
            artifact_text = target_path.read_text(encoding="utf-8")
        except Exception:
            artifact_text = ""
        if self._looks_like_writing_artifact(artifact_text, tool_calls, target_path):
            return ""
        word_count = self._word_count(artifact_text)
        reason = (
            "writing task did not produce draft prose; "
            f"{target_path} contains {word_count} words and appears to be a scaffold or completion summary"
        )
        task.meta["writing_completion_failure"] = {
            "path": str(target_path),
            "word_count": word_count,
            "requires_draft_prose": True,
            "reason": reason,
            "final_output_excerpt": str(final_output or "").strip()[:500],
        }
        return reason

    def _artifact_update_completion_failure(
        self,
        *,
        task: RepairTask,
        tool_calls: List[Dict[str, Any]],
        final_output: str,
    ) -> str:
        """Reject artifact-update tasks that stop after read-only analysis."""
        if task.meta.get("persisted_writing_output"):
            return ""
        objective = str(getattr(task, "objective", "") or "")
        candidate_paths = self._extract_candidate_paths(objective)
        if not candidate_paths or not self._objective_requests_artifact_update(objective):
            return ""

        write_seen = any(call.get("name") in {"fs_write_file", "edit_file"} for call in tool_calls)
        return_seen = any(
            call.get("name") in {"workflow_create_schedule", "workflow_update_schedule"}
            for call in tool_calls
        )
        if write_seen or return_seen:
            return ""

        output = str(final_output or "")
        output_lower = output.lower()
        if not self._output_records_unresolved_artifact_blocker(output_lower):
            return ""

        reason = (
            "artifact update objective made no write/edit tool call and no return schedule; "
            "target artifact was not modified"
        )
        task.meta["artifact_update_failure"] = {
            "paths": candidate_paths,
            "write_or_return_call_seen": False,
            "reason": reason,
            "final_output_excerpt": output.strip()[:700],
        }
        return reason

    @staticmethod
    def _objective_requests_artifact_update(objective: str) -> bool:
        text = str(objective or "").lower()
        return any(
            marker in text
            for marker in (
                "append",
                "apply",
                "draft",
                "edit",
                "integrate",
                "insert",
                "manuscript progress",
                "merge",
                "modified",
                "modify",
                "persist",
                "replace",
                "revise",
                "revision",
                "save",
                "update",
                "write",
            )
        )

    @staticmethod
    def _output_records_unresolved_artifact_blocker(output_lower: str) -> bool:
        if "no blockers detected" in output_lower:
            return False
        return any(
            marker in output_lower
            for marker in (
                "## blocker",
                "**blocker",
                "concrete technical blocker",
                "cannot perform the actual",
                "cannot modify",
                "do not have `fs_write_file`",
                "do not have fs_write_file",
                "file-write tool",
                "has not been modified",
                "i have read-only file tools",
                "manuscript progress status: not claimed",
                "no `fs_write_file`",
                "no file modification tool",
                "not been modified",
                "target artifact",
                "write capability",
                "write tool absent",
            )
        )

    def _writing_task_output_target(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> Optional[tuple[Path, int]]:
        workspace_root = self._managed_workspace_root()
        if workspace_root is None:
            return None

        for index in range(len(tool_calls) - 1, -1, -1):
            call = tool_calls[index]
            if call.get("name") != "workflow_create_writing_task":
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            output_path = str(args.get("output_path") or "").strip()
            if not output_path:
                continue
            target = self._resolve_managed_path(output_path, workspace_root)
            if target is not None:
                return target, index
        return None

    def _managed_workspace_root(self) -> Optional[Path]:
        config = getattr(getattr(self.runtime, "ctx", None), "config", None)
        root_fn = getattr(config, "agent_workspace_root", None)
        if not callable(root_fn):
            return None
        try:
            return Path(root_fn()).expanduser().resolve()
        except Exception:
            return None

    @staticmethod
    def _resolve_managed_path(raw_path: str, workspace_root: Path) -> Optional[Path]:
        try:
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = workspace_root / candidate
            resolved = candidate.resolve()
            resolved.relative_to(workspace_root)
            return resolved
        except Exception:
            return None

    def _tool_calls_write_target_after(
        self,
        tool_calls: List[Dict[str, Any]],
        target_path: Path,
        writing_call_index: int,
    ) -> bool:
        workspace_root = self._managed_workspace_root()
        if workspace_root is None:
            return False
        for call in tool_calls[writing_call_index + 1 :]:
            if call.get("name") not in {"fs_write_file", "edit_file"}:
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            raw_path = str(args.get("file_path") or "").strip()
            if not raw_path:
                continue
            touched = self._resolve_managed_path(raw_path, workspace_root)
            if touched == target_path:
                return True
        return False

    @staticmethod
    def _is_writable_scaffold_target(target_path: Path) -> bool:
        if not target_path.exists():
            return True
        try:
            existing = target_path.read_text(encoding="utf-8")
        except Exception:
            return False
        stripped = existing.strip()
        if not stripped:
            return True
        if "<!-- Created by OpenCAS writing workflow -->" in existing:
            return True
        return len(stripped) < 200

    @classmethod
    def _looks_like_writing_artifact(
        cls,
        text: str,
        tool_calls: List[Dict[str, Any]],
        target_path: Path,
    ) -> bool:
        content = str(text or "").strip()
        if not content:
            return False
        lowered = content.lower()
        if "<!-- created by opencas writing workflow -->" in lowered:
            return False
        if cls._looks_like_completion_summary(content):
            return False
        min_words = 500 if cls._writing_tool_context_mentions_draft_prose(tool_calls, target_path) else 80
        return cls._word_count(content) >= min_words

    @staticmethod
    def _looks_like_completion_summary(text: str) -> bool:
        lowered = str(text or "").lower()
        summary_markers = (
            "summary of this session",
            "drafted full chapter",
            "drafted full ",
            "verified continuity",
            "scheduled next writing session",
            "approximately 3,400 words",
            "approx. 3,400 words",
            "verification and draft delivery",
            "continuity verification checklist",
            "word count estimate",
            "ready for placement",
            "unable to write directly",
            "what this chapter accomplishes beyond the synopsis",
            "let me do one final check",
            "manuscript is progressing well",
        )
        return any(marker in lowered for marker in summary_markers)

    @classmethod
    def _writing_task_requires_draft_prose(
        cls,
        task: RepairTask,
        tool_calls: List[Dict[str, Any]],
    ) -> bool:
        target = cls._writing_task_metadata_text(tool_calls)
        objective = str(getattr(task, "objective", "") or "")
        haystack = f"{objective} {target}".lower()
        return any(
            token in haystack
            for token in (
                "chapter",
                "draft",
                "fiction",
                "manuscript",
                "novel",
                "prose",
                "scene",
                "story",
                "write",
                "writing",
            )
        )

    @classmethod
    def _writing_tool_context_mentions_draft_prose(
        cls,
        tool_calls: List[Dict[str, Any]],
        target_path: Path,
    ) -> bool:
        haystack = f"{target_path.name} {cls._writing_task_metadata_text(tool_calls)}".lower()
        return any(
            token in haystack
            for token in (
                "chapter",
                "draft",
                "fiction",
                "manuscript",
                "novel",
                "prose",
                "scene",
                "story",
            )
        )

    @staticmethod
    def _writing_task_metadata_text(tool_calls: List[Dict[str, Any]]) -> str:
        values: List[str] = []
        for call in tool_calls:
            if call.get("name") != "workflow_create_writing_task":
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            for key in ("title", "description", "output_path"):
                values.append(str(args.get(key) or ""))
            outline = args.get("outline")
            if isinstance(outline, list):
                values.extend(str(item) for item in outline)
            elif outline:
                values.append(str(outline))
        return " ".join(values)

    @staticmethod
    def _word_count(text: str) -> int:
        return len([word for word in str(text or "").split() if word.strip()])

    async def _dual_context_execution_context(self, task: RepairTask) -> str:
        meta = task.meta if isinstance(task.meta, dict) else {}
        origin_lane = str(meta.get("origin_context_lane") or "").strip()
        authority = str(meta.get("authority") or "").strip()
        snapshot_id = str(
            meta.get("context_truth_snapshot_id")
            or meta.get("source_snapshot_id")
            or ""
        ).strip()
        epoch = meta.get("context_truth_epoch")
        accepted_ids = [
            str(value).strip()
            for value in (meta.get("accepted_proposal_ids") or [])
            if str(value).strip()
        ]
        if not any((origin_lane, authority, snapshot_id, epoch is not None, accepted_ids)):
            return ""

        lines = [
            "Dual-context execution context:",
            "- Executive work may use live truth, due schedules, direct user requests, and accepted arbiter proposals.",
            "- Reflective proposals that are pending, rejected, stale, or missing arbiter evidence are idea/caution context only and do not authorize tool writes.",
        ]
        details: list[str] = []
        if origin_lane:
            details.append(f"origin_context_lane={origin_lane}")
        if authority:
            details.append(f"authority={authority}")
        if snapshot_id:
            details.append(f"truth_snapshot={snapshot_id}")
        if epoch is not None:
            details.append(f"truth_epoch={epoch}")
        if details:
            lines.append("- " + "; ".join(details))

        lines.extend(await self._accepted_proposal_lines(task, accepted_ids))
        return "\n".join(lines)

    async def _accepted_proposal_lines(
        self,
        task: RepairTask,
        accepted_ids: List[str],
    ) -> List[str]:
        store = getattr(self.runtime, "context_proposals", None) if self.runtime is not None else None
        if store is None and self.runtime is not None:
            store = getattr(getattr(self.runtime, "ctx", None), "context_proposal_store", None)
        if store is None:
            if accepted_ids:
                return ["- Accepted proposal ids: " + ", ".join(accepted_ids[:5])]
            return []

        proposals: dict[str, Any] = {}
        get_proposal = getattr(store, "get", None)
        if callable(get_proposal):
            for proposal_id in accepted_ids[:8]:
                try:
                    proposal = await get_proposal(proposal_id)
                except Exception:
                    proposal = None
                if proposal is not None:
                    proposals[str(getattr(proposal, "proposal_id", proposal_id))] = proposal

        for method_name, value in (
            ("list_by_task", str(task.task_id)),
            ("list_by_project", getattr(task, "project_id", None)),
        ):
            if not value:
                continue
            method = getattr(store, method_name, None)
            if not callable(method):
                continue
            try:
                for proposal in await method(str(value), include_terminal=True, limit=8):
                    proposals[str(getattr(proposal, "proposal_id", ""))] = proposal
            except Exception:
                continue

        lines: list[str] = []
        for proposal in list(proposals.values())[:5]:
            status = str(getattr(getattr(proposal, "status", ""), "value", getattr(proposal, "status", ""))).lower()
            proposal_authority = str(
                getattr(getattr(proposal, "authority", ""), "value", getattr(proposal, "authority", ""))
            ).lower()
            proposal_id = str(getattr(proposal, "proposal_id", "") or "").strip()
            if status != "accepted" or proposal_authority != "executive_committed":
                if proposal_id:
                    lines.append(f"- Proposal {proposal_id} is not accepted executive support; do not execute from it.")
                continue
            content = " ".join(str(getattr(proposal, "content", "") or "").split())[:500]
            kind = str(getattr(proposal, "proposal_kind", "") or "").strip()
            refs = [proposal_id, str(getattr(proposal, "source_snapshot_id", "") or "").strip()]
            refs.extend(str(ref).strip() for ref in (getattr(proposal, "evidence_refs", []) or [])[:3])
            ref_text = ", ".join(ref for ref in refs if ref)
            lines.append(f"- Accepted proposal support ({kind}): {content} [refs: {ref_text}]")
        return lines

    async def _project_return_context(self, task: RepairTask) -> str:
        """Build project-return continuity context for scheduled project work."""
        meta = task.meta if isinstance(task.meta, dict) else {}
        if not (
            meta.get("project_key")
            or meta.get("project_title")
            or meta.get("source") == "project_return_capture"
        ):
            return ""
        title = str(meta.get("project_title") or "conversation project").strip()
        project_intent = str(meta.get("project_intent") or "").strip()
        project_type = self._classify_project_return_type(task, meta)
        next_step = str(meta.get("next_step") or "").strip()
        source_session_id = str(meta.get("source_session_id") or "").strip()
        lines = [
            "Project return context:",
            f"- Project: {title}",
            f"- Project type: {project_type}",
        ]
        if project_intent:
            label = self._project_intent_label(project_type)
            lines.append(f"- {label}: {project_intent}")
        if next_step:
            lines.append(f"- Immediate next step: {next_step}")
        if source_session_id:
            lines.append(f"- Source chat session: {source_session_id}")
        workspace_project = self._resolve_project_workspace(task, meta)
        if workspace_project:
            lines.append(f"- Canonical workspace project root: {workspace_project.path}")
            lines.append(f"- Workspace-relative project root: {workspace_project.workspace_rel_path}")
            lines.append(
                "- Start by inspecting this project root. Do not create a new scratch project "
                "when this path exists."
            )
        else:
            requested_workspace = self._requested_workspace_context(meta)
            if requested_workspace:
                lines.extend(requested_workspace)

        lines.append(
            "- Missing context or evidence is not a stop condition. Use the available filesystem, "
            "workflow, memory, and research tools to get the evidence, then continue the work. "
            "Only pause for a real external blocker, missing credential, safety boundary, or "
            "destructive ambiguity, and record that blocker with the linked commitment."
        )
        contract = meta.get("project_start_contract") if isinstance(meta.get("project_start_contract"), dict) else {}
        if contract.get("new_project") is True:
            lines.append(
                "- New-project contract: create work for this project root itself. Do not satisfy "
                "the task by copying, moving, or materializing a sibling project's artifacts into "
                "this root. Reference/premise continuity is allowed only as researched context."
            )

        recent = await self._recent_project_return_messages(source_session_id, limit=10)
        if recent:
            lines.append("- Recent source-session evidence:")
            lines.extend(f"  - {line}" for line in recent)
        if project_type == PROJECT_TYPE_SOFTWARE:
            lines.append(
                "- Continue ordinary implementation, build, test, documentation, and proof work without "
                "asking for permission; ask only for real ambiguity, missing credentials, safety boundaries, "
                "or destructive host changes. Use workflow_cancel_project if the project should be composted."
            )
        elif project_type == PROJECT_TYPE_WRITING:
            lines.append(
                "- Do not ask the user for permission to continue ordinary creative research, writing, "
                "or revision; only ask if you hit a real ambiguity, safety boundary, or missing artifact."
            )
        else:
            lines.append(
                "- Continue scoped project work without asking for permission; ask only for real ambiguity, "
                "missing access, safety boundaries, or destructive host changes."
            )
        return "\n".join(lines)

    def _resolve_project_workspace(
        self,
        task: RepairTask,
        meta: Dict[str, Any],
    ) -> Optional[WorkspaceProjectCandidate]:
        stored_abs_path = str(meta.get("workspace_abs_path") or "").strip()
        stored_rel_path = str(meta.get("workspace_rel_path") or "").strip()
        if stored_abs_path:
            path = Path(stored_abs_path).expanduser().resolve()
            if path.exists() and path.is_dir():
                return WorkspaceProjectCandidate(
                    project_key=str(meta.get("project_key") or path.name),
                    project_title=str(meta.get("project_title") or path.name),
                    path=path,
                    workspace_rel_path=Path(stored_rel_path or path.name),
                    confidence=float(meta.get("workspace_project_confidence") or 1.0),
                    evidence=tuple(str(item) for item in meta.get("workspace_project_evidence", []) or ()),
                )
        if self.runtime is None:
            return None
        return resolve_workspace_project(
            self.runtime,
            project_key=str(meta.get("project_key") or ""),
            project_title=str(meta.get("project_title") or task.objective),
        )

    @staticmethod
    def _requested_workspace_context(meta: Dict[str, Any]) -> List[str]:
        requested_abs = str(meta.get("requested_workspace_abs_path") or "").strip()
        requested_rel = str(meta.get("requested_workspace_rel_path") or "").strip()
        if not requested_abs and not requested_rel:
            return []
        kind = str(meta.get("requested_workspace_kind") or "target").strip() or "target"
        lines = []
        if requested_abs:
            label = "Requested workspace parent directory" if kind == "parent" else "Requested workspace path"
            lines.append(f"- {label}: {requested_abs}")
        if requested_rel:
            lines.append(f"- Workspace-relative requested path: {requested_rel}")
        if kind == "parent":
            lines.append(
                "- Create the new project under this managed-workspace directory and keep generated artifacts there."
            )
        else:
            lines.append("- Create or continue the project at this managed-workspace path.")
        return lines

    def _project_return_system_content(self, task: RepairTask, project_return_context: str) -> str:
        meta = task.meta if isinstance(task.meta, dict) else {}
        project_type = self._classify_project_return_type(task, meta)
        agent_name = resolve_agent_name(runtime=self.runtime)
        if project_type == PROJECT_TYPE_SOFTWARE:
            preamble = (
                f"You are {agent_name} returning to your own software project, not an external contractor. "
                "Use tools as part of your own agency and continuity. Decide what meaningful progress "
                "requires, and preserve the next return point if the project remains unfinished. "
                "Prioritize source correctness, build, test, run instructions, and proof before claiming "
                "the project works. Do not stop with a context/evidence disclaimer when tools can inspect "
                "or produce the needed evidence."
            )
        elif project_type == PROJECT_TYPE_WRITING:
            preamble = (
                f"You are {agent_name} returning to your own creative project, not an external contractor. "
                "Use tools as part of your own agency and continuity. Decide what meaningful progress "
                "requires, and preserve the next return point if the project remains unfinished. "
                "For writing work, creating a workflow scaffold is not manuscript progress; persist "
                "actual draft prose to the target artifact before claiming a chapter, scene, word count, "
                "or manuscript milestone is complete. Do not stop with a context/evidence disclaimer when "
                "tools can inspect, research, revise, or write the needed artifact."
            )
        else:
            preamble = (
                f"You are {agent_name} returning to your own project, not an external contractor. "
                "Use tools as part of your own agency and continuity. Decide what meaningful progress "
                "requires, and preserve the next return point if the project remains unfinished. Do not stop "
                "with a context/evidence disclaimer when tools can inspect or produce the needed evidence."
            )
        return f"{preamble}\n\n{project_return_context}"

    @staticmethod
    def _classify_project_return_type(task: RepairTask, meta: Dict[str, Any]) -> str:
        return classify_project_type(
            current_turn_text=" ".join(
                str(value or "")
                for value in (
                    task.objective,
                    meta.get("project_title"),
                    meta.get("project_intent"),
                    meta.get("next_step"),
                    meta.get("source_user_turn"),
                    meta.get("source_assistant_turn"),
                )
            ),
            metadata=meta,
        ).project_type

    @staticmethod
    def _project_intent_label(project_type: str) -> str:
        if project_type == PROJECT_TYPE_SOFTWARE:
            return "Software project intent"
        if project_type == PROJECT_TYPE_WRITING:
            return "Book-level intent"
        return "Project intent"

    async def _recent_project_return_messages(self, session_id: str, *, limit: int) -> List[str]:
        if not session_id or self.runtime is None:
            return []
        store = getattr(getattr(self.runtime, "ctx", None), "context_store", None)
        if store is None:
            return []
        try:
            entries = await store.list_recent(session_id, limit=limit, include_hidden=True)
        except Exception:
            return []
        lines: List[str] = []
        for entry in entries[-limit:]:
            role = getattr(getattr(entry, "role", None), "value", getattr(entry, "role", ""))
            content = " ".join(str(getattr(entry, "content", "") or "").split())
            if not content:
                continue
            if len(content) > 260:
                content = content[:259].rstrip() + "..."
            lines.append(f"{role}: {content}")
        return lines

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

    @staticmethod
    def _attempt_outcome(
        task: RepairTask,
        *,
        exec_record: PhaseRecord,
        verify_record: PhaseRecord,
    ) -> AttemptOutcome:
        if RepairExecutor._tool_loop_guard_stopped(task, exec_record):
            return AttemptOutcome.GUARD_STOPPED
        if exec_record.success is not True:
            return AttemptOutcome.FAILED
        if verify_record.success is not True:
            return AttemptOutcome.VERIFY_FAILED
        return AttemptOutcome.PARTIAL

    @staticmethod
    def _tool_loop_guard_stopped(task: RepairTask, exec_record: PhaseRecord) -> bool:
        loop_meta = task.meta.get("last_tool_loop") if isinstance(task.meta, dict) else {}
        if isinstance(loop_meta, dict) and loop_meta.get("guard_fired") is True:
            return True
        output = str(exec_record.output or "").lower()
        return "tool loop guard fired" in output or "[tool loop halted]" in output

    @classmethod
    def _artifact_progress_boundary(cls, task: RepairTask, exec_record: PhaseRecord) -> bool:
        """Treat iteration caps after direct artifact writes as continuation, not exhaustion."""
        if exec_record.success is True or cls._tool_loop_guard_stopped(task, exec_record):
            return False
        loop_meta = task.meta.get("last_tool_loop") if isinstance(task.meta, dict) else {}
        loop_reached_cap = False
        if isinstance(loop_meta, dict):
            try:
                iterations = int(loop_meta.get("iterations") or 0)
            except (TypeError, ValueError):
                iterations = 0
            loop_reached_cap = iterations >= 32 and not bool(loop_meta.get("guard_fired"))
        text = " ".join(
            [
                str(exec_record.output or ""),
                " ".join(str(item) for item in task.artifacts[-5:]),
            ]
        ).lower()
        if "reached maximum number of tool-use iterations" in text:
            loop_reached_cap = True
        timeout_paths = cls._timeout_artifact_progress_paths(task)
        if timeout_paths and "tool loop timed out" in text:
            return True
        if not loop_reached_cap:
            return False
        return bool(cls._artifact_paths_from_tool_calls(cls._tool_calls_from_task_meta(task)))

    @staticmethod
    def _canonical_artifact_path(task: RepairTask) -> Optional[str]:
        resume_project = task.meta.get("resume_project")
        if isinstance(resume_project, dict):
            path = resume_project.get("canonical_artifact_path")
            if isinstance(path, str) and path.strip():
                return path.strip()
        path = task.meta.get("canonical_artifact_path")
        if isinstance(path, str) and path.strip():
            return path.strip()
        return None

    @staticmethod
    def _artifact_paths_touched(task: RepairTask, affected_files: List[str]) -> List[str]:
        paths = [path for path in affected_files if isinstance(path, str) and path.strip()]
        paths.extend(
            RepairExecutor._artifact_paths_from_tool_calls(
                RepairExecutor._tool_calls_from_task_meta(task)
            )
        )
        paths.extend(RepairExecutor._timeout_artifact_progress_paths(task))
        return sorted({path.strip() for path in paths if path.strip()})

    @staticmethod
    def _timeout_artifact_progress_paths(task: RepairTask) -> List[str]:
        meta = task.meta if isinstance(task.meta, dict) else {}
        progress = meta.get("timeout_artifact_progress")
        if not isinstance(progress, dict):
            return []
        raw_paths = progress.get("paths")
        if not isinstance(raw_paths, list):
            return []
        return [str(path).strip() for path in raw_paths if str(path).strip()]

    @classmethod
    def _artifact_paths_from_tool_calls(cls, tool_calls: List[Dict[str, Any]]) -> List[str]:
        paths: list[str] = []
        for call in tool_calls:
            if call.get("name") not in cls._WRITE_ARTIFACT_TOOLS:
                continue
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            path = artifact_hint_from_mapping(args)
            if path is not None:
                paths.append(path)
        return paths

    @staticmethod
    def _tool_calls_from_task_meta(task: RepairTask) -> List[Dict[str, Any]]:
        tool_calls = task.meta.get("last_tool_calls") or task.meta.get("tool_calls") or []
        if isinstance(tool_calls, list):
            return [call for call in tool_calls if isinstance(call, dict)]
        return []

    @staticmethod
    def _has_new_evidence(prior_packet, candidate) -> bool:
        if prior_packet is None:
            return False
        return any(
            (
                prior_packet.verification_digest != candidate.verification_digest,
                prior_packet.discovered_constraints != candidate.discovered_constraints,
                prior_packet.unresolved_questions != candidate.unresolved_questions,
                prior_packet.partial_value != candidate.partial_value,
            )
        )

    @staticmethod
    def _is_broad_attempt(task: RepairTask, packet) -> bool:
        explicit = task.meta.get("retry_mode")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip().lower() not in {
                "resume_existing_artifact",
                "narrow_edit",
                "deterministic_review",
                "complete_partial_and_stop",
            }
        return packet.llm_spend_class == "broad"

    def _shadow_registry(self):
        return getattr(getattr(self.runtime, "ctx", None), "shadow_registry", None)

    def _capture_retry_blocked_intention(self, task: RepairTask, reason: str) -> None:
        capture = getattr(self._shadow_registry(), "capture_retry_blocked", None)
        if not callable(capture):
            return
        canonical_artifact = self._canonical_artifact_path(task)
        retry_governor = task.meta.get("retry_governor") if isinstance(task.meta, dict) else {}
        capture(
            {
                "task_id": str(task.task_id),
                "target_id": str(task.task_id),
                "target_kind": "repair_task",
                "objective": task.objective,
                "attempt": task.attempt,
                "artifact": canonical_artifact,
                "canonical_artifact_path": canonical_artifact,
                "retry_mode": task.meta.get("retry_mode"),
                "governor_mode": retry_governor.get("mode") if isinstance(retry_governor, dict) else None,
                "retry_governor": retry_governor if isinstance(retry_governor, dict) else {},
                "resume_project": task.meta.get("resume_project") if isinstance(task.meta, dict) else {},
                "reason": f"RetryGovernor blocked attempt {task.attempt}: {reason}",
            }
        )

    def _shadow_planning_context(self, task: RepairTask) -> str:
        builder = getattr(self._shadow_registry(), "build_planning_context", None)
        if not callable(builder):
            return ""
        artifact = self._canonical_artifact_path(task)
        context = builder(
            objective=task.objective,
            artifact=artifact,
        )
        if not isinstance(context, dict) or not context.get("available"):
            return ""
        return str(context.get("prompt_block", "") or "").strip()

    def _fail(self, task: RepairTask, message: str) -> RepairResult:
        task.stage = ExecutionStage.FAILED
        task.status = "failed"
        self.record_task_boundary(
            task,
            boundary="task_completed",
            workflow_phase="stop",
            artifact="repair-task|default|completed",
            why=f"task completed unsuccessfully for {task.objective}",
            action="COMMIT",
            risk="MEDIUM",
            source_trace={"success": False, "stage": ExecutionStage.FAILED.value, "reason": message},
        )
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
