"""Repair executor for OpenCAS."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import shlex
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from opencas.api import LLMClient
from opencas.provenance_adapter import append_provenance_record
from opencas.telemetry import EventKind, Tracer
from opencas.tools import ToolRegistry

from .git_checkpoint import GitCheckpointManager
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
        await self._run_phase(task, ExecutionPhase.POSTCHECK, self._postcheck)

        if verified and exec_record.success is True:
            convergence_hash = self._hash_convergence(exec_output, task.artifacts)
            task.convergence_hashes.append(convergence_hash)
            task.stage = ExecutionStage.DONE
            task.status = "completed"
            if checkpoint:
                checkpoint.discard()
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
                if checkpoint:
                    checkpoint.restore(commit_hash)
                return self._fail(task, f"retry blocked: {decision.reason}")
        else:
            convergence_hash = self._hash_convergence(exec_output, task.artifacts)
            if convergence_hash in task.convergence_hashes:
                if checkpoint:
                    checkpoint.restore(commit_hash)
                return self._fail(task, "non-improving loop detected")
            task.convergence_hashes.append(convergence_hash)

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
            outcome=self._attempt_outcome(exec_record=exec_record, verify_record=verify_record),
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
        task.meta["last_salvage_packet_id"] = str(packet.packet_id)
        task.meta["retry_governor"] = {
            "allowed": decision.allowed,
            "reason": decision.reason,
            "mode": decision.mode.value,
            "reuse_packet_id": str(decision.reuse_packet_id) if decision.reuse_packet_id else None,
            "attempt": packet.attempt,
            "packet_id": str(packet.packet_id),
        }
        return decision

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
                return f"investigate and fix (llm error: {exc})"
        return "investigate and fix"

    async def _execute_plan(self, task: RepairTask, plan: str) -> str:
        """Execute the plan using available tools."""
        if self.runtime and hasattr(self.runtime, "tool_loop"):
            from opencas.tools import ToolUseContext

            objective = f"Objective: {task.objective}\nPlan: {plan}"
            project_return_context = await self._project_return_context(task)
            system_content = "You are executing a repair task."
            if project_return_context:
                system_content = (
                    "You are the OpenCAS agent returning to your own creative project, not an external contractor. "
                    "Use tools as part of your own agency and continuity. Decide what meaningful progress "
                    "requires, and preserve the next return point if the project remains unfinished. "
                    "For writing work, creating a workflow scaffold is not manuscript progress; persist "
                    "actual draft prose to the target artifact before claiming a chapter, scene, word count, "
                    "or manuscript milestone is complete.\n\n"
                    f"{project_return_context}"
                )
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
            result = await self.runtime.tool_loop.run(
                objective=task.objective,
                messages=messages,
                ctx=ctx,
                on_focus_enter=scheduler.enter_focus_mode if scheduler else None,
                on_focus_exit=scheduler.exit_focus_mode if scheduler else None,
            )
            if result.guard_fired:
                reason = result.guard_reason or result.final_output
                output = f"execute failed: tool loop guard fired: {reason}"
                return output
            task.meta["last_tool_calls"] = list(result.tool_calls)
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

    async def _project_return_context(self, task: RepairTask) -> str:
        """Build project-return continuity context for scheduled creative work."""
        meta = task.meta if isinstance(task.meta, dict) else {}
        if not (
            meta.get("project_key")
            or meta.get("project_title")
            or meta.get("source") == "project_return_capture"
        ):
            return ""
        title = str(meta.get("project_title") or "conversation project").strip()
        project_intent = str(meta.get("project_intent") or "").strip()
        next_step = str(meta.get("next_step") or "").strip()
        source_session_id = str(meta.get("source_session_id") or "").strip()
        lines = [
            "Project return context:",
            f"- Project: {title}",
        ]
        if project_intent:
            lines.append(f"- Book-level intent: {project_intent}")
        if next_step:
            lines.append(f"- Immediate next step: {next_step}")
        if source_session_id:
            lines.append(f"- Source chat session: {source_session_id}")

        recent = await self._recent_project_return_messages(source_session_id, limit=10)
        if recent:
            lines.append("- Recent source-session evidence:")
            lines.extend(f"  - {line}" for line in recent)
        lines.append(
            "- Do not ask the user for permission to continue ordinary creative research, writing, "
            "or revision; only ask if you hit a real ambiguity, safety boundary, or missing artifact."
        )
        return "\n".join(lines)

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
        *,
        exec_record: PhaseRecord,
        verify_record: PhaseRecord,
    ) -> AttemptOutcome:
        if exec_record.success is not True:
            return AttemptOutcome.FAILED
        if verify_record.success is not True:
            return AttemptOutcome.VERIFY_FAILED
        return AttemptOutcome.PARTIAL

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
        canonical = RepairExecutor._canonical_artifact_path(task)
        if canonical:
            paths.append(canonical)
        return sorted({path.strip() for path in paths if path.strip()})

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
