"""Agentic harness for long-horizon objectives and research notebooks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opencas.api import LLMClient
from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.project_orchestrator import ProjectOrchestrator
from opencas.autonomy.work_store import WorkStore
from opencas.execution import BoundedAssistantAgent, RepairTask
from opencas.telemetry import EventKind, Tracer

from .models import (
    DeliverableSchema,
    NotebookEntry,
    NotebookEntryKind,
    ObjectiveLoop,
    ObjectiveLoopContract,
    ObjectiveStatus,
    ResearchNotebook,
)
from .store import HarnessStore

_OBJECTIVE_CONTRACT_META_KEY = "objective_contract"
_REQUIRED_OBJECTIVE_CONTRACT_FIELDS = (
    "goal",
    "expected_output",
    "success_check",
    "stop_condition",
)


class AgenticHarness:
    """Manages research notebooks and objective loops, producing RepairTasks."""

    def __init__(
        self,
        store: HarnessStore,
        llm: Optional[LLMClient] = None,
        tracer: Optional[Tracer] = None,
        work_store: Optional[WorkStore] = None,
        baa: Optional[BoundedAssistantAgent] = None,
        project_orchestrator: Optional[ProjectOrchestrator] = None,
        project_resume_resolver: Optional[Any] = None,
        shadow_registry: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.tracer = tracer
        self.work_store = work_store
        self.baa = baa
        self.project_orchestrator = project_orchestrator
        self.project_resume_resolver = project_resume_resolver
        self.shadow_registry = shadow_registry

    async def create_notebook(
        self,
        title: str,
        description: str = "",
        deliverable_schema: Optional[DeliverableSchema] = None,
    ) -> ResearchNotebook:
        """Create a new research notebook."""
        notebook = ResearchNotebook(
            title=title,
            description=description,
            deliverable_schema=deliverable_schema,
        )
        await self.store.save_notebook(notebook)
        self._trace("notebook_created", {"notebook_id": str(notebook.notebook_id), "title": title})
        return notebook

    async def add_notebook_entry(
        self,
        notebook_id: str,
        kind: NotebookEntryKind,
        content: str,
        source_episode_ids: Optional[List[str]] = None,
        source_task_ids: Optional[List[str]] = None,
    ) -> Optional[NotebookEntry]:
        """Add an entry to a research notebook."""
        notebook = await self.store.get_notebook(notebook_id)
        if notebook is None:
            return None
        entry = NotebookEntry(
            kind=kind,
            content=content,
            source_episode_ids=source_episode_ids or [],
            source_task_ids=source_task_ids or [],
        )
        await self.store.add_entry(notebook_id, entry)
        notebook.updated_at = datetime.now(timezone.utc)
        await self.store.save_notebook(notebook)
        self._trace("notebook_entry_added", {"notebook_id": notebook_id, "entry_id": str(entry.entry_id)})
        return entry

    async def create_objective_loop(
        self,
        title: str,
        description: str = "",
        notebook_id: Optional[str] = None,
        completion_criteria: Optional[List[str]] = None,
        *,
        expected_output: Optional[str] = None,
        success_check: Optional[str] = None,
        stop_condition: Optional[str] = None,
        max_attempt_budget: int = 1,
        reframe_path: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> ObjectiveLoop:
        """Create a new objective loop, optionally attached to a notebook."""
        if self.project_resume_resolver is not None:
            existing = await self.project_resume_resolver.find_matching_active_loop(title)
            if existing is not None:
                self._trace(
                    "objective_loop_reused_existing",
                    {
                        "existing_loop_id": str(existing.loop_id),
                        "requested_title": title,
                    },
                )
                return existing
        loop_meta = dict(meta or {})
        if any(value is not None for value in (expected_output, success_check, stop_condition)) or reframe_path:
            loop_meta[_OBJECTIVE_CONTRACT_META_KEY] = ObjectiveLoopContract(
                goal=title,
                expected_output=str(expected_output or "").strip(),
                success_check=str(success_check or "").strip(),
                stop_condition=str(stop_condition or "").strip(),
                max_attempt_budget=max(1, int(max_attempt_budget or 1)),
                reframe_path=str(reframe_path or "").strip(),
            ).model_dump(mode="json")
        loop = ObjectiveLoop(
            title=title,
            description=description,
            notebook_id=notebook_id,
            completion_criteria=completion_criteria or [],
            meta=loop_meta,
        )
        await self.store.save_loop(loop)
        self._trace("objective_loop_created", {"loop_id": str(loop.loop_id), "title": title})
        return loop

    async def run_objective_cycle(self, max_active_loops: int = 3) -> Dict[str, Any]:
        """Run one cycle for active objective loops: plan, generate tasks, submit."""
        if self.project_resume_resolver is not None:
            await self.project_resume_resolver.suppress_duplicate_active_objective_loops()
        active_loops = await self.store.list_loops(status=ObjectiveStatus.ACTIVE, limit=max_active_loops)
        pending_loops = await self.store.list_loops(status=ObjectiveStatus.PENDING, limit=max_active_loops)

        # Promote pending to active if under capacity
        loops_to_process: List[ObjectiveLoop] = []
        parked_loops: List[str] = []
        for loop in active_loops:
            if await self._park_loop_if_missing_contract(loop):
                parked_loops.append(str(loop.loop_id))
                continue
            loops_to_process.append(loop)
        for loop in pending_loops:
            if len(loops_to_process) >= max_active_loops:
                break
            if await self._park_loop_if_missing_contract(loop):
                parked_loops.append(str(loop.loop_id))
                continue
            loop.status = ObjectiveStatus.ACTIVE
            await self.store.save_loop(loop)
            loops_to_process.append(loop)

        submitted_tasks: List[str] = []
        created_work_objects: List[str] = []

        for loop in loops_to_process:
            tasks, work_objects = await self._process_loop(loop)
            submitted_tasks.extend(tasks)
            created_work_objects.extend(work_objects)

        self._trace(
            "objective_cycle_completed",
            {
                "loops_processed": len(loops_to_process),
                "submitted_tasks": len(submitted_tasks),
                "created_work_objects": len(created_work_objects),
                "parked_loops": len(parked_loops),
            },
        )
        return {
            "loops_processed": len(loops_to_process),
            "submitted_tasks": submitted_tasks,
            "created_work_objects": created_work_objects,
            "parked_loops": parked_loops,
        }

    async def _process_loop(self, loop: ObjectiveLoop) -> tuple[List[str], List[str]]:
        """Plan next steps for a loop and emit tasks/work objects."""
        submitted_tasks: List[str] = []
        created_work_objects: List[str] = []
        if self.project_resume_resolver is not None:
            resume_snapshot = await self.project_resume_resolver.resolve(loop.title)
            if resume_snapshot is not None:
                loop.meta = dict(loop.meta or {})
                loop.meta["resume_project"] = resume_snapshot.to_meta()
                if resume_snapshot.retry_state == "blocked_low_divergence":
                    loop.updated_at = datetime.now(timezone.utc)
                    await self.store.save_loop(loop)
                    self._trace(
                        "objective_loop_switched_to_salvage_resume",
                        {
                            "loop_id": str(loop.loop_id),
                            "signature": resume_snapshot.signature,
                            "best_next_step": resume_snapshot.best_next_step,
                        },
                    )
                    return submitted_tasks, created_work_objects

                if resume_snapshot.has_live_workstream:
                    loop.updated_at = datetime.now(timezone.utc)
                    await self.store.save_loop(loop)
                    self._trace(
                        "objective_loop_resumed_existing_project",
                        {
                            "loop_id": str(loop.loop_id),
                            "signature": resume_snapshot.signature,
                            "primary_loop_id": resume_snapshot.primary_loop_id,
                            "active_work_count": resume_snapshot.active_work_count,
                        },
                    )
                    return submitted_tasks, created_work_objects

        notebook = None
        if loop.notebook_id:
            notebook = await self.store.get_notebook(loop.notebook_id)

        # Decide next action for the loop
        plan = await self._generate_loop_plan(loop, notebook)

        # Record plan as experiment entry
        if notebook:
            await self.add_notebook_entry(
                notebook_id=str(notebook.notebook_id),
                kind=NotebookEntryKind.EXPERIMENT,
                content=f"Cycle plan: {plan}",
            )

        # Create a WorkObject for the plan
        work = WorkObject(
            content=plan,
            stage=WorkStage.MICRO_TASK,
            project_id=str(loop.loop_id),
            meta=self._loop_work_meta(loop),
        )
        if self.work_store:
            await self.work_store.save(work)
            created_work_objects.append(str(work.work_id))

        # If plan looks like a multi-step project, decompose via orchestrator
        if self.project_orchestrator and len(plan.split()) > 8:
            project_work = WorkObject(
                content=plan,
                stage=WorkStage.PROJECT,
                project_id=str(loop.loop_id),
                meta=self._loop_work_meta(loop),
            )
            if self.work_store:
                await self.work_store.save(project_work)
                created_work_objects.append(str(project_work.work_id))
            await self.project_orchestrator.decompose(project_work)
            loop.generated_task_ids.append(str(project_work.work_id))
        elif self.baa:
            # Direct RepairTask for simpler plans
            repair_task = RepairTask(
                objective=plan,
                project_id=str(loop.loop_id),
                max_attempts=self._contract_max_attempts(loop),
                meta=self._loop_work_meta(loop),
            )
            await self.baa.submit(repair_task)
            submitted_tasks.append(str(repair_task.task_id))
            loop.generated_task_ids.append(str(repair_task.task_id))

        loop.updated_at = datetime.now(timezone.utc)
        await self.store.save_loop(loop)
        return submitted_tasks, created_work_objects

    async def _generate_loop_plan(
        self,
        loop: ObjectiveLoop,
        notebook: Optional[ResearchNotebook],
    ) -> str:
        """Use LLM or heuristics to decide the next step for a loop."""
        context_lines: List[str] = [
            f"Objective: {loop.title}",
            f"Description: {loop.description}",
        ]
        if notebook and notebook.entries:
            context_lines.append("Recent notebook entries:")
            for entry in notebook.entries[:5]:
                context_lines.append(f"- [{entry.kind.value}] {entry.content[:160]}")
        if loop.completion_criteria:
            context_lines.append(f"Completion criteria: {', '.join(loop.completion_criteria)}")
        contract = self._objective_contract_payload(loop)
        if contract:
            context_lines.extend(
                [
                    "Objective contract:",
                    f"- Goal: {contract['goal']}",
                    f"- Expected output: {contract['expected_output']}",
                    f"- Success check: {contract['success_check']}",
                    f"- Stop condition: {contract['stop_condition']}",
                    f"- Max attempt budget: {contract['max_attempt_budget']}",
                ]
            )
            reframe_path = str(contract.get("reframe_path", "") or "").strip()
            if reframe_path:
                context_lines.append(f"- Reframe path: {reframe_path}")

        prompt = (
            "You are an autonomous research assistant. Given the objective and notebook context, "
            "decide the single most valuable next step. Return a concise 1-2 sentence plan.\n\n"
            + "\n".join(context_lines)
        )
        shadow_context = self._shadow_planning_context(loop)
        if shadow_context:
            prompt += f"\n\n{shadow_context}"

        if self.llm:
            try:
                messages = [
                    {"role": "system", "content": "You are a planning assistant for an autonomous agent."},
                    {"role": "user", "content": prompt},
                ]
                response = await self.llm.chat_completion(
                    messages,
                    complexity="standard",
                    source="harness_planning",
                )
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                plan = content.strip()
                if plan:
                    return plan
            except Exception as exc:
                self._trace("loop_plan_llm_error", {"loop_id": str(loop.loop_id), "error": str(exc)})

        # Fallback heuristic
        return f"Investigate and make progress on: {loop.title}"

    def _shadow_planning_context(self, loop: ObjectiveLoop) -> str:
        builder = getattr(self.shadow_registry, "build_planning_context", None)
        if not callable(builder):
            return ""
        context = builder(
            objective=loop.title,
            artifact=self._artifact_hint_from_loop(loop),
        )
        if not isinstance(context, dict) or not context.get("available"):
            return ""
        return str(context.get("prompt_block", "") or "").strip()

    async def _park_loop_if_missing_contract(self, loop: ObjectiveLoop) -> bool:
        contract, missing_fields = self._validated_objective_contract(loop)
        if contract is not None:
            loop.meta = dict(loop.meta or {})
            loop.meta[_OBJECTIVE_CONTRACT_META_KEY] = contract
            return False

        drafted_contract = await self._draft_missing_objective_contract(loop)
        if drafted_contract is not None:
            loop.meta = dict(loop.meta or {})
            loop.meta[_OBJECTIVE_CONTRACT_META_KEY] = drafted_contract
            loop.meta["objective_contract_status"] = "drafted_by_agent"
            loop.updated_at = datetime.now(timezone.utc)
            await self.store.save_loop(loop)
            self._trace(
                "objective_loop_contract_drafted",
                {"loop_id": str(loop.loop_id)},
            )
            return False

        loop.status = ObjectiveStatus.PAUSED
        loop.updated_at = datetime.now(timezone.utc)
        loop.meta = dict(loop.meta or {})
        loop.meta.update(
            {
                "paused_reason": "missing_objective_contract",
                "objective_contract_status": "missing",
                "missing_contract_fields": missing_fields,
                "reframe_hint": (
                    "Reframe this objective with an objective_contract containing "
                    "goal, expected_output, success_check, and stop_condition before resuming."
                ),
            }
        )
        await self.store.save_loop(loop)
        self._trace(
            "objective_loop_parked_missing_contract",
            {"loop_id": str(loop.loop_id), "missing_contract_fields": missing_fields},
        )
        return True

    async def _draft_missing_objective_contract(
        self,
        loop: ObjectiveLoop,
    ) -> Optional[Dict[str, Any]]:
        """Ask the agent's model to draft a task-specific objective contract."""
        if self.llm is None:
            return None

        context_lines = [
            f"Objective title: {loop.title}",
            f"Description: {loop.description or '(none)'}",
        ]
        if loop.completion_criteria:
            context_lines.append(
                "Completion criteria already known: " + "; ".join(loop.completion_criteria)
            )
        prompt = (
            "Draft your own outcome contract for this objective. "
            "Do not choose from a canned contract list; infer what would count as done "
            "for this specific project. Return JSON only with these keys: "
            "goal, expected_output, success_check, stop_condition, max_attempt_budget, "
            "and optional reframe_path.\n\n"
            + "\n".join(context_lines)
        )
        try:
            response = await self.llm.chat_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are the OpenCAS agent drafting an outcome contract for your own "
                            "autonomous work before execution begins."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                complexity="standard",
                source="harness_contract_drafting",
            )
        except Exception as exc:
            self._trace(
                "objective_loop_contract_draft_error",
                {"loop_id": str(loop.loop_id), "error": str(exc)},
            )
            return None

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        raw = _extract_json_object(content)
        if raw is None:
            self._trace(
                "objective_loop_contract_draft_invalid",
                {"loop_id": str(loop.loop_id), "reason": "no_json_object"},
            )
            return None
        raw.setdefault("goal", loop.title)
        candidate_meta = dict(loop.meta or {})
        candidate_meta[_OBJECTIVE_CONTRACT_META_KEY] = raw
        candidate_loop = loop.model_copy(update={"meta": candidate_meta})
        contract, _missing = self._validated_objective_contract(candidate_loop)
        return contract

    def _validated_objective_contract(
        self,
        loop: ObjectiveLoop,
    ) -> tuple[Optional[Dict[str, Any]], List[str]]:
        raw = {}
        meta = loop.meta if isinstance(loop.meta, dict) else {}
        if isinstance(meta.get(_OBJECTIVE_CONTRACT_META_KEY), dict):
            raw = dict(meta[_OBJECTIVE_CONTRACT_META_KEY])

        missing = [
            field
            for field in _REQUIRED_OBJECTIVE_CONTRACT_FIELDS
            if not str(raw.get(field, "") or "").strip()
        ]
        if missing:
            return None, missing

        try:
            max_attempt_budget = max(1, int(raw.get("max_attempt_budget") or 1))
        except (TypeError, ValueError):
            max_attempt_budget = 1

        contract = ObjectiveLoopContract(
            goal=str(raw.get("goal", "")).strip(),
            expected_output=str(raw.get("expected_output", "")).strip(),
            success_check=str(raw.get("success_check", "")).strip(),
            stop_condition=str(raw.get("stop_condition", "")).strip(),
            max_attempt_budget=max_attempt_budget,
            reframe_path=str(raw.get("reframe_path", "") or "").strip(),
        ).model_dump(mode="json")
        return contract, []

    def _objective_contract_payload(self, loop: ObjectiveLoop) -> Optional[Dict[str, Any]]:
        contract, _missing = self._validated_objective_contract(loop)
        return contract

    def _contract_max_attempts(self, loop: ObjectiveLoop) -> int:
        contract = self._objective_contract_payload(loop)
        if not contract:
            return 1
        return max(1, int(contract.get("max_attempt_budget") or 1))

    def _loop_work_meta(self, loop: ObjectiveLoop) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "harness_origin": "objective_loop",
            "loop_id": str(loop.loop_id),
            "notebook_id": loop.notebook_id,
        }
        loop_meta = loop.meta if isinstance(loop.meta, dict) else {}
        resume_project = loop_meta.get("resume_project")
        if isinstance(resume_project, dict):
            meta["resume_project"] = dict(resume_project)
        contract = self._objective_contract_payload(loop)
        if contract:
            meta[_OBJECTIVE_CONTRACT_META_KEY] = contract
        return meta

    @staticmethod
    def _artifact_hint_from_loop(loop: ObjectiveLoop) -> Optional[str]:
        meta = loop.meta if isinstance(loop.meta, dict) else {}
        resume_project = meta.get("resume_project")
        if isinstance(resume_project, dict):
            artifact = str(resume_project.get("canonical_artifact_path", "") or "").strip()
            if artifact:
                return artifact
        artifact = str(meta.get("canonical_artifact_path", "") or "").strip()
        return artifact or None

    async def complete_loop(self, loop_id: str, success: bool = True) -> Optional[ObjectiveLoop]:
        """Mark an objective loop as completed or failed."""
        loop = await self.store.get_loop(loop_id)
        if loop is None:
            return None
        loop.status = ObjectiveStatus.COMPLETED if success else ObjectiveStatus.FAILED
        loop.updated_at = datetime.now(timezone.utc)
        await self.store.save_loop(loop)
        self._trace("objective_loop_completed" if success else "objective_loop_failed", {"loop_id": loop_id})
        return loop

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"AgenticHarness: {event}",
                payload,
            )


def _extract_json_object(content: Any) -> Optional[Dict[str, Any]]:
    text = str(content or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
