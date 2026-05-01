"""Project orchestrator for decomposing PROJECT-stage work into dependency-aware tasks."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from opencas.api import LLMClient
from opencas.execution import BoundedAssistantAgent, RepairTask
from opencas.execution.models import ExecutionStage
from opencas.infra import BaaCompletedEvent, EventBus

from .models import ProjectPlan, WorkObject, WorkStage
from .work_store import WorkStore


class ProjectOrchestrator:
    """Decomposes project-level work into RepairTasks with dependency tracking."""

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        baa: Optional[BoundedAssistantAgent] = None,
        work_store: Optional[WorkStore] = None,
        event_bus: Optional[EventBus] = None,
        shadow_registry: Optional[Any] = None,
    ) -> None:
        self.llm = llm
        self.baa = baa
        self.work_store = work_store
        self.event_bus = event_bus
        self.shadow_registry = shadow_registry
        if self.event_bus:
            self.event_bus.subscribe(BaaCompletedEvent, self._on_baa_completed)

    async def decompose(self, work: WorkObject) -> ProjectPlan:
        """Decompose a PROJECT-stage work object into a plan and child tasks."""
        if work.stage != WorkStage.PROJECT:
            work.stage = WorkStage.PROJECT

        plan = await self._generate_plan(work)
        work_objects, repair_tasks = self._create_tasks(work, plan)

        # Persist work objects and tasks
        if self.work_store:
            await self.work_store.save(work)
            for wo in work_objects:
                await self.work_store.save(wo)

        # Submit ready tasks (no unresolved dependencies)
        if self.baa:
            for task in repair_tasks:
                if not task.depends_on:
                    await self._submit_task(task)
                    await self._mark_repair_task_submitted(str(task.task_id))

        return plan

    async def _generate_plan(self, work: WorkObject) -> ProjectPlan:
        """Use LLM to generate a dependency-aware project plan."""
        shadow_context = self._shadow_planning_context(
            objective=work.content,
            meta=work.meta,
        )
        prompt = (
            f"Decompose the following project into 1-5 concrete tasks. "
            f"Return ONLY valid JSON with no markdown formatting.\n\n"
            f"Project: {work.content}\n\n"
            f"JSON schema:\n"
            f'{{"tasks": [{{"name": "string", "description": "string", "dependencies": [0]}}], "summary": "string"}}\n'
            f'"dependencies" are zero-based indices of prerequisite tasks in the tasks list.\n'
        )
        if shadow_context:
            prompt += f"\n\n{shadow_context}"

        raw_text = ""
        if self.llm:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a project planning assistant. "
                            "Break down projects into small, executable tasks with explicit dependencies."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
                response = await self.llm.chat_completion(
                    messages,
                    complexity="high",
                    source="project_decomposition",
                )
                raw_text = (
                    response.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
            except Exception as exc:
                raw_text = f'{{"tasks": [{{"name": "execute project", "description": "{work.content}", "dependencies": []}}], "summary": "fallback due to {exc}"}}'
        else:
            raw_text = (
                '{"tasks": [{"name": "execute project", "description": "'
                + work.content.replace('"', '\\"')
                + '", "dependencies": []}], "summary": "fallback (no LLM)"}'
            )

        parsed = self._parse_plan(raw_text, work)
        return parsed

    def _parse_plan(self, raw: str, work: WorkObject) -> ProjectPlan:
        """Parse LLM JSON output into a ProjectPlan."""
        # Strip markdown fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove first and last fence lines
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {
                "tasks": [
                    {
                        "name": "execute project",
                        "description": work.content,
                        "dependencies": [],
                    }
                ],
                "summary": "fallback due to JSON parse error",
            }

        tasks = data.get("tasks", [])
        if not tasks:
            tasks = [
                {
                    "name": "execute project",
                    "description": work.content,
                    "dependencies": [],
                }
            ]

        return ProjectPlan(
            project_work_id=str(work.work_id),
            tasks=tasks,
            summary=data.get("summary", ""),
        )

    def _create_tasks(
        self, project_work: WorkObject, plan: ProjectPlan
    ) -> tuple[List[WorkObject], List[RepairTask]]:
        """Create WorkObjects and RepairTasks from a project plan."""
        work_objects: List[WorkObject] = []
        repair_tasks: List[RepairTask] = []
        task_id_map: Dict[int, str] = {}

        # First pass: create WorkObjects and RepairTasks
        for idx, task_def in enumerate(plan.tasks):
            task_wo = WorkObject(
                content=f"{task_def.get('name', 'task')}: {task_def.get('description', '')}",
                stage=WorkStage.MICRO_TASK,
                project_id=str(project_work.work_id),
                meta={
                    "plan_task_index": idx,
                    "plan_summary": plan.summary,
                    **({"resume_project": dict(project_work.meta.get("resume_project", {}))} if isinstance(project_work.meta.get("resume_project"), dict) else {}),
                },
            )
            work_objects.append(task_wo)
            task_id_map[idx] = str(task_wo.work_id)

        # Second pass: resolve dependencies
        for idx, task_def in enumerate(plan.tasks):
            dep_indices = task_def.get("dependencies", [])
            task_wo = work_objects[idx]
            blocked_by: List[str] = []
            for d in dep_indices:
                if isinstance(d, int) and 0 <= d < len(work_objects):
                    task_wo.dependency_ids.append(task_id_map[d])
                    blocked_by.append(task_id_map[d])
            task_wo.blocked_by = blocked_by

        # Create RepairTasks from WorkObjects
        for task_wo in work_objects:
            repair_task = RepairTask(
                objective=task_wo.content,
                project_id=str(project_work.work_id),
                depends_on=list(task_wo.dependency_ids),
                scratch_dir=None,
                meta={
                    **({"resume_project": dict(project_work.meta.get("resume_project", {}))} if isinstance(project_work.meta.get("resume_project"), dict) else {}),
                },
            )
            # Store mapping in meta for event handling
            task_wo.meta["repair_task_id"] = str(repair_task.task_id)
            task_wo.meta["repair_task_submitted"] = False
            task_wo.meta["repair_task_status"] = "pending"
            repair_tasks.append(repair_task)

        return work_objects, repair_tasks

    def _shadow_planning_context(
        self,
        *,
        objective: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        builder = getattr(self.shadow_registry, "build_planning_context", None)
        if not callable(builder):
            return ""
        context = builder(
            objective=objective,
            artifact=self._artifact_hint_from_meta(meta),
        )
        if not isinstance(context, dict) or not context.get("available"):
            return ""
        return str(context.get("prompt_block", "") or "").strip()

    @staticmethod
    def _artifact_hint_from_meta(meta: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(meta, dict):
            return None
        resume_project = meta.get("resume_project")
        if isinstance(resume_project, dict):
            artifact = str(resume_project.get("canonical_artifact_path", "") or "").strip()
            if artifact:
                return artifact
        artifact = str(meta.get("canonical_artifact_path", "") or "").strip()
        return artifact or None

    async def _submit_task(self, task: RepairTask) -> None:
        """Submit a repair task to the BAA."""
        if not self.baa:
            return
        task.stage = ExecutionStage.QUEUED
        task.status = "queued"
        await self.baa.submit(task)

    async def _on_baa_completed(self, event: BaaCompletedEvent) -> None:
        """Unblock dependent work objects when a repair task completes."""
        if not self.work_store:
            return

        # Find the WorkObject associated with this repair task
        wo = await self._find_work_by_repair_task(event.task_id)
        if not wo:
            return

        # Update WorkObject stage based on success
        wo.stage = WorkStage.ARTIFACT if event.success else WorkStage.MICRO_TASK
        wo.meta["repair_task_submitted"] = True
        wo.meta["repair_task_status"] = event.stage
        wo.meta["baa_result"] = {
            "success": event.success,
            "stage": event.stage,
            "output": event.output,
        }
        await self.work_store.save(wo)

        if not event.success:
            return

        # Unblock dependents
        modified = await self.work_store.unblock_dependencies(str(wo.work_id))
        if modified and self.baa:
            await self._submit_ready_project_tasks(wo.project_id)

    async def _find_work_by_repair_task(self, repair_task_id: str) -> Optional[WorkObject]:
        """Find a WorkObject by its associated repair_task_id in meta."""
        if not self.work_store:
            return None
        return await self.work_store.find_by_repair_task_id(repair_task_id)

    async def _submit_ready_project_tasks(self, project_id: Optional[str]) -> None:
        """Submit newly ready project tasks that have not already been queued."""
        if not self.work_store or not self.baa or not project_id:
            return
        ready_items = await self.work_store.list_ready(limit=250)
        for item in ready_items:
            if item.project_id != project_id:
                continue
            if item.stage != WorkStage.MICRO_TASK:
                continue
            if item.meta.get("repair_task_submitted"):
                continue
            repair_task_id = item.meta.get("repair_task_id")
            if not repair_task_id:
                continue
            rt = RepairTask(
                task_id=uuid.UUID(repair_task_id),
                objective=item.content,
                project_id=item.project_id,
                depends_on=list(item.dependency_ids),
                scratch_dir=None,
            )
            await self._submit_task(rt)
            await self._mark_repair_task_submitted(repair_task_id)

    async def _mark_repair_task_submitted(self, repair_task_id: str) -> None:
        """Mark the mapped work item as already submitted to the BAA."""
        if not self.work_store:
            return
        work = await self._find_work_by_repair_task(repair_task_id)
        if work is None:
            return
        work.meta["repair_task_submitted"] = True
        work.meta["repair_task_status"] = ExecutionStage.QUEUED.value
        await self.work_store.save(work)
