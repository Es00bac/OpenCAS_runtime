"""Agentic harness for long-horizon objectives and research notebooks."""

from __future__ import annotations

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
    ObjectiveStatus,
    ResearchNotebook,
)
from .store import HarnessStore


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
    ) -> None:
        self.store = store
        self.llm = llm
        self.tracer = tracer
        self.work_store = work_store
        self.baa = baa
        self.project_orchestrator = project_orchestrator

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
    ) -> ObjectiveLoop:
        """Create a new objective loop, optionally attached to a notebook."""
        loop = ObjectiveLoop(
            title=title,
            description=description,
            notebook_id=notebook_id,
            completion_criteria=completion_criteria or [],
        )
        await self.store.save_loop(loop)
        self._trace("objective_loop_created", {"loop_id": str(loop.loop_id), "title": title})
        return loop

    async def run_objective_cycle(self, max_active_loops: int = 3) -> Dict[str, Any]:
        """Run one cycle for active objective loops: plan, generate tasks, submit."""
        active_loops = await self.store.list_loops(status=ObjectiveStatus.ACTIVE, limit=max_active_loops)
        pending_loops = await self.store.list_loops(status=ObjectiveStatus.PENDING, limit=max_active_loops)

        # Promote pending to active if under capacity
        loops_to_process = list(active_loops)
        for loop in pending_loops:
            if len(loops_to_process) >= max_active_loops:
                break
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
            },
        )
        return {
            "loops_processed": len(loops_to_process),
            "submitted_tasks": submitted_tasks,
            "created_work_objects": created_work_objects,
        }

    async def _process_loop(self, loop: ObjectiveLoop) -> tuple[List[str], List[str]]:
        """Plan next steps for a loop and emit tasks/work objects."""
        submitted_tasks: List[str] = []
        created_work_objects: List[str] = []

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
            meta={
                "harness_origin": "objective_loop",
                "loop_id": str(loop.loop_id),
                "notebook_id": loop.notebook_id,
            },
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
                meta={
                    "harness_origin": "objective_loop",
                    "loop_id": str(loop.loop_id),
                    "notebook_id": loop.notebook_id,
                },
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
                meta={
                    "harness_origin": "objective_loop",
                    "loop_id": str(loop.loop_id),
                    "notebook_id": loop.notebook_id,
                },
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

        prompt = (
            "You are an autonomous research assistant. Given the objective and notebook context, "
            "decide the single most valuable next step. Return a concise 1-2 sentence plan.\n\n"
            + "\n".join(context_lines)
        )

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
