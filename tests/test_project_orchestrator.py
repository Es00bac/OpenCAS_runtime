"""Tests for ProjectOrchestrator."""

import pytest
import pytest_asyncio

from opencas.autonomy.models import WorkObject, WorkStage
from opencas.autonomy.project_orchestrator import ProjectOrchestrator
from opencas.autonomy.work_store import WorkStore
from opencas.execution import BoundedAssistantAgent
from opencas.execution.store import TaskStore
from opencas.infra import EventBus
from opencas.tools import ToolRegistry


@pytest_asyncio.fixture
async def stores(tmp_path):
    ws = WorkStore(tmp_path / "work.db")
    ts = TaskStore(tmp_path / "tasks.db")
    await ws.connect()
    await ts.connect()
    yield ws, ts
    await ws.close()
    await ts.close()


@pytest_asyncio.fixture
async def orchestrator(stores):
    work_store, task_store = stores
    tools = ToolRegistry()
    baa = BoundedAssistantAgent(tools=tools, store=task_store)
    bus = EventBus()
    return ProjectOrchestrator(
        llm=None,
        baa=baa,
        work_store=work_store,
        event_bus=bus,
    )


@pytest.mark.asyncio
async def test_decompose_fallback_without_llm(orchestrator, stores):
    work_store, _ = stores
    project = WorkObject(content="build a tiny web server", stage=WorkStage.PROJECT)
    plan = await orchestrator.decompose(project)
    assert len(plan.tasks) >= 1
    assert plan.project_work_id == str(project.work_id)

    # Verify work objects persisted
    all_work = await work_store.list_all()
    assert len(all_work) >= 2  # project + at least one task


@pytest.mark.asyncio
async def test_decompose_creates_dependency_graph(orchestrator, stores):
    work_store, _ = stores
    project = WorkObject(content="deploy app", stage=WorkStage.PROJECT)
    plan = await orchestrator.decompose(project)

    tasks = [w for w in await work_store.list_all() if w.project_id == str(project.work_id) and w.work_id != project.work_id]
    # At least one task should exist
    assert len(tasks) >= 1
    # All tasks should have repair_task_id in meta
    for t in tasks:
        assert "repair_task_id" in t.meta


@pytest.mark.asyncio
async def test_dependency_resolution(orchestrator):
    project = WorkObject(content="test project", stage=WorkStage.PROJECT)
    # Mock plan with explicit dependencies
    raw_json = (
        '{'
        '"tasks": ['
        '{"name": "setup", "description": "init repo", "dependencies": []},'
        '{"name": "build", "description": "compile", "dependencies": [0]}'
        '],'
        '"summary": "two-step plan"'
        '}'
    )
    plan = orchestrator._parse_plan(raw_json, project)
    assert len(plan.tasks) == 2
    assert plan.tasks[1]["dependencies"] == [0]

    wos, rts = orchestrator._create_tasks(project, plan)
    assert len(wos) == 2
    assert wos[1].blocked_by == [str(wos[0].work_id)]
    assert rts[1].depends_on == [str(wos[0].work_id)]


@pytest.mark.asyncio
async def test_on_baa_completed_unblocks_dependents(orchestrator, stores):
    work_store, task_store = stores
    project = WorkObject(content=" chained project", stage=WorkStage.PROJECT)

    # Manually construct two tasks with dependency
    from opencas.execution.models import RepairTask
    wo1 = WorkObject(content="step 1", stage=WorkStage.MICRO_TASK, project_id=str(project.work_id))
    rt1 = RepairTask(objective="step 1", project_id=str(project.work_id))
    wo1.meta["repair_task_id"] = str(rt1.task_id)

    wo2 = WorkObject(content="step 2", stage=WorkStage.MICRO_TASK, project_id=str(project.work_id), blocked_by=[str(wo1.work_id)])
    rt2 = RepairTask(objective="step 2", project_id=str(project.work_id), depends_on=[str(wo1.work_id)])
    wo2.meta["repair_task_id"] = str(rt2.task_id)

    await work_store.save(wo1)
    await work_store.save(wo2)

    from opencas.infra import BaaCompletedEvent
    event = BaaCompletedEvent(
        task_id=str(rt1.task_id),
        success=True,
        stage="done",
        objective="step 1",
    )
    await orchestrator._on_baa_completed(event)

    fetched = await work_store.get(str(wo2.work_id))
    assert fetched.blocked_by == []


@pytest.mark.asyncio
async def test_on_baa_completed_does_not_unblock_on_failure(orchestrator, stores):
    work_store, _ = stores
    project = WorkObject(content="failed project", stage=WorkStage.PROJECT)

    from opencas.execution.models import RepairTask

    wo1 = WorkObject(content="step 1", stage=WorkStage.MICRO_TASK, project_id=str(project.work_id))
    rt1 = RepairTask(objective="step 1", project_id=str(project.work_id))
    wo1.meta["repair_task_id"] = str(rt1.task_id)
    wo1.meta["repair_task_submitted"] = True

    wo2 = WorkObject(
        content="step 2",
        stage=WorkStage.MICRO_TASK,
        project_id=str(project.work_id),
        blocked_by=[str(wo1.work_id)],
    )
    wo2.meta["repair_task_id"] = str(RepairTask(objective="step 2", project_id=str(project.work_id)).task_id)

    await work_store.save(wo1)
    await work_store.save(wo2)

    from opencas.infra import BaaCompletedEvent

    event = BaaCompletedEvent(
        task_id=str(rt1.task_id),
        success=False,
        stage="failed",
        objective="step 1",
    )
    await orchestrator._on_baa_completed(event)

    fetched = await work_store.get(str(wo2.work_id))
    assert fetched.blocked_by == [str(wo1.work_id)]


@pytest.mark.asyncio
async def test_on_baa_completed_only_submits_newly_ready_dependents(stores):
    work_store, _ = stores
    submitted: list[str] = []

    class FakeBaa:
        async def submit(self, task):
            submitted.append(str(task.task_id))

    orchestrator = ProjectOrchestrator(
        llm=None,
        baa=FakeBaa(),
        work_store=work_store,
        event_bus=None,
    )

    project = WorkObject(content="parallel roots", stage=WorkStage.PROJECT)
    from opencas.execution.models import RepairTask

    wo1 = WorkObject(content="root 1", stage=WorkStage.MICRO_TASK, project_id=str(project.work_id))
    rt1 = RepairTask(objective="root 1", project_id=str(project.work_id))
    wo1.meta["repair_task_id"] = str(rt1.task_id)
    wo1.meta["repair_task_submitted"] = True

    wo2 = WorkObject(content="root 2", stage=WorkStage.MICRO_TASK, project_id=str(project.work_id))
    rt2 = RepairTask(objective="root 2", project_id=str(project.work_id))
    wo2.meta["repair_task_id"] = str(rt2.task_id)
    wo2.meta["repair_task_submitted"] = True

    wo3 = WorkObject(
        content="child",
        stage=WorkStage.MICRO_TASK,
        project_id=str(project.work_id),
        blocked_by=[str(wo1.work_id)],
        dependency_ids=[str(wo1.work_id)],
    )
    rt3 = RepairTask(
        objective="child",
        project_id=str(project.work_id),
        depends_on=[str(wo1.work_id)],
    )
    wo3.meta["repair_task_id"] = str(rt3.task_id)
    wo3.meta["repair_task_submitted"] = False

    await work_store.save(wo1)
    await work_store.save(wo2)
    await work_store.save(wo3)

    from opencas.infra import BaaCompletedEvent

    event = BaaCompletedEvent(
        task_id=str(rt1.task_id),
        success=True,
        stage="done",
        objective="root 1",
    )
    await orchestrator._on_baa_completed(event)

    assert submitted == [str(rt3.task_id)]
    refreshed_wo3 = await work_store.get(str(wo3.work_id))
    assert refreshed_wo3.meta["repair_task_submitted"] is True
