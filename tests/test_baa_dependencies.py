"""Tests for BAA dependency-aware scheduling."""

from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.api import provenance_store as ps
from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.commitment_store import CommitmentStore
from opencas.autonomy.models import WorkObject, WorkStage
from opencas.autonomy.work_store import WorkStore
from opencas.context.models import MessageRole
from opencas.execution import BoundedAssistantAgent, ExecutionStage, RepairResult, RepairTask
from opencas.execution.lanes import CommandLane
from opencas.execution.lifecycle import LifecycleStage
from opencas.execution.store import TaskStore
from opencas.memory import EpisodeKind
from opencas.scheduling.models import ScheduleAction, ScheduleKind, ScheduleRecurrence, ScheduleStatus
from opencas.tools import ToolRegistry


@pytest_asyncio.fixture
async def store(tmp_path):
    ts = TaskStore(tmp_path / "tasks.db")
    await ts.connect()
    yield ts
    await ts.close()


@pytest_asyncio.fixture
async def work_store(tmp_path):
    ws = WorkStore(tmp_path / "work.db")
    await ws.connect()
    yield ws
    await ws.close()


@pytest_asyncio.fixture
async def baa(store):
    tools = ToolRegistry()
    agent = BoundedAssistantAgent(tools=tools, store=store, max_concurrent=1)
    await agent.start()
    yield agent
    await agent.stop()


def _runtime_with_work_store(work_store):
    class _Ctx:
        def __init__(self, ws):
            self.work_store = ws

    class _Runtime:
        def __init__(self, ws):
            self.ctx = _Ctx(ws)

    return _Runtime(work_store)


def test_artifact_progress_recovery_accepts_timeout_workspace_progress_marker():
    task = RepairTask(
        objective="continue writing project",
        meta={
            "timeout_artifact_progress": {
                "paths": ["/workspace/novels/book/drafts/chapter_31.md"],
                "reason": "tool_loop_execution_timeout",
            }
        },
    )
    result = RepairResult(
        task_id=task.task_id,
        success=False,
        stage=ExecutionStage.RECOVERING,
        output="Artifact progress boundary reached; will continue.",
        artifacts=[],
    )

    assert BoundedAssistantAgent._artifact_progress_boundary_recovery(task, result)


class _ContextStore:
    def __init__(self):
        self.entries = []

    async def append(self, session_id, role, content, meta=None):
        self.entries.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "meta": meta or {},
            }
        )


class _MemoryStore:
    def __init__(self):
        self.episodes = []

    async def save_episode(self, episode):
        self.episodes.append(episode)


class _PausedExecutive:
    def recommend_pause(self) -> bool:
        return True

    def pause_reason(self) -> str:
        return "fatigue"


class _OverloadedExecutive:
    def recommend_pause(self) -> bool:
        return True

    def pause_reason(self) -> str:
        return "overload"


class _ScheduleService:
    def __init__(self):
        self.created = []
        self.items = []

    async def create_schedule(self, **kwargs):
        self.created.append(kwargs)
        item = SimpleNamespace(
            schedule_id=f"schedule-{len(self.created)}",
            status=kwargs.get("status", ScheduleStatus.ACTIVE),
            action=kwargs.get("action"),
            commitment_id=kwargs.get("commitment_id"),
            tags=kwargs.get("tags", []),
            meta=kwargs.get("meta", {}),
        )
        self.items.append(item)
        return item

    def list_items(self, status=None, limit=1000):
        items = self.items
        if status is not None:
            items = [item for item in items if item.status == status]
        return items[:limit]


class _ProgressBoundaryExecutor:
    def __init__(self):
        self.calls = 0

    def record_task_boundary(self, *args, **kwargs):
        return None

    async def run(self, task):
        self.calls += 1
        if self.calls == 1:
            task.meta["last_tool_loop"] = {
                "iterations": 32,
                "guard_fired": False,
                "tool_call_count": 1,
            }
            task.meta["last_tool_calls"] = [
                {
                    "id": "call-1",
                    "name": "fs_write_file",
                    "args": {"file_path": "/tmp/progress-artifact.md"},
                }
            ]
            task.artifacts.append("exec:Reached maximum number of tool-use iterations.")
            task.stage = ExecutionStage.RECOVERING
            task.status = "retrying"
            return RepairResult(
                task_id=task.task_id,
                success=False,
                stage=ExecutionStage.RECOVERING,
                output="Artifact progress boundary reached; will continue.",
                artifacts=task.artifacts,
            )
        task.stage = ExecutionStage.DONE
        task.status = "completed"
        return RepairResult(
            task_id=task.task_id,
            success=True,
            stage=ExecutionStage.DONE,
            output="completed after progress boundary continuation",
            artifacts=task.artifacts,
        )


class _FailingExecutor:
    def __init__(self, reason: str):
        self.reason = reason

    def record_task_boundary(self, *args, **kwargs):
        return None

    async def run(self, task):
        task.stage = ExecutionStage.FAILED
        task.status = "failed"
        return RepairResult(
            task_id=task.task_id,
            success=False,
            stage=ExecutionStage.FAILED,
            output=self.reason,
            artifacts=task.artifacts,
        )


class _RecoveryScheduleService:
    def __init__(self):
        self.created = []
        self.items = []

    async def create_schedule(self, **kwargs):
        self.created.append(kwargs)
        item = SimpleNamespace(
            schedule_id=f"recovery-schedule-{len(self.created)}",
            status=kwargs.get("status", ScheduleStatus.ACTIVE),
            action=kwargs.get("action"),
            commitment_id=kwargs.get("commitment_id"),
            tags=kwargs.get("tags", []),
            meta=kwargs.get("meta", {}),
        )
        self.items.append(item)
        return item

    async def list_items(self, status=None, limit=1000):
        items = self.items
        if status is not None:
            items = [item for item in items if item.status == status]
        return items[:limit]


@pytest.mark.asyncio
async def test_task_held_until_dependency_completes(baa, store):
    dep = RepairTask(objective="dependency task")
    child = RepairTask(objective="child task", depends_on=[str(dep.task_id)])

    # Submit child first; it should be held
    f_child = await baa.submit(child)
    assert str(child.task_id) in baa._held
    assert child.meta["held_reason"] == "waiting_for_dependencies"

    # Submit dependency; it should run immediately
    f_dep = await baa.submit(dep)
    r_dep = await f_dep
    assert r_dep.success is True

    # Child should now be released and run
    r_child = await f_child
    assert r_child.success is True
    assert "held_reason" not in child.meta


@pytest.mark.asyncio
async def test_schedule_task_held_while_executive_paused(store):
    runtime = SimpleNamespace(executive=_PausedExecutive())
    baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)
    task = RepairTask(objective="Return to project index", meta={"source": "schedule"})

    future = await baa.submit(task)
    stored = await store.get(str(task.task_id))

    assert not future.done()
    assert str(task.task_id) in baa._held
    assert baa.queue_size == 0
    assert task.meta["held_reason"] == "executive_recommended_pause"
    assert task.meta["executive_pause_reason"] == "fatigue"
    assert task.status == "held"
    assert stored is not None
    assert stored.status == "held"
    assert stored.meta["held_reason"] == "executive_recommended_pause"
    assert stored.meta["last_baa_awareness_signature"]


@pytest.mark.asyncio
async def test_baa_awareness_records_project_work_in_source_session_memory(store):
    context_store = _ContextStore()
    memory_store = _MemoryStore()
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            context_store=context_store,
            config=SimpleNamespace(session_id="default-session"),
        ),
        memory=memory_store,
    )
    baa = BoundedAssistantAgent(
        tools=ToolRegistry(),
        store=store,
        runtime=runtime,
        max_concurrent=1,
    )
    task = RepairTask(
        objective="Return to the workspace novel and continue revision until completion evidence exists.",
        meta={
            "source": "schedule",
            "source_session_id": "chat-session-123",
            "project_title": "The Test Novel",
            "project_type": "writing",
            "workspace_rel_path": "novels/the-test-novel",
            "workspace_project_confidence": 1.0,
        },
    )
    result = RepairResult(
        task_id=task.task_id,
        success=True,
        stage=ExecutionStage.DONE,
        output="Revised the manuscript and updated the name research notes.",
        artifacts=[
            "workspace/novels/the-test-novel/revision/full_clean_draft.md",
            "workspace/novels/the-test-novel/research/name_place_research.md",
        ],
    )

    await baa._record_baa_awareness(
        task,
        status="completed",
        reason="execution completed successfully",
        result=result,
        lane=CommandLane.BAA,
    )

    [entry] = context_store.entries
    assert entry["session_id"] == "chat-session-123"
    assert entry["role"] == MessageRole.SYSTEM
    assert "Recent project work memory" in entry["content"]
    assert "Progress update: I completed this background task. I have 2 artifacts recorded." in entry["content"]
    assert "The Test Novel" in entry["content"]
    assert "book, novel, manuscript" in entry["content"]

    [episode] = memory_store.episodes
    assert episode.kind == EpisodeKind.OBSERVATION
    assert episode.session_id == "chat-session-123"
    assert episode.salience == 7.5
    assert "Recent project work memory" in episode.content
    assert episode.payload["payload"]["project_title"] == "The Test Novel"


@pytest.mark.asyncio
async def test_user_project_return_with_workspace_evidence_bypasses_overload_pause(store):
    runtime = SimpleNamespace(executive=_OverloadedExecutive())
    baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)
    await baa.start()
    task = RepairTask(
        objective="Return to project with concrete workspace evidence",
        meta={
            "source": "schedule",
            "authority": "schedule_due",
            "source_session_id": "chat-session-123",
            "workspace_abs_path": "/tmp/example-project",
            "workspace_project_confidence": 1.0,
        },
    )

    future = await baa.submit(task)
    result = await future
    stored = await store.get(str(task.task_id))

    await baa.stop()
    assert result.success is True
    assert str(task.task_id) not in baa._held
    assert stored is not None
    assert stored.status != "held"
    assert stored.meta.get("held_reason") != "executive_recommended_pause"


@pytest.mark.asyncio
async def test_immediate_operator_project_start_bypasses_overload_before_workspace_exists(store):
    runtime = SimpleNamespace(executive=_OverloadedExecutive())
    baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)
    await baa.start()
    task = RepairTask(
        objective="Start the requested writing project under the requested workspace parent",
        meta={
            "source": "schedule",
            "authority": "schedule_due",
            "source_session_id": "chat-session-123",
            "requested_workspace_abs_path": "/tmp/workspace/novels",
            "requested_workspace_rel_path": "workspace/novels",
            "requested_workspace_kind": "parent",
            "start_policy": "immediate_operator_work_request",
        },
    )

    future = await baa.submit(task)
    result = await future
    stored = await store.get(str(task.task_id))

    await baa.stop()
    assert result.success is True
    assert str(task.task_id) not in baa._held
    assert stored is not None
    assert stored.status != "held"
    assert stored.meta.get("held_reason") != "executive_recommended_pause"


@pytest.mark.asyncio
async def test_restored_user_project_return_requeues_after_overload_pause(store):
    runtime = SimpleNamespace(executive=_OverloadedExecutive())
    task = RepairTask(
        objective="Return to project after restart",
        status="held",
        meta={
            "source": "schedule",
            "authority": "schedule_due",
            "source_session_id": "chat-session-123",
            "workspace_abs_path": "/tmp/example-project",
            "workspace_project_confidence": 1.0,
            "held_reason": "executive_recommended_pause",
            "executive_pause_reason": "overload",
        },
    )
    await store.save(task)

    baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)
    await baa.start()
    result = await baa._futures[str(task.task_id)]
    stored = await store.get(str(task.task_id))

    await baa.stop()
    assert result.success is True
    assert str(task.task_id) not in baa._held
    assert stored is not None
    assert stored.status != "held"
    assert stored.meta.get("held_reason") != "executive_recommended_pause"


@pytest.mark.asyncio
async def test_artifact_progress_boundary_requeues_without_consuming_failure_cap(store):
    baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, max_concurrent=1)
    fake_executor = _ProgressBoundaryExecutor()
    baa.executor = fake_executor
    await baa.start()
    task = RepairTask(objective="Long artifact-producing task", max_attempts=1)

    future = await baa.submit(task)
    result = await future
    stored = await store.get(str(task.task_id))

    await baa.stop()
    assert result.success is True
    assert fake_executor.calls == 2
    assert stored is not None
    assert stored.meta.get("recovery_count") in (None, 0)
    assert stored.meta["progress_boundary_recovery_count"] == 1


@pytest.mark.asyncio
async def test_schedule_origin_failure_creates_recovery_schedule(store):
    schedule_service = _RecoveryScheduleService()
    runtime = SimpleNamespace(schedule_service=schedule_service)
    baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)
    baa.executor = _FailingExecutor("Execution failed after 3 attempts: dns_resolution_failed")
    await baa.start()
    task = RepairTask(
        objective="Write the scheduled news digest.",
        meta={
            "source": "schedule",
            "authority": "schedule_due",
            "schedule_id": "schedule-123",
            "schedule_title": "Daily digest",
            "scheduled_for": "2026-05-19T15:00:00+00:00",
        },
    )

    future = await baa.submit(task)
    result = await future
    stored = await store.get(str(task.task_id))

    await baa.stop()
    assert result.success is False
    assert stored is not None
    assert stored.status == "failed"
    assert len(schedule_service.created) == 1
    created = schedule_service.created[0]
    assert created["kind"] == ScheduleKind.TASK
    assert created["action"] == ScheduleAction.SUBMIT_BAA
    assert created["recurrence"] == ScheduleRecurrence.NONE
    assert created["title"].startswith("Recover scheduled task:")
    assert "Diagnose why the scheduled task failed" in created["objective"]
    assert "Try a materially different" in created["objective"]
    assert created["meta"]["source"] == "schedule_failure_recovery"
    assert created["meta"]["recovery_from_task_id"] == str(task.task_id)
    assert created["meta"]["recovery_from_schedule_id"] == "schedule-123"
    assert created["meta"]["schedule_recovery_depth"] == 1
    assert "dns_resolution_failed" in created["meta"]["previous_task_output"]


@pytest.mark.asyncio
async def test_schedule_failure_recovery_stops_at_depth_cap(store):
    schedule_service = _RecoveryScheduleService()
    runtime = SimpleNamespace(schedule_service=schedule_service)
    baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)
    baa.executor = _FailingExecutor("Recovery cap exceeded after repeated provider failures.")
    await baa.start()
    task = RepairTask(
        objective="Recover the scheduled digest.",
        meta={
            "source": "schedule_failure_recovery",
            "authority": "schedule_due",
            "schedule_id": "schedule-123",
            "schedule_recovery_depth": 3,
        },
    )

    future = await baa.submit(task)
    result = await future

    await baa.stop()
    assert result.success is False
    assert schedule_service.created == []


@pytest.mark.asyncio
async def test_restored_executive_hold_does_not_duplicate_agent_awareness(store):
    runtime = SimpleNamespace(
        executive=_PausedExecutive(),
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="system:automated"),
            context_store=_ContextStore(),
        ),
        memory=_MemoryStore(),
    )
    baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)
    task = RepairTask(objective="Identify canonical writing project 4246 artifact", meta={"source": "schedule"})

    await baa.submit(task)

    assert len(runtime.ctx.context_store.entries) == 1
    stored = await store.get(str(task.task_id))
    assert stored is not None
    assert stored.status == "held"

    restored = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)
    await restored.start()
    try:
        assert len(runtime.ctx.context_store.entries) == 1
        restored_task = await store.get(str(task.task_id))
        assert restored_task is not None
        assert restored_task.status == "held"
        assert restored_task.meta["held_reason"] == "executive_recommended_pause"
    finally:
        await restored.stop()


@pytest.mark.asyncio
async def test_task_with_precompleted_dependency_runs_immediately(baa, store):
    dep = RepairTask(objective="already done")
    await baa.submit(dep)
    r = await baa._futures[str(dep.task_id)]
    assert r.success is True

    child = RepairTask(objective="child", depends_on=[str(dep.task_id)])
    f_child = await baa.submit(child)
    assert str(child.task_id) not in baa._held
    r_child = await f_child
    assert r_child.success is True


@pytest.mark.asyncio
async def test_task_held_until_work_dependency_becomes_artifact(tmp_path, work_store):
    ts = TaskStore(tmp_path / "tasks.db")
    await ts.connect()
    tools = ToolRegistry()
    baa = BoundedAssistantAgent(
        tools=tools,
        store=ts,
        max_concurrent=1,
        runtime=_runtime_with_work_store(work_store),
    )
    await baa.start()

    dep_work = WorkObject(content="prepare dependency", stage=WorkStage.MICRO_TASK)
    await work_store.save(dep_work)

    child = RepairTask(objective="child", depends_on=[str(dep_work.work_id)])
    f_child = await baa.submit(child)
    assert str(child.task_id) in baa._held
    assert child.meta["held_reason"] == "waiting_for_dependencies"

    dep_work.stage = WorkStage.ARTIFACT
    await work_store.save(dep_work)
    released = await baa.try_release_held()

    assert released == 1
    assert str(child.task_id) not in baa._held
    assert "held_reason" not in child.meta
    r_child = await f_child
    assert r_child.success is True

    await baa.stop()
    await ts.close()


@pytest.mark.asyncio
async def test_micro_task_work_dependency_does_not_release_early(tmp_path, work_store):
    ts = TaskStore(tmp_path / "tasks.db")
    await ts.connect()
    tools = ToolRegistry()
    baa = BoundedAssistantAgent(
        tools=tools,
        store=ts,
        max_concurrent=1,
        runtime=_runtime_with_work_store(work_store),
    )
    await baa.start()

    dep_work = WorkObject(content="still in progress", stage=WorkStage.MICRO_TASK)
    await work_store.save(dep_work)

    child = RepairTask(objective="child", depends_on=[str(dep_work.work_id)])
    await baa.submit(child)
    await baa._try_release_held()

    assert str(child.task_id) in baa._held

    await baa.stop()
    await ts.close()


@pytest.mark.asyncio
async def test_held_tasks_restored_from_store(tmp_path):
    ts = TaskStore(tmp_path / "tasks.db")
    await ts.connect()
    dep = RepairTask(objective="dep")
    child = RepairTask(objective="child", depends_on=[str(dep.task_id)])
    await ts.save(child)

    tools = ToolRegistry()
    baa2 = BoundedAssistantAgent(tools=tools, store=ts, max_concurrent=1)
    await baa2.start()
    assert str(child.task_id) in baa2._held
    restored_child = baa2._held[str(child.task_id)]
    assert restored_child.meta["held_reason"] == "waiting_for_dependencies"

    f_dep = await baa2.submit(dep)
    await f_dep

    # Child should have been released after dep finished
    assert str(child.task_id) not in baa2._held
    assert "held_reason" not in restored_child.meta
    await baa2.stop()
    await ts.close()


@pytest.mark.asyncio
async def test_task_held_for_approval_and_resolved(baa):
    import asyncio

    from opencas.execution.models import ExecutionStage

    task = RepairTask(objective="needs approval")
    task_id = str(task.task_id)
    task.stage = ExecutionStage.NEEDS_APPROVAL

    future = asyncio.get_running_loop().create_future()
    baa._futures[task_id] = future
    baa._held[task_id] = task

    resolved = await baa.resolve_hold(task_id)
    assert resolved is True
    assert task_id not in baa._held
    assert task.stage.value == "queued"

    r = await future
    assert r.success is True


@pytest.mark.asyncio
async def test_task_held_for_clarification_and_resolved(baa):
    import asyncio

    from opencas.execution.models import ExecutionStage

    task = RepairTask(objective="needs clarification")
    task_id = str(task.task_id)
    task.stage = ExecutionStage.NEEDS_CLARIFICATION

    future = asyncio.get_running_loop().create_future()
    baa._futures[task_id] = future
    baa._held[task_id] = task

    resolved = await baa.resolve_hold(task_id)
    assert resolved is True
    assert task_id not in baa._held
    assert task.stage.value == "queued"

    r = await future
    assert r.success is True


@pytest.mark.asyncio
async def test_resolve_hold_missing_task_returns_false(baa):
    resolved = await baa.resolve_hold("nonexistent-task-id")
    assert resolved is False


def test_active_count_excludes_held_and_queued_tasks():
    tools = ToolRegistry()
    agent = BoundedAssistantAgent(tools=tools, max_concurrent=1)
    agent._futures = {"a": object(), "b": object(), "c": object()}
    agent._held = {"held": RepairTask(objective="held")}
    agent._lanes.submit(CommandLane.BAA, object())
    assert agent.active_count == 1


@pytest.mark.asyncio
async def test_task_transition_to_operator_input_records_waiting_provenance(store, tmp_path):
    tools = ToolRegistry()
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="session-1", state_dir=tmp_path),
        )
    )
    baa = BoundedAssistantAgent(tools=tools, store=store, runtime=runtime, max_concurrent=1)

    task = RepairTask(objective="needs approval")
    task.stage = ExecutionStage.QUEUED

    await baa._transition_task(task, LifecycleStage.NEEDS_APPROVAL, "need operator input")

    records_path = tmp_path / "provenance.transitions.jsonl"
    records = [
        ps.parse_provenance_transition(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = records[-1]

    assert record.kind == ps.ProvenanceTransitionKind.WAITING
    assert record.status == "blocked"
    assert record.details["source_artifact"] == f"repair|baa|{task.task_id}"
    assert record.details["trigger_action"] == "baa.transition_task"
    assert record.details["target_entity"] == str(task.task_id)
    assert record.details["origin_action_id"] == str(task.task_id)
    assert any(event.get("event_type") == "BLOCKED" for event in task.meta["provenance_events"])
    assert any(
        event.get("triggering_artifact") == f"repair-task|default|{task.task_id}"
        for event in task.meta["provenance_events"]
    )


@pytest.mark.asyncio
async def test_task_transition_to_operator_input_notifies_owner(store, tmp_path):
    tools = ToolRegistry()
    contacts = []

    async def _contact_owner(**kwargs):
        contacts.append(kwargs)
        return {"status": "sent", "channel": "telegram"}

    runtime = SimpleNamespace(
        initiative_contact_owner=_contact_owner,
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="session-1", state_dir=tmp_path),
        ),
    )
    baa = BoundedAssistantAgent(tools=tools, store=store, runtime=runtime, max_concurrent=1)
    task = RepairTask(
        objective="Return to project \"kPony\".",
        meta={
            "source": "schedule",
            "project_key": "kpony",
            "project_title": "kPony",
            "project_type": "software",
            "workspace_rel_path": "workspace/kPony",
        },
    )

    await baa._transition_task(task, LifecycleStage.NEEDS_CLARIFICATION, "missing account credentials")

    assert contacts
    assert contacts[0]["channel"] == "telegram"
    assert contacts[0]["urgency"] == "high"
    assert "kPony" in contacts[0]["message"]
    assert "missing account credentials" in contacts[0]["message"]
    assert "workspace/kPony" in contacts[0]["message"]


@pytest.mark.asyncio
async def test_worker_exception_fails_task_and_notifies_owner(store, tmp_path):
    tools = ToolRegistry()
    contacts = []

    async def _contact_owner(**kwargs):
        contacts.append(kwargs)
        return {"status": "sent", "channel": "telegram"}

    runtime = SimpleNamespace(
        initiative_contact_owner=_contact_owner,
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="session-1", state_dir=tmp_path),
        ),
    )
    baa = BoundedAssistantAgent(tools=tools, store=store, runtime=runtime, max_concurrent=1)

    async def _boom(_task):
        raise RuntimeError("executor exploded")

    baa.executor.run = _boom
    baa._running = True
    task = RepairTask(
        objective="Return to project \"kPony\".",
        meta={
            "source": "schedule",
            "project_key": "kpony",
            "project_title": "kPony",
            "project_type": "software",
            "workspace_rel_path": "workspace/kPony",
        },
    )
    future = await baa.submit(task, lane=CommandLane.BAA)
    baa._lanes.submit(CommandLane.BAA, RepairTask(objective="__sentinel__"))

    await baa._worker_coro(CommandLane.BAA)

    assert future.done()
    result = future.result()
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED
    assert "executor exploded" in result.output
    stored_result = await store.get_result(str(task.task_id))
    assert stored_result is not None
    assert stored_result.stage == ExecutionStage.FAILED
    assert contacts
    assert contacts[0]["channel"] == "telegram"
    assert "executor exploded" in contacts[0]["message"]


@pytest.mark.asyncio
async def test_owner_notification_explains_provider_rate_limit(store, tmp_path):
    tools = ToolRegistry()
    contacts = []

    async def _contact_owner(**kwargs):
        contacts.append(kwargs)
        return {"status": "sent", "channel": "telegram"}

    runtime = SimpleNamespace(
        initiative_contact_owner=_contact_owner,
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="session-1", state_dir=tmp_path),
        ),
    )
    baa = BoundedAssistantAgent(tools=tools, store=store, runtime=runtime, max_concurrent=1)
    task = RepairTask(
        objective="Daily news digest",
        meta={"source": "schedule", "project_title": "Daily News"},
    )

    await baa._notify_owner_task_attention(
        task,
        status="failed",
        reason="HTTP 429 rate limit from api.z.ai",
        urgency="high",
    )

    assert contacts
    assert "HTTP 429 rate limit" in contacts[0]["message"]
    assert "Provider rate limit" in contacts[0]["message"]


@pytest.mark.asyncio
async def test_failed_project_task_blocks_linked_commitment_and_notifies_owner(store, tmp_path):
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    contacts = []

    async def _contact_owner(**kwargs):
        contacts.append(kwargs)
        return {"status": "sent", "channel": "telegram"}

    try:
        commitment = Commitment(
            content="Return to project: kPony",
            tags=["project_return", "self_directed"],
            meta={"source": "project_return_capture", "project_key": "kpony", "project_title": "kPony"},
        )
        await commitment_store.save(commitment)
        runtime = SimpleNamespace(
            commitment_store=commitment_store,
            initiative_contact_owner=_contact_owner,
            ctx=SimpleNamespace(
                config=SimpleNamespace(session_id="session-1", state_dir=tmp_path),
            ),
        )
        baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)

        async def _failed(task):
            return RepairResult(
                task_id=task.task_id,
                success=False,
                stage=ExecutionStage.FAILED,
                output="verification failed after retries",
                artifacts=[],
            )

        baa.executor.run = _failed
        task = RepairTask(
            objective="Return to project \"kPony\".",
            commitment_id=str(commitment.commitment_id),
            meta={
                "source": "schedule",
                "project_key": "kpony",
                "project_title": "kPony",
                "project_type": "software",
            },
        )

        await baa._run_bounded(task)

        updated = await commitment_store.get(str(commitment.commitment_id))
        assert updated is not None
        assert updated.status == CommitmentStatus.BLOCKED
        assert updated.meta["blocked_task_id"] == str(task.task_id)
        assert "verification failed" in updated.meta["blocked_reason"]
        assert contacts
        assert "verification failed after retries" in contacts[0]["message"]
    finally:
        await commitment_store.close()


@pytest.mark.asyncio
async def test_recoverable_project_task_failure_schedules_automatic_continuation(store, tmp_path):
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    schedule_service = _ScheduleService()
    contacts = []

    async def _contact_owner(**kwargs):
        contacts.append(kwargs)
        return {"status": "sent", "channel": "telegram"}

    try:
        commitment = Commitment(
            content="Return to project: Impossible Lantern",
            tags=["project_return", "self_directed"],
            meta={
                "source": "project_return_capture",
                "project_key": "impossible-lantern",
                "project_title": "Impossible Lantern",
                "project_type": "writing",
                "source_session_id": "session-project-1",
                "workspace_abs_path": "/mnt/xtra/OpenCAS/workspace/projects/impossible-lantern",
                "workspace_rel_path": "workspace/projects/impossible-lantern",
                "workspace_project_confidence": 0.95,
                "requested_workspace_abs_path": "/mnt/xtra/OpenCAS/workspace/projects",
                "requested_workspace_rel_path": "workspace/projects",
                "requested_workspace_kind": "parent",
                "start_policy": "immediate_operator_work_request",
                "project_intent": "finish the requested project without waiting for more owner prompting.",
                "next_step": "continue from the last persisted artifact",
                "creative_completion_contract": "complete the requested draft",
                "target_word_count": 1200,
            },
        )
        await commitment_store.save(commitment)
        runtime = SimpleNamespace(
            commitment_store=commitment_store,
            schedule_service=schedule_service,
            initiative_contact_owner=_contact_owner,
            ctx=SimpleNamespace(
                config=SimpleNamespace(session_id="session-1", state_dir=tmp_path),
            ),
        )
        baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)

        async def _recoverable_failure(task):
            return RepairResult(
                task_id=task.task_id,
                success=False,
                stage=ExecutionStage.FAILED,
                output="retry blocked: tool loop guard stopped broad attempt; preserve partial value and reframe narrowly before retry",
                artifacts=["workspace/projects/impossible-lantern/draft.md"],
            )

        baa.executor.run = _recoverable_failure
        task = RepairTask(
            objective="Continue the linked project until completion evidence is available.",
            commitment_id=str(commitment.commitment_id),
            meta={
                "source": "schedule",
                "project_key": "impossible-lantern",
                "project_title": "Impossible Lantern",
                "project_type": "writing",
            },
        )

        await baa._run_bounded(task)

        updated = await commitment_store.get(str(commitment.commitment_id))
        assert updated is not None
        assert updated.status == CommitmentStatus.ACTIVE
        assert updated.meta["resume_policy"] == "auto_on_recoverable_task_blocker"
        assert updated.meta["last_recoverable_task_id"] == str(task.task_id)
        assert "tool loop guard" in updated.meta["last_recoverable_task_failure"]
        assert contacts == []
        assert len(schedule_service.created) == 1
        created = schedule_service.created[0]
        assert "project_return" in created["tags"]
        assert created["kind"] == ScheduleKind.TASK
        assert created["action"] == ScheduleAction.SUBMIT_BAA
        assert created["commitment_id"] == str(commitment.commitment_id)
        assert created["meta"]["source"] == "project_followthrough_recovery"
        assert created["meta"]["source_session_id"] == "session-project-1"
        assert created["meta"]["workspace_rel_path"] == "workspace/projects/impossible-lantern"
        assert created["meta"]["requested_workspace_rel_path"] == "workspace/projects"
        assert created["meta"]["requested_workspace_kind"] == "parent"
        assert created["meta"]["start_policy"] == "immediate_operator_work_request"
        assert created["meta"]["project_intent"].startswith("finish the requested project")
        assert created["meta"]["next_step"] == "continue from the last persisted artifact"
        assert created["meta"]["creative_completion_contract"] == "complete the requested draft"
        assert created["meta"]["target_word_count"] == 1200
        assert created["meta"]["recovery_from_task_id"] == str(task.task_id)
        assert "narrow" in created["objective"].lower()
        assert "completion evidence" in created["objective"].lower()
    finally:
        await commitment_store.close()


@pytest.mark.asyncio
async def test_project_contract_failure_schedules_recovery_instead_of_blocking(store, tmp_path):
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    schedule_service = _ScheduleService()

    try:
        commitment = Commitment(
            content="Return to project: Glass Tide",
            tags=["project_return", "self_directed"],
            meta={
                "source": "project_return_capture",
                "project_key": "glass-tide",
                "project_title": "Glass Tide",
                "project_type": "writing",
                "project_intent": "finish the requested new project without sibling materialization.",
                "next_step": "create original project artifacts in the reserved root",
            },
        )
        await commitment_store.save(commitment)
        runtime = SimpleNamespace(
            commitment_store=commitment_store,
            schedule_service=schedule_service,
            ctx=SimpleNamespace(
                config=SimpleNamespace(session_id="session-1", state_dir=tmp_path),
            ),
        )
        baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)

        async def _contract_failure(task):
            return RepairResult(
                task_id=task.task_id,
                success=False,
                stage=ExecutionStage.FAILED,
                output=(
                    "execute failed: new-project contract forbids satisfying the task by "
                    "materializing sibling project artifacts into the target root"
                ),
                artifacts=[],
            )

        baa.executor.run = _contract_failure
        task = RepairTask(
            objective="Continue the linked new project until valid completion evidence is available.",
            commitment_id=str(commitment.commitment_id),
            meta={
                "source": "schedule",
                "project_key": "glass-tide",
                "project_title": "Glass Tide",
                "project_type": "writing",
            },
        )

        await baa._run_bounded(task)

        updated = await commitment_store.get(str(commitment.commitment_id))
        assert updated is not None
        assert updated.status == CommitmentStatus.ACTIVE
        assert updated.meta["resume_policy"] == "auto_on_recoverable_task_blocker"
        assert "new-project contract" in updated.meta["last_recoverable_task_failure"]
        assert len(schedule_service.created) == 1
        created = schedule_service.created[0]
        assert created["meta"]["source"] == "project_followthrough_recovery"
        assert created["meta"]["recoverable_blocker"].startswith("execute failed: new-project contract")
        assert "changed" in created["objective"].lower() or "narrower" in created["objective"].lower()
    finally:
        await commitment_store.close()


@pytest.mark.asyncio
async def test_successful_partial_project_task_schedules_next_continuation(store, tmp_path):
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    schedule_service = _ScheduleService()

    try:
        commitment = Commitment(
            content="Return to project: Glass Compiler",
            tags=["project_return", "self_directed"],
            meta={
                "source": "project_return_capture",
                "project_key": "glass-compiler",
                "project_title": "Glass Compiler",
                "project_type": "software",
                "source_session_id": "session-project-2",
                "requested_workspace_abs_path": "/mnt/xtra/OpenCAS/workspace/software",
                "requested_workspace_rel_path": "workspace/software",
                "requested_workspace_kind": "parent",
                "return_policy": "immediate_autonomous_work",
                "project_intent": "finish the requested software project without waiting for more owner prompting.",
                "next_step": "continue implementation and verification",
                "project_start_contract": "use the requested workspace and continue until completion evidence",
            },
        )
        await commitment_store.save(commitment)
        runtime = SimpleNamespace(
            commitment_store=commitment_store,
            schedule_service=schedule_service,
            ctx=SimpleNamespace(
                config=SimpleNamespace(session_id="session-1", state_dir=tmp_path),
            ),
        )
        baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)

        async def _partial_success(task):
            return RepairResult(
                task_id=task.task_id,
                success=True,
                stage=ExecutionStage.DONE,
                output="Implemented the next slice and left the project incomplete.",
                artifacts=["workspace/projects/glass-compiler/src/main.py"],
            )

        baa.executor.run = _partial_success
        task = RepairTask(
            objective="Advance the linked project.",
            commitment_id=str(commitment.commitment_id),
            meta={
                "source": "schedule",
                "project_key": "glass-compiler",
                "project_title": "Glass Compiler",
                "project_type": "software",
            },
        )

        await baa._run_bounded(task)

        updated = await commitment_store.get(str(commitment.commitment_id))
        assert updated is not None
        assert updated.status == CommitmentStatus.ACTIVE
        assert updated.meta["resume_policy"] == "auto_until_completion_evidence"
        assert updated.meta["last_successful_continuation_task_id"] == str(task.task_id)
        assert len(schedule_service.created) == 1
        created = schedule_service.created[0]
        assert created["commitment_id"] == str(commitment.commitment_id)
        assert "project_return" in created["tags"]
        assert created["meta"]["source"] == "project_followthrough_success_continuation"
        assert created["meta"]["source_session_id"] == "session-project-2"
        assert created["meta"]["requested_workspace_rel_path"] == "workspace/software"
        assert created["meta"]["return_policy"] == "immediate_autonomous_work"
        assert created["meta"]["project_start_contract"].startswith("use the requested workspace")
        assert created["meta"]["previous_task_id"] == str(task.task_id)
        assert "Continue" in created["title"]
        assert "Only stop" in created["objective"]
    finally:
        await commitment_store.close()


@pytest.mark.asyncio
async def test_authorized_retry_exhausted_project_return_schedules_recovery_instead_of_blocking(store, tmp_path):
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    schedule_service = _ScheduleService()
    contacts = []

    async def _contact_owner(**kwargs):
        contacts.append(kwargs)
        return {"status": "sent", "channel": "telegram"}

    try:
        commitment = Commitment(
            content="Return to project: North Star Draft",
            tags=["project_return", "self_directed"],
            meta={
                "source": "project_return_capture",
                "source_session_id": "session-retry-exhausted",
                "project_key": "north-star-draft",
                "project_title": "North Star Draft",
                "project_type": "writing",
                "workspace_rel_path": "workspace/projects/north-star-draft",
                "workspace_project_confidence": 0.9,
                "project_intent": "finish the operator-requested draft.",
                "next_step": "resume from the last saved section",
            },
        )
        await commitment_store.save(commitment)
        runtime = SimpleNamespace(
            commitment_store=commitment_store,
            schedule_service=schedule_service,
            initiative_contact_owner=_contact_owner,
            ctx=SimpleNamespace(config=SimpleNamespace(session_id="session-1", state_dir=tmp_path)),
        )
        baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)

        async def _retry_exhausted(task):
            return RepairResult(
                task_id=task.task_id,
                success=False,
                stage=ExecutionStage.FAILED,
                output="Recovery cap exceeded after 3 retries.",
                artifacts=["workspace/projects/north-star-draft/notes.md"],
            )

        baa.executor.run = _retry_exhausted
        task = RepairTask(
            objective="Continue the linked project until completion evidence is available.",
            commitment_id=str(commitment.commitment_id),
            meta={"source": "schedule", "project_key": "north-star-draft", "authority": "schedule_due"},
        )

        await baa._run_bounded(task)

        updated = await commitment_store.get(str(commitment.commitment_id))
        assert updated is not None
        assert updated.status == CommitmentStatus.ACTIVE
        assert "blocked_reason" not in updated.meta
        assert updated.meta["resume_policy"] == "auto_on_recoverable_task_blocker"
        assert contacts == []
        assert len(schedule_service.created) == 1
        created = schedule_service.created[0]
        assert created["meta"]["source"] == "project_followthrough_recovery"
        assert created["meta"]["source_session_id"] == "session-retry-exhausted"
        assert created["meta"]["workspace_rel_path"] == "workspace/projects/north-star-draft"
        assert "project_return" in created["tags"]
    finally:
        await commitment_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_reason,expected_marker",
    [
        ("Recovery cap exceeded after 3 retries: credential required for provider access.", "credential required"),
        ("Recovery cap exceeded after 3 retries: safety policy requires operator review.", "safety policy"),
        ("Recovery cap exceeded after 3 retries: missing required user input.", "missing required user input"),
    ],
)
async def test_authorized_project_return_external_attention_failures_still_block(
    store,
    tmp_path,
    failure_reason,
    expected_marker,
):
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    schedule_service = _ScheduleService()
    contacts = []

    async def _contact_owner(**kwargs):
        contacts.append(kwargs)
        return {"status": "sent", "channel": "telegram"}

    try:
        commitment = Commitment(
            content="Return to project: Credentialed Build",
            tags=["project_return", "self_directed"],
            meta={
                "source": "project_return_capture",
                "source_session_id": "session-credential",
                "project_key": "credentialed-build",
                "project_title": "Credentialed Build",
                "workspace_rel_path": "workspace/projects/credentialed-build",
                "workspace_project_confidence": 0.9,
            },
        )
        await commitment_store.save(commitment)
        runtime = SimpleNamespace(
            commitment_store=commitment_store,
            schedule_service=schedule_service,
            initiative_contact_owner=_contact_owner,
            ctx=SimpleNamespace(config=SimpleNamespace(session_id="session-1", state_dir=tmp_path)),
        )
        baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)

        async def _external_attention_failure(task):
            return RepairResult(
                task_id=task.task_id,
                success=False,
                stage=ExecutionStage.FAILED,
                output=failure_reason,
                artifacts=[],
            )

        baa.executor.run = _external_attention_failure
        task = RepairTask(
            objective="Continue the linked project.",
            commitment_id=str(commitment.commitment_id),
            meta={"source": "schedule", "authority": "schedule_due"},
        )

        await baa._run_bounded(task)

        updated = await commitment_store.get(str(commitment.commitment_id))
        assert updated is not None
        assert updated.status == CommitmentStatus.BLOCKED
        assert expected_marker in updated.meta["blocked_reason"]
        assert schedule_service.created == []
        assert contacts
    finally:
        await commitment_store.close()


@pytest.mark.asyncio
async def test_legacy_blocked_authorized_project_return_reconciles_recovery_schedule(store, tmp_path):
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    schedule_service = _ScheduleService()

    try:
        task = RepairTask(
            objective="Continue the old project return.",
            commitment_id="placeholder",
            meta={"source": "schedule", "authority": "schedule_due"},
            artifacts=["workspace/projects/legacy-return/draft.md"],
        )
        await store.save(task)
        await store.save_result(
            RepairResult(
                task_id=task.task_id,
                success=False,
                stage=ExecutionStage.FAILED,
                output="Execution failed after 3 attempts.",
                artifacts=task.artifacts,
            )
        )
        commitment = Commitment(
            content="Return to project: Legacy Return",
            status=CommitmentStatus.BLOCKED,
            linked_task_ids=[str(task.task_id)],
            tags=["project_return", "self_directed"],
            meta={
                "source": "project_return_capture",
                "source_session_id": "legacy-session-1",
                "project_key": "legacy-return",
                "project_title": "Legacy Return",
                "project_type": "writing",
                "workspace_rel_path": "workspace/projects/legacy-return",
                "workspace_project_confidence": 0.9,
                "requested_workspace_rel_path": "workspace/projects",
                "requested_workspace_kind": "parent",
                "start_policy": "immediate_operator_work_request",
                "return_policy": "immediate_autonomous_work",
                "project_intent": "finish the legacy project return without another owner prompt.",
                "next_step": "resume from the last saved draft",
                "blocked_reason": "Execution failed after 3 attempts.",
                "blocked_task_id": str(task.task_id),
            },
        )
        task.commitment_id = str(commitment.commitment_id)
        await store.save(task)
        await commitment_store.save(commitment)
        runtime = SimpleNamespace(
            commitment_store=commitment_store,
            schedule_service=schedule_service,
            ctx=SimpleNamespace(config=SimpleNamespace(session_id="session-1", state_dir=tmp_path)),
        )
        baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)

        repaired = await baa.reconcile_legacy_followthrough_commitments()

        assert repaired == 1
        updated = await commitment_store.get(str(commitment.commitment_id))
        assert updated is not None
        assert updated.status == CommitmentStatus.ACTIVE
        assert updated.meta["resume_policy"] == "auto_on_recoverable_task_blocker"
        assert updated.meta["previous_blocked_reason"] == "Execution failed after 3 attempts."
        assert "blocked_reason" not in updated.meta
        assert len(schedule_service.created) == 1
        created = schedule_service.created[0]
        assert created["commitment_id"] == str(commitment.commitment_id)
        assert "project_return" in created["tags"]
        assert created["meta"]["source"] == "project_followthrough_legacy_recovery"
        assert created["meta"]["commitment_id"] == str(commitment.commitment_id)
        assert created["meta"]["source_session_id"] == "legacy-session-1"
        assert created["meta"]["workspace_rel_path"] == "workspace/projects/legacy-return"
        assert created["meta"]["requested_workspace_rel_path"] == "workspace/projects"
        assert created["meta"]["requested_workspace_kind"] == "parent"
        assert created["meta"]["start_policy"] == "immediate_operator_work_request"
        assert created["meta"]["return_policy"] == "immediate_autonomous_work"
        assert created["meta"]["project_intent"].startswith("finish the legacy project return")
        assert created["meta"]["next_step"] == "resume from the last saved draft"
        assert created["meta"]["previous_task_id"] == str(task.task_id)
        assert created["meta"]["recovery_from_task_id"] == str(task.task_id)
        assert "narrower" in created["objective"].lower() or "narrow" in created["objective"].lower()
    finally:
        await commitment_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked_reason",
    [
        "Execution failed after 3 attempts: credential required for provider access.",
        "Execution failed after 3 attempts: safety policy requires operator review.",
        "Execution failed after 3 attempts: missing required user input.",
        "Execution failed after 3 attempts: permission denied reading workspace.",
        "Execution failed after 3 attempts: workspace inaccessible.",
    ],
)
async def test_legacy_external_blocker_remains_blocked_without_recovery_schedule(store, tmp_path, blocked_reason):
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    schedule_service = _ScheduleService()

    try:
        commitment = Commitment(
            content="Return to project: External Blocker",
            status=CommitmentStatus.BLOCKED,
            tags=["project_return", "self_directed"],
            meta={
                "source": "project_return_capture",
                "source_session_id": "legacy-session-2",
                "project_title": "External Blocker",
                "workspace_rel_path": "workspace/projects/external-blocker",
                "workspace_project_confidence": 0.9,
                "blocked_reason": blocked_reason,
            },
        )
        await commitment_store.save(commitment)
        runtime = SimpleNamespace(
            commitment_store=commitment_store,
            schedule_service=schedule_service,
            ctx=SimpleNamespace(config=SimpleNamespace(session_id="session-1", state_dir=tmp_path)),
        )
        baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)

        repaired = await baa.reconcile_legacy_followthrough_commitments()

        assert repaired == 0
        updated = await commitment_store.get(str(commitment.commitment_id))
        assert updated is not None
        assert updated.status == CommitmentStatus.BLOCKED
        assert updated.meta["blocked_reason"] == blocked_reason
        assert schedule_service.created == []
    finally:
        await commitment_store.close()


@pytest.mark.asyncio
async def test_legacy_authorized_recovery_schedule_prevents_duplicate(store, tmp_path):
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    schedule_service = _ScheduleService()

    try:
        commitment = Commitment(
            content="Return to project: Duplicate Guard",
            status=CommitmentStatus.BLOCKED,
            tags=["project_return", "self_directed"],
            meta={
                "source": "project_return_capture",
                "source_session_id": "legacy-session-3",
                "project_title": "Duplicate Guard",
                "workspace_rel_path": "workspace/projects/duplicate-guard",
                "workspace_project_confidence": 0.9,
                "blocked_reason": "Execution failed after 3 attempts.",
            },
        )
        await commitment_store.save(commitment)
        schedule_service.items.append(
            SimpleNamespace(
                schedule_id="existing-schedule",
                status=ScheduleStatus.ACTIVE,
                action=ScheduleAction.SUBMIT_BAA,
                commitment_id=str(commitment.commitment_id),
                meta={
                    "source": "project_followthrough_legacy_recovery",
                    "source_session_id": "legacy-session-3",
                    "workspace_rel_path": "workspace/projects/duplicate-guard",
                },
            )
        )
        runtime = SimpleNamespace(
            commitment_store=commitment_store,
            schedule_service=schedule_service,
            ctx=SimpleNamespace(config=SimpleNamespace(session_id="session-1", state_dir=tmp_path)),
        )
        baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)

        repaired = await baa.reconcile_legacy_followthrough_commitments()

        assert repaired == 0
        updated = await commitment_store.get(str(commitment.commitment_id))
        assert updated is not None
        assert updated.status == CommitmentStatus.BLOCKED
        assert schedule_service.created == []
    finally:
        await commitment_store.close()


@pytest.mark.asyncio
async def test_successful_baa_task_records_agent_visible_context_and_memory(store, tmp_path):
    context_store = _ContextStore()
    memory_store = _MemoryStore()
    runtime = SimpleNamespace(
        memory=memory_store,
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="session-baa", state_dir=tmp_path),
            context_store=context_store,
        ),
    )
    baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)

    async def _successful(task):
        return RepairResult(
            task_id=task.task_id,
            success=True,
            stage=ExecutionStage.DONE,
            output="Researched the local dashboard failure and wrote the summary.",
            artifacts=["workspace/research/dashboard-summary.md"],
        )

    baa.executor.run = _successful
    task = RepairTask(objective="Research the local dashboard failure")

    await baa._run_bounded(task)

    started = [entry for entry in context_store.entries if entry["meta"].get("status") == "started"]
    assert started
    assert "Progress update: I started this background task and am continuing the work." in started[-1]["content"]

    completed = [entry for entry in context_store.entries if entry["meta"].get("status") == "completed"]
    assert completed
    assert completed[-1]["session_id"] == "session-baa"
    assert completed[-1]["role"] == MessageRole.SYSTEM
    assert completed[-1]["meta"]["event_kind"] == "baa_task_activity"
    assert completed[-1]["meta"]["source"] == "baa"
    assert completed[-1]["meta"]["source_id"] == str(task.task_id)
    assert "Research the local dashboard failure" in completed[-1]["content"]
    assert "Progress update: I completed this background task. I have 1 artifact recorded." in completed[-1]["content"]
    assert "dashboard-summary.md" in completed[-1]["content"]

    completed_episodes = [
        episode
        for episode in memory_store.episodes
        if episode.payload.get("status") == "completed"
    ]
    assert completed_episodes
    assert completed_episodes[-1].kind == EpisodeKind.OBSERVATION
    assert completed_episodes[-1].session_id == "session-baa"
    assert completed_episodes[-1].payload["payload"]["task_id"] == str(task.task_id)


@pytest.mark.asyncio
async def test_failed_baa_task_records_agent_visible_context_and_memory(store, tmp_path):
    context_store = _ContextStore()
    memory_store = _MemoryStore()
    runtime = SimpleNamespace(
        memory=memory_store,
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="session-baa", state_dir=tmp_path),
            context_store=context_store,
        ),
    )
    baa = BoundedAssistantAgent(tools=ToolRegistry(), store=store, runtime=runtime, max_concurrent=1)

    async def _failed(task):
        return RepairResult(
            task_id=task.task_id,
            success=False,
            stage=ExecutionStage.FAILED,
            output="Background research failed because the provider returned 429.",
            artifacts=[],
        )

    baa.executor.run = _failed
    task = RepairTask(objective="Research provider outage")

    await baa._run_bounded(task)

    failed = [entry for entry in context_store.entries if entry["meta"].get("status") == "failed"]
    assert failed
    assert failed[-1]["role"] == MessageRole.SYSTEM
    assert failed[-1]["meta"]["event_kind"] == "baa_task_activity"
    assert failed[-1]["meta"]["source_id"] == str(task.task_id)
    assert "Research provider outage" in failed[-1]["content"]
    assert "Progress update: I hit a real blocker on this background task" in failed[-1]["content"]
    assert "provider returned 429" in failed[-1]["content"]

    failed_episodes = [
        episode
        for episode in memory_store.episodes
        if episode.payload.get("status") == "failed"
    ]
    assert failed_episodes
    assert failed_episodes[-1].kind == EpisodeKind.OBSERVATION
    assert failed_episodes[-1].payload["payload"]["task_id"] == str(task.task_id)
