"""Tests for the agentic harness and research notebook layer."""

import pytest
import pytest_asyncio

from opencas.harness import (
    AgenticHarness,
    HarnessStore,
    NotebookEntryKind,
    ObjectiveLoop,
    ObjectiveStatus,
    ResearchNotebook,
)


@pytest_asyncio.fixture
async def harness_store(tmp_path):
    store = HarnessStore(tmp_path / "harness.db")
    await store.connect()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def harness(harness_store):
    h = AgenticHarness(store=harness_store)
    yield h


@pytest.mark.asyncio
async def test_create_notebook(harness):
    notebook = await harness.create_notebook(
        title="Test Notebook",
        description="A test notebook",
    )
    assert isinstance(notebook, ResearchNotebook)
    assert notebook.title == "Test Notebook"
    assert notebook.description == "A test notebook"

    fetched = await harness.store.get_notebook(str(notebook.notebook_id))
    assert fetched is not None
    assert fetched.title == "Test Notebook"


@pytest.mark.asyncio
async def test_add_notebook_entry(harness):
    notebook = await harness.create_notebook(title="Entry Test")
    entry = await harness.add_notebook_entry(
        notebook_id=str(notebook.notebook_id),
        kind=NotebookEntryKind.OBSERVATION,
        content="Something observed",
    )
    assert entry is not None
    assert entry.kind == NotebookEntryKind.OBSERVATION
    assert entry.content == "Something observed"

    fetched = await harness.store.get_notebook(str(notebook.notebook_id))
    assert fetched is not None
    assert len(fetched.entries) == 1
    assert fetched.entries[0].content == "Something observed"


@pytest.mark.asyncio
async def test_create_objective_loop(harness):
    loop = await harness.create_objective_loop(
        title="Test Loop",
        description="A test objective",
        completion_criteria=["criterion one"],
    )
    assert isinstance(loop, ObjectiveLoop)
    assert loop.title == "Test Loop"
    assert loop.status == ObjectiveStatus.PENDING
    assert loop.completion_criteria == ["criterion one"]

    fetched = await harness.store.get_loop(str(loop.loop_id))
    assert fetched is not None
    assert fetched.title == "Test Loop"


@pytest.mark.asyncio
async def test_run_objective_cycle_promotes_pending(harness):
    loop = await harness.create_objective_loop(title="Cycle Test")
    result = await harness.run_objective_cycle(max_active_loops=3)
    assert result["loops_processed"] == 1

    fetched = await harness.store.get_loop(str(loop.loop_id))
    assert fetched is not None
    assert fetched.status == ObjectiveStatus.ACTIVE


@pytest.mark.asyncio
async def test_complete_loop(harness):
    loop = await harness.create_objective_loop(title="Complete Test")
    completed = await harness.complete_loop(str(loop.loop_id), success=True)
    assert completed is not None
    assert completed.status == ObjectiveStatus.COMPLETED

    failed = await harness.complete_loop(str(loop.loop_id), success=False)
    # Re-fetch should show failed
    assert failed is not None
    assert failed.status == ObjectiveStatus.FAILED


@pytest.mark.asyncio
async def test_list_loops_by_status(harness_store):
    h = AgenticHarness(store=harness_store)
    active = await h.create_objective_loop(title="Active")
    active.status = ObjectiveStatus.ACTIVE
    await h.store.save_loop(active)

    pending = await h.create_objective_loop(title="Pending")
    completed = await h.create_objective_loop(title="Completed")
    completed.status = ObjectiveStatus.COMPLETED
    await h.store.save_loop(completed)

    actives = await h.store.list_loops(status=ObjectiveStatus.ACTIVE)
    assert len(actives) == 1
    assert actives[0].title == "Active"

    pendings = await h.store.list_loops(status=ObjectiveStatus.PENDING)
    assert len(pendings) == 1
    assert pendings[0].title == "Pending"

    all_loops = await h.store.list_loops()
    assert len(all_loops) == 3


@pytest.mark.asyncio
async def test_harness_generates_repair_task_meta(harness_store):
    class FakeBAA:
        def __init__(self):
            self.submitted = []

        async def submit(self, task):
            self.submitted.append(task)

    baa = FakeBAA()
    h = AgenticHarness(store=harness_store, baa=baa)
    loop = await h.create_objective_loop(title="Meta Test")

    result = await h.run_objective_cycle(max_active_loops=3)
    assert result["loops_processed"] == 1

    assert len(baa.submitted) == 1
    task = baa.submitted[0]
    assert task.meta.get("harness_origin") == "objective_loop"
    assert task.meta.get("loop_id") == str(loop.loop_id)


@pytest.mark.asyncio
async def test_notebook_with_deliverable_schema(harness):
    from opencas.harness import DeliverableSchema

    schema = DeliverableSchema(
        name="test_schema",
        acceptance_criteria=["criterion"],
        expected_artifacts=["artifact.txt"],
    )
    notebook = await harness.create_notebook(
        title="Schema Test",
        deliverable_schema=schema,
    )
    fetched = await harness.store.get_notebook(str(notebook.notebook_id))
    assert fetched is not None
    assert fetched.deliverable_schema is not None
    assert fetched.deliverable_schema.name == "test_schema"
    assert fetched.deliverable_schema.acceptance_criteria == ["criterion"]
