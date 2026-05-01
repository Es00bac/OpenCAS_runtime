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
    loop = await harness.create_objective_loop(
        title="Cycle Test",
        expected_output="A concise answer describing the next concrete step.",
        success_check="The answer names a specific next action.",
        stop_condition="Stop after one focused cycle.",
    )
    result = await harness.run_objective_cycle(max_active_loops=3)
    assert result["loops_processed"] == 1

    fetched = await harness.store.get_loop(str(loop.loop_id))
    assert fetched is not None
    assert fetched.status == ObjectiveStatus.ACTIVE


@pytest.mark.asyncio
async def test_run_objective_cycle_pauses_contractless_loop_before_processing(harness_store):
    h = AgenticHarness(store=harness_store)
    loop = ObjectiveLoop(
        title="Loop Without Contract",
        description="This loop should not consume planning work.",
        status=ObjectiveStatus.ACTIVE,
    )
    await harness_store.save_loop(loop)

    async def fail_if_planned(*args, **kwargs):
        raise AssertionError("contractless loop should be parked before planning")

    h._generate_loop_plan = fail_if_planned

    result = await h.run_objective_cycle(max_active_loops=3)

    assert result["loops_processed"] == 0
    assert result["parked_loops"] == [str(loop.loop_id)]
    fetched = await harness_store.get_loop(str(loop.loop_id))
    assert fetched is not None
    assert fetched.status == ObjectiveStatus.PAUSED
    assert fetched.meta["paused_reason"] == "missing_objective_contract"
    assert set(fetched.meta["missing_contract_fields"]) == {
        "goal",
        "expected_output",
        "success_check",
        "stop_condition",
    }
    assert "reframe" in fetched.meta["reframe_hint"].lower()


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

    await h.create_objective_loop(title="Pending")
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
    loop = await h.create_objective_loop(
        title="Meta Test",
        expected_output="A focused repair task with one expected artifact.",
        success_check="The repair task has a bounded objective.",
        stop_condition="Stop after one submitted task.",
    )

    result = await h.run_objective_cycle(max_active_loops=3)
    assert result["loops_processed"] == 1

    assert len(baa.submitted) == 1
    task = baa.submitted[0]
    assert task.meta.get("harness_origin") == "objective_loop"
    assert task.meta.get("loop_id") == str(loop.loop_id)
    assert task.meta["objective_contract"]["expected_output"] == "A focused repair task with one expected artifact."


@pytest.mark.asyncio
async def test_harness_drafts_missing_objective_contract_with_llm(harness_store):
    class FakeBAA:
        def __init__(self):
            self.submitted = []

        async def submit(self, task):
            self.submitted.append(task)

    class FakeLLM:
        async def chat_completion(self, messages, **kwargs):
            if kwargs.get("source") == "harness_contract_drafting":
                return {
                    "choices": [
                        {
                            "message": {
                                "content": """
                                {
                                  "goal": "Revise Chronicle 4246 as the OpenCAS agent's own manuscript project.",
                                  "expected_output": "A concrete manuscript revision saved in the workspace.",
                                  "success_check": "The target chapter file contains new or revised prose.",
                                  "stop_condition": "Stop after one verified manuscript edit or a clear blocker.",
                                  "max_attempt_budget": 2,
                                  "reframe_path": "If no chapter can be chosen, inspect the manuscript index first."
                                }
                                """
                            }
                        }
                    ]
                }
            return {"choices": [{"message": {"content": "Revise one Chronicle chapter and verify the file changed."}}]}

    baa = FakeBAA()
    h = AgenticHarness(store=harness_store, llm=FakeLLM(), baa=baa)
    loop = await h.create_objective_loop(
        title="Continue Chronicle 4246",
        description="Return to the manuscript and keep improving it.",
    )

    result = await h.run_objective_cycle(max_active_loops=1)

    assert result["parked_loops"] == []
    assert result["loops_processed"] == 1
    assert len(baa.submitted) == 1
    task = baa.submitted[0]
    assert task.meta["objective_contract"]["expected_output"] == (
        "A concrete manuscript revision saved in the workspace."
    )
    fetched = await harness_store.get_loop(str(loop.loop_id))
    assert fetched.meta["objective_contract"]["success_check"] == (
        "The target chapter file contains new or revised prose."
    )


@pytest.mark.asyncio
async def test_harness_injects_objective_contract_into_loop_plan_prompt(harness_store):
    llm_calls = []

    class FakeLLM:
        async def chat_completion(self, messages, **kwargs):
            llm_calls.append({"messages": messages, "kwargs": kwargs})
            return {"choices": [{"message": {"content": "Write one findings note from the current logs."}}]}

    h = AgenticHarness(store=harness_store, llm=FakeLLM())
    loop = await h.create_objective_loop(
        title="Investigate approval loop stalls",
        description="Find why approval handling keeps looping without useful output.",
        expected_output="A findings note at workspace/reports/approval-loop.md.",
        success_check="The note names one blocker and one next action.",
        stop_condition="Stop after one focused evidence-gathering cycle if no new logs appear.",
        max_attempt_budget=2,
        reframe_path="Ask the operator for the missing log path.",
    )

    plan = await h._generate_loop_plan(loop, None)

    assert plan == "Write one findings note from the current logs."
    prompt = llm_calls[0]["messages"][1]["content"]
    assert "Objective contract:" in prompt
    assert "Expected output: A findings note at workspace/reports/approval-loop.md." in prompt
    assert "Success check: The note names one blocker and one next action." in prompt
    assert "Stop condition: Stop after one focused evidence-gathering cycle if no new logs appear." in prompt
    assert "Max attempt budget: 2" in prompt
    assert "Reframe path: Ask the operator for the missing log path." in prompt


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

@pytest.mark.asyncio
async def test_harness_switches_to_resume_mode_when_latest_salvage_blocks_broad_retry(harness_store):
    class MockResumeSnapshot:
        def __init__(self):
            self.retry_state = "blocked_low_divergence"
            self.signature = "chronicle-4246"
            self.best_next_step = "fix the dialogue tags"
            self.has_live_workstream = False
        def to_meta(self):
            return {
                "signature": self.signature,
                "retry_state": self.retry_state,
                "best_next_step": self.best_next_step,
            }

    class MockResumeResolver:
        async def suppress_duplicate_active_objective_loops(self): pass
        async def resolve(self, query: str):
            return MockResumeSnapshot()

    harness = AgenticHarness(
        store=harness_store,
        project_resume_resolver=MockResumeResolver(),
    )

    # If the harness tries to generate a plan, it should fail
    async def mock_generate_loop_plan(*args, **kwargs):
        raise AssertionError("Should not generate a loop plan")
    harness._generate_loop_plan = mock_generate_loop_plan

    loop = ObjectiveLoop(
        title="Continue Chronicle 4246",
        description="Write the next chapter.",
        status=ObjectiveStatus.ACTIVE,
        meta={
            "objective_contract": {
                "goal": "Continue Chronicle 4246",
                "expected_output": "A deterministic resume decision for the existing manuscript.",
                "success_check": "The loop switches to salvage resume instead of broad replanning.",
                "stop_condition": "Stop immediately when the latest salvage packet blocks broad retry.",
            }
        },
    )
    await harness_store.save_loop(loop)

    result = await harness.run_objective_cycle(max_active_loops=1)

    assert result["loops_processed"] == 1
    assert result["submitted_tasks"] == []
    assert result["created_work_objects"] == []
    resumed = await harness.store.get_loop(str(loop.loop_id))
    assert resumed.meta["resume_project"]["best_next_step"] == "fix the dialogue tags"


@pytest.mark.asyncio
async def test_harness_injects_shadow_registry_guidance_into_loop_plan_prompt(harness_store):
    llm_calls = []
    shadow_calls = []

    class FakeLLM:
        async def chat_completion(self, messages, **kwargs):
            llm_calls.append({"messages": messages, "kwargs": kwargs})
            return {"choices": [{"message": {"content": "Make one narrow revision to the chronicle draft."}}]}

    class FakeShadowRegistry:
        def build_planning_context(self, **kwargs):
            shadow_calls.append(kwargs)
            return {
                "available": True,
                "prompt_block": (
                    "Related blocked-intention clusters:\n"
                    "- 2x retry_blocked around retry:workspace/Chronicles/4246/chronicle_4246.md\n"
                    "Safer alternatives:\n"
                    "- Prefer deterministic review of the existing artifact rather than broad replanning."
                ),
            }

    harness = AgenticHarness(
        store=harness_store,
        llm=FakeLLM(),
        shadow_registry=FakeShadowRegistry(),
    )
    loop = ObjectiveLoop(
        title="Continue Chronicle 4246",
        description="Write the next chapter from the existing manuscript.",
        status=ObjectiveStatus.ACTIVE,
        meta={
            "resume_project": {
                "canonical_artifact_path": "workspace/Chronicles/4246/chronicle_4246.md",
            }
        },
    )

    plan = await harness._generate_loop_plan(loop, None)

    assert plan == "Make one narrow revision to the chronicle draft."
    assert shadow_calls == [
        {
            "objective": "Continue Chronicle 4246",
            "artifact": "workspace/Chronicles/4246/chronicle_4246.md",
        }
    ]
    prompt = llm_calls[0]["messages"][1]["content"]
    assert "Related blocked-intention clusters:" in prompt
    assert "Prefer deterministic review of the existing artifact" in prompt
