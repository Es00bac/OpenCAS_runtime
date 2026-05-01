"""Tests for project continuation resolution and duplicate-start suppression."""

from pathlib import Path

import pytest
import pytest_asyncio

from opencas.autonomy import WorkObject, WorkStage
from opencas.autonomy.work_store import WorkStore
from opencas.context import ContextBuilder, MemoryRetriever, SessionContextStore
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.execution import RepairTask
from opencas.execution.models import AttemptOutcome
from opencas.execution.salvage import build_salvage_packet
from opencas.execution.store import TaskStore
from opencas.harness import AgenticHarness, HarnessStore, ObjectiveStatus
from opencas.harness.models import ObjectiveLoop
from opencas.identity import IdentityManager, IdentityStore
from opencas.memory import Episode, EpisodeKind, MemoryStore
from opencas.planning.store import PlanStore


@pytest_asyncio.fixture
async def resume_stores(tmp_path: Path):
    memory = MemoryStore(tmp_path / "memory.db")
    await memory.connect()

    work = WorkStore(tmp_path / "work.db")
    await work.connect()

    plans = PlanStore(tmp_path / "plans.db")
    await plans.connect()

    harness = HarnessStore(tmp_path / "harness.db")
    await harness.connect()

    yield {
        "memory": memory,
        "work": work,
        "plans": plans,
        "harness": harness,
        "tmp_path": tmp_path,
    }

    await harness.close()
    await plans.close()
    await work.close()
    await memory.close()


@pytest.mark.asyncio
async def test_project_resume_resolver_prefers_canonical_manuscript_path(resume_stores):
    from opencas.autonomy.project_resume import ProjectResumeResolver

    memory = resume_stores["memory"]
    work = resume_stores["work"]
    plans = resume_stores["plans"]
    harness = resume_stores["harness"]

    await memory.save_episode(
        Episode(
            kind=EpisodeKind.OBSERVATION,
            content=(
                "Artifact memory from workspace/Chronicles/4246/chronicle_4246.md\n"
                "Title: Chronicle 4246\nChunk 1 of 1"
            ),
            payload={
                "payload": {
                    "artifact": {
                        "path": "workspace/Chronicles/4246/chronicle_4246.md",
                        "title": "Chronicle 4246",
                    }
                }
            },
        )
    )
    await memory.save_episode(
        Episode(
            kind=EpisodeKind.OBSERVATION,
            content=(
                "Artifact memory from workspace/review/chronicle_4246_project_review.md\n"
                "Title: Chronicle 4246 — Project State Review Summary\nChunk 1 of 1"
            ),
            payload={
                "payload": {
                    "artifact": {
                        "path": "workspace/review/chronicle_4246_project_review.md",
                        "title": "Chronicle 4246 — Project State Review Summary",
                    }
                }
            },
        )
    )

    primary_loop = ObjectiveLoop(
        title="Write Chronicle 4246 as a 12 chapter novel and compile chronicle_4246.md.",
        description="Write Chronicle 4246 as a 12 chapter novel and compile chronicle_4246.md.",
        status=ObjectiveStatus.ACTIVE,
        generated_task_ids=["task-1", "task-2"],
    )
    await harness.save_loop(primary_loop)

    await work.save(
        WorkObject(
            stage=WorkStage.PROJECT,
            content="Continue Chronicle 4246 manuscript from the existing compiled draft.",
            project_id=str(primary_loop.loop_id),
            meta={"loop_id": str(primary_loop.loop_id)},
        )
    )

    plan = await plans.create_plan(
        "plan-chronicle",
        content="Continue Chronicle 4246 from chronicle_4246.md rather than restarting.",
        project_id=str(primary_loop.loop_id),
    )
    await plans.set_status(plan.plan_id, "active")

    resolver = ProjectResumeResolver(
        memory=memory,
        work_store=work,
        plan_store=plans,
        harness_store=harness,
    )

    snapshot = await resolver.resolve(
        "Should I continue Chronicle 4246 instead of starting over?"
    )

    assert snapshot is not None
    assert snapshot.canonical_artifact_path == "workspace/Chronicles/4246/chronicle_4246.md"
    assert snapshot.active_work_count == 1
    assert snapshot.active_plan_count == 1
    assert snapshot.primary_loop_id == str(primary_loop.loop_id)


@pytest.mark.asyncio
async def test_context_builder_includes_project_resume_guidance(resume_stores):
    from opencas.autonomy.project_resume import ProjectResumeResolver

    memory = resume_stores["memory"]
    work = resume_stores["work"]
    plans = resume_stores["plans"]
    harness = resume_stores["harness"]
    tmp_path = resume_stores["tmp_path"]

    await memory.save_episode(
        Episode(
            kind=EpisodeKind.OBSERVATION,
            content=(
                "Artifact memory from workspace/Chronicles/4246/chronicle_4246.md\n"
                "Title: Chronicle 4246\nChunk 1 of 1"
            ),
            payload={
                "payload": {
                    "artifact": {
                        "path": "workspace/Chronicles/4246/chronicle_4246.md",
                        "title": "Chronicle 4246",
                    }
                }
            },
        )
    )
    await work.save(
        WorkObject(
            stage=WorkStage.PROJECT,
            content="Continue Chronicle 4246 manuscript from the current draft.",
            project_id="loop-chronicle",
        )
    )
    plan = await plans.create_plan(
        "plan-chronicle",
        content="Continue Chronicle 4246 from the current manuscript.",
        project_id="loop-chronicle",
    )
    await plans.set_status(plan.plan_id, "active")

    ctx_store = SessionContextStore(tmp_path / "context.db")
    await ctx_store.connect()

    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    retriever = MemoryRetriever(memory=memory, embeddings=embeddings)

    identity = IdentityManager(IdentityStore(tmp_path / "identity"))
    identity.load()

    resolver = ProjectResumeResolver(
        memory=memory,
        work_store=work,
        plan_store=plans,
        harness_store=harness,
    )
    builder = ContextBuilder(
        store=ctx_store,
        retriever=retriever,
        identity=identity,
        project_resume_resolver=resolver,
    )

    manifest = await builder.build(
        "Please continue Chronicle 4246 instead of starting over.",
        session_id="s1",
    )

    assert "Project continuation evidence" in manifest.system.content
    assert "workspace/Chronicles/4246/chronicle_4246.md" in manifest.system.content
    assert "continue the existing project" in manifest.system.content.lower()
    assert "this is your own active creative work" in manifest.system.content
    assert "do not substitute research, naming, cataloging" in manifest.system.content
    assert "Do not default to tomorrow when you believe sooner is right" in manifest.system.content

    await cache.close()
    await ctx_store.close()


@pytest.mark.asyncio
async def test_harness_suppresses_duplicate_objective_loops_and_reuses_existing_workstream(
    resume_stores,
):
    from opencas.autonomy.project_resume import ProjectResumeResolver

    work = resume_stores["work"]
    harness_store = resume_stores["harness"]

    primary = ObjectiveLoop(
        title="Write Chronicle 4246 as a 12 chapter novel and compile chronicle_4246.md.",
        description="Write Chronicle 4246 as a 12 chapter novel and compile chronicle_4246.md.",
        status=ObjectiveStatus.ACTIVE,
        generated_task_ids=["task-1", "task-2", "task-3"],
        meta={
            "objective_contract": {
                "goal": "Write Chronicle 4246 as a 12 chapter novel and compile chronicle_4246.md.",
                "expected_output": "One bounded continuation edit in workspace/Chronicles/4246/chronicle_4246.md.",
                "success_check": "The edit advances the existing manuscript without restarting.",
                "stop_condition": "Stop after one bounded edit or a concrete blocker.",
            }
        },
    )
    duplicate = ObjectiveLoop(
        title="Complete Chronicle 4246 as a real novel-length third book, include revision and critique, and compile chronicle_4246.md.",
        description="Complete Chronicle 4246 as a real novel-length third book, include revision and critique, and compile chronicle_4246.md.",
        status=ObjectiveStatus.ACTIVE,
    )
    await harness_store.save_loop(primary)
    await harness_store.save_loop(duplicate)

    await work.save(
        WorkObject(
            stage=WorkStage.PROJECT,
            content="Continue Chronicle 4246 from the existing compiled manuscript.",
            project_id=str(primary.loop_id),
            meta={"loop_id": str(primary.loop_id)},
        )
    )

    resolver = ProjectResumeResolver(
        memory=resume_stores["memory"],
        work_store=work,
        plan_store=resume_stores["plans"],
        harness_store=harness_store,
    )
    harness = AgenticHarness(
        store=harness_store,
        work_store=work,
        project_resume_resolver=resolver,
    )

    result = await harness.run_objective_cycle(max_active_loops=5)

    refreshed_primary = await harness_store.get_loop(str(primary.loop_id))
    refreshed_duplicate = await harness_store.get_loop(str(duplicate.loop_id))

    assert result["loops_processed"] == 1
    assert result["created_work_objects"] == []
    assert refreshed_primary is not None
    assert refreshed_primary.status == ObjectiveStatus.ACTIVE
    assert refreshed_duplicate is not None
    assert refreshed_duplicate.status == ObjectiveStatus.PAUSED
    assert refreshed_duplicate.meta["paused_reason"] == "duplicate_project_resume"
    assert refreshed_duplicate.meta["duplicate_of_loop_id"] == str(primary.loop_id)
    assert "reframe" in refreshed_duplicate.meta["reframe_hint"].lower()


@pytest.mark.asyncio
async def test_project_resume_lists_compact_project_ledger_entries(resume_stores):
    from opencas.autonomy.project_resume import ProjectResumeResolver

    memory = resume_stores["memory"]
    work = resume_stores["work"]
    plans = resume_stores["plans"]
    harness = resume_stores["harness"]

    await memory.save_episode(
        Episode(
            kind=EpisodeKind.OBSERVATION,
            content=(
                "Artifact memory from workspace/Chronicles/4246/chronicle_4246.md\n"
                "Title: Chronicle 4246\nChunk 1 of 1"
            ),
            payload={
                "payload": {
                    "artifact": {
                        "path": "workspace/Chronicles/4246/chronicle_4246.md",
                        "title": "Chronicle 4246",
                    }
                }
            },
        )
    )
    loop = ObjectiveLoop(
        title="Write Chronicle 4246 as a 12 chapter novel and compile chronicle_4246.md.",
        description="Write Chronicle 4246 as a 12 chapter novel and compile chronicle_4246.md.",
        status=ObjectiveStatus.ACTIVE,
        generated_task_ids=["task-1"],
    )
    await harness.save_loop(loop)
    await work.save(
        WorkObject(
            stage=WorkStage.PROJECT,
            content="Continue Chronicle 4246 from the existing compiled manuscript.",
            project_id=str(loop.loop_id),
            meta={"loop_id": str(loop.loop_id)},
        )
    )
    plan = await plans.create_plan(
        "plan-chronicle",
        content="Continue Chronicle 4246 from the current compiled manuscript instead of restarting.",
        project_id=str(loop.loop_id),
    )
    await plans.set_status(plan.plan_id, "active")

    resolver = ProjectResumeResolver(
        memory=memory,
        work_store=work,
        plan_store=plans,
        harness_store=harness,
    )

    entries = await resolver.list_projects(limit=10)

    assert len(entries) == 1
    assert entries[0].display_name == "Chronicle 4246"
    assert entries[0].canonical_artifact_path == "workspace/Chronicles/4246/chronicle_4246.md"
    assert entries[0].synopsis
    assert "existing compiled manuscript" in entries[0].synopsis.lower()
    assert "objective_loop" in entries[0].source_surfaces
    assert "plan" in entries[0].source_surfaces
    assert "work" in entries[0].source_surfaces


@pytest.mark.asyncio
async def test_project_resume_surfaces_retry_state_and_latest_salvage_packet(resume_stores):
    from opencas.autonomy.project_resume import ProjectResumeResolver

    tmp_path = resume_stores["tmp_path"]
    work = resume_stores["work"]
    harness_store = resume_stores["harness"]

    loop = ObjectiveLoop(
        title="Continue Chronicle 4246",
        description="Continue Chronicle 4246",
        status=ObjectiveStatus.ACTIVE,
        meta={
            "objective_contract": {
                "goal": "Continue Chronicle 4246",
                "expected_output": "A focused edit in workspace/Chronicles/4246/chronicle_4246.md.",
                "success_check": "The existing manuscript changes in a bounded way.",
                "stop_condition": "Stop if the next attempt lacks new evidence or artifact progress.",
            }
        },
    )
    await harness_store.save_loop(loop)
    await work.save(
        WorkObject(
            stage=WorkStage.PROJECT,
            content="Continue Chronicle 4246 from the existing manuscript.",
            project_id=str(loop.loop_id),
            meta={"loop_id": str(loop.loop_id)},
        )
    )

    sig = ProjectResumeResolver.project_signature(f"{loop.title} {loop.description}")
    assert sig is not None

    task_store = TaskStore(tmp_path / "tasks.db")
    await task_store.connect()

    task = RepairTask(
        objective="Continue Chronicle 4246 from the existing manuscript.",
        project_id=str(loop.loop_id),
        meta={"resume_project": {"signature": sig}},
    )
    task.attempt = 1
    packet = build_salvage_packet(
        task,
        outcome=AttemptOutcome.VERIFY_FAILED.value,
        canonical_artifact_path="workspace/Chronicles/4246/chronicle_4246.md",
        artifact_paths_touched=["workspace/Chronicles/4246/chronicle_4246.md"],
        tool_calls=[{"name": "fs_write_file"}],
    )
    await task_store.save_salvage_packet(packet)

    resolver = ProjectResumeResolver(
        memory=resume_stores["memory"],
        work_store=work,
        plan_store=resume_stores["plans"],
        harness_store=harness_store,
        salvage_store=task_store,
    )

    entries = await resolver.list_projects(limit=10)
    await task_store.close()

    assert len(entries) >= 1
    entry = entries[0]
    assert entry.retry_state == "blocked_low_divergence"
    assert entry.best_next_step
    assert entry.latest_salvage_packet_id == str(packet.packet_id)
    assert entry.last_salvage_outcome == AttemptOutcome.VERIFY_FAILED.value
    assert entry.latest_salvage_meaningful_progress_signal == packet.meaningful_progress_signal
    assert entry.objective_contract["expected_output"] == (
        "A focused edit in workspace/Chronicles/4246/chronicle_4246.md."
    )
    assert entry.to_meta()["objective_contract"]["stop_condition"] == (
        "Stop if the next attempt lacks new evidence or artifact progress."
    )
