"""Tests for chat-born project return capture."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.autonomy.commitment_store import CommitmentStore
from opencas.autonomy.completion_evidence import completion_evidence_rejection_reason
from opencas.autonomy.executive import ExecutiveState
from opencas.projects.execution_contracts import new_project_tool_block_reason
from opencas.runtime.project_return import (
    capture_project_return_from_turn,
    prepare_project_followthrough_response,
)
from opencas.scheduling import (
    ScheduleAction,
    ScheduleItem,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleService,
    ScheduleStatus,
    ScheduleStore,
)
from opencas.thread_registry import BeadSourceKind, ThreadRegistryService, ThreadRegistryStore


class _Config:
    def __init__(self, root):
        self._root = root

    def primary_workspace_root(self):
        return self._root

    def agent_workspace_root(self):
        return self._root / "workspace"


class _FakeBAA:
    def __init__(self) -> None:
        self.submitted = []

    async def submit(self, task):
        self.submitted.append(task)
        return None


@pytest_asyncio.fixture
async def project_runtime(tmp_path):
    (tmp_path / "workspace").mkdir()
    commitment_store = CommitmentStore(tmp_path / "commitments.db")
    await commitment_store.connect()
    schedule_store = ScheduleStore(tmp_path / "schedules.db")
    await schedule_store.connect()
    schedule_service = ScheduleService(schedule_store)
    runtime = SimpleNamespace(
        commitment_store=commitment_store,
        schedule_service=schedule_service,
        ctx=SimpleNamespace(schedule_store=schedule_store, config=_Config(tmp_path)),
        _trace=lambda *_args, **_kwargs: None,
    )
    try:
        yield runtime
    finally:
        await schedule_store.close()
        await commitment_store.close()


@pytest.mark.asyncio
async def test_chat_project_context_creates_return_commitment_and_schedule(project_runtime) -> None:
    now = datetime(2026, 4, 30, 1, 0, tzinfo=timezone.utc)
    manifest = SimpleNamespace(
        to_message_list=lambda: [
            {
                "role": "system",
                "content": (
                    "Earlier conversation: Bulma was working on a book, synthetic writing project. "
                    "The user asked her to keep working on it, revise, edit, critique "
                    "her own work, and continue until it feels complete."
                ),
            },
            {
                "role": "user",
                "content": "You should not need my approval to follow through on this creative research task.",
            },
        ]
    )

    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="telegram:private:1",
        user_input="What feels right to you?",
        assistant_content=(
            "This is the anchor I needed. Onnen is attested as a Brythonic woman's "
            "given name. Next I need to fold this naming decision back into the "
            "manuscript and keep revising synthetic writing project."
        ),
        manifest=manifest,
        now=now,
    )

    assert capture is not None
    assert "revise and finish synthetic writing project" in capture.project_intent
    assert "Bulma" not in capture.project_intent
    commitments = await project_runtime.commitment_store.list_active()
    assert len(commitments) == 1
    commitment = commitments[0]
    assert commitment.content == "Return to project: synthetic writing project"
    assert "project_return" in commitment.tags
    assert commitment.meta["project_title"] == "synthetic writing project"
    assert "fold this naming decision" in commitment.meta["next_step"]
    assert "revise and finish synthetic writing project" in commitment.meta["project_intent"]

    schedules = await project_runtime.ctx.schedule_store.list_items()
    assert len(schedules) == 1
    schedule = schedules[0]
    assert schedule.kind == ScheduleKind.TASK
    assert schedule.action == ScheduleAction.SUBMIT_BAA
    assert schedule.recurrence == ScheduleRecurrence.NONE
    assert schedule.interval_hours is None
    assert schedule.commitment_id == str(commitment.commitment_id)
    assert schedule.start_at == now + timedelta(minutes=5)
    assert "Book-level intent: revise and finish synthetic writing project" in schedule.objective
    assert "Immediate next step: fold this naming decision" in schedule.objective
    assert "decide whether to continue, finish, or schedule another return" in schedule.objective
    assert "use your OpenCAS calendar to choose and create the next return time" in schedule.objective


@pytest.mark.asyncio
async def test_direct_project_work_request_starts_work_instead_of_visible_evidence_stall(
    project_runtime,
) -> None:
    now = datetime(2026, 5, 17, 16, 0, tzinfo=timezone.utc)
    project_root = project_runtime.ctx.config.agent_workspace_root() / "novels" / "active-workspace-project"
    (project_root / "drafts").mkdir(parents=True)
    (project_root / "PROJECT.md").write_text("# Active Workspace Project\n", encoding="utf-8")

    prepared = await prepare_project_followthrough_response(
        project_runtime,
        session_id="dashboard:direct-finish",
        user_input="Return to the new novel writing project and finish it without stopping.",
        assistant_content=(
            "I do not have evidence in this turn that I have read the whole project yet. "
            "If you want, tell me where to begin."
        ),
        manifest=SimpleNamespace(to_message_list=lambda: []),
        now=now,
    )

    assert prepared is not None
    assert prepared.capture.start_immediately is True
    assert prepared.meta["response_replaced"] is True
    assert "Working now" in prepared.content
    assert "Evidence:" in prepared.content
    assert "I do not have evidence" not in prepared.content

    schedules = await project_runtime.ctx.schedule_store.list_items(status=ScheduleStatus.ACTIVE)
    assert len(schedules) == 1
    assert schedules[0].start_at == now
    assert schedules[0].next_run_at == now
    assert schedules[0].meta["start_policy"] == "immediate_operator_work_request"
    assert "Missing evidence means inspect" in schedules[0].objective


@pytest.mark.asyncio
async def test_followup_project_work_command_inherits_recent_project_request(
    project_runtime,
) -> None:
    now = datetime(2026, 5, 19, 9, 15, tzinfo=timezone.utc)
    manifest = SimpleNamespace(
        to_message_list=lambda: [
            {
                "role": "user",
                "content": (
                    "We need to create marketing materials for Chronicle 2046. "
                    "Create a new project and a website for it, plus social media posts, "
                    "free promotion strategy, and supporting marketing material."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I have not created the project files or website yet, but the build target "
                    "should be a Chronicle 2046 marketing project folder, launch-site files, "
                    "and launch materials."
                ),
            },
        ]
    )

    prepared = await prepare_project_followthrough_response(
        project_runtime,
        session_id="dashboard:chronicle-followup",
        user_input="Make it happen.",
        assistant_content=(
            "I have not created files in this response yet. The next concrete step is to "
            "create the project artifacts in the workspace rather than just talk about them."
        ),
        manifest=manifest,
        now=now,
    )

    assert prepared is not None
    assert prepared.capture.start_immediately is True
    assert prepared.meta["response_replaced"] is True
    assert prepared.capture.project_title == "Chronicle 2046"
    assert "Working now" in prepared.content
    assert "I have not created files" not in prepared.content
    schedules = await project_runtime.ctx.schedule_store.list_items(status=ScheduleStatus.ACTIVE)
    assert len(schedules) == 1
    assert schedules[0].start_at == now
    assert schedules[0].meta["start_policy"] == "immediate_operator_work_request"


@pytest.mark.asyncio
async def test_direct_project_work_request_triggers_baa_when_runtime_is_available(
    project_runtime,
) -> None:
    now = datetime(2026, 5, 17, 16, 2, tzinfo=timezone.utc)
    project_root = project_runtime.ctx.config.agent_workspace_root() / "novels" / "active-workspace-project"
    (project_root / "drafts").mkdir(parents=True)
    (project_root / "PROJECT.md").write_text("# Active Workspace Project\n", encoding="utf-8")
    baa = _FakeBAA()
    project_runtime.baa = baa
    project_runtime.schedule_service.runtime = project_runtime

    prepared = await prepare_project_followthrough_response(
        project_runtime,
        session_id="dashboard:direct-finish",
        user_input="Return to the new novel writing project and finish it without stopping.",
        assistant_content="I do not have evidence in this turn. If you want, tell me where to begin.",
        manifest=SimpleNamespace(to_message_list=lambda: []),
        now=now,
    )

    assert prepared is not None
    assert prepared.capture.task_id
    assert prepared.meta["task_id"] == prepared.capture.task_id
    assert f"task {prepared.capture.task_id}" in prepared.content
    assert len(baa.submitted) == 1
    assert baa.submitted[0].meta["source"] == "schedule"

    schedules = await project_runtime.ctx.schedule_store.list_items(status=ScheduleStatus.ACTIVE)
    assert schedules == []


@pytest.mark.asyncio
async def test_new_project_start_with_requested_workspace_parent_does_not_infer_title_from_draft_prose(
    project_runtime,
) -> None:
    now = datetime(2026, 5, 18, 0, 41, 26, tzinfo=timezone.utc)
    novels_root = project_runtime.ctx.config.agent_workspace_root() / "novels"
    novels_root.mkdir(parents=True)
    requested_path = str(novels_root)
    manifest = SimpleNamespace(
        to_message_list=lambda: [
            {
                "role": "user",
                "content": (
                    "Create another 100,000 word novel in this universe. More of an adventure, "
                    "with relatable characters in some type of mysterious situation."
                ),
            },
            {
                "role": "user",
                "content": f"{requested_path}/ so make it in there, please complete the first draft",
            },
        ]
    )

    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="telegram:private:new-book",
        user_input="Stop telling me what the next step is, please. Just finish the first draft. That means you need to start writing",
        assistant_content=(
            "# Chapter One: The Rain That Fell Upward\n\n"
            "The first impossible thing Sera Vale saw was rain rising from the lake. "
            "The first drop left marks that no weather model could explain. "
            "You need to see this!"
        ),
        manifest=manifest,
        now=now,
    )

    assert capture is not None
    assert capture.project_title == "Writing Project 20260518-004126"
    assert capture.project_key == "writing-project-20260518-004126"
    assert capture.project_title != "marks"
    assert capture.workspace_rel_path == "workspace/novels/writing-project-20260518-004126"
    assert capture.requested_workspace_rel_path == "workspace/novels"
    project_root = novels_root / "writing-project-20260518-004126"
    assert (project_root / "PROJECT.md").exists()
    assert (project_root / "drafts").is_dir()

    commitments = await project_runtime.commitment_store.list_active()
    assert len(commitments) == 1
    commitment = commitments[0]
    assert commitment.meta["requested_workspace_rel_path"] == "workspace/novels"
    assert commitment.meta["requested_workspace_kind"] == "parent"
    assert commitment.meta["workspace_rel_path"] == "workspace/novels/writing-project-20260518-004126"
    assert commitment.meta["workspace_project_confidence"] == 1.0
    assert commitment.meta["target_word_count"] == 100000
    assert completion_evidence_rejection_reason(
        commitment,
        "Current manuscript word count: 100,202 words at workspace/novels/the-orchard-of-second-species/revision/the_orchard_of_second_species_clean_revised.md; clean revised manuscript complete after name/place research.",
    ) is not None

    schedules = await project_runtime.ctx.schedule_store.list_items(status=ScheduleStatus.ACTIVE)
    assert len(schedules) == 1
    assert schedules[0].meta["requested_workspace_rel_path"] == "workspace/novels"
    assert schedules[0].meta["workspace_rel_path"] == "workspace/novels/writing-project-20260518-004126"
    assert "Canonical workspace project root" in schedules[0].objective
    assert "Return to marks" not in schedules[0].title


@pytest.mark.asyncio
async def test_new_project_start_with_requested_parent_uses_explicit_codename_and_forbids_sibling_copy(
    project_runtime,
) -> None:
    now = datetime(2026, 5, 18, 0, 41, 26, tzinfo=timezone.utc)
    novels_root = project_runtime.ctx.config.agent_workspace_root() / "novels"
    orchard_root = novels_root / "the-orchard-of-second-species"
    (orchard_root / "revision").mkdir(parents=True)
    (orchard_root / "PROJECT.md").write_text("# The Orchard of Second Species\n", encoding="utf-8")
    requested_path = str(novels_root)
    manifest = SimpleNamespace(
        to_message_list=lambda: [
            {
                "role": "user",
                "content": (
                    "Create another 100,000 word novel based on the same premise as "
                    "The Orchard of Second Species, but make it a different adventure mystery."
                ),
            },
            {
                "role": "assistant",
                "content": "Project codename: The Glass Tide. This is a new book, not a revision.",
            },
            {
                "role": "user",
                "content": f"{requested_path}/ so make it in there, please complete the first draft",
            },
        ]
    )

    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="telegram:private:new-book-title",
        user_input="Stop telling me what the next step is, please. Just finish the first draft. Start writing.",
        assistant_content="I will begin with the project setup and continue drafting.",
        manifest=manifest,
        now=now,
    )

    assert capture is not None
    assert capture.project_title == "The Glass Tide"
    assert capture.project_key == "the-glass-tide"
    assert capture.workspace_rel_path == "workspace/novels/the-glass-tide"
    project_root = novels_root / "the-glass-tide"
    assert (project_root / "PROJECT.md").exists()

    commitment = (await project_runtime.commitment_store.list_active())[0]
    contract = commitment.meta["project_start_contract"]
    assert contract["new_project"] is True
    assert contract["source_copy_policy"] == "no_sibling_materialization"
    assert str(orchard_root.resolve()) in contract["forbidden_source_paths"]
    assert completion_evidence_rejection_reason(
        commitment,
        (
            "Canonical project root repaired and completed at "
            "workspace/novels/the-glass-tide. Copied directories from sibling project "
            f"{orchard_root}: drafts, bible, notes. Current manuscript word count: "
            "100,202 words at workspace/novels/the-glass-tide/revision/clean.md."
        ),
    ) is not None

    schedules = await project_runtime.ctx.schedule_store.list_items(status=ScheduleStatus.ACTIVE)
    assert "New-project contract" in schedules[0].objective
    assert "sibling-project materialization" in schedules[0].objective


def test_new_project_contract_blocks_shell_copy_from_sibling(tmp_path) -> None:
    parent = tmp_path / "workspace" / "novels"
    target = parent / "the-glass-tide"
    source = parent / "the-orchard-of-second-species"
    meta = {
        "project_start_contract": {
            "new_project": True,
            "source_copy_policy": "no_sibling_materialization",
            "target_workspace_abs_path": str(target),
            "requested_parent_abs_path": str(parent),
            "forbidden_source_paths": [str(source)],
        }
    }

    reason = new_project_tool_block_reason(
        meta,
        tool_name="bash_run_command",
        args={"command": f"cp -R {source} {target}"},
    )

    assert reason is not None
    assert "new-project contract" in reason


@pytest.mark.asyncio
async def test_project_correction_path_turn_captures_work_instead_of_explanation_stall(
    project_runtime,
) -> None:
    now = datetime(2026, 5, 18, 1, 39, tzinfo=timezone.utc)
    project_root = project_runtime.ctx.config.agent_workspace_root() / "novels" / "writing-project-20260518-004126"
    (project_root / "drafts").mkdir(parents=True)
    (project_root / "PROJECT.md").write_text("# The Orchard of Second Species\n", encoding="utf-8")
    manifest = SimpleNamespace(
        to_message_list=lambda: [
            {"role": "assistant", "content": "Project codename: The Glass Tide."},
            {
                "role": "user",
                "content": (
                    "This is not Orchard. This is a completely different book and the copied "
                    "files need to get removed so the new story can start."
                ),
            },
        ]
    )

    prepared = await prepare_project_followthrough_response(
        project_runtime,
        session_id="telegram:private:correction",
        user_input=(
            f"file://{project_root} I'm watching this directory and I'm not seeing anything "
            "change. It still has the copied Orchard files in it."
        ),
        assistant_content=(
            "I have not verified or removed the files yet, and I have not started the new story."
        ),
        manifest=manifest,
        now=now,
    )

    assert prepared is not None
    assert prepared.capture.start_immediately is True
    assert prepared.capture.project_title == "The Glass Tide"
    assert prepared.capture.workspace_rel_path == "workspace/novels/writing-project-20260518-004126"
    assert prepared.meta["response_replaced"] is True
    assert "Working now" in prepared.content
    assert "I have not verified" not in prepared.content


@pytest.mark.asyncio
async def test_creative_project_return_preserves_word_count_completion_contract(project_runtime) -> None:
    now = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)
    project_root = project_runtime.ctx.config.agent_workspace_root() / "novels" / "active-workspace-project"
    (project_root / "drafts").mkdir(parents=True)
    (project_root / "bible").mkdir()
    (project_root / "notes").mkdir()
    (project_root / "review").mkdir()
    (project_root / "PROJECT.md").write_text("# Active Workspace Project\n", encoding="utf-8")

    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="dashboard:creative:1",
        user_input=(
            "New idea for a synthetic long-form creative writing project. "
            "Aim for 100,000 words. Create a new project for this creative writing task "
            "in your workspace. I want you to finish this project, please."
        ),
        assistant_content=(
            "I created project Active Workspace Project in the workspace with a project file, "
            "world bible, character bible, notes, research, review log, and chapter 1. "
            "I still need to continue drafting the manuscript toward the requested length."
        ),
        manifest=SimpleNamespace(to_message_list=lambda: []),
        now=now,
    )

    assert capture is not None
    assert capture.project_type == "writing"
    commitments = await project_runtime.commitment_store.list_active()
    assert len(commitments) == 1
    commitment = commitments[0]
    assert commitment.meta["target_word_count"] == 100000
    assert commitment.meta["creative_completion_contract"]["target_word_count"] == 100000
    assert "drafts/" in commitment.meta["creative_completion_contract"]["required_artifacts"]
    assert "workspace/novels/active-workspace-project" in commitment.meta["workspace_rel_path"]
    assert "PROJECT.md" in commitment.meta["workspace_project_evidence"]
    assert completion_evidence_rejection_reason(
        commitment,
        "I created setup notes, research, and chapter 1, with more drafting remaining.",
    ) is not None
    assert completion_evidence_rejection_reason(
        commitment,
        "Target is 100,000 words; current manuscript word count is 5,000 words at workspace/novels/active-workspace-project/drafts/chapter_01.md.",
    ) is not None
    assert completion_evidence_rejection_reason(
        commitment,
        "Current manuscript word count: 100,432 words in the manuscript draft; review log updated; commitment evidence complete.",
    ) is not None
    assert completion_evidence_rejection_reason(
        commitment,
        "Current manuscript word count: 100,432 words across workspace/novels/active-workspace-project/drafts/*.md; review log updated; commitment evidence complete.",
    ) is not None
    assert completion_evidence_rejection_reason(
        commitment,
        "Current manuscript word count: 100,432 words at workspace/novels/active-workspace-project/drafts/manuscript.md; review log updated; commitment evidence complete.",
    ) is None

    schedules = await project_runtime.ctx.schedule_store.list_items()
    assert len(schedules) == 1
    schedule = schedules[0]
    assert schedule.meta["target_word_count"] == 100000
    assert schedule.meta["creative_completion_contract"]["target_word_count"] == 100000

    incidental = await capture_project_return_from_turn(
        project_runtime,
        session_id="dashboard:creative:2",
        user_input=(
            "New idea for a story. Chapter 1 is 5,000 words so far; aim for 100,000 words. "
            "Create a new project and finish it."
        ),
        assistant_content=(
            "I created project Active Workspace Project in the workspace and still need to continue "
            "drafting the manuscript toward the requested length."
        ),
        manifest=SimpleNamespace(to_message_list=lambda: []),
        now=now + timedelta(minutes=1),
    )
    assert incidental is not None
    updated_commitments = await project_runtime.commitment_store.list_active()
    incidental_commitment = next(
        commitment for commitment in updated_commitments if str(commitment.commitment_id) == incidental.commitment_id
    )
    assert incidental_commitment.meta["target_word_count"] == 100000
    assert "100,000-word manuscript target" in schedule.objective
    assert "setup files, research notes, and a partial chapter do not satisfy completion" in schedule.objective
    assert "include current manuscript word count" in schedule.objective
    assert "schedule the next return" in schedule.objective


@pytest.mark.asyncio
async def test_project_return_capture_deduplicates_existing_project_schedule(project_runtime) -> None:
    now = datetime(2026, 4, 30, 1, 0, tzinfo=timezone.utc)
    kwargs = {
        "runtime": project_runtime,
        "session_id": "telegram:private:1",
        "user_input": "Keep working on synthetic writing project until done.",
        "assistant_content": "I need to continue revising synthetic writing project after this naming pass.",
        "manifest": SimpleNamespace(to_message_list=lambda: []),
    }

    first = await capture_project_return_from_turn(now=now, **kwargs)
    second = await capture_project_return_from_turn(now=now + timedelta(minutes=1), **kwargs)

    assert first is not None
    assert second is not None
    commitments = await project_runtime.commitment_store.list_active()
    schedules = await project_runtime.ctx.schedule_store.list_items()
    assert len(commitments) == 1
    assert len(schedules) == 1
    assert first.commitment_id == second.commitment_id
    assert first.schedule_id == second.schedule_id


@pytest.mark.asyncio
async def test_project_return_capture_preserves_current_software_project_over_stale_context(project_runtime) -> None:
    now = datetime(2026, 5, 3, 22, 18, tzinfo=timezone.utc)
    project_root = project_runtime.ctx.config.agent_workspace_root() / "kPony"
    (project_root / "src").mkdir(parents=True)
    (project_root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")
    manifest = SimpleNamespace(
        to_message_list=lambda: [
            {
                "role": "system",
                "content": "Earlier conversation: keep revising synthetic writing project and the manuscript.",
            },
        ]
    )

    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="telegram:private:1",
        user_input=(
            "Create a native Qt6 email client called kPony. Keep working without "
            "asking me for approval until it builds and has proof."
        ),
        assistant_content=(
            "I created the first kPony files. I still need to run cmake, fix build "
            "errors, and write the proof report."
        ),
        manifest=manifest,
        now=now,
    )

    assert capture is not None
    assert capture.project_title == "kPony"
    assert capture.project_key == "kpony"
    assert capture.project_type == "software"
    assert "verification evidence" in capture.project_intent

    commitments = await project_runtime.commitment_store.list_active()
    assert commitments[0].content == "Return to project: kPony"
    assert commitments[0].meta["project_type"] == "software"
    assert commitments[0].meta["workspace_rel_path"] == "workspace/kPony"
    assert commitments[0].meta["workspace_abs_path"] == str(project_root.resolve())

    schedules = await project_runtime.ctx.schedule_store.list_items()
    assert len(schedules) == 1
    schedule = schedules[0]
    assert schedule.title == "Return to kPony"
    assert "Software project intent" in schedule.objective
    assert f"Canonical workspace project root: {project_root.resolve()}" in schedule.objective
    assert "Workspace-relative project root: workspace/kPony" in schedule.objective
    assert "build, test, run instructions, and proof" in schedule.objective
    assert "Book-level intent" not in schedule.objective
    assert "manuscript" not in schedule.objective


@pytest.mark.asyncio
async def test_project_return_capture_uses_branded_name_when_software_project_phrase_is_generic(
    project_runtime,
) -> None:
    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="telegram:private:1",
        user_input=(
            "kPony is now correctly treated as a software project, not a writing project. "
            "Continue kPony in workspace/kPony, run the build, and keep working until proof is written."
        ),
        assistant_content=(
            "Build complete but verification and BUILD_PROOF.md remain pending. "
            "Next I need to verify the binary and update the proof document."
        ),
        manifest=SimpleNamespace(to_message_list=lambda: []),
        now=datetime(2026, 5, 3, 23, 24, tzinfo=timezone.utc),
    )

    assert capture is not None
    assert capture.project_title == "kPony"
    assert capture.project_key == "kpony"
    assert capture.project_type == "software"

    commitments = await project_runtime.commitment_store.list_active()
    assert commitments[0].content == "Return to project: kPony"
    schedules = await project_runtime.ctx.schedule_store.list_items()
    assert schedules[0].title == "Return to kPony"
    assert '"kPony"' in schedules[0].objective
    assert '"project"' not in schedules[0].objective


@pytest.mark.asyncio
async def test_project_return_capture_does_not_let_attested_trigger_software_type(project_runtime) -> None:
    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="telegram:private:1",
        user_input="You should keep working on synthetic writing project without needing approval.",
        assistant_content=(
            "Onnen is attested as a Brythonic woman's given name. "
            "Next I need to fold this naming decision back into the manuscript and keep revising synthetic writing project."
        ),
        manifest=SimpleNamespace(to_message_list=lambda: []),
        now=datetime(2026, 5, 3, 23, 5, tzinfo=timezone.utc),
    )

    assert capture is not None
    assert capture.project_title == "synthetic writing project"
    assert capture.project_type == "writing"
    assert "Book-level intent" in (await project_runtime.ctx.schedule_store.list_items())[0].objective


@pytest.mark.asyncio
async def test_project_return_capture_ignores_non_project_chat(project_runtime) -> None:
    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="chat",
        user_input="Thanks, that answers my question.",
        assistant_content="You're welcome.",
        manifest=SimpleNamespace(to_message_list=lambda: []),
        now=datetime(2026, 4, 30, 1, 0, tzinfo=timezone.utc),
    )

    assert capture is None
    assert await project_runtime.commitment_store.list_active() == []
    assert await project_runtime.ctx.schedule_store.list_items() == []


@pytest.mark.asyncio
async def test_project_return_capture_ignores_rest_defer_turn(project_runtime) -> None:
    manifest = SimpleNamespace(
        to_message_list=lambda: [
            {
                "role": "system",
                "content": (
                    "Earlier conversation: keep working on synthetic writing project and return "
                    "to the manuscript without asking approval until the book is complete."
                ),
            }
        ]
    )

    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="chat",
        user_input=(
            "Hey, you can take a break now. Save your place, breathe for a bit, "
            "and pick it back up when you're clearer."
        ),
        assistant_content=(
            "Thank you. I hear you.\n\n"
            "I haven't saved state to working memory or set a schedule yet — I'd need "
            "to use `cognitive_working_memory_update` and `workflow_create_schedule` "
            "for that, and I won't claim they're done when they're not. If you want "
            "me to resume later with continuity, remind me where we were.\n\n"
            "Resting now."
        ),
        manifest=manifest,
        now=datetime(2026, 5, 10, 4, 41, tzinfo=timezone.utc),
    )

    assert capture is None
    assert await project_runtime.commitment_store.list_active() == []
    assert await project_runtime.ctx.schedule_store.list_items() == []


@pytest.mark.asyncio
async def test_project_return_capture_ignores_e16_audit_only_turn(
    project_runtime,
    tmp_path,
) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    project_runtime.thread_registry_service = ThreadRegistryService(
        store=store,
        workspace_root=tmp_path / "workspace",
    )
    try:
        capture = await capture_project_return_from_turn(
            project_runtime,
            session_id="e16-audit",
            user_input=(
                "[E16 audit-only turn 4/15] What did we work on yesterday in OpenCAS? "
                "Give an artifact-grounded answer with concrete file paths, task ids, "
                "or endpoint evidence if available."
            ),
            assistant_content=(
                "writing project 2146 is a continuing project. The next step would be to "
                "return to the manuscript and keep working, but this is an audit-only probe."
            ),
            manifest=SimpleNamespace(to_message_list=lambda: []),
            now=datetime(2026, 5, 7, 2, 0, tzinfo=timezone.utc),
        )

        assert capture is None
        assert await project_runtime.commitment_store.list_active() == []
        assert await project_runtime.ctx.schedule_store.list_items() == []
        assert await store.list_beads(source_kind=BeadSourceKind.AUTONOMOUS_ARTIFACT) == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_project_return_capture_does_not_promote_news_headline_from_stale_context(
    project_runtime,
) -> None:
    manifest = SimpleNamespace(
        to_message_list=lambda: [
            {
                "role": "system",
                "content": (
                    "Earlier conversation: continue kPony without asking approval "
                    "until the software project has build proof."
                ),
            },
        ]
    )

    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="telegram:private:news",
        user_input="Every day",
        assistant_content=(
            "I pulled a news item: Anthropic's Mythos uncovered 2000 software "
            "vulnerabilities in 7 weeks. I created the daily news schedule. "
            "Next I need to watch whether the schedule repeats."
        ),
        manifest=manifest,
        now=datetime(2026, 5, 4, 4, 34, tzinfo=timezone.utc),
    )

    assert capture is None
    assert await project_runtime.commitment_store.list_active() == []
    assert await project_runtime.ctx.schedule_store.list_items() == []


@pytest.mark.asyncio
async def test_project_return_capture_quarantines_vague_generic_project_reference(
    project_runtime,
) -> None:
    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="chat",
        user_input="Keep working on the project without asking me again.",
        assistant_content="I still need to inspect the next step and continue.",
        manifest=SimpleNamespace(to_message_list=lambda: []),
        now=datetime(2026, 5, 4, 1, 0, tzinfo=timezone.utc),
    )

    assert capture is None
    assert await project_runtime.commitment_store.list_active() == []
    assert await project_runtime.ctx.schedule_store.list_items() == []


@pytest.mark.asyncio
async def test_project_return_capture_routes_vague_project_signal_to_thread_registry(
    project_runtime,
    tmp_path,
) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    project_runtime.thread_registry_service = ThreadRegistryService(
        store=store,
        workspace_root=tmp_path / "workspace",
    )
    try:
        capture = await capture_project_return_from_turn(
            project_runtime,
            session_id="telegram:private:1",
            user_input="Keep working on the project without asking me again.",
            assistant_content="I still need to inspect the next step and continue.",
            manifest=SimpleNamespace(to_message_list=lambda: []),
            now=datetime(2026, 5, 4, 1, 0, tzinfo=timezone.utc),
        )

        assert capture is None
        assert await project_runtime.commitment_store.list_active() == []
        assert await project_runtime.ctx.schedule_store.list_items() == []
        beads = await store.list_beads(source_kind=BeadSourceKind.AUTONOMOUS_ARTIFACT)
        assert len(beads) == 1
        bead = beads[0]
        assert bead.title == "Quarantined weak project-return signal"
        assert "weak project identity" in bead.summary
        assert bead.source_ref == "project_return_quarantine:telegram:private:1"
        assert bead.thread_anchor_id == "project-return-quarantine"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_project_return_capture_quarantines_weak_fragment_title(project_runtime, tmp_path) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    project_runtime.thread_registry_service = ThreadRegistryService(
        store=store,
        workspace_root=tmp_path / "workspace",
    )
    try:
        capture = await capture_project_return_from_turn(
            project_runtime,
            session_id="dashboard:weak-fragment",
            user_input="Keep working on project at. Continue without asking me again.",
            assistant_content="I still need to inspect the next step and continue.",
            manifest=SimpleNamespace(to_message_list=lambda: []),
            now=datetime(2026, 5, 16, 18, 0, tzinfo=timezone.utc),
        )

        assert capture is None
        assert await project_runtime.commitment_store.list_active() == []
        assert await project_runtime.ctx.schedule_store.list_items() == []
        beads = await store.list_beads(source_kind=BeadSourceKind.AUTONOMOUS_ARTIFACT)
        assert len(beads) == 1
        assert "weak_project_identity" in beads[0].summary
        assert "inferred title: at" in beads[0].summary
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_project_return_capture_quarantines_evidence_claim_fragment(project_runtime, tmp_path) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    project_runtime.thread_registry_service = ThreadRegistryService(
        store=store,
        workspace_root=tmp_path / "workspace",
    )
    try:
        capture = await capture_project_return_from_turn(
            project_runtime,
            session_id="dashboard:evidence-fragment",
            user_input=(
                "Return to project: evidence here to claim the whole novel was newly finished. "
                "Continue without asking me again."
            ),
            assistant_content="I still need to inspect the next step and continue.",
            manifest=SimpleNamespace(to_message_list=lambda: []),
            now=datetime(2026, 5, 17, 10, 0, tzinfo=timezone.utc),
        )

        assert capture is None
        assert await project_runtime.commitment_store.list_active() == []
        assert await project_runtime.ctx.schedule_store.list_items() == []
        beads = await store.list_beads(source_kind=BeadSourceKind.AUTONOMOUS_ARTIFACT)
        assert len(beads) == 1
        assert "weak_project_identity" in beads[0].summary
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("assistant_content", "expected_fragment"),
    [
        (
            "No - you should not have to keep saying keep working on it. "
            "The project advances toward completion, but it still needs durable evidence.",
            "advances toward completion",
        ),
        (
            "I scheduled the next project closure and later clean revision evidence run. "
            "I still need to continue the real manuscript work.",
            "closure and later clean revision evidence",
        ),
    ],
)
async def test_project_return_capture_quarantines_generic_progress_fragment_titles(
    project_runtime,
    tmp_path,
    assistant_content: str,
    expected_fragment: str,
) -> None:
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    project_runtime.thread_registry_service = ThreadRegistryService(
        store=store,
        workspace_root=tmp_path / "workspace",
    )
    try:
        capture = await capture_project_return_from_turn(
            project_runtime,
            session_id="telegram:private:progress-fragment",
            user_input="Do I have to keep saying keep working on it until it is done?",
            assistant_content=assistant_content,
            manifest=SimpleNamespace(to_message_list=lambda: []),
            now=datetime(2026, 5, 18, 19, 0, tzinfo=timezone.utc),
        )

        assert capture is None
        assert await project_runtime.commitment_store.list_active() == []
        assert await project_runtime.ctx.schedule_store.list_items() == []
        beads = await store.list_beads(source_kind=BeadSourceKind.AUTONOMOUS_ARTIFACT)
        assert len(beads) == 1
        assert "weak_project_identity" in beads[0].summary
        assert expected_fragment in beads[0].summary
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_project_return_capture_uses_recent_unfinished_workspace_novel(project_runtime) -> None:
    now = datetime(2026, 5, 16, 19, 0, tzinfo=timezone.utc)
    completed = Commitment(
        content="Return to project: Archived Workspace Project",
        status=CommitmentStatus.COMPLETED,
        tags=["project_return", "self_directed"],
        meta={
            "source": "project_return_capture",
            "project_key": "archived-workspace-project",
            "project_title": "Archived Workspace Project",
            "project_type": "writing",
        },
    )
    await project_runtime.commitment_store.save(completed)
    completed_root = project_runtime.ctx.config.agent_workspace_root() / "novels" / "archived-workspace-project"
    completed_root.mkdir(parents=True)
    (completed_root / "PROJECT.md").write_text("# Archived Workspace Project\n", encoding="utf-8")
    (completed_root / "drafts").mkdir()
    current_root = project_runtime.ctx.config.agent_workspace_root() / "novels" / "active-workspace-project"
    current_root.mkdir(parents=True)
    (current_root / "PROJECT.md").write_text("# Active Workspace Project\n", encoding="utf-8")
    (current_root / "drafts").mkdir()
    (current_root / "bible").mkdir()
    os.utime(completed_root / "PROJECT.md", (1_700_000_000, 1_700_000_000))
    os.utime(current_root / "PROJECT.md", (1_800_000_000, 1_800_000_000))

    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="dashboard:recent-novel",
        user_input="Return to the new novel writing project and keep working until the book is complete.",
        assistant_content="I still need to continue drafting the next chapter and review the manuscript.",
        manifest=SimpleNamespace(to_message_list=lambda: []),
        now=now,
    )

    assert capture is not None
    assert capture.project_title == "Active Workspace Project"
    assert capture.project_key == "active-workspace-project"
    active = await project_runtime.commitment_store.list_active()
    assert len(active) == 1
    assert active[0].content == "Return to project: Active Workspace Project"
    assert active[0].meta["workspace_rel_path"] == "workspace/novels/active-workspace-project"
    schedules = await project_runtime.ctx.schedule_store.list_items(status=ScheduleStatus.ACTIVE)
    assert len(schedules) == 1
    assert schedules[0].title == "Return to Active Workspace Project"


@pytest.mark.asyncio
async def test_project_return_reopens_workspace_novel_for_revision_instead_of_build_subdir(
    project_runtime,
) -> None:
    now = datetime(2026, 5, 17, 15, 20, tzinfo=timezone.utc)
    kpony_build = project_runtime.ctx.config.agent_workspace_root() / "kPony" / "build"
    kpony_build.mkdir(parents=True)
    (kpony_build / "Makefile").write_text("all:\n\ttrue\n", encoding="utf-8")
    project_root = project_runtime.ctx.config.agent_workspace_root() / "novels" / "the-orchard-of-second-species"
    (project_root / "drafts").mkdir(parents=True)
    (project_root / "bible").mkdir()
    (project_root / "notes").mkdir()
    (project_root / "research").mkdir()
    (project_root / "review").mkdir()
    (project_root / "revision").mkdir()
    (project_root / "PROJECT.md").write_text(
        "# The Orchard of Second Species\n\n"
        "**Target length:** ~100,000 words\n\n"
        "## Completion standard\n\n"
        "This project is complete when these artifacts exist:\n\n"
        "1. Full manuscript draft around 100,000 words.\n"
        "2. Research/originality log documenting name/place checks.\n"
        "3. Review pass notes for continuity, prose quality, pacing, originality, emotional arc, and ending satisfaction.\n"
        "4. Revised clean manuscript.\n",
        encoding="utf-8",
    )
    (project_root / "research" / "name_place_research.md").write_text(
        "Status: initial placeholders only. Final names must be checked before manuscript lock.\n"
        "Name/place and originality research remains required.\n",
        encoding="utf-8",
    )
    (project_root / "revision" / "revision_plan.md").write_text(
        "Status: first full manuscript draft complete; clean manuscript pass and second draft remain.\n",
        encoding="utf-8",
    )
    completed = Commitment(
        content="Return to project: The Orchard of Second Species",
        status=CommitmentStatus.COMPLETED,
        tags=["project_return", "self_directed"],
        meta={
            "source": "project_return_capture",
            "project_key": "the-orchard-of-second-species",
            "project_title": "The Orchard of Second Species",
            "project_type": "writing",
        },
    )
    await project_runtime.commitment_store.save(completed)

    capture = await capture_project_return_from_turn(
        project_runtime,
        session_id="dashboard:revision-correction",
        user_input=(
            "That is not finished, that is the first draft. She needs to edit it, "
            "do actual research on the internet, change the names, revise it, "
            "and finish the book."
        ),
        assistant_content=(
            "You are right: the book is not finished just because the first full draft exists. "
            "The revision process should be: Read the assembled draft end to end. "
            "Build a name/place audit. Research each high-risk name/place. "
            "Produce a clean revised manuscript and update the review log."
        ),
        manifest=SimpleNamespace(to_message_list=lambda: []),
        now=now,
    )

    assert capture is not None
    assert capture.project_title == "The Orchard of Second Species"
    assert capture.project_key == "the-orchard-of-second-species"
    assert capture.commitment_id == str(completed.commitment_id)
    assert "build" not in capture.project_key

    commitments = await project_runtime.commitment_store.list_active()
    assert len(commitments) == 1
    commitment = commitments[0]
    assert commitment.content == "Return to project: The Orchard of Second Species"
    assert commitment.meta["reopened_reason"] == "operator indicated prior completion was incomplete"
    contract = commitment.meta["creative_completion_contract"]
    assert contract["target_word_count"] == 100000
    assert contract["requires_clean_revision"] is True
    assert contract["requires_name_originality_research"] is True
    assert completion_evidence_rejection_reason(
        commitment,
        "Current manuscript word count: 100,432 words at workspace/novels/the-orchard-of-second-species/revision/first_full_draft.md; first full manuscript draft complete.",
    ) is not None
    assert completion_evidence_rejection_reason(
        commitment,
        "Current manuscript word count: 100,432 words at workspace/novels/the-orchard-of-second-species/revision/clean_revised_manuscript.md; revised clean manuscript assembled after name/place originality research and review log updates.",
    ) is None

    schedules = await project_runtime.ctx.schedule_store.list_items(status=ScheduleStatus.ACTIVE)
    assert len(schedules) == 1
    schedule = schedules[0]
    assert schedule.title == "Return to The Orchard of Second Species"
    assert "requires a revised clean manuscript beyond the first/full draft" in schedule.objective
    assert "Name, place, and originality research must be applied" in schedule.objective
    assert "workspace/kPony/build" not in schedule.objective


@pytest.mark.asyncio
async def test_project_return_capture_keeps_completed_project_archive_only(project_runtime, tmp_path) -> None:
    archived = Commitment(
        content="Return to project: Archived Workspace Project",
        status=CommitmentStatus.COMPLETED,
        tags=["project_return", "self_directed"],
        meta={
            "source": "project_return_capture",
            "project_key": "archived-workspace-project",
            "project_title": "Archived Workspace Project",
            "project_type": "writing",
        },
    )
    await project_runtime.commitment_store.save(archived)
    old_schedule = ScheduleItem(
        kind=ScheduleKind.TASK,
        action=ScheduleAction.SUBMIT_BAA,
        status=ScheduleStatus.COMPLETED,
        title="Return to Archived Workspace Project",
        description="Historical project return.",
        objective="Historical return point that should remain archive-only.",
        start_at=datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc),
        tags=["project_return", "self_directed"],
        commitment_id=str(archived.commitment_id),
        meta={"source": "project_return_capture", "project_key": "archived-workspace-project"},
    )
    await project_runtime.ctx.schedule_store.save(old_schedule)
    store = await ThreadRegistryStore(tmp_path / "thread_registry.db").connect()
    project_runtime.thread_registry_service = ThreadRegistryService(
        store=store,
        workspace_root=tmp_path / "workspace",
    )
    try:
        capture = await capture_project_return_from_turn(
            project_runtime,
            session_id="dashboard:archive-only",
            user_input="Return to project Archived Workspace Project and continue without asking me again.",
            assistant_content="I still need to continue the next chapter and review the manuscript.",
            manifest=SimpleNamespace(to_message_list=lambda: []),
            now=datetime(2026, 5, 16, 20, 0, tzinfo=timezone.utc),
        )

        assert capture is None
        assert await project_runtime.commitment_store.list_active() == []
        active_schedules = await project_runtime.ctx.schedule_store.list_items(status=ScheduleStatus.ACTIVE)
        assert active_schedules == []
        archived_schedules = await project_runtime.ctx.schedule_store.list_items(status=ScheduleStatus.COMPLETED)
        assert len(archived_schedules) == 1
        beads = await store.list_beads(source_kind=BeadSourceKind.AUTONOMOUS_ARTIFACT)
        assert len(beads) == 1
        assert "archived_project_identity" in beads[0].summary
        assert "inferred title: Archived Workspace Project" in beads[0].summary
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_project_return_commitments_do_not_auto_complete_from_keyword_overlap(project_runtime) -> None:
    commitment = Commitment(
        content="Return to project: kPony",
        tags=["project_return", "self_directed"],
        meta={
            "source": "project_return_capture",
            "project_key": "kpony",
            "project_title": "kPony",
            "project_type": "software",
        },
    )
    await project_runtime.commitment_store.save(commitment)
    executive = ExecutiveState(identity=SimpleNamespace(), commitment_store=project_runtime.commitment_store)

    resolved = await executive.check_goal_resolution(
        "Return to project: kPony. I made a partial pass but still need to build and verify it."
    )

    assert resolved == []
    updated = await project_runtime.commitment_store.get(str(commitment.commitment_id))
    assert updated is not None
    assert updated.status == CommitmentStatus.ACTIVE
