"""Tests for RepairExecutor explicit phases."""

import asyncio
import subprocess
from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.autonomy.models import ActionRiskTier
from opencas.context.models import MessageRole
from opencas.execution import ExecutionPhase, ExecutionStage, PhaseRecord, RepairExecutor, RepairTask
from opencas.execution.store import TaskStore
from opencas.tools import FileSystemToolAdapter, ShellToolAdapter, ToolRegistry, ToolUseResult


@pytest_asyncio.fixture
async def executor(tmp_path):
    tools = ToolRegistry()
    workspace = str(tmp_path)
    fs = FileSystemToolAdapter(allowed_roots=[workspace])
    tools.register("fs_read_file", "Read file", fs, ActionRiskTier.READONLY)
    tools.register("fs_write_file", "Write file", fs, ActionRiskTier.WORKSPACE_WRITE)
    shell = ShellToolAdapter(cwd=workspace, timeout=30.0)
    tools.register("bash_run_command", "Run command", shell, ActionRiskTier.SHELL_LOCAL)
    return RepairExecutor(tools=tools)


@pytest.mark.asyncio
async def test_executor_records_all_phases(executor):
    task = RepairTask(objective="test task")
    result = await executor.run(task)
    assert result.success is True
    phase_names = [p.phase for p in task.phases]
    assert ExecutionPhase.DETECT in phase_names
    assert ExecutionPhase.SNAPSHOT in phase_names
    assert ExecutionPhase.PLAN in phase_names
    assert ExecutionPhase.EXECUTE in phase_names
    assert ExecutionPhase.VERIFY in phase_names
    assert ExecutionPhase.POSTCHECK in phase_names


@pytest.mark.asyncio
async def test_executor_persists_phase_start_before_handler_finishes(tmp_path):
    store = await TaskStore(tmp_path / "tasks.db").connect()
    try:
        executor = RepairExecutor(tools=ToolRegistry(), store=store)
        task = RepairTask(objective="slow phase")
        await store.save(task)
        started = asyncio.Event()
        finish = asyncio.Event()

        async def slow_handler(_task):
            started.set()
            await finish.wait()
            return "slow done"

        phase_task = asyncio.create_task(
            executor._run_phase(task, ExecutionPhase.EXECUTE, slow_handler)
        )
        await asyncio.wait_for(started.wait(), timeout=1.0)

        in_progress = await store.get(str(task.task_id))
        assert in_progress is not None
        assert in_progress.phases[-1].phase == ExecutionPhase.EXECUTE
        assert in_progress.phases[-1].success is None
        assert in_progress.phases[-1].ended_at is None

        finish.set()
        record = await asyncio.wait_for(phase_task, timeout=1.0)
        assert record.success is True

        completed = await store.get(str(task.task_id))
        assert completed is not None
        assert completed.phases[-1].success is True
        assert completed.phases[-1].output == "slow done"
        assert completed.phases[-1].ended_at is not None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_executor_tool_loop_timeout_returns_recoverable_execute_failure(executor):
    class SlowToolLoop:
        async def run(self, **_kwargs):
            await asyncio.sleep(30)

    runtime = SimpleNamespace(tool_loop=SlowToolLoop(), scheduler=None)
    executor.runtime = runtime
    task = RepairTask(
        objective="background project task",
        meta={"tool_loop_timeout_seconds": 0.01},
    )

    output = await executor._execute_plan(task, "continue")

    assert "tool loop timed out" in output
    assert output.startswith("execute failed:")
    assert task.meta["tool_loop_timeout"]["reason"] == "tool_loop_execution_timeout"


@pytest.mark.asyncio
async def test_executor_tool_loop_timeout_records_workspace_artifact_progress(executor, tmp_path):
    workspace = tmp_path / "workspace" / "novels" / "book"
    workspace.mkdir(parents=True)
    artifact = workspace / "drafts" / "chapter_31.md"

    class SlowWritingToolLoop:
        async def run(self, **_kwargs):
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("new manuscript prose\n", encoding="utf-8")
            await asyncio.sleep(30)

    runtime = SimpleNamespace(tool_loop=SlowWritingToolLoop(), scheduler=None)
    executor.runtime = runtime
    task = RepairTask(
        objective="continue writing project",
        meta={
            "tool_loop_timeout_seconds": 0.01,
            "workspace_abs_path": str(workspace),
        },
    )

    output = await executor._execute_plan(task, "continue")
    record = PhaseRecord(phase=ExecutionPhase.EXECUTE, success=False, output=output)

    assert "tool loop timed out" in output
    assert task.meta["tool_loop_timeout"]["artifact_progress_paths"] == [str(artifact)]
    assert RepairExecutor._artifact_progress_boundary(task, record)
    assert str(artifact) in RepairExecutor._artifact_paths_touched(task, [])


@pytest.mark.asyncio
async def test_executor_snapshot_phase_when_scratch_dir_set(executor, tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    subprocess.run(["git", "init"], cwd=scratch, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=scratch,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=scratch,
        check=True,
        capture_output=True,
    )
    task = RepairTask(
        objective="check file.txt",
        scratch_dir=str(scratch),
    )
    result = await executor.run(task)
    assert result.success is True
    snap_phase = [p for p in task.phases if p.phase == ExecutionPhase.SNAPSHOT][0]
    assert snap_phase.success is True


@pytest.mark.asyncio
async def test_executor_detects_files_from_objective(executor):
    task = RepairTask(objective="read config.json and data.csv")
    await executor.run(task)
    detect_phase = [p for p in task.phases if p.phase == ExecutionPhase.DETECT][0]
    assert "config.json" in detect_phase.output
    assert "data.csv" in detect_phase.output


@pytest.mark.asyncio
async def test_executor_execute_failure_heuristic_fails_task(executor):
    """If _execute_plan returns an empty or failure-marked string, the task should fail."""
    task = RepairTask(objective="failing task", max_attempts=1)
    # Patch _execute_plan to return the exact empty failure string from the brief
    executor._execute_plan = lambda _task, _plan: ""
    result = await executor.run(task)
    exec_phase = [p for p in task.phases if p.phase == ExecutionPhase.EXECUTE][0]
    assert exec_phase.success is False
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED


@pytest.mark.asyncio
async def test_executor_tool_loop_halted_fails_task(executor):
    task = RepairTask(objective="halted task", max_attempts=1)
    executor._execute_plan = lambda _task, _plan: "[Tool loop halted] exceeded rounds"
    result = await executor.run(task)
    exec_phase = [p for p in task.phases if p.phase == ExecutionPhase.EXECUTE][0]
    assert exec_phase.success is False
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED


@pytest.mark.asyncio
async def test_executor_max_tool_iterations_fails_task(executor):
    task = RepairTask(objective="iteration capped task", max_attempts=1)
    executor._execute_plan = lambda _task, _plan: "Reached maximum number of tool-use iterations."
    result = await executor.run(task)
    exec_phase = [p for p in task.phases if p.phase == ExecutionPhase.EXECUTE][0]
    assert exec_phase.success is False
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED


@pytest.mark.asyncio
async def test_executor_max_iterations_with_artifact_progress_recovers(executor, tmp_path):
    target = tmp_path / "chapter.md"
    task = RepairTask(objective="long artifact task", max_attempts=1)

    def _execute_with_artifact_progress(_task, _plan):
        _task.meta["last_tool_loop"] = {
            "iterations": 32,
            "guard_fired": False,
            "tool_call_count": 1,
        }
        _task.meta["last_tool_calls"] = [
            {"id": "call-1", "name": "fs_write_file", "args": {"file_path": str(target)}}
        ]
        return "Reached maximum number of tool-use iterations."

    executor._execute_plan = _execute_with_artifact_progress

    result = await executor.run(task)

    exec_phase = [p for p in task.phases if p.phase == ExecutionPhase.EXECUTE][0]
    assert exec_phase.success is False
    assert result.success is False
    assert result.stage == ExecutionStage.RECOVERING
    assert result.output == "Artifact progress boundary reached; will continue."


@pytest.mark.asyncio
async def test_executor_runtime_guard_fired_result_fails_task(executor):
    async def _run(**kwargs):
        return ToolUseResult(
            final_output="I made partial progress before pausing after a long tool run.",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "runtime_status",
                    "args": {},
                }
            ],
            guard_fired=True,
            guard_reason="Tool loop circuit breaker: exceeded 24 consecutive tool calls in this session.",
        )

    executor.runtime = SimpleNamespace(
        tool_loop=SimpleNamespace(run=_run),
        scheduler=None,
    )

    task = RepairTask(objective="halted task", max_attempts=1)
    result = await executor.run(task)
    exec_phase = [p for p in task.phases if p.phase == ExecutionPhase.EXECUTE][0]
    assert exec_phase.success is False
    assert "tool loop guard fired" in exec_phase.output.lower()
    assert task.meta["last_tool_calls"][0]["name"] == "runtime_status"
    assert task.meta["last_tool_loop"]["guard_fired"] is True
    assert result.success is False
    assert result.stage == ExecutionStage.FAILED


@pytest.mark.asyncio
async def test_executor_frames_project_return_as_self_continuity(executor):
    captured = {}

    class _FakeContextStore:
        async def list_recent(self, session_id, limit=50, include_hidden=False):
            captured["session_lookup"] = {
                "session_id": session_id,
                "limit": limit,
                "include_hidden": include_hidden,
            }
            return [
                SimpleNamespace(
                    role=MessageRole.USER,
                    content="Keep working on writing project 4246 until it feels complete.",
                ),
                SimpleNamespace(
                    role=MessageRole.ASSISTANT,
                    content="I need to fold the Onnen naming decision back into the manuscript.",
                ),
            ]

    async def _run(**kwargs):
        captured["messages"] = kwargs["messages"]
        captured["objective"] = kwargs["objective"]
        return ToolUseResult(final_output="returned to the project")

    executor.runtime = SimpleNamespace(
        tool_loop=SimpleNamespace(run=_run),
        scheduler=None,
        ctx=SimpleNamespace(
            context_store=_FakeContextStore(),
            identity=SimpleNamespace(self_model=SimpleNamespace(name="TestAgent")),
        ),
    )
    task = RepairTask(
        objective="Return to project \"writing project 4246\".",
        meta={
            "source": "schedule",
            "project_key": "writing-project-4246",
            "project_title": "writing project 4246",
            "source_session_id": "telegram:private:1",
            "project_intent": (
                "revise and finish the writing project 4246 manuscript until explicit completion evidence exists, "
                "using critique as input without narrowing the project to naming research"
            ),
            "next_step": "Fold the Onnen naming decision into the manuscript.",
        },
    )

    output = await executor._execute_plan(task, "review context and continue")

    assert output == "returned to the project"
    assert captured["session_lookup"]["session_id"] == "telegram:private:1"
    system_message = captured["messages"][0]["content"]
    assert "You are TestAgent returning to your own creative project" in system_message
    assert "You are an OpenCAS agent" not in system_message
    assert "not an external contractor" in system_message
    assert "writing project 4246" in system_message
    assert "revise and finish the writing project 4246 manuscript" in system_message
    assert "fold the Onnen naming decision" in system_message
    assert "Keep working on writing project 4246" in system_message
    assert "creating a workflow scaffold is not manuscript progress" in system_message
    assert "before claiming a chapter, scene, word count, or manuscript milestone" in system_message


@pytest.mark.asyncio
async def test_executor_rejects_new_project_sibling_materialization_trace(executor, tmp_path):
    workspace = tmp_path / "workspace"
    target = workspace / "novels" / "the-glass-tide"
    source = workspace / "novels" / "the-orchard-of-second-species"
    target.mkdir(parents=True)
    source.mkdir(parents=True)

    async def _run(**_kwargs):
        return ToolUseResult(
            final_output=(
                f"Copied directories from sibling project {source} into {target}. "
                "Current manuscript word count: 100,202 words."
            ),
            tool_calls=[
                {
                    "name": "bash_run_command",
                    "args": {
                        "command": (
                            f"python3 - <<'PY'\n"
                            f"import shutil\n"
                            f"shutil.copytree('{source}', '{target}', dirs_exist_ok=True)\n"
                            f"PY"
                        )
                    },
                }
            ],
        )

    executor.runtime = SimpleNamespace(
        tool_loop=SimpleNamespace(run=_run),
        scheduler=None,
        ctx=SimpleNamespace(
            context_store=None,
            identity=SimpleNamespace(self_model=SimpleNamespace(name="TestAgent")),
            config=SimpleNamespace(
                primary_workspace_root=lambda: tmp_path,
                agent_workspace_root=lambda: workspace,
            ),
        ),
    )
    task = RepairTask(
        objective="Return to project \"The Glass Tide\" and finish the new project.",
        meta={
            "source": "schedule",
            "project_key": "the-glass-tide",
            "project_title": "The Glass Tide",
            "project_type": "writing",
            "workspace_abs_path": str(target),
            "workspace_rel_path": "workspace/novels/the-glass-tide",
            "requested_workspace_abs_path": str(workspace / "novels"),
            "requested_workspace_kind": "parent",
            "project_start_contract": {
                "new_project": True,
                "source_copy_policy": "no_sibling_materialization",
                "target_workspace_abs_path": str(target),
                "requested_parent_abs_path": str(workspace / "novels"),
                "forbidden_source_paths": [str(source)],
            },
        },
    )

    output = await executor._execute_plan(task, "inspect the target and continue")

    assert output.startswith("execute failed:")
    assert "sibling" in output
    assert task.meta["project_contract_failure"]["tool_call_count"] == 1


@pytest.mark.asyncio
async def test_executor_frames_software_project_return_without_creative_manuscript_language(executor):
    captured = {}
    workspace = executor.tools._tools["fs_read_file"].adapter.allowed_roots[0] / "workspace"  # type: ignore[attr-defined]
    project_root = workspace / "kPony"
    (project_root / "src").mkdir(parents=True)
    (project_root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.20)\n", encoding="utf-8")

    async def _run(**kwargs):
        captured["messages"] = kwargs["messages"]
        return ToolUseResult(final_output="continued software project")

    executor.runtime = SimpleNamespace(
        tool_loop=SimpleNamespace(run=_run),
        scheduler=None,
        ctx=SimpleNamespace(
            context_store=None,
            identity=SimpleNamespace(self_model=SimpleNamespace(name="TestAgent")),
            config=SimpleNamespace(
                primary_workspace_root=lambda: workspace.parent,
                agent_workspace_root=lambda: workspace,
            ),
        ),
    )
    task = RepairTask(
        objective="Return to project \"kPony\".",
        meta={
            "source": "schedule",
            "project_key": "kpony",
            "project_title": "kPony",
            "project_type": "software",
            "project_intent": "continue building kPony until it builds and has proof",
            "next_step": "Run cmake and fix any Qt6 build errors.",
        },
    )

    output = await executor._execute_plan(task, "build and verify")

    assert output == "continued software project"
    system_message = captured["messages"][0]["content"]
    assert "software project" in system_message
    assert "You are TestAgent returning to your own software project" in system_message
    assert "You are an OpenCAS agent" not in system_message
    assert f"Canonical workspace project root: {project_root}" in system_message
    assert "Workspace-relative project root: workspace/kPony" in system_message
    assert "Do not create a new scratch project" in system_message
    assert "build, test" in system_message
    assert "creative project" not in system_message
    assert "manuscript" not in system_message


@pytest.mark.asyncio
async def test_executor_persists_unsaved_writing_task_final_output(executor, tmp_path):
    workspace = tmp_path / "workspace"
    output_path = workspace / "Writing Projects" / "4246" / "chapter_02_v2.md"
    final_output = "# Chapter 2\n\n" + ("The revised chapter continues with concrete dramatized prose.\n" * 120)

    async def _run(**kwargs):
        return ToolUseResult(
            final_output=final_output,
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "workflow_create_writing_task",
                    "args": {
                        "title": "Chapter 2 Revision",
                        "output_path": str(output_path),
                    },
                }
            ],
        )

    executor.runtime = SimpleNamespace(
        tool_loop=SimpleNamespace(run=_run),
        scheduler=None,
        ctx=SimpleNamespace(
            config=SimpleNamespace(agent_workspace_root=lambda: workspace)
        ),
    )

    task = RepairTask(objective="Write the revised Chapter 2.")
    output = await executor._execute_plan(task, "write the chapter")

    assert output == final_output
    assert output_path.read_text(encoding="utf-8") == final_output.rstrip() + "\n"
    assert f"file:{output_path}" in task.artifacts
    assert task.meta["persisted_writing_output"]["path"] == str(output_path)
    assert task.meta["persisted_writing_output"]["reason"] == "final_output_after_writing_task"


@pytest.mark.asyncio
async def test_executor_persists_only_artifact_body_from_wrapped_writing_response(executor, tmp_path):
    workspace = tmp_path / "workspace"
    output_path = workspace / "Writing Projects" / "4246" / "chapter_03_v2.md"
    prose = "Cauldron's heat rolled under the stone while Maren watched the commons breathe.\n" * 130
    final_output = (
        "Now let me deliver the composed prose.\n\n"
        "---\n\n"
        "## writing project 4246 — Chapter 3: Cauldron's Ghost\n\n"
        "**Status:** Composed prose, not yet persisted to artifact. Blocker: no `fs_write_file` tool.\n\n"
        "---\n\n"
        "### Scene 1 — Thermal Commons\n\n"
        f"{prose}\n"
        "---\n\n"
        "## Session Status Report\n\n"
        "**What was accomplished:**\n"
        "- Composed the chapter.\n\n"
        "**What was NOT accomplished:**\n"
        "- The expanded prose was not persisted.\n"
    )

    async def _run(**kwargs):
        return ToolUseResult(
            final_output=final_output,
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "workflow_create_writing_task",
                    "args": {
                        "title": "Chapter 3",
                        "description": "Draft Chapter 3 as full dramatized prose.",
                        "output_path": str(output_path),
                    },
                }
            ],
        )

    executor.runtime = SimpleNamespace(
        tool_loop=SimpleNamespace(run=_run),
        scheduler=None,
        ctx=SimpleNamespace(
            config=SimpleNamespace(agent_workspace_root=lambda: workspace)
        ),
    )

    task = RepairTask(objective="Draft Chapter 3 as full dramatized prose.")
    await executor._execute_plan(task, "write the chapter")

    persisted = output_path.read_text(encoding="utf-8")
    assert persisted.startswith("## writing project 4246")
    assert "Now let me deliver" not in persisted
    assert "not yet persisted" not in persisted
    assert "Session Status Report" not in persisted
    assert "Cauldron's heat rolled" in persisted
    assert task.meta["persisted_writing_output"]["path"] == str(output_path)


@pytest.mark.asyncio
async def test_executor_rejects_false_writing_task_completion_summary(executor, tmp_path):
    workspace = tmp_path / "workspace"
    output_path = workspace / "Writing Projects" / "4246" / "chapter_03_v2.md"
    output_path.parent.mkdir(parents=True)
    scaffold = (
        "# Chapter 3\n\n"
        "> Draft Chapter 3 as full dramatized prose.\n\n"
        "<!-- Created by OpenCAS writing workflow -->\n"
    )
    output_path.write_text(scaffold, encoding="utf-8")
    final_output = (
        "Good. Let me do one final check and verify continuity.\n\n"
        "**Summary of this session's work:**\n"
        "- Drafted full Chapter 3, approximately 3,400 words.\n"
        "- Verified continuity across all chapters.\n"
        "- Scheduled next writing session for Chapter 4.\n"
    )

    async def _run(**kwargs):
        return ToolUseResult(
            final_output=final_output,
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "workflow_create_writing_task",
                    "args": {
                        "title": "Chapter 3",
                        "description": "Draft Chapter 3 as full dramatized prose.",
                        "output_path": str(output_path),
                    },
                }
            ],
        )

    executor.runtime = SimpleNamespace(
        tool_loop=SimpleNamespace(run=_run),
        scheduler=None,
        ctx=SimpleNamespace(
            config=SimpleNamespace(agent_workspace_root=lambda: workspace)
        ),
    )

    task = RepairTask(objective="Draft Chapter 3 as full dramatized prose.")
    output = await executor._execute_plan(task, "write the chapter")

    assert output.startswith("execute failed: writing task did not produce draft prose")
    assert output_path.read_text(encoding="utf-8") == scaffold
    assert "persisted_writing_output" not in task.meta
    assert task.meta["writing_completion_failure"]["path"] == str(output_path)


@pytest.mark.asyncio
async def test_executor_rejects_verification_report_as_writing_artifact(executor, tmp_path):
    workspace = tmp_path / "workspace"
    output_path = workspace / "Writing Projects" / "4246" / "story_4246_ch4_draft.md"
    final_output = (
        "The scaffold file exists but I need a way to write content.\n\n"
        "Given my current tool constraints, I'll present the complete Chapter 4 draft here.\n\n"
        "---\n\n"
        "## Verification and Draft Delivery\n\n"
        "Here is the complete **Chapter 4 — The Sisters of the Reef** draft. I was unable "
        "to write directly to the filesystem with my available tools, but the full text is "
        f"ready for placement at `{output_path}`.\n\n"
        "### Word count estimate: ~3,200 words\n\n"
        "### Continuity verification checklist:\n\n"
        "| Check | Status |\n"
        "|-------|--------|\n"
        "| Names | All new names used consistently |\n\n"
        "### What this chapter accomplishes beyond the synopsis:\n\n"
        + ("It describes the intended chapter function without containing the chapter prose.\n" * 100)
    )

    async def _run(**kwargs):
        return ToolUseResult(
            final_output=final_output,
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "workflow_create_writing_task",
                    "args": {
                        "title": "writing project 4246 — Chapter 4 draft",
                        "description": "Draft Chapter 4 as full novel prose.",
                        "output_path": str(output_path),
                    },
                }
            ],
        )

    executor.runtime = SimpleNamespace(
        tool_loop=SimpleNamespace(run=_run),
        scheduler=None,
        ctx=SimpleNamespace(
            config=SimpleNamespace(agent_workspace_root=lambda: workspace)
        ),
    )

    task = RepairTask(objective="Draft Chapter 4 of writing project 4246 as full novel prose.")
    output = await executor._execute_plan(task, "write the chapter")

    assert output.startswith("execute failed: writing task did not produce draft prose")
    assert not output_path.exists()
    assert "persisted_writing_output" not in task.meta
    assert task.meta["writing_completion_failure"]["path"] == str(output_path)


@pytest.mark.asyncio
async def test_executor_does_not_overwrite_existing_writing_artifact(executor, tmp_path):
    workspace = tmp_path / "workspace"
    output_path = workspace / "Writing Projects" / "4246" / "chapter_02_v2.md"
    output_path.parent.mkdir(parents=True)
    existing = "# Chapter 2\n\n" + ("Existing saved manuscript prose.\n" * 20)
    output_path.write_text(existing, encoding="utf-8")

    async def _run(**kwargs):
        return ToolUseResult(
            final_output="# Chapter 2\n\n" + ("A later summary that should not overwrite.\n" * 20),
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "workflow_create_writing_task",
                    "args": {
                        "title": "Chapter 2 Revision",
                        "output_path": str(output_path),
                    },
                }
            ],
        )

    executor.runtime = SimpleNamespace(
        tool_loop=SimpleNamespace(run=_run),
        scheduler=None,
        ctx=SimpleNamespace(
            config=SimpleNamespace(agent_workspace_root=lambda: workspace)
        ),
    )

    task = RepairTask(objective="Write the revised Chapter 2.")
    await executor._execute_plan(task, "write the chapter")

    assert output_path.read_text(encoding="utf-8") == existing
    assert "persisted_writing_output" not in task.meta


@pytest.mark.asyncio
async def test_executor_rejects_artifact_update_blocker_without_write_or_return(executor, tmp_path):
    workspace = tmp_path / "workspace"
    expansion_path = workspace / "Writing Projects" / "4246" / "story_4246_ch3_expansion.md"
    manuscript_path = workspace / "Writing Projects" / "4246" / "story_4246.md"
    expansion_path.parent.mkdir(parents=True)
    expansion_path.write_text("## Chapter 3\n\nExpanded prose.\n", encoding="utf-8")
    manuscript_path.write_text("## Chapter 3\n\nCompressed prose.\n", encoding="utf-8")

    final_output = (
        "The expanded Chapter 3 fits cleanly into the manuscript.\n\n"
        "## Blocker: No file-write tool available\n\n"
        "I have read-only file tools. I do not have `fs_write_file`, `fs_edit_file`, "
        "or any file modification tool in my current tool set. I cannot perform the actual "
        "integration edit to `story_4246.md`.\n\n"
        "**Manuscript progress status: NOT claimed.** Target artifact `story_4246.md` "
        "has not been modified. The next OpenCAS pass with write capability can execute the edit."
    )

    async def _run(**kwargs):
        return ToolUseResult(
            final_output=final_output,
            tool_calls=[
                {"id": "call-1", "name": "fs_read_file", "args": {"file_path": str(expansion_path)}},
                {"id": "call-2", "name": "fs_read_file", "args": {"file_path": str(manuscript_path)}},
                {
                    "id": "call-3",
                    "name": "grep_search",
                    "args": {"path": str(manuscript_path), "pattern": "^## Chapter 3"},
                },
            ],
        )

    executor.runtime = SimpleNamespace(
        tool_loop=SimpleNamespace(run=_run),
        scheduler=None,
        ctx=SimpleNamespace(
            config=SimpleNamespace(agent_workspace_root=lambda: workspace)
        ),
    )

    task = RepairTask(
        objective=(
            f"Review {expansion_path} and integrate the expanded Chapter 3 prose into "
            f"{manuscript_path} if it fits the current manuscript. If it needs critique before "
            "integration, record the concrete blocker and schedule a near future OpenCAS return. "
            "Do not claim manuscript progress unless the target artifact is actually modified."
        )
    )

    output = await executor._execute_plan(task, "review and integrate")

    assert output.startswith("execute failed:")
    assert "no write/edit tool call" in output
    assert task.meta["artifact_update_failure"]["write_or_return_call_seen"] is False
    assert manuscript_path.read_text(encoding="utf-8") == "## Chapter 3\n\nCompressed prose.\n"


@pytest.mark.asyncio
async def test_executor_plan_includes_shadow_registry_guidance(tmp_path):
    class _FakeLLM:
        def __init__(self) -> None:
            self.messages = None

        async def chat_completion(self, messages, **kwargs):
            self.messages = messages
            return {"choices": [{"message": {"content": "inspect the existing artifact and apply one narrow patch"}}]}

    llm = _FakeLLM()
    executor = RepairExecutor(
        tools=ToolRegistry(),
        llm=llm,
        runtime=SimpleNamespace(
            ctx=SimpleNamespace(
                shadow_registry=SimpleNamespace(
                    build_planning_context=lambda **kwargs: {
                        "available": True,
                        "prompt_block": (
                            "Related blocked-intention clusters:\n"
                            "- 2x retry_blocked for workspace/writing/4246/story_4246.md\n"
                            "Safer alternatives:\n"
                            "- Prefer deterministic review of the existing artifact.\n"
                            "- Prefer one narrow edit and stop after verification.\n"
                        ),
                    }
                )
            )
        ),
    )
    task = RepairTask(
        objective="Continue writing project 4246 from the existing manuscript.",
        meta={
            "resume_project": {
                "canonical_artifact_path": "workspace/writing/4246/story_4246.md",
            }
        },
    )

    plan = await executor._plan(task)

    assert "narrow patch" in plan
    assert llm.messages is not None
    assert "Related blocked-intention clusters" in llm.messages[1]["content"]
    assert "deterministic review" in llm.messages[1]["content"].lower()
