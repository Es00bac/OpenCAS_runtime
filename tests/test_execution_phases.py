"""Tests for RepairExecutor explicit phases."""

from types import SimpleNamespace

import pytest
import pytest_asyncio

from opencas.autonomy.models import ActionRiskTier
from opencas.context.models import MessageRole
from opencas.execution import ExecutionPhase, ExecutionStage, RepairExecutor, RepairTask
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
async def test_executor_snapshot_phase_when_scratch_dir_set(executor, tmp_path):
    task = RepairTask(
        objective="check file.txt",
        scratch_dir=str(tmp_path / "scratch"),
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
async def test_executor_runtime_guard_fired_result_fails_task(executor):
    async def _run(**kwargs):
        return ToolUseResult(
            final_output="I made partial progress before pausing after a long tool run.",
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
                    content="Keep working on Chronicle 4246 until it feels complete.",
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
        ctx=SimpleNamespace(context_store=_FakeContextStore()),
    )
    task = RepairTask(
        objective="Return to project \"Chronicle 4246\".",
        meta={
            "source": "schedule",
            "project_key": "chronicle-4246",
            "project_title": "Chronicle 4246",
            "source_session_id": "telegram:private:1",
            "project_intent": (
                "revise and finish the Chronicle 4246 manuscript until the OpenCAS agent is satisfied, "
                "using critique as input without narrowing the project to naming research"
            ),
            "next_step": "Fold the Onnen naming decision into the manuscript.",
        },
    )

    output = await executor._execute_plan(task, "review context and continue")

    assert output == "returned to the project"
    assert captured["session_lookup"]["session_id"] == "telegram:private:1"
    system_message = captured["messages"][0]["content"]
    assert "You are the OpenCAS agent returning to your own creative project" in system_message
    assert "not an external contractor" in system_message
    assert "Chronicle 4246" in system_message
    assert "revise and finish the Chronicle 4246 manuscript" in system_message
    assert "fold the Onnen naming decision" in system_message
    assert "Keep working on Chronicle 4246" in system_message
    assert "creating a workflow scaffold is not manuscript progress" in system_message
    assert "before claiming a chapter, scene, word count, or manuscript milestone" in system_message


@pytest.mark.asyncio
async def test_executor_persists_unsaved_writing_task_final_output(executor, tmp_path):
    workspace = tmp_path / "workspace"
    output_path = workspace / "Chronicles" / "4246" / "chapter_02_v2.md"
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
    output_path = workspace / "Chronicles" / "4246" / "chapter_03_v2.md"
    prose = "Cauldron's heat rolled under the stone while Maren watched the commons breathe.\n" * 130
    final_output = (
        "Now let me deliver the composed prose.\n\n"
        "---\n\n"
        "## Chronicle 4246 — Chapter 3: Cauldron's Ghost\n\n"
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
    assert persisted.startswith("## Chronicle 4246")
    assert "Now let me deliver" not in persisted
    assert "not yet persisted" not in persisted
    assert "Session Status Report" not in persisted
    assert "Cauldron's heat rolled" in persisted
    assert task.meta["persisted_writing_output"]["path"] == str(output_path)


@pytest.mark.asyncio
async def test_executor_rejects_false_writing_task_completion_summary(executor, tmp_path):
    workspace = tmp_path / "workspace"
    output_path = workspace / "Chronicles" / "4246" / "chapter_03_v2.md"
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
    output_path = workspace / "Chronicles" / "4246" / "chronicle_4246_ch4_draft.md"
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
                        "title": "Chronicle 4246 — Chapter 4 draft",
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

    task = RepairTask(objective="Draft Chapter 4 of Chronicle 4246 as full novel prose.")
    output = await executor._execute_plan(task, "write the chapter")

    assert output.startswith("execute failed: writing task did not produce draft prose")
    assert not output_path.exists()
    assert "persisted_writing_output" not in task.meta
    assert task.meta["writing_completion_failure"]["path"] == str(output_path)


@pytest.mark.asyncio
async def test_executor_does_not_overwrite_existing_writing_artifact(executor, tmp_path):
    workspace = tmp_path / "workspace"
    output_path = workspace / "Chronicles" / "4246" / "chapter_02_v2.md"
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
    expansion_path = workspace / "Chronicles" / "4246" / "chronicle_4246_ch3_expansion.md"
    manuscript_path = workspace / "Chronicles" / "4246" / "chronicle_4246.md"
    expansion_path.parent.mkdir(parents=True)
    expansion_path.write_text("## Chapter 3\n\nExpanded prose.\n", encoding="utf-8")
    manuscript_path.write_text("## Chapter 3\n\nCompressed prose.\n", encoding="utf-8")

    final_output = (
        "The expanded Chapter 3 fits cleanly into the manuscript.\n\n"
        "## Blocker: No file-write tool available\n\n"
        "I have read-only file tools. I do not have `fs_write_file`, `fs_edit_file`, "
        "or any file modification tool in my current tool set. I cannot perform the actual "
        "integration edit to `chronicle_4246.md`.\n\n"
        "**Manuscript progress status: NOT claimed.** Target artifact `chronicle_4246.md` "
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
                            "- 2x retry_blocked for workspace/Chronicles/4246/chronicle_4246.md\n"
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
        objective="Continue Chronicle 4246 from the existing manuscript.",
        meta={
            "resume_project": {
                "canonical_artifact_path": "workspace/Chronicles/4246/chronicle_4246.md",
            }
        },
    )

    plan = await executor._plan(task)

    assert "narrow patch" in plan
    assert llm.messages is not None
    assert "Related blocked-intention clusters" in llm.messages[1]["content"]
    assert "deterministic review" in llm.messages[1]["content"].lower()
