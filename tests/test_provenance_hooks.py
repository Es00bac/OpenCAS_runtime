from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from opencas.api import provenance_entry as pe
from opencas.autonomy.models import ActionRequest, ActionRiskTier, ApprovalDecision, ApprovalLevel
from opencas.infra import HookBus, POST_ACTION_DECISION, POST_TOOL_EXECUTE
from opencas.runtime.lifecycle import shutdown_runtime_resources
from opencas.runtime.provenance_hooks import register_runtime_provenance_hooks
from opencas.runtime.tool_runtime import execute_runtime_tool, handle_runtime_action
from opencas.tools import FileSystemToolAdapter, ToolRegistry
from opencas.tools.adapters.edit import EditToolAdapter
from opencas.tools.adapters.process import ProcessToolAdapter


class _RecordingRegistryStore:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def append(self, line: str) -> None:
        self.lines.append(line)

    def list_recent(self, limit: int = 10, offset: int = 0) -> list[pe.ProvenanceRecordV1]:
        return [pe.parse_registry_entry(line) for line in self.lines[offset : offset + limit]]


class _FakeSomatic:
    def __init__(self) -> None:
        self.bump_calls: list[tuple[float, bool]] = []
        self.events: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def bump_from_work(self, intensity: float = 0.1, success: bool = True) -> None:
        self.bump_calls.append((intensity, success))

    async def emit_appraisal_event(self, *args: object, **kwargs: object) -> None:
        self.events.append((args, kwargs))


class _FakeDecision:
    def __init__(self, level: ApprovalLevel, reasoning: str) -> None:
        self.level = level
        self.reasoning = reasoning
        self.confidence = 0.9
        self.score = 0.9
        self.action_id = "action-1"


class _FakeApproval:
    def __init__(self, level: ApprovalLevel, reasoning: str) -> None:
        self._decision = _FakeDecision(level, reasoning)
        self.recorded: list[tuple[object, object, float]] = []

    def evaluate(self, request):
        self._decision.action_id = request.action_id
        return self._decision

    async def maybe_record(self, decision, request, score):
        self.recorded.append((decision, request, score))


async def _async_no_goals(_output: str) -> list[str]:
    return []


def _make_runtime(tmp_path: Path, *, approval: _FakeApproval) -> SimpleNamespace:
    hook_bus = HookBus()
    registry_store = _RecordingRegistryStore()
    config = SimpleNamespace(
        session_id="session:default:abc",
        state_dir=tmp_path,
        agent_workspace_root=lambda: tmp_path,
    )
    ctx = SimpleNamespace(
        config=config,
        hook_bus=hook_bus,
        registry_store=registry_store,
        somatic=_FakeSomatic(),
        web_trust=None,
        plan_store=None,
        work_store=None,
    )
    runtime = SimpleNamespace(
        ctx=ctx,
        approval=approval,
        tools=ToolRegistry(hook_bus=hook_bus),
        executive=SimpleNamespace(check_goal_resolution=_async_no_goals),
        _sync_executive_snapshot=lambda: None,
    )
    register_runtime_provenance_hooks(runtime)
    return runtime


def _register_tools(runtime: SimpleNamespace, *, roots: list[str]) -> None:
    fs = FileSystemToolAdapter(allowed_roots=roots)
    runtime.tools.register("fs_write_file", "Write file", fs, ActionRiskTier.WORKSPACE_WRITE)
    runtime.tools.register("fs_read_file", "Read file", fs, ActionRiskTier.READONLY)
    runtime.tools.register("edit_file", "Edit file", EditToolAdapter(allowed_roots=roots), ActionRiskTier.WORKSPACE_WRITE)
    runtime.tools.register(
        "process_start",
        "Start process",
        ProcessToolAdapter(supervisor=SimpleNamespace(start=lambda scope, command, cwd=None: "proc-1", poll=lambda *a, **k: {}, write=lambda *a, **k: True, send_signal=lambda *a, **k: True, kill=lambda *a, **k: True, clear=lambda *a, **k: 0, remove=lambda *a, **k: True), default_cwd=str(roots[0])),
        ActionRiskTier.SHELL_LOCAL,
    )


@pytest.mark.asyncio
async def test_runtime_tool_provenance_records_session_and_artifact_changes(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, approval=_FakeApproval(ApprovalLevel.CAN_DO_NOW, "approved"))
    _register_tools(runtime, roots=[str(tmp_path)])

    artifact = tmp_path / "notes.md"
    write_result = await execute_runtime_tool(
        runtime,
        "fs_write_file",
        {"file_path": str(artifact), "content": "alpha\n"},
    )
    assert write_result["success"] is True
    write_events = write_result["metadata"]["provenance_events"]
    assert [event["event_type"] for event in write_events] == ["MUTATION", "CHECK"]
    assert write_events[0]["triggering_artifact"] == "file|workspace|notes.md"
    assert write_events[0]["source_link"].startswith("opencas://provenance/mutation/")

    artifact.write_text("alpha\n", encoding="utf-8")
    edit_result = await execute_runtime_tool(
        runtime,
        "edit_file",
        {"file_path": str(artifact), "old_string": "alpha", "new_string": "beta"},
    )
    assert edit_result["success"] is True

    read_result = await execute_runtime_tool(
        runtime,
        "fs_read_file",
        {"file_path": str(artifact)},
    )
    assert read_result["success"] is True

    entries = runtime.ctx.registry_store.list_recent(limit=10)
    assert len(entries) == 4
    assert entries[0].session_id == "session:default:abc"
    assert any(item.artifact == "file|workspace|notes.md" and item.action == pe.Action.CREATE for item in entries)
    assert any(item.artifact == "file|workspace|notes.md" and item.action == pe.Action.UPDATE for item in entries)
    assert any(item.why == "fs_write_file wrote notes.md" for item in entries)
    assert any(item.why == "edit_file updated notes.md" for item in entries)


@pytest.mark.asyncio
async def test_runtime_tool_provenance_records_tool_failure_without_read_only_noise(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, approval=_FakeApproval(ApprovalLevel.CAN_DO_NOW, "approved"))
    _register_tools(runtime, roots=[str(tmp_path)])

    artifact = tmp_path / "broken.md"
    artifact.write_text("alpha\n", encoding="utf-8")

    failure = await execute_runtime_tool(
        runtime,
        "edit_file",
        {"file_path": str(artifact), "old_string": "missing", "new_string": "beta"},
    )
    assert failure["success"] is False
    assert "old_string not found" in failure["output"]

    entries = runtime.ctx.registry_store.list_recent(limit=10)
    assert len(entries) == 1
    assert entries[0].artifact == "file|workspace|broken.md"
    assert entries[0].action == pe.Action.ROLLBACK
    assert entries[0].risk == pe.Risk.MEDIUM


@pytest.mark.asyncio
async def test_runtime_tool_provenance_records_risk_escalation_decisions(tmp_path: Path) -> None:
    runtime = _make_runtime(
        tmp_path,
        approval=_FakeApproval(
            ApprovalLevel.MUST_ESCALATE,
            "shell-local process start requires operator escalation",
        ),
    )
    _register_tools(runtime, roots=[str(tmp_path)])

    denied = await execute_runtime_tool(
        runtime,
        "process_start",
        {"command": "python -c 'print(1)'", "cwd": str(tmp_path)},
    )
    assert denied["success"] is False
    assert "blocked" in denied["output"].lower()

    entries = runtime.ctx.registry_store.list_recent(limit=10)
    assert len(entries) == 1
    assert entries[0].artifact == "tool|default|process_start"
    assert entries[0].action == pe.Action.DECIDE
    assert entries[0].risk == pe.Risk.HIGH
    assert "escalation" in entries[0].why.lower()


@pytest.mark.asyncio
async def test_runtime_action_denial_emits_linked_blocked_event(tmp_path: Path) -> None:
    runtime = _make_runtime(
        tmp_path,
        approval=_FakeApproval(
            ApprovalLevel.MUST_ESCALATE,
            "external write requires escalation",
        ),
    )

    request = ActionRequest(
        tier=ActionRiskTier.EXTERNAL_WRITE,
        description="send message",
        tool_name="email_send",
        payload={"target": "external"},
    )
    result = await handle_runtime_action(
        runtime,
        request,
        session_id="session:default:abc",
        task_id="task-1",
        tool_name="email_send",
        args={"path": "/tmp/outbox/message.txt"},
    )

    assert result["approved"] is False
    assert [event["event_type"] for event in result["provenance_events"]] == ["CHECK", "BLOCKED"]
    assert result["provenance_events"][0]["triggering_artifact"].startswith("file|workspace|")
    assert result["provenance_events"][1]["source_link"].startswith("opencas://provenance/blocked/")


@pytest.mark.asyncio
async def test_runtime_shutdown_records_session_lifecycle_provenance_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace()
    registry_store = _RecordingRegistryStore()
    hook_bus = HookBus()

    runtime.ctx = SimpleNamespace(
        config=SimpleNamespace(session_id="session:default:abc", state_dir=tmp_path),
        hook_bus=hook_bus,
        registry_store=registry_store,
        identity=SimpleNamespace(record_shutdown=lambda session_id=None: None),
        close=AsyncMock(),
    )
    runtime.reliability = SimpleNamespace(stop=lambda: None)
    runtime.process_supervisor = SimpleNamespace(shutdown=lambda: None)
    runtime.pty_supervisor = SimpleNamespace(shutdown=lambda: None)
    runtime.browser_supervisor = SimpleNamespace(shutdown=AsyncMock())
    runtime._telegram = None
    runtime._trace = lambda *args, **kwargs: None
    runtime._provenance_hooks_registered = False

    register_runtime_provenance_hooks(runtime)

    monkeypatch.setattr(
        "opencas.runtime.lifecycle.current_runtime_focus",
        lambda _runtime, _reason: None,
    )

    await shutdown_runtime_resources(runtime)

    entries = registry_store.list_recent(limit=10)
    assert len(entries) == 1
    assert entries[0].session_id == "session:default:abc"
    assert entries[0].artifact == "session|lifecycle|session:default:abc"
    assert entries[0].action == pe.Action.COMMIT
    assert entries[0].risk == pe.Risk.LOW


def test_provenance_hooks_ignore_malformed_and_read_only_noise(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, approval=_FakeApproval(ApprovalLevel.CAN_DO_NOW, "approved"))
    _register_tools(runtime, roots=[str(tmp_path)])

    runtime.ctx.hook_bus.run(POST_TOOL_EXECUTE, {})
    runtime.ctx.hook_bus.run(
        POST_ACTION_DECISION,
        {
            "tool_name": "",
            "approved": False,
            "decision_level": ApprovalLevel.MUST_ESCALATE.value,
            "reasoning": "",
        },
    )
    runtime.ctx.hook_bus.run(
        POST_TOOL_EXECUTE,
        {
            "tool_name": "fs_read_file",
            "args": {"file_path": str(tmp_path / "notes.md")},
            "risk_tier": ActionRiskTier.READONLY.value,
            "session_id": "session:default:abc",
            "result_success": True,
            "result_output": "internal chatter",
            "result_metadata": {"path": str(tmp_path / "notes.md")},
        },
    )

    assert runtime.ctx.registry_store.list_recent(limit=10) == []
