from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.autonomy.models import ActionRiskTier, ApprovalLevel
from opencas.governance import BlockReason, ShadowRegistry, ShadowRegistryStore
from opencas.infra import HookBus
from opencas.runtime.shadow_registry_hooks import register_runtime_shadow_registry_hooks
from opencas.runtime.tool_runtime import execute_runtime_tool
from opencas.tools import FileSystemToolAdapter, ToolRegistry
from opencas.tools.adapters.process import ProcessToolAdapter
from opencas.tools.adapters.shell import ShellToolAdapter
from opencas.tools.validation import create_default_tool_validation_pipeline


class _FakeSomatic:
    def bump_from_work(self, intensity: float = 0.1, success: bool = True) -> None:
        return None

    async def emit_appraisal_event(self, *args, **kwargs) -> None:
        return None


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

    def evaluate(self, request):
        self._decision.action_id = request.action_id
        return self._decision

    async def maybe_record(self, decision, request, score):
        return None


async def _async_no_goals(_output: str) -> list[str]:
    return []


def _make_runtime(tmp_path: Path, *, approval: _FakeApproval) -> SimpleNamespace:
    hook_bus = HookBus()
    config = SimpleNamespace(
        session_id="session:default:abc",
        state_dir=tmp_path,
        agent_workspace_root=lambda: tmp_path,
    )
    shadow_registry = ShadowRegistry(ShadowRegistryStore(tmp_path / "shadow_registry"))
    ctx = SimpleNamespace(
        config=config,
        hook_bus=hook_bus,
        shadow_registry=shadow_registry,
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
    register_runtime_shadow_registry_hooks(runtime)
    return runtime


def _register_process_tool(runtime: SimpleNamespace, *, roots: list[str]) -> None:
    runtime.tools.register(
        "process_start",
        "Start process",
        ProcessToolAdapter(
            supervisor=SimpleNamespace(
                start=lambda scope, command, cwd=None: "proc-1",
                poll=lambda *a, **k: {},
                write=lambda *a, **k: True,
                send_signal=lambda *a, **k: True,
                kill=lambda *a, **k: True,
                clear=lambda *a, **k: 0,
                remove=lambda *a, **k: True,
            ),
            default_cwd=str(roots[0]),
        ),
        ActionRiskTier.SHELL_LOCAL,
    )


@pytest.mark.asyncio
async def test_shadow_registry_captures_approval_denials(tmp_path: Path) -> None:
    runtime = _make_runtime(
        tmp_path,
        approval=_FakeApproval(
            ApprovalLevel.MUST_ESCALATE,
            "shell-local process start requires operator escalation",
        ),
    )
    _register_process_tool(runtime, roots=[str(tmp_path)])

    denied = await execute_runtime_tool(
        runtime,
        "process_start",
        {"command": "python -c 'print(1)'", "cwd": str(tmp_path)},
    )
    assert denied["success"] is False

    entries = runtime.ctx.shadow_registry.list_recent(limit=10)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.block_reason == BlockReason.APPROVAL_DENIED
    assert entry.tool_name == "process_start"
    assert entry.decision_level == ApprovalLevel.MUST_ESCALATE.value


@pytest.mark.asyncio
async def test_shadow_registry_captures_validation_blocks(tmp_path: Path) -> None:
    runtime = _make_runtime(
        tmp_path,
        approval=_FakeApproval(ApprovalLevel.CAN_DO_NOW, "approved"),
    )
    runtime.tools.validation_pipeline = create_default_tool_validation_pipeline(roots=[str(tmp_path)])
    runtime.tools.register(
        "bash_run_command",
        "Run shell",
        ShellToolAdapter(cwd=str(tmp_path), timeout=5.0),
        ActionRiskTier.SHELL_LOCAL,
    )

    blocked = await execute_runtime_tool(
        runtime,
        "bash_run_command",
        {"command": "rm -rf /"},
    )
    assert blocked["success"] is False

    entries = runtime.ctx.shadow_registry.list_recent(limit=10)
    assert len(entries) == 1
    assert entries[0].block_reason == BlockReason.VALIDATION_BLOCKED
    assert "validation" in entries[0].block_context.lower()


@pytest.mark.asyncio
async def test_shadow_registry_captures_shell_safety_blocks(tmp_path: Path) -> None:
    runtime = _make_runtime(
        tmp_path,
        approval=_FakeApproval(ApprovalLevel.CAN_DO_NOW, "approved"),
    )
    roots = [str(tmp_path)]
    runtime.tools.register(
        "bash_run_command",
        "Run shell",
        ShellToolAdapter(cwd=str(tmp_path), timeout=5.0),
        ActionRiskTier.SHELL_LOCAL,
    )

    blocked = await execute_runtime_tool(
        runtime,
        "bash_run_command",
        {"command": "rm -rf /"},
    )
    assert blocked["success"] is False

    entries = runtime.ctx.shadow_registry.list_recent(limit=10)
    assert len(entries) == 1
    assert entries[0].block_reason == BlockReason.SAFETY_BLOCKED
    assert entries[0].intent_summary == "shell:rm -rf /"
