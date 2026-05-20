"""Regression coverage for F-004 runtime/registry hook double emission."""

from types import SimpleNamespace

import pytest

from opencas.autonomy.models import ActionRiskTier, ApprovalDecision, ApprovalLevel
from opencas.infra.hook_bus import HookBus, HookResult, POST_TOOL_EXECUTE, PRE_TOOL_EXECUTE
from opencas.runtime.tool_runtime import execute_runtime_tool
from opencas.tools import ToolRegistry, ToolResult


@pytest.mark.asyncio
async def test_runtime_tool_execution_emits_pre_and_post_hooks_once() -> None:
    bus = HookBus()
    registry = ToolRegistry(hook_bus=bus)
    observed_args = []
    pre_count = {"count": 0}
    post_count = {"count": 0}

    def pre_handler(_name, ctx):
        pre_count["count"] += 1
        mutated = dict(ctx)
        args = dict(mutated.get("args") or {})
        args["n"] = int(args.get("n", 0)) + 1
        mutated["args"] = args
        return HookResult(True, mutated_context=mutated)

    def post_handler(_name, _ctx):
        post_count["count"] += 1
        return HookResult(True)

    async def adapter(_name, args):
        observed_args.append(dict(args))
        return ToolResult(True, "ok", {"args": dict(args)})

    bus.register(PRE_TOOL_EXECUTE, pre_handler)
    bus.register(POST_TOOL_EXECUTE, post_handler)
    registry.register(
        "mutating_tool",
        "Mutating test tool",
        adapter,
        risk_tier=ActionRiskTier.WORKSPACE_WRITE,
    )

    class FakeApproval:
        def evaluate(self, request):
            return ApprovalDecision(
                action_id=request.action_id,
                level=ApprovalLevel.CAN_DO_NOW,
                score=1.0,
                confidence=1.0,
                reasoning="test allow",
            )

        async def maybe_record_decision_feedback(self, *args, **kwargs):
            return None

    runtime = SimpleNamespace(
        tools=registry,
        approval=FakeApproval(),
        auto_review=None,
        tracer=None,
        ctx=SimpleNamespace(
            hook_bus=bus,
            plugin_lifecycle=None,
            config=SimpleNamespace(session_id="s"),
            somatic=None,
        ),
        executive=None,
        _sync_executive_snapshot=lambda: None,
    )

    result = await execute_runtime_tool(
        runtime,
        "mutating_tool",
        {"n": 0},
        session_id="s",
        task_id="task-1",
        audit_only=True,
    )

    assert result["success"] is True
    assert pre_count["count"] == 1
    assert post_count["count"] == 1
    assert len(observed_args) == 1
    assert observed_args[0]["n"] == 1
