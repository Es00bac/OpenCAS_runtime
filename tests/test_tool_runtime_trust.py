from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.autonomy.models import ActionRiskTier, ApprovalDecision, ApprovalLevel
from opencas.cognition import CognitiveEventKind, CognitiveStateStore
from opencas.runtime.tool_runtime import execute_runtime_tool
from opencas.somatic import SomaticState
from opencas.tools import ToolRegistry, ToolResult


class _Approval:
    def __init__(self) -> None:
        self.records = []

    def evaluate(self, request):
        return ApprovalDecision(
            level=ApprovalLevel.CAN_DO_NOW,
            action_id=request.action_id,
            confidence=1.0,
            reasoning="test approval",
            score=0.01,
        )

    async def maybe_record(self, decision, request, score):
        self.records.append((decision, request, score))
        return None


class _Somatic:
    def __init__(self) -> None:
        self.state = SomaticState()
        self.bump_calls = []
        self.appraisal_events = []

    def bump_from_work(self, **kwargs):
        self.bump_calls.append(kwargs)
        return None

    async def emit_appraisal_event(self, *args, **kwargs):
        self.appraisal_events.append((args, kwargs))
        return SimpleNamespace()


class _Executive:
    def __init__(self) -> None:
        self.goal_resolution_outputs = []

    async def check_goal_resolution(self, output):
        self.goal_resolution_outputs.append(output)
        return []


class _AffectiveExaminations:
    def __init__(self) -> None:
        self.calls = []

    async def examine_tool_result(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(pressure_metadata=lambda: {"pressure": "observed"})


class _TrustEngine:
    def __init__(self) -> None:
        self.successes = []
        self.failures = []

    async def record_success(self, *, tool_name=None, tier=None):
        self.successes.append((tool_name, tier))

    async def record_failure(self, *, tool_name=None, tier=None, is_recoverable=True):
        self.failures.append((tool_name, tier, is_recoverable))


def _runtime_with_tool(
    result: ToolResult,
    *,
    cognitive_state_store=None,
    affective_examinations=None,
):
    registry = ToolRegistry()

    async def _adapter(name, args):
        return result

    registry.register(
        "workspace_tool",
        "write a workspace artifact",
        _adapter,
        ActionRiskTier.WORKSPACE_WRITE,
    )
    trust_engine = _TrustEngine()
    approval = _Approval()
    somatic = _Somatic()
    executive = _Executive()
    runtime = SimpleNamespace(
        tools=registry,
        approval=approval,
        auto_review=None,
        trust_engine=trust_engine,
        cognitive_state_store=cognitive_state_store,
        executive=executive,
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="s1"),
            hook_bus=None,
            plugin_lifecycle=None,
            somatic=somatic,
            affective_examinations=affective_examinations,
            sandbox=SimpleNamespace(allowed_roots=[]),
        ),
        _trace=lambda *args, **kwargs: None,
        _sync_executive_snapshot=lambda: None,
    )
    registry.runtime = runtime
    return runtime, trust_engine


@pytest.mark.asyncio
async def test_execute_runtime_tool_records_trust_success() -> None:
    runtime, trust_engine = _runtime_with_tool(ToolResult(success=True, output="ok", metadata={}))

    result = await execute_runtime_tool(runtime, "workspace_tool", {}, session_id="s1")

    assert result["success"] is True
    assert trust_engine.successes == [("workspace_tool", ActionRiskTier.WORKSPACE_WRITE)]
    assert trust_engine.failures == []


@pytest.mark.asyncio
async def test_execute_runtime_tool_records_trust_failure() -> None:
    runtime, trust_engine = _runtime_with_tool(ToolResult(success=False, output="no", metadata={}))

    result = await execute_runtime_tool(runtime, "workspace_tool", {}, session_id="s1")

    assert result["success"] is False
    assert trust_engine.failures == [("workspace_tool", ActionRiskTier.WORKSPACE_WRITE, True)]
    assert trust_engine.successes == []


@pytest.mark.asyncio
async def test_execute_runtime_tool_updates_cognitive_state_on_failure(tmp_path) -> None:
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    try:
        runtime, _trust_engine = _runtime_with_tool(
            ToolResult(success=False, output="missing prerequisite", metadata={"error_type": "test"}),
            cognitive_state_store=cognitive,
        )

        result = await execute_runtime_tool(runtime, "workspace_tool", {"path": "artifact.md"}, session_id="s1")

        working = await cognitive.list_working_memory(limit=10)
        surprise = await cognitive.list_recent_events(kind=CognitiveEventKind.SURPRISE, limit=10)
        assert result["success"] is False
        assert any(item.slot == "last_tool_result" and "workspace_tool failed" in item.content for item in working)
        retry = next(item for item in working if item.slot == "counterfactual_retry")
        assert "Recommended changed hypothesis" in retry.content
        assert retry.payload["counterfactual"]["recommended"]["strategy"] == "verify_prerequisite"
        assert any(
            "workspace_tool" in event.summary
            and event.payload["counterfactual"]["recommended"]["strategy"] == "verify_prerequisite"
            for event in surprise
        )
    finally:
        await cognitive.close()


@pytest.mark.asyncio
async def test_audit_only_runtime_tool_does_not_write_runtime_side_effect_channels(tmp_path) -> None:
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    try:
        affective = _AffectiveExaminations()
        runtime, trust_engine = _runtime_with_tool(
            ToolResult(success=True, output="audit output", metadata={}),
            cognitive_state_store=cognitive,
            affective_examinations=affective,
        )

        result = await execute_runtime_tool(
            runtime,
            "workspace_tool",
            {"path": "artifact.md"},
            session_id="audit-session",
            audit_only=True,
        )

        assert result["success"] is True
        assert await cognitive.list_working_memory(limit=10) == []
        assert await cognitive.list_recent_events(session_id="audit-session", limit=10) == []
        assert runtime.approval.records == []
        assert trust_engine.successes == []
        assert trust_engine.failures == []
        assert runtime.ctx.somatic.bump_calls == []
        assert runtime.ctx.somatic.appraisal_events == []
        assert runtime.executive.goal_resolution_outputs == []
        assert affective.calls == []
    finally:
        await cognitive.close()
