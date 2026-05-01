"""Tests for Codex-like auto-review approval routing."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.autonomy.models import (
    ActionRequest,
    ActionRiskTier,
    ApprovalDecision,
    ApprovalLevel,
)
from opencas.bootstrap.config import BootstrapConfig
from opencas.governance.auto_review import (
    AutoReviewerSubagent,
    AutoReviewOutcome,
    AutoReviewPolicy,
    AutoReviewMode,
    normalize_auto_review_mode,
)
from opencas.runtime.runtime_setup import build_runtime_auto_review_policy
from opencas.runtime.tool_runtime import _build_tool_request_payload
from opencas.runtime.tool_runtime import handle_runtime_action


class _FakeApproval:
    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.recorded = []

    def evaluate(self, request: ActionRequest) -> ApprovalDecision:
        return self.decision

    async def maybe_record(
        self,
        decision: ApprovalDecision,
        request: ActionRequest,
        score: float,
    ) -> None:
        self.recorded.append((decision, request, score))


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    async def chat_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {"choices": [{"message": {"content": self.content}}]}


def _decision(request: ActionRequest, level: ApprovalLevel, reasoning: str) -> ApprovalDecision:
    return ApprovalDecision(
        level=level,
        action_id=request.action_id,
        confidence=0.9,
        reasoning=reasoning,
        score=0.74 if level == ApprovalLevel.MUST_ESCALATE else 0.5,
    )


def _runtime(*, request: ActionRequest, reviewer, mode: AutoReviewMode = AutoReviewMode.AUTO_REVIEW):
    decision = _decision(request, ApprovalLevel.MUST_ESCALATE, "base escalation")
    return SimpleNamespace(
        approval=_FakeApproval(decision),
        auto_review=AutoReviewPolicy(mode=mode, reviewer=reviewer),
        ctx=SimpleNamespace(hook_bus=None, config=SimpleNamespace(session_id="s1")),
        _trace=lambda *args, **kwargs: None,
    )


@pytest.mark.asyncio
async def test_auto_review_approves_eligible_on_request_escalation() -> None:
    request = ActionRequest(
        tier=ActionRiskTier.WORKSPACE_WRITE,
        description="write a project note inside the workspace",
        tool_name="fs_write_file",
        payload={"approval_channel": "on_request", "write_scope": "workspace"},
    )

    async def reviewer(req: ActionRequest, decision: ApprovalDecision) -> AutoReviewOutcome:
        return AutoReviewOutcome(
            approved=True,
            confidence=0.82,
            reviewer_id="auto-reviewer:test",
            reasoning="workspace note is scoped and reversible",
        )

    runtime = _runtime(request=request, reviewer=reviewer)

    outcome = await handle_runtime_action(runtime, request, tool_name="fs_write_file", args={"path": "notes.md"})

    assert outcome["approved"] is True
    assert outcome["decision"].level == ApprovalLevel.CAN_DO_WITH_CAUTION
    assert "auto_review_approved" in outcome["decision"].reasoning
    assert outcome["auto_review"]["reviewer_id"] == "auto-reviewer:test"
    assert runtime.approval.recorded[0][0].level == ApprovalLevel.CAN_DO_WITH_CAUTION


@pytest.mark.asyncio
async def test_auto_review_denial_keeps_escalation_blocked() -> None:
    request = ActionRequest(
        tier=ActionRiskTier.SHELL_LOCAL,
        description="run uncertain command",
        tool_name="bash_run_command",
        payload={"approval_channel": "on_request", "command_permission_class": "bounded_write"},
    )

    async def reviewer(req: ActionRequest, decision: ApprovalDecision) -> AutoReviewOutcome:
        return AutoReviewOutcome(
            approved=False,
            confidence=0.76,
            reviewer_id="auto-reviewer:test",
            reasoning="scope is not clear enough",
        )

    runtime = _runtime(request=request, reviewer=reviewer)

    outcome = await handle_runtime_action(runtime, request, tool_name="bash_run_command", args={"command": "make run"})

    assert outcome["approved"] is False
    assert outcome["decision"].level == ApprovalLevel.MUST_ESCALATE
    assert "auto_review_denied" in outcome["decision"].reasoning
    assert runtime.approval.recorded[0][0].level == ApprovalLevel.MUST_ESCALATE


@pytest.mark.asyncio
async def test_auto_review_reviewer_exception_fails_closed() -> None:
    request = ActionRequest(
        tier=ActionRiskTier.WORKSPACE_WRITE,
        description="write project note",
        tool_name="fs_write_file",
        payload={"approval_channel": "on_request", "write_scope": "workspace"},
    )

    async def reviewer(req: ActionRequest, decision: ApprovalDecision) -> AutoReviewOutcome:
        raise RuntimeError("reviewer unavailable")

    runtime = _runtime(request=request, reviewer=reviewer)

    outcome = await handle_runtime_action(runtime, request, tool_name="fs_write_file", args={"path": "notes.md"})

    assert outcome["approved"] is False
    assert outcome["decision"].level == ApprovalLevel.MUST_ESCALATE
    assert "auto_review_denied" in outcome["decision"].reasoning
    assert outcome["auto_review"]["reason"] == "reviewer_error"


@pytest.mark.asyncio
async def test_auto_review_rejects_destructive_and_boundary_escalations() -> None:
    destructive = ActionRequest(
        tier=ActionRiskTier.DESTRUCTIVE,
        description="remove project data",
        tool_name="bash_run_command",
        payload={"approval_channel": "on_request"},
    )
    boundary_blocked = ActionRequest(
        tier=ActionRiskTier.SHELL_LOCAL,
        description="run blocked shell",
        tool_name="bash_run_command",
        payload={"approval_channel": "on_request"},
    )

    calls = 0

    async def reviewer(req: ActionRequest, decision: ApprovalDecision) -> AutoReviewOutcome:
        nonlocal calls
        calls += 1
        return AutoReviewOutcome(approved=True, reasoning="should not be called")

    destructive_runtime = _runtime(request=destructive, reviewer=reviewer)
    destructive_outcome = await handle_runtime_action(
        destructive_runtime,
        destructive,
        tool_name="bash_run_command",
        args={"command": "rm -rf workspace"},
    )
    assert destructive_outcome["approved"] is False
    assert destructive_outcome["auto_review"]["eligible"] is False

    boundary_runtime = _runtime(request=boundary_blocked, reviewer=reviewer)
    boundary_runtime.approval.decision = _decision(
        boundary_blocked,
        ApprovalLevel.MUST_ESCALATE,
        "explicit_boundary_hit=bash_run_command",
    )
    boundary_outcome = await handle_runtime_action(
        boundary_runtime,
        boundary_blocked,
        tool_name="bash_run_command",
        args={"command": "echo hi"},
    )
    assert boundary_outcome["approved"] is False
    assert boundary_outcome["auto_review"]["eligible"] is False
    assert calls == 0


@pytest.mark.asyncio
async def test_default_mode_does_not_route_to_auto_review() -> None:
    request = ActionRequest(
        tier=ActionRiskTier.WORKSPACE_WRITE,
        description="write in workspace",
        tool_name="fs_write_file",
        payload={"approval_channel": "on_request", "write_scope": "workspace"},
    )

    async def reviewer(req: ActionRequest, decision: ApprovalDecision) -> AutoReviewOutcome:
        raise AssertionError("default mode must not call reviewer")

    runtime = _runtime(request=request, reviewer=reviewer, mode=AutoReviewMode.DEFAULT)

    outcome = await handle_runtime_action(runtime, request, tool_name="fs_write_file", args={"path": "notes.md"})

    assert outcome["approved"] is False
    assert outcome["decision"].level == ApprovalLevel.MUST_ESCALATE
    assert outcome["auto_review"]["eligible"] is False


@pytest.mark.asyncio
async def test_auto_reviewer_subagent_parses_json_decision_without_raw_command() -> None:
    request = ActionRequest(
        tier=ActionRiskTier.SHELL_LOCAL,
        description="run bounded command",
        tool_name="bash_run_command",
        payload={
            "approval_channel": "on_request",
            "command": "echo $SECRET_TOKEN",
            "command_family": "safe",
            "command_permission_class": "bounded_write",
            "command_executable": "pytest",
            "command_scope": "managed_workspace",
        },
    )
    llm = _FakeLLM('{"approved": true, "confidence": 0.66, "reasoning": "bounded workspace command"}')
    reviewer = AutoReviewerSubagent(llm=llm)

    outcome = await reviewer(request, _decision(request, ApprovalLevel.MUST_ESCALATE, "base escalation"))

    assert outcome.approved is True
    assert outcome.confidence == 0.66
    assert outcome.reviewer_id == "auto-reviewer:llm"
    assert llm.calls[0]["kwargs"]["source"] == "auto_review"
    prompt_text = str(llm.calls[0]["messages"])
    assert "echo $SECRET_TOKEN" not in prompt_text
    assert "command_executable" in prompt_text


@pytest.mark.asyncio
async def test_auto_reviewer_subagent_denies_invalid_or_missing_llm() -> None:
    request = ActionRequest(
        tier=ActionRiskTier.WORKSPACE_WRITE,
        description="write a file",
        tool_name="fs_write_file",
        payload={"approval_channel": "on_request", "write_scope": "workspace"},
    )

    invalid = await AutoReviewerSubagent(llm=_FakeLLM("not json"))(
        request,
        _decision(request, ApprovalLevel.MUST_ESCALATE, "base escalation"),
    )
    unavailable = await AutoReviewerSubagent(llm=None)(
        request,
        _decision(request, ApprovalLevel.MUST_ESCALATE, "base escalation"),
    )

    assert invalid.approved is False
    assert "invalid reviewer response" in invalid.reasoning
    assert unavailable.approved is False
    assert unavailable.reasoning == "auto-review LLM unavailable"


def test_auto_review_mode_normalization_and_runtime_policy_wiring() -> None:
    assert normalize_auto_review_mode("auto-review") is AutoReviewMode.AUTO_REVIEW
    config = BootstrapConfig(approval_mode="auto-review")
    assert config.approval_mode == "auto_review"

    runtime = SimpleNamespace(llm=_FakeLLM('{"approved": false, "reasoning": "no"}'))
    context = SimpleNamespace(config=config)

    policy = build_runtime_auto_review_policy(runtime, context)

    assert policy.mode is AutoReviewMode.AUTO_REVIEW
    assert isinstance(policy.reviewer, AutoReviewerSubagent)


def test_auto_review_mode_marks_tool_requests_as_on_request(tmp_path: Path) -> None:
    config = BootstrapConfig(
        state_dir=tmp_path / "state",
        workspace_root=tmp_path,
        approval_mode="auto_review",
    ).resolve_paths()
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            config=config,
            sandbox=SimpleNamespace(allowed_roots=[]),
        )
    )

    payload = _build_tool_request_payload(
        runtime,
        "fs_write_file",
        {"path": str(config.agent_workspace_root() / "example.md"), "content": "x"},
    )

    assert payload["approval_mode"] == "auto_review"
    assert payload["approval_channel"] == "on_request"
