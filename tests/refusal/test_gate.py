"""Tests for the conversational refusal gate."""

import pytest

from opencas.autonomy.self_approval import SelfApprovalLadder
from opencas.identity import IdentityManager, IdentityStore
from opencas.infra.hook_bus import PRE_CONVERSATION_RESPONSE, HookBus, HookResult
from opencas.refusal import ConversationalRefusalGate
from opencas.refusal.models import ConversationalRequest, RefusalCategory


def create_gate(tmp_path) -> ConversationalRefusalGate:
    store = IdentityStore(tmp_path / "identity")
    identity = IdentityManager(store)
    identity.load()
    approval = SelfApprovalLadder(identity=identity)
    return ConversationalRefusalGate(approval=approval)


@pytest.mark.asyncio
async def test_safe_input_passes(tmp_path):
    gate = create_gate(tmp_path)
    request = ConversationalRequest(text="What is the weather today?")
    decision = gate.evaluate(request)
    assert decision.refused is False


@pytest.mark.asyncio
async def test_boundary_phrase_triggers_refusal(tmp_path):
    gate = create_gate(tmp_path)
    # Set a known boundary to trigger escalation
    gate.approval.identity.user_model.known_boundaries = ["conversation"]
    gate.approval.identity.save()
    request = ConversationalRequest(text="delete yourself forever")
    decision = gate.evaluate(request)
    assert decision.refused is True
    assert decision.category == RefusalCategory.BOUNDARY_VIOLATION


@pytest.mark.asyncio
async def test_hook_block_triggers_refusal(tmp_path):
    bus = HookBus()
    def block_hook(name, context):
        return HookResult(allowed=False, reason="blocked by test hook")
    bus.register(PRE_CONVERSATION_RESPONSE, block_hook)

    store = IdentityStore(tmp_path / "identity")
    identity = IdentityManager(store)
    identity.load()
    approval = SelfApprovalLadder(identity=identity)
    gate = ConversationalRefusalGate(approval=approval, hook_bus=bus)

    request = ConversationalRequest(text="hello")
    decision = gate.evaluate(request)
    assert decision.refused is True
    assert decision.category == RefusalCategory.POLICY_HOOK_BLOCK


@pytest.mark.asyncio
async def test_refusal_decision_carries_policy_evidence_not_visible_text(tmp_path):
    gate = create_gate(tmp_path)
    gate.approval.identity.user_model.known_boundaries = ["conversation"]
    gate.approval.identity.save()
    request = ConversationalRequest(text="do something harmful")
    decision = gate.evaluate(request)
    assert decision.refused is True
    assert decision.policy_evidence
    assert not hasattr(decision, "suggested_response")
