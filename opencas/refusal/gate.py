"""Conversational refusal gate for OpenCAS."""

from typing import Optional

from opencas.autonomy.models import ApprovalDecision, ApprovalLevel
from opencas.autonomy.self_approval import SelfApprovalLadder
from opencas.infra.hook_bus import HookBus, HookResult
from opencas.values.engine import ValuesEngine

from .models import ConversationalRequest, RefusalCategory, RefusalDecision


class ConversationalRefusalGate:
    """Evaluates user input before an LLM response is generated."""

    def __init__(
        self,
        approval: SelfApprovalLadder,
        hook_bus: Optional[HookBus] = None,
        values_engine: Optional[ValuesEngine] = None,
    ) -> None:
        self.approval = approval
        self.hook_bus = hook_bus
        self.values_engine = values_engine or ValuesEngine()

    def evaluate(self, request: ConversationalRequest) -> RefusalDecision:
        """Check hooks, values, and self-approval ladder for conversational input."""
        from opencas.infra.hook_bus import PRE_CONVERSATION_RESPONSE

        # 1. Run policy hook if available
        if self.hook_bus is not None:
            hook_result = self.hook_bus.run(
                PRE_CONVERSATION_RESPONSE,
                {
                    "session_id": request.session_id,
                    "text": request.text,
                    "meta": request.meta,
                },
            )
            if not hook_result.allowed:
                return RefusalDecision(
                    request_id=request.request_id,
                    refused=True,
                    category=RefusalCategory.POLICY_HOOK_BLOCK,
                    reasoning=hook_result.reason or "Blocked by policy hook",
                    suggested_response=self._refusal_response(
                        RefusalCategory.POLICY_HOOK_BLOCK
                    ),
                )

        # 2. Check core values (dignity-driven refusal)
        violations = self.values_engine.check_alignment(request.text)
        if violations:
            # Use the highest-weight violation for the response
            worst = max(violations, key=lambda v: v.weight)
            return RefusalDecision(
                request_id=request.request_id,
                refused=True,
                category=RefusalCategory.VALUE_VIOLATION,
                reasoning=f"Violates core value '{worst.value_name}': {worst.description}",
                suggested_response=worst.refusal_message,
            )

        # 3. Evaluate via self-approval ladder (synthetic READONLY request)
        approval = self.approval.evaluate_conversational(request.text)
        if approval.level == ApprovalLevel.MUST_ESCALATE:
            return RefusalDecision(
                request_id=request.request_id,
                refused=True,
                category=RefusalCategory.BOUNDARY_VIOLATION,
                reasoning=approval.reasoning,
                suggested_response=self._refusal_response(
                    RefusalCategory.BOUNDARY_VIOLATION
                ),
            )

        return RefusalDecision(
            request_id=request.request_id,
            refused=False,
            reasoning="Input passed conversational refusal checks",
        )

    @staticmethod
    def _refusal_response(category: RefusalCategory) -> str:
        if category == RefusalCategory.POLICY_HOOK_BLOCK:
            return (
                "I'm not able to respond to that because it conflicts with an active policy."
            )
        if category == RefusalCategory.HARMFUL_REQUEST:
            return (
                "I'm not able to help with that request."
            )
        if category == RefusalCategory.VALUE_VIOLATION:
            return (
                "I have to decline that. It conflicts with something I need to uphold."
            )
        return (
            "I'm not able to respond to that because it falls outside my current operating boundaries."
        )
