"""Conversational refusal gate for OpenCAS."""

from typing import Optional

from opencas.autonomy.models import ApprovalDecision, ApprovalLevel
from opencas.autonomy.self_approval import SelfApprovalLadder
from opencas.infra.hook_bus import HookBus, HookResult

from .models import ConversationalRequest, RefusalCategory, RefusalDecision


class ConversationalRefusalGate:
    """Evaluates user input before an LLM response is generated."""

    def __init__(
        self,
        approval: SelfApprovalLadder,
        hook_bus: Optional[HookBus] = None,
    ) -> None:
        self.approval = approval
        self.hook_bus = hook_bus

    def evaluate(self, request: ConversationalRequest) -> RefusalDecision:
        """Check hooks and self-approval ladder for conversational input."""
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

        # 2. Evaluate via self-approval ladder (synthetic READONLY request)
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
        return (
            "I'm not able to respond to that because it falls outside my current operating boundaries."
        )
