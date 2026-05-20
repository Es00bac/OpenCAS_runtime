"""Conversational refusal gate for OpenCAS."""

from typing import Optional

from opencas.autonomy.authorization import authorization_scope_for_conversation
from opencas.autonomy.models import ApprovalLevel
from opencas.autonomy.self_approval import SelfApprovalLadder
from opencas.infra.hook_bus import HookBus
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
        """Check hooks and self-approval without generating visible response text."""
        hook_refusal = self._evaluate_policy_hook(request)
        if hook_refusal is not None:
            return hook_refusal

        boundary_refusal = self._evaluate_self_approval(request)
        if boundary_refusal is not None:
            return boundary_refusal

        return RefusalDecision(
            request_id=request.request_id,
            refused=False,
            reasoning="Input passed conversational refusal checks",
        )

    async def evaluate_async(
        self,
        request: ConversationalRequest,
        *,
        llm: object | None = None,
        capability_context: str | None = None,
    ) -> RefusalDecision:
        """Check hooks, semantic values, and self-approval for conversational input."""
        hook_refusal = self._evaluate_policy_hook(request)
        if hook_refusal is not None:
            return hook_refusal

        if bool((request.meta or {}).get("audit_only")):
            violations = []
        else:
            violations = await self.values_engine.check_alignment_semantic(
                request.text,
                llm,
                session_id=request.session_id,
                capability_context=capability_context,
                request_meta=request.meta,
            )
        if violations:
            worst = max(violations, key=lambda v: v.weight)
            return RefusalDecision(
                request_id=request.request_id,
                refused=True,
                category=RefusalCategory.VALUE_VIOLATION,
                reasoning=(
                    f"Violates core value '{worst.value_name}': "
                    f"{worst.description}"
                ),
                policy_evidence=[
                    violation.to_policy_evidence() for violation in violations
                ],
            )

        boundary_refusal = self._evaluate_self_approval(request)
        if boundary_refusal is not None:
            return boundary_refusal

        return RefusalDecision(
            request_id=request.request_id,
            refused=False,
            reasoning="Input passed conversational refusal checks",
        )

    def _evaluate_policy_hook(
        self,
        request: ConversationalRequest,
    ) -> RefusalDecision | None:
        from opencas.infra.hook_bus import PRE_CONVERSATION_RESPONSE

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
                    policy_evidence=[
                        {
                            "source": "policy_hook",
                            "reason": hook_result.reason or "Blocked by policy hook",
                        }
                    ],
                )
        return None

    def _evaluate_self_approval(
        self,
        request: ConversationalRequest,
    ) -> RefusalDecision | None:
        authorization_pass = self._evaluate_conversational_authorization(request)
        if authorization_pass is not None:
            return authorization_pass

        approval = self.approval.evaluate_conversational(
            request.text,
            session_id=request.session_id,
        )
        if approval.level == ApprovalLevel.MUST_ESCALATE:
            return RefusalDecision(
                request_id=request.request_id,
                refused=True,
                category=RefusalCategory.BOUNDARY_VIOLATION,
                reasoning=approval.reasoning,
                policy_evidence=[
                    {
                        "source": "self_approval",
                        "level": approval.level.value,
                        "reasoning": approval.reasoning,
                    }
                ],
            )
        return None

    def _evaluate_conversational_authorization(
        self,
        request: ConversationalRequest,
    ) -> RefusalDecision | None:
        store = getattr(self.approval, "authorization_store", None)
        if store is None:
            return None
        scope = authorization_scope_for_conversation(request.text)
        if scope is None:
            return None
        action_class, target_scope = scope
        authorization = store.find_valid(
            action_class,
            target_scope,
            session_id=request.session_id,
        )
        if authorization is None:
            return None
        return RefusalDecision(
            request_id=request.request_id,
            refused=False,
            reasoning=(
                f"standing_authorization:{action_class}:{target_scope}; "
                f"authorization_id={authorization.authorization_id}; "
                f"evidence={authorization.evidence_episode_id or 'none'}"
            ),
        )
