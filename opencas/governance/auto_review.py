"""Auto-review routing for approval decisions that would otherwise escalate."""

from __future__ import annotations

import inspect
import json
import re
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from opencas.autonomy.models import ActionRequest, ActionRiskTier, ApprovalDecision, ApprovalLevel


class AutoReviewMode(str, Enum):
    """Approval routing mode for escalated on-request actions."""

    DEFAULT = "default"
    AUTO_REVIEW = "auto_review"


class AutoReviewOutcome(BaseModel):
    """Decision returned by an auto-reviewer."""

    approved: bool
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reviewer_id: str = "auto-reviewer"
    reasoning: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)


ReviewerFn = Callable[[ActionRequest, ApprovalDecision], AutoReviewOutcome | Awaitable[AutoReviewOutcome]]


def normalize_auto_review_mode(value: AutoReviewMode | str | None) -> AutoReviewMode:
    """Normalize user/config spelling for approval review mode."""
    if isinstance(value, AutoReviewMode):
        return value
    cleaned = str(value or AutoReviewMode.DEFAULT.value).strip().lower().replace("-", "_")
    return AutoReviewMode(cleaned)


class AutoReviewerSubagent:
    """Small LLM-backed reviewer for eligible on-request approval escalations.

    The subagent receives only policy metadata, not raw commands or full tool
    arguments, so the review can improve autonomy without leaking secrets.
    """

    def __init__(
        self,
        *,
        llm: Any = None,
        reviewer_id: str = "auto-reviewer:llm",
    ) -> None:
        self.llm = llm
        self.reviewer_id = reviewer_id

    async def __call__(
        self,
        request: ActionRequest,
        decision: ApprovalDecision,
    ) -> AutoReviewOutcome:
        if self.llm is None or not hasattr(self.llm, "chat_completion"):
            return AutoReviewOutcome(
                approved=False,
                confidence=0.0,
                reviewer_id=self.reviewer_id,
                reasoning="auto-review LLM unavailable",
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the OpenCAS auto-reviewer subagent. Review only eligible "
                    "on-request approval escalations. Approve only when the action is "
                    "workspace-scoped, bounded, non-destructive, and reversible enough "
                    "for autonomous execution. Deny when scope, destination, side "
                    "effects, or user intent remain unclear. Return JSON only with "
                    "approved:boolean, confidence:number, and reasoning:string."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "request": _review_request_payload(request),
                        "decision": {
                            "level": decision.level.value,
                            "score": round(float(decision.score), 3),
                            "confidence": round(float(decision.confidence), 3),
                            "reasoning": decision.reasoning,
                        },
                    },
                    sort_keys=True,
                ),
            },
        ]

        try:
            response = await self.llm.chat_completion(
                messages=messages,
                complexity="low",
                source="auto_review",
                execution_mode="governance",
            )
        except Exception as exc:
            return AutoReviewOutcome(
                approved=False,
                confidence=0.0,
                reviewer_id=self.reviewer_id,
                reasoning=f"auto-review LLM failed: {type(exc).__name__}",
            )

        content = _extract_response_content(response)
        try:
            parsed = _parse_json_object(content)
        except ValueError as exc:
            return AutoReviewOutcome(
                approved=False,
                confidence=0.0,
                reviewer_id=self.reviewer_id,
                reasoning=f"invalid reviewer response: {exc}",
                meta={"raw_length": len(content)},
            )

        return AutoReviewOutcome(
            approved=bool(parsed.get("approved", False)),
            confidence=_coerce_confidence(parsed.get("confidence", 0.5)),
            reviewer_id=self.reviewer_id,
            reasoning=str(parsed.get("reasoning") or parsed.get("reason") or "").strip()
            or "reviewer supplied no reasoning",
            meta={"source": "llm_subagent"},
        )


class AutoReviewPolicy:
    """Route eligible on-request escalation decisions through an auto-reviewer."""

    def __init__(
        self,
        *,
        mode: AutoReviewMode | str = AutoReviewMode.DEFAULT,
        reviewer: Optional[ReviewerFn] = None,
    ) -> None:
        self.mode = normalize_auto_review_mode(mode)
        self.reviewer = reviewer

    def eligibility(self, request: ActionRequest, decision: ApprovalDecision) -> Tuple[bool, str]:
        """Return whether *request* can be routed to auto-review and why."""
        if self.mode != AutoReviewMode.AUTO_REVIEW:
            return False, "mode_default"
        if self.reviewer is None:
            return False, "reviewer_missing"
        if decision.level not in {
            ApprovalLevel.MUST_ESCALATE,
            ApprovalLevel.CAN_DO_AFTER_MORE_EVIDENCE,
        }:
            return False, "decision_not_escalated"
        if not _is_on_request_channel(request):
            return False, "approval_channel_not_on_request"
        if request.tier in {ActionRiskTier.DESTRUCTIVE, ActionRiskTier.EXTERNAL_WRITE}:
            return False, f"tier_ineligible:{request.tier.value}"
        if _decision_has_hard_boundary(decision):
            return False, "explicit_boundary_hit"
        dangerous_reason = _dangerous_payload_reason(request.payload or {})
        if dangerous_reason:
            return False, dangerous_reason
        return True, "eligible"

    async def review(
        self,
        request: ActionRequest,
        decision: ApprovalDecision,
    ) -> tuple[ApprovalDecision, Dict[str, Any]]:
        """Return the final decision plus structured auto-review metadata."""
        eligible, reason = self.eligibility(request, decision)
        meta: Dict[str, Any] = {
            "mode": self.mode.value,
            "eligible": eligible,
            "reason": reason,
        }
        if not eligible:
            return decision, meta

        assert self.reviewer is not None
        try:
            raw = self.reviewer(request, decision)
            outcome = await raw if inspect.isawaitable(raw) else raw
            if not isinstance(outcome, AutoReviewOutcome):
                outcome = AutoReviewOutcome.model_validate(outcome)
        except Exception as exc:
            outcome = AutoReviewOutcome(
                approved=False,
                confidence=0.0,
                reviewer_id="auto-reviewer",
                reasoning=f"reviewer error: {type(exc).__name__}",
                meta={"reason": "reviewer_error"},
            )

        meta.update(outcome.model_dump(mode="json"))
        if outcome.meta.get("reason"):
            meta["reason"] = str(outcome.meta["reason"])
        if outcome.approved:
            reviewed = decision.model_copy(
                update={
                    "level": ApprovalLevel.CAN_DO_WITH_CAUTION,
                    "confidence": outcome.confidence,
                    "score": min(decision.score, 0.44),
                    "reasoning": _append_review_reason(
                        decision.reasoning,
                        "auto_review_approved",
                        outcome,
                    ),
                }
            )
            return reviewed, meta

        reviewed = decision.model_copy(
            update={
                "confidence": max(decision.confidence, outcome.confidence),
                "reasoning": _append_review_reason(
                    decision.reasoning,
                    "auto_review_denied",
                    outcome,
                ),
            }
        )
        return reviewed, meta


def _is_on_request_channel(request: ActionRequest) -> bool:
    payload = request.payload or {}
    for key in ("approval_channel", "approval_route", "sandbox_approval", "approval_policy"):
        if str(payload.get(key, "")).lower().replace("-", "_") == "on_request":
            return True
    return False


_SAFE_REVIEW_PAYLOAD_KEYS = (
    "approval_channel",
    "approval_route",
    "sandbox_approval",
    "approval_policy",
    "approval_mode",
    "write_scope",
    "command_family",
    "command_permission_class",
    "command_executable",
    "command_subcommand",
    "command_scope",
    "command_effective_family",
    "command_effective_permission_class",
    "web_action_class",
    "web_domain",
)


def _review_request_payload(request: ActionRequest) -> Dict[str, Any]:
    payload = request.payload or {}
    safe_payload = {
        key: payload[key]
        for key in _SAFE_REVIEW_PAYLOAD_KEYS
        if key in payload and _safe_scalar(payload[key])
    }
    return {
        "tier": request.tier.value,
        "description": request.description,
        "tool_name": request.tool_name,
        "target_path": request.target_path,
        "payload": safe_payload,
        "memory_evidence_count": len(request.memory_evidence_ids),
    }


def _safe_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _extract_response_content(response: Any) -> str:
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    return str(content or "").strip()


def _parse_json_object(text: str) -> Dict[str, Any]:
    clean = text.strip()
    if not clean:
        raise ValueError("empty response")
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    if not clean.startswith("{"):
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match:
            clean = match.group(0)
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise ValueError("reviewer JSON was not an object")
    return parsed


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.5
    return max(0.0, min(1.0, confidence))


def _decision_has_hard_boundary(decision: ApprovalDecision) -> bool:
    reasoning = decision.reasoning.lower()
    hard_markers = (
        "explicit_boundary_hit",
        "blocked domain",
        "web_trust_blocked",
        "must escalate to operator",
    )
    return any(marker in reasoning for marker in hard_markers)


def _dangerous_payload_reason(payload: Dict[str, Any]) -> str:
    dangerous_values = {"dangerous", "filesystem_destructive", "privilege_escalation"}
    for key in (
        "command_permission_class",
        "command_effective_permission_class",
        "command_family",
        "command_effective_family",
    ):
        value = str(payload.get(key, "")).lower()
        if value in dangerous_values:
            return f"payload_ineligible:{key}={value}"
    return ""


def _append_review_reason(
    original: str,
    marker: str,
    outcome: AutoReviewOutcome,
) -> str:
    reviewer = outcome.reviewer_id or "auto-reviewer"
    reason = outcome.reasoning or "no reasoning supplied"
    return f"{original} | {marker}({reviewer}): {reason}"
