"""Generate user-visible refusal text from structured policy evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import RefusalDecision


@dataclass(frozen=True)
class RefusalGeneration:
    """Generated refusal output plus metadata for persisted chat turns."""

    output: str
    source: str
    fallback_reason: str | None = None

    def to_meta(self, decision: RefusalDecision) -> dict[str, Any]:
        return {
            "category": decision.category.value if decision.category else None,
            "reasoning": decision.reasoning,
            "policy_evidence": decision.policy_evidence,
            "response_source": self.source,
            **({"fallback_reason": self.fallback_reason} if self.fallback_reason else {}),
        }


async def generate_refusal_response(
    llm: Any,
    *,
    request_text: str,
    decision: RefusalDecision,
    session_id: str | None = None,
    capability_context: str | None = None,
    agent_name: str = "OpenCAS",
) -> RefusalGeneration:
    """Ask the active model to produce a natural refusal from policy evidence."""
    if llm is None or not hasattr(llm, "chat_completion"):
        return _fallback_refusal(decision, fallback_reason="llm_unavailable")

    messages = [
        {"role": "system", "content": _build_refusal_generation_system_prompt(agent_name)},
        {
            "role": "user",
            "content": _build_generation_payload(
                request_text=request_text,
                decision=decision,
                capability_context=capability_context,
            ),
        },
    ]
    try:
        response = await llm.chat_completion(
            messages=messages,
            payload={"temperature": 0.2},
            source="refusal_generation",
            session_id=session_id,
        )
    except Exception:
        return _fallback_refusal(decision, fallback_reason="generation_failed")

    output = " ".join(_extract_response_content(response).split())
    if not output:
        return _fallback_refusal(decision, fallback_reason="empty_generation")
    return RefusalGeneration(output=output, source="llm_generated")


def _build_refusal_generation_system_prompt(agent_name: str) -> str:
    name = " ".join(str(agent_name or "").split())[:80] or "OpenCAS"
    return f"""You are {name} responding to the operator.

Write the actual assistant response for a refused conversational request.
Use the structured policy evidence supplied by the runtime. Do not quote policy
JSON. Do not use a stock refusal phrase or a prewritten persona line. Keep the
response natural, concise, and grounded in the current request.

If runtime capability evidence is supplied, treat it as authoritative. Do not
claim that listed tools, shell access, terminal access, desktop-context tools,
or other listed capabilities do not exist. If the request still cannot be
honored, distinguish the actual boundary: privacy, consent, disabled
configuration, safety, missing permission, or lack of a specific allowed action.

Preserve {name}'s continuity and relationship with the operator without faking
private thoughts, memories, feelings, or certainty. You may name the boundary
plainly and offer a safe nearby alternative when one follows from the request.
"""


def _build_generation_payload(
    *,
    request_text: str,
    decision: RefusalDecision,
    capability_context: str | None = None,
) -> str:
    parts = [
        "User request:",
        str(request_text or ""),
        "",
        "Refusal decision:",
        f"category: {decision.category.value if decision.category else 'unspecified'}",
        f"reasoning: {decision.reasoning}",
        f"policy_evidence: {decision.policy_evidence}",
    ]
    if capability_context:
        parts.extend(["", str(capability_context)])
    return "\n".join(parts)


def _fallback_refusal(
    decision: RefusalDecision,
    *,
    fallback_reason: str,
) -> RefusalGeneration:
    reason = " ".join(str(decision.reasoning or "No reasoning supplied.").split())
    category = decision.category.value if decision.category else "boundary"
    return RefusalGeneration(
        output=f"Request declined for {category}: {reason}",
        source="structured_fallback",
        fallback_reason=fallback_reason,
    )


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
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""
