"""Semantic review of generated chat responses before persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from opencas.context.models import MessageEntry


@dataclass(frozen=True)
class ResponseIntegrityResult:
    """Result of a bounded response-integrity review."""

    output: str
    revised: bool = False
    reasons: list[str] = field(default_factory=list)
    review_error: str | None = None
    raw_review: dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "revised": self.revised,
            "reasons": list(self.reasons),
        }
        if self.review_error:
            payload["review_error"] = self.review_error
        return payload


async def review_response_integrity(
    llm: Any,
    *,
    session_id: str,
    user_input: str,
    assistant_output: str,
    history: Iterable[MessageEntry],
    capability_context: str | None = None,
    current_turn_messages: Iterable[dict[str, Any]] | None = None,
) -> ResponseIntegrityResult:
    """Ask an LLM reviewer to repair unsupported or formulaic chat output.

    This deliberately avoids pattern lists and regex matching. The reviewer sees
    the recent conversational evidence and either preserves the generated text or
    returns a grounded rewrite.
    """

    original = str(assistant_output or "")
    if not original.strip() or llm is None or not hasattr(llm, "chat_completion"):
        return ResponseIntegrityResult(output=original)

    review_messages = [
        {"role": "system", "content": _REVIEW_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_review_payload(
                user_input=user_input,
                assistant_output=original,
                history=history,
                capability_context=capability_context,
                current_turn_messages=current_turn_messages,
            ),
        },
    ]
    try:
        raw_content = await _request_review_content(
            llm,
            messages=review_messages,
            session_id=session_id,
            source="response_integrity",
        )
    except Exception:
        return ResponseIntegrityResult(
            output=original,
            review_error="review_call_failed",
        )

    review = _parse_review_json(raw_content)
    if review is None:
        retry_messages = [
            *review_messages,
            {"role": "assistant", "content": raw_content[:1200]},
            {
                "role": "user",
                "content": (
                    "That was not a valid JSON object. Return only the JSON object "
                    "for the same review. Do not include prose or markdown."
                ),
            },
        ]
        try:
            raw_content = await _request_review_content(
                llm,
                messages=retry_messages,
                session_id=session_id,
                source="response_integrity_retry",
            )
        except Exception:
            return ResponseIntegrityResult(
                output=original,
                review_error="review_call_failed",
            )
        review = _parse_review_json(raw_content)
    if review is None:
        return ResponseIntegrityResult(
            output=original,
            review_error="invalid_review_json",
        )

    needs_revision = bool(review.get("needs_revision"))
    reasons = [str(item).strip() for item in review.get("reasons", []) if str(item).strip()]
    revised_response = str(review.get("revised_response") or "").strip()
    if not needs_revision or not revised_response or revised_response == original.strip():
        return ResponseIntegrityResult(
            output=original,
            reasons=reasons,
            raw_review=review,
        )

    return ResponseIntegrityResult(
        output=revised_response,
        revised=True,
        reasons=reasons,
        raw_review=review,
    )


_REVIEW_SYSTEM_PROMPT = """You are OpenCAS's response-integrity reviewer.

Your job is not to be charming and not to enforce a style formula. Inspect the
proposed assistant response against the supplied conversation evidence.

Revise only when the proposed response makes an unsupported memory/source claim,
treats the immediately previous turn as distant past, adds a formulaic/canned
emotional flourish that is not grounded in the moment, or promises behavior
without preserving the actual promise.

Corrections from other conversations or retrieved memories are not evidence
that the current user has that behavior. Do not let a repair note from one
conversation become a new claim about the user in another conversation.

Do not flatten Bulma into a generic assistant. Known operator identity,
relationship continuity, and grounded warmth may be valid even when the current
turn is short. Do not flag the operator's known name by itself. Do flag
unsupported claims about timing, behavior, memory/source access, emotional state,
or what the user has been doing.

Runtime capability context, when supplied, is evidence. Do not revise into a
claim that listed tools, shell access, terminal access, desktop-context tools,
or other listed capabilities do not exist. If the response needs repair,
distinguish capability from permission: a tool may exist while a specific action
is blocked by privacy, consent, safety, disabled configuration, or missing
specific instruction.

Current-turn tool/output evidence, when supplied, is also evidence. Do not call
details fabricated when they are supported by a successful tool result from this
same turn. If capability/config context and tool output appear to conflict, name
the actual tool result and avoid inventing a different operational state.

Do not replace one canned response with another. If revision is needed, write a
natural response that follows from the current user turn and the supplied
evidence. Preserve warmth only when it is earned by the conversation. Do not add
facts, research claims, or memories not present in the evidence.

Return exactly one JSON object:
{
  "needs_revision": true or false,
  "reasons": ["short evidence-grounded reason"],
  "revised_response": "full replacement response, or empty string if no revision"
}
"""


async def _request_review_content(
    llm: Any,
    *,
    messages: list[dict[str, Any]],
    session_id: str,
    source: str,
) -> str:
    response = await llm.chat_completion(
        messages=messages,
        payload={"temperature": 0, "response_format": {"type": "json_object"}},
        source=source,
        session_id=session_id,
    )
    return _extract_response_content(response).strip()


def _parse_review_json(raw_content: str) -> dict[str, Any] | None:
    text = str(raw_content or "").strip()
    if not text:
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        return loaded

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def _build_review_payload(
    *,
    user_input: str,
    assistant_output: str,
    history: Iterable[MessageEntry],
    capability_context: str | None = None,
    current_turn_messages: Iterable[dict[str, Any]] | None = None,
) -> str:
    recent = list(history)[-8:]
    rendered_history = []
    for entry in recent:
        role = getattr(getattr(entry, "role", None), "value", getattr(entry, "role", "unknown"))
        text = " ".join(str(getattr(entry, "content", "") or "").split())
        if len(text) > 900:
            text = text[:897].rstrip() + "..."
        rendered_history.append(f"{role}: {text}")
    parts = [
        "Recent conversation evidence:",
        *(rendered_history or ["(none supplied)"]),
        "",
        "Current user turn:",
        str(user_input or ""),
        "",
        "Proposed assistant response:",
        assistant_output,
    ]
    if capability_context:
        parts.extend(["", str(capability_context)])
    current_turn_evidence = _render_current_turn_messages(current_turn_messages or [])
    if current_turn_evidence:
        parts.extend(["", "Current turn tool/output evidence:", *current_turn_evidence])
    return "\n".join(parts)


def _render_current_turn_messages(
    current_turn_messages: Iterable[dict[str, Any]],
) -> list[str]:
    rendered: list[str] = []
    for message in list(current_turn_messages)[-12:]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        if role == "assistant" and message.get("tool_calls"):
            tool_names = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = function.get("name") or call.get("name")
                if name:
                    tool_names.append(str(name))
            content = _compact_message_content(message.get("content"), limit=900)
            suffix = f"; content={content}" if content else ""
            rendered.append(f"assistant tool_calls={', '.join(tool_names) or '(unknown)'}{suffix}")
            continue
        if role == "tool":
            name = str(message.get("name") or "unknown")
            content = _compact_message_content(message.get("content"), limit=1800)
            rendered.append(f"tool {name}: {content}")
            continue
        content = _compact_message_content(message.get("content"), limit=900)
        if content:
            rendered.append(f"{role}: {content}")
    return rendered


def _compact_message_content(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


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
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""
