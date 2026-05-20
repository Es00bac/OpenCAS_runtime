"""Semantic review of generated chat responses before persistence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from opencas.context.models import MessageEntry
from opencas.generation.policy import GenerationDomain, GenerationPhase, GenerationPolicyRequest


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
    agent_name: str = "OpenCAS",
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
    if not original.strip():
        return ResponseIntegrityResult(output=original)

    continuity_nudge = _continuity_absence_nudge(
        user_input=user_input,
        assistant_output=original,
        capability_context=capability_context,
    )
    if continuity_nudge is not None:
        return continuity_nudge

    authorship_nudge = _workspace_authorship_denial_nudge(
        user_input=user_input,
        assistant_output=original,
        current_turn_messages=current_turn_messages or [],
    )
    if authorship_nudge is not None:
        return authorship_nudge

    autobiography_nudge = _autobiographical_recall_nudge(
        assistant_output=original,
        current_turn_messages=current_turn_messages or [],
    )
    if autobiography_nudge is not None:
        return autobiography_nudge

    filesystem_recall_nudge = _filesystem_request_recall_only_nudge(
        user_input=user_input,
        assistant_output=original,
        current_turn_messages=current_turn_messages or [],
    )
    if filesystem_recall_nudge is not None:
        return filesystem_recall_nudge

    filesystem_write_nudge = _filesystem_write_request_unexecuted_nudge(
        user_input=user_input,
        assistant_output=original,
        current_turn_messages=current_turn_messages or [],
    )
    if filesystem_write_nudge is not None:
        return filesystem_write_nudge

    web_evidence_nudge = _external_web_request_no_evidence_nudge(
        user_input=user_input,
        assistant_output=original,
        current_turn_messages=current_turn_messages or [],
    )
    if web_evidence_nudge is not None:
        return web_evidence_nudge

    recall_compaction = _verbose_recall_nudge(
        assistant_output=original,
        current_turn_messages=current_turn_messages or [],
    )
    if recall_compaction is not None:
        return recall_compaction

    permission_stall_nudge = _unnecessary_permission_stall_nudge(
        user_input=user_input,
        assistant_output=original,
    )
    if permission_stall_nudge is not None:
        return permission_stall_nudge

    self_state_nudge = _grounded_self_state_context_nudge(
        user_input=user_input,
        assistant_output=original,
        current_turn_messages=current_turn_messages or [],
        capability_context=capability_context,
    )
    if self_state_nudge is not None:
        return self_state_nudge

    turn_scope_nudge = _turn_scoped_no_evidence_nudge(original)
    if turn_scope_nudge is not None:
        return turn_scope_nudge

    if llm is None or not hasattr(llm, "chat_completion"):
        return ResponseIntegrityResult(output=original)

    review_system_prompt = _review_system_prompt(agent_name=agent_name)
    review_messages = [
        {"role": "system", "content": review_system_prompt},
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

    revised_self_state_nudge = _grounded_self_state_context_nudge(
        user_input=user_input,
        assistant_output=revised_response,
        current_turn_messages=current_turn_messages or [],
        capability_context=capability_context,
    )
    if revised_self_state_nudge is not None:
        return ResponseIntegrityResult(
            output=revised_self_state_nudge.output,
            revised=True,
            reasons=[*reasons, *revised_self_state_nudge.reasons],
            raw_review=review,
        )

    revised_turn_scope_nudge = _turn_scoped_no_evidence_nudge(revised_response)
    if revised_turn_scope_nudge is not None:
        return ResponseIntegrityResult(
            output=revised_turn_scope_nudge.output,
            revised=True,
            reasons=[*reasons, *revised_turn_scope_nudge.reasons],
            raw_review=review,
        )

    revised_setup_nudge = _unsupported_revised_action_setup_nudge(revised_response)
    if revised_setup_nudge is not None:
        return ResponseIntegrityResult(
            output=revised_setup_nudge.output,
            revised=True,
            reasons=[*reasons, *revised_setup_nudge.reasons],
            raw_review=review,
        )

    if not _revised_response_needs_second_model_review(revised_response):
        return ResponseIntegrityResult(
            output=revised_response,
            revised=True,
            reasons=reasons,
            raw_review=review,
        )

    second_pass = await _review_revised_response(
        llm,
        session_id=session_id,
        user_input=user_input,
        assistant_output=revised_response,
        history=history,
        agent_name=agent_name,
        capability_context=capability_context,
        current_turn_messages=current_turn_messages,
    )
    if second_pass is not None:
        second_needs_revision = bool(second_pass.get("needs_revision"))
        second_revised_response = str(second_pass.get("revised_response") or "").strip()
        second_reasons = [
            str(item).strip()
            for item in second_pass.get("reasons", [])
            if str(item).strip()
        ]
        if (
            second_needs_revision
            and second_revised_response
            and second_revised_response != revised_response
        ):
            return ResponseIntegrityResult(
                output=second_revised_response,
                revised=True,
                reasons=[*reasons, *second_reasons],
                raw_review={"initial": review, "second_pass": second_pass},
            )

    return ResponseIntegrityResult(
        output=revised_response,
        revised=True,
        reasons=reasons,
        raw_review=review,
    )


_UNSUPPORTED_REVISED_ACTION_SETUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:first,\s*)?let me (?:check|look|inspect|query|open|read|verify)\b", re.IGNORECASE),
    re.compile(r"\b(?:give me|one)\s+(?:a\s+)?moment\b", re.IGNORECASE),
    re.compile(r"\bI(?:'ll| will)\s+(?:check|look|inspect|query|open|read|verify)\b", re.IGNORECASE),
)


def _unsupported_revised_action_setup_nudge(text: str) -> ResponseIntegrityResult | None:
    """Repair a reviewer rewrite that tries to start tool work it cannot perform."""

    original = str(text or "").strip()
    if not original:
        return None
    for pattern in _UNSUPPORTED_REVISED_ACTION_SETUP_PATTERNS:
        match = pattern.search(original)
        if not match:
            continue
        prefix = original[: match.start()].strip(" :;-")
        if not prefix:
            prefix = "I have not checked the needed evidence yet"
        if not prefix.endswith((".", "!", "?")):
            prefix += "."
        return ResponseIntegrityResult(
            output=(
                f"{prefix} The next grounded step is to use the relevant evidence tool "
                "or create a durable task, schedule, receipt, or handoff before claiming "
                "the check is underway."
            ),
            revised=True,
            reasons=["unsupported_revised_action_setup"],
        )
    return None


def _revised_response_needs_second_model_review(text: str) -> bool:
    """Reserve a second LLM review for residual high-risk revision shapes only."""

    lowered = str(text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "i will get back",
            "i'll get back",
            "i will keep working",
            "i'll keep working",
            "report later",
            "when it is done",
            "when it's done",
        )
    )


_REVIEW_SYSTEM_PROMPT_TEMPLATE = """You are OpenCAS's response-integrity reviewer.

Your job is not to be charming and not to enforce a style formula. Inspect the
proposed assistant response against the supplied conversation evidence.

Revise only when the proposed response makes an unsupported memory/source claim,
treats the immediately previous turn as distant past, adds a formulaic/canned
emotional flourish that is not grounded in the moment, or promises behavior
without preserving the actual promise.

Corrections from other conversations or retrieved memories are not evidence
that the current user has that behavior. Do not let a repair note from one
conversation become a new claim about the user in another conversation.

Do not flatten {agent_name} into a generic assistant. Known operator identity,
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

Supplied tool/output evidence is also evidence. Do not call
details fabricated when they are supported by a successful tool result from this
response cycle. If capability/config context and tool output appear to conflict, name
the actual tool result and avoid inventing a different operational state.

Text attachment content included with the current user message is supplied
evidence too. If the supplied evidence contains an attached file excerpt or
full attachment body, the assistant may truthfully analyze that included text
without a separate fs_read_file call. Require fs_read_file only when the
attachment content was not included, was too truncated to support the claim, or
the response claims filesystem facts beyond the included attachment evidence.

Grounded self-state answers are allowed. If the operator asks how {agent_name}
is feeling, what state {agent_name} is in, whether {agent_name} daydreams or
dreams, or what projects/tasks {agent_name} is carrying, use supplied somatic,
wellbeing, self-inspection, cognitive-context, runtime-status, daydream,
schedule, task, receipt, and inner-life/runtime-truth evidence as the grounding.
Do not revise those answers into a generic denial of inner life or independent
agency when the runtime evidence supports a bounded OpenCAS answer. Keep the
distinction clear: grounded OpenCAS state and records are valid evidence;
uninterrupted human-style consciousness, private chain-of-thought, and
unsupported claims remain invalid.

A blocked shell command is not the same thing as no shell access, and it is not
the same thing as no file access. If the proposed response says it cannot
create, update, inspect, or verify a file/proof document because shell was
blocked, but runtime capability context lists filesystem tools such as
fs_write_file, edit_file, fs_read_file, fs_list_dir, or glob_search, revise it.
The repair should distinguish the blocked command from available fallback tools
and should not claim those tools are unavailable.

Claims about completed or current actions require evidence. If the proposed
response says the assistant checked, queried, inspected, verified, created,
scheduled, sent, saved, changed, restarted, started a terminal or PTY session,
or is now going to "give me a moment" to do one of those things, the supplied
evidence must contain supporting tool/output evidence or a cited durable record. If
no such evidence is supplied, revise the response so it states what has not
been checked yet and what evidence would be needed. Do not revise an unsupported
action claim into another unsupported action claim. Do not let a final assistant
message imply background action will happen after the message unless a durable
schedule, task, receipt, or handoff record exists in the supplied evidence.

The reviewer cannot invoke tools. A revised response must be a truthful final
message, not a setup for a tool call that will never happen. Do not write a
revision that says "let me check", "let me look", "first I will inspect", "one
moment", "give me a moment", or ends with a colon introducing an absent tool
call unless the supplied evidence already contains that tool call
or a durable follow-up record. In that case, revise to the grounded truth: what
has not been checked yet, what tool or durable record is needed, and what is
available now.

Claims to follow up later require durable execution evidence. If the proposed
response says the assistant will keep working after the reply, get back to the
operator when done, report later, or otherwise complete work autonomously after
the message, the supplied evidence must include the schedule, task, commitment,
receipt, or handoff that will cause that follow-up to happen without another
operator prompt. If it does not, revise the response to avoid the unsupported
promise and name the missing durable record.

Do not replace one canned response with another. If revision is needed, write a
natural response that follows from the current user message and the supplied
evidence. Preserve warmth only when it is earned by the conversation. Do not add
facts, research claims, or memories not present in the evidence.

Return exactly one JSON object:
{
  "needs_revision": true or false,
  "reasons": ["short evidence-grounded reason"],
  "revised_response": "full replacement response, or empty string if no revision"
}
"""


def _review_system_prompt(*, agent_name: str = "OpenCAS") -> str:
    name = " ".join(str(agent_name or "OpenCAS").split()) or "OpenCAS"
    return _REVIEW_SYSTEM_PROMPT_TEMPLATE.replace("{agent_name}", name)


def _workspace_authorship_denial_nudge(
    *,
    user_input: str = "",
    assistant_output: str,
    current_turn_messages: Iterable[dict[str, Any]],
) -> ResponseIntegrityResult | None:
    path = _workspace_authorship_denial_path(assistant_output)
    if not path:
        path = _workspace_underclaim_path(
            user_input=user_input,
            assistant_output=assistant_output,
        )
    if path:
        if _current_turn_called_tool(current_turn_messages, "artifact_lookup"):
            return None
        return _artifact_lookup_nudge(path)

    workflow_kind = _workflow_underclaim_kind(
        user_input=user_input,
        assistant_output=assistant_output,
    )
    if workflow_kind is not None:
        required_tools = _WORKFLOW_UNDERCLAIM_LOOKUPS[workflow_kind]
        if _current_turn_called_any_tool(current_turn_messages, required_tools):
            return None
        tool_list = " or ".join(required_tools)
        revised = (
            f"I have not checked {tool_list} for that {workflow_kind} yet, so I cannot "
            f"honestly say I do not recognize it. The next grounded step is to retrieve "
            "the workflow record and answer from that evidence."
        )
        return ResponseIntegrityResult(
            output=revised,
            revised=True,
            reasons=[f"requires_{workflow_kind}_lookup"],
        )

    return None


def _artifact_lookup_nudge(path: str) -> ResponseIntegrityResult:
    if path == "the referenced workspace artifact":
        revised = (
            "I have not checked artifact_lookup for the referenced workspace artifact yet, "
            "so I cannot honestly say I do not recognize it or did not work on it. The "
            "next grounded step is to call artifact_lookup with the artifact path and "
            "answer from that timeline."
        )
    else:
        revised = (
            f"I have not checked artifact_lookup for {path} yet, so I cannot honestly "
            "say there is no evidence I wrote, created, or worked on it. The next "
            f"grounded step is to call artifact_lookup(path={json.dumps(path)}) and "
            "answer from that timeline."
        )
    return ResponseIntegrityResult(
        output=revised,
        revised=True,
        reasons=["requires_artifact_lookup"],
    )


def _workspace_underclaim_path(*, user_input: str, assistant_output: str) -> str | None:
    if not _looks_like_underclaim(assistant_output):
        return None
    combined = f"{user_input}\n{assistant_output}"
    path = _extract_workspace_path(combined)
    if path:
        return path
    if not _mentions_workspace_artifact(user_input):
        return None
    if _mentions_workspace_artifact(combined):
        return "the referenced workspace artifact"
    return None


_WORKFLOW_UNDERCLAIM_LOOKUPS: dict[str, tuple[str, ...]] = {
    "schedule": ("workflow_list_schedules", "workflow_get_schedule", "workflow_status"),
    "commitment": ("workflow_list_commitments", "workflow_get_commitment", "workflow_status"),
    "plan": ("workflow_list_plans", "workflow_get_plan", "workflow_status"),
    "task": ("workflow_list_tasks", "workflow_get_task", "workflow_status"),
}


def _workflow_underclaim_kind(*, user_input: str, assistant_output: str) -> str | None:
    if not _looks_like_underclaim(assistant_output):
        return None
    combined = f"{user_input}\n{assistant_output}".lower()
    for kind in _WORKFLOW_UNDERCLAIM_LOOKUPS:
        if re.search(rf"\b{re.escape(kind)}s?\b", combined):
            return kind
    return None


def _looks_like_underclaim(text: str) -> bool:
    patterns = (
        r"\bI\s+(?:do\s+not|don't)\s+(?:recognize|remember|know)\b",
        r"\b(?:I\s+am\s+not|I'm\s+not)\s+sure\s+(?:if|whether)\s+I\s+"
        r"(?:worked\s+on|wrote|created|made|generated|authored|scheduled|planned)\b",
        r"\bI\s+(?:cannot|can't)\s+(?:recognize|confirm|verify)\b",
        r"\bI\s+have\s+no\s+(?:record|evidence)\s+of\s+(?:that|this|it)\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _continuity_absence_nudge(
    *,
    user_input: str,
    assistant_output: str,
    capability_context: str | None,
) -> ResponseIntegrityResult | None:
    if not _looks_like_continuity_absence_question(user_input):
        return None
    snapshot = _parse_continuity_clock_snapshot(capability_context or "")
    if not snapshot:
        return None
    elapsed = snapshot.get("elapsed_since_last_shutdown_or_persistence", "").strip()
    if (
        elapsed
        and elapsed != "unavailable"
        and elapsed.lower() in assistant_output.lower()
        and _continuity_clock_leads_answer(assistant_output, elapsed)
    ):
        return ResponseIntegrityResult(
            output=assistant_output,
            reasons=["continuity_clock_present"],
        )

    if elapsed and elapsed != "unavailable":
        parts = [f"The continuity clock says the measured offline gap is {elapsed}."]
        started = snapshot.get("last_offline_started_at", "").strip()
        boot = snapshot.get("last_boot_time", "").strip()
        if started or boot:
            anchor_bits = []
            if started:
                anchor_bits.append(f"last offline anchor {started}")
            if boot:
                anchor_bits.append(f"boot {boot}")
            parts.append("Clock evidence: " + "; ".join(anchor_bits) + ".")
    else:
        parts = [
            "I do not have a prior persistence anchor, so I cannot compute a truthful elapsed absence from continuity state."
        ]
    score = snapshot.get("continuous_present_score", "").strip()
    if score:
        parts.append(f"Continuous present score is {score}.")
    breadcrumb = snapshot.get("latest_breadcrumb", "").strip()
    if breadcrumb:
        parts.append(f"The latest persisted continuity breadcrumb is: {breadcrumb}.")
    parts.append(
        "Older autobiographical recall can help with project details after the clock answer, but it cannot replace the continuity clock for elapsed time."
    )
    return ResponseIntegrityResult(
        output=" ".join(parts),
        revised=True,
        reasons=["continuity_clock_omitted"],
    )


def _continuity_clock_leads_answer(assistant_output: str, elapsed: str) -> bool:
    prefix = str(assistant_output or "").strip().lower()[:220]
    elapsed_text = str(elapsed or "").lower()
    if elapsed_text not in prefix:
        return False
    return prefix.startswith(
        (
            "the continuity clock",
            "continuity clock",
            "the measured offline gap",
            "measured offline gap",
            "the offline gap",
            "offline gap",
            "i just came back online",
        )
    )


def _looks_like_continuity_absence_question(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        phrase in lowered
        for phrase in (
            "how long have i been gone",
            "how long was i gone",
            "how long were you offline",
            "how long have you been offline",
            "what were we doing last",
            "what were you doing last",
            "since restart",
            "before the restart",
        )
    )


def _parse_continuity_clock_snapshot(context: str) -> dict[str, str]:
    line = ""
    for candidate in str(context or "").splitlines():
        if "Continuity clock:" in candidate:
            line = candidate
            break
    if not line:
        return {}
    payload = line.split("Continuity clock:", 1)[1]
    parsed: dict[str, str] = {}
    for key in (
        "elapsed_since_last_shutdown_or_persistence",
        "last_offline_started_at",
        "last_boot_time",
        "last_persisted_at",
        "continuous_present_score",
        "prior_persistence",
    ):
        match = re.search(
            rf"(?:^|;)\s*{re.escape(key)}=(?P<value>[^;]+)",
            payload,
        )
        if match:
            parsed[key] = match.group("value").strip()
    breadcrumb_match = re.search(r"(?:^|;)\s*latest_breadcrumb=(?P<value>.+)$", payload)
    if breadcrumb_match:
        parsed["latest_breadcrumb"] = breadcrumb_match.group("value").strip()
    return parsed


def _workspace_authorship_denial_path(text: str) -> str | None:
    patterns = (
        r"\bI\s+(?:do\s+not|don't)\s+have\s+evidence\s+(?:that\s+I|of\s+having|of)\s+"
        r"(?:wrote|written|created|made|generated|authored)\s+(?P<path>\S+)",
        r"\bI\s+have\s+no\s+(?:record|evidence)\s+(?:of\s+having\s+)?"
        r"(?:I\s+)?(?:wrote|written|created|made|generated|authored)\s+(?P<path>\S+)",
        r"\bI\s+cannot\s+confirm\s+I\s+(?:wrote|created|made|generated|authored)\s+"
        r"(?P<path>\S+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        path = match.group("path").strip().strip("`'\".,;:)")
        if _is_workspace_path(path):
            return path
    return None


def _is_workspace_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith("workspace/") or "/workspace/" in normalized


def _extract_workspace_path(text: str) -> str | None:
    for candidate in re.findall(r"(?P<path>(?:/|\.)?\S*workspace/\S+)", text):
        path = candidate.strip().strip("`'\".,;:)")
        if _is_workspace_path(path):
            return path
    return None


def _mentions_workspace_artifact(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:workspace|artifact|file|document|doc)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _current_turn_called_tool(
    current_turn_messages: Iterable[dict[str, Any]],
    tool_name: str,
) -> bool:
    return _current_turn_called_any_tool(current_turn_messages, (tool_name,))


def _current_turn_called_any_tool(
    current_turn_messages: Iterable[dict[str, Any]],
    tool_names: Iterable[str],
) -> bool:
    wanted = {str(item) for item in tool_names}
    for message in current_turn_messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("name") or "") in wanted:
            return True
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = function.get("name") or call.get("name")
            if str(name or "") in wanted:
                return True
    return False


def _autobiographical_recall_nudge(
    *,
    assistant_output: str,
    current_turn_messages: Iterable[dict[str, Any]],
) -> ResponseIntegrityResult | None:
    if not _looks_like_memory_denial_after_recall(assistant_output):
        return None
    packet = _current_turn_autobiographical_packet(current_turn_messages)
    if packet is None and not _current_turn_has_raw_self_memory(current_turn_messages):
        return None
    essence = ""
    if isinstance(packet, dict):
        essence = _compact_message_content(packet.get("essence"), limit=360)
    if not essence:
        essence = "I found self-evidence in the retrieved records."
    return ResponseIntegrityResult(
        output=(
            "I did find autobiographical evidence in my stored records. I reconstruct "
            "this from stored autobiographical records rather than continuous human "
            f"consciousness, but the evidence is grounded: {essence}"
        ),
        revised=True,
        reasons=["autobiographical_recall_denial"],
    )


def _verbose_recall_nudge(
    *,
    assistant_output: str,
    current_turn_messages: Iterable[dict[str, Any]],
) -> ResponseIntegrityResult | None:
    packet = _current_turn_autobiographical_packet(current_turn_messages)
    if packet is None:
        return None
    text = str(assistant_output or "")
    inline_excerpt_count = len(
        re.findall(
            r"\b(?:episode excerpt|excerpt|score:)\b|\[(?:ACTION|TURN)\]",
            text,
            re.IGNORECASE,
        )
    )
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    redundant = any(
        _paragraph_overlap(paragraphs[i], paragraphs[i + 1]) > 0.30
        for i in range(len(paragraphs) - 1)
    )
    if len(text) < 4000 and inline_excerpt_count <= 3 and not redundant:
        return None
    essence = _compact_message_content(packet.get("essence"), limit=520)
    evidence = packet.get("strongest_evidence") if isinstance(packet, dict) else []
    top = ""
    if isinstance(evidence, list) and evidence:
        first = evidence[0] if isinstance(evidence[0], dict) else {}
        top = str(first.get("label") or first.get("excerpt") or "").strip()
    output = essence or "I recovered the strongest autobiographical evidence."
    if top and top.casefold() not in output.casefold():
        output += f" Strongest evidence: {top}."
    return ResponseIntegrityResult(
        output=output,
        revised=True,
        reasons=["verbose_recall"],
    )


def _filesystem_request_recall_only_nudge(
    *,
    user_input: str,
    assistant_output: str,
    current_turn_messages: Iterable[dict[str, Any]],
) -> ResponseIntegrityResult | None:
    messages = list(current_turn_messages or [])
    if not _looks_like_filesystem_read_request(user_input):
        return None
    if _current_turn_autobiographical_packet(messages) is None:
        return None
    if _has_filesystem_evidence(messages):
        return None
    if not _looks_like_recall_only_filesystem_answer(assistant_output):
        return None
    return ResponseIntegrityResult(
        output=(
            "Autobiographical recall is only orientation for this request. The "
            "operator asked me to read filesystem/project material, so I need to "
            "use filesystem evidence before answering: start with fs_list_dir, "
            "glob_search, or grep_search to locate the relevant path, then use "
            "fs_read_file on the actual file contents and answer from those results."
        ),
        revised=True,
        reasons=["requires_filesystem_read"],
    )


def _looks_like_filesystem_read_request(text: str) -> bool:
    normalized = str(text or "").lower()
    file_markers = (
        r"\bfile(?:s)?\b",
        r"\bfolder(?:s)?\b",
        r"\bdirector(?:y|ies)\b",
        r"\bworkspace/",
        r"\bwriting/",
        r"\bchapter(?:s)?\b",
        r"\bmanuscript\b",
        r"\bdraft(?:s)?\b",
        r"\bchat logs?\b",
        r"\blog(?:s)?\b",
        r"\bsource\b",
        r"\brepo(?:sitory)?\b",
        r"\bcode\b",
    )
    if not any(re.search(marker, normalized) for marker in file_markers):
        return False
    action_markers = (
        r"\bread(?:\s+through)?\b",
        r"\bgo\s+read\b",
        r"\bopen\b",
        r"\binspect\b",
        r"\bexamine\b",
        r"\bcheck\b",
        r"\blist\b",
        r"\blook\s+(?:at|through|in)\b",
        r"\bfind\b",
        r"\bwhat(?:'s| is)\s+there\b",
    )
    if any(re.search(marker, normalized) for marker in action_markers):
        return True
    return bool(re.search(r"\bread\s+(?:it|them|that)\b", normalized))


def _has_filesystem_evidence(current_turn_messages: Iterable[dict[str, Any]]) -> bool:
    return _current_turn_called_any_tool(
        current_turn_messages,
        (
            "fs_read_file",
            "fs_list_dir",
            "grep_search",
            "glob_search",
            "workspace_get_file_gist",
            "workspace_list_directory_gists",
            "read_file",
            "list_directory",
        ),
    )


def _looks_like_recall_only_filesystem_answer(text: str) -> bool:
    normalized = str(text or "").lower()
    recall_markers = (
        r"\bi\s+can\s+reconstruct\b",
        r"\bstrongest\s+evidence\b",
        r"\bautobiographical\s+evidence\b",
        r"\bacross\s+\d+\s+sessions\b",
        r"\bfs_write_file\s+touched\b",
        r"\bstored\s+(?:records|memory|memories)\b",
    )
    return any(re.search(marker, normalized) for marker in recall_markers)


def _filesystem_write_request_unexecuted_nudge(
    *,
    user_input: str,
    assistant_output: str,
    current_turn_messages: Iterable[dict[str, Any]],
) -> ResponseIntegrityResult | None:
    messages = list(current_turn_messages or [])
    if not _looks_like_filesystem_write_request(user_input):
        return None
    if _has_filesystem_write_evidence(messages):
        return None
    if not _looks_like_unexecuted_filesystem_write_answer(assistant_output):
        return None
    return ResponseIntegrityResult(
        output=(
            "I have not made the requested file edit yet. The available filesystem "
            "write tools need to be used in this response cycle: locate and read the "
            "target as needed, call edit_file or fs_write_file to make the change, "
            "then verify the result before answering."
        ),
        revised=True,
        reasons=["requires_filesystem_write"],
    )


def _looks_like_filesystem_write_request(text: str) -> bool:
    normalized = str(text or "").lower()
    target_markers = (
        r"\bfile(?:s)?\b",
        r"\bfolder(?:s)?\b",
        r"\bdirector(?:y|ies)\b",
        r"\bworkspace/",
        r"\bwriting/",
        r"\bchapter(?:s)?\b",
        r"\bmanuscript\b",
        r"\bdraft(?:s)?\b",
        r"\brepo(?:sitory)?\b",
        r"\bcode\b",
    )
    if not any(re.search(marker, normalized) for marker in target_markers):
        return False
    action_markers = (
        r"\bedit\b",
        r"\bwrite\b",
        r"\brewrite\b",
        r"\bpatch\b",
        r"\bupdate\b",
        r"\bmodify\b",
        r"\bchange\b",
        r"\bfix\b",
        r"\brepair\b",
        r"\bintegrate\b",
        r"\bapply\b",
        r"\bfinish\b",
        r"\bcontinue\b",
        r"\bimplement\b",
        r"\bmake\s+(?:the\s+)?(?:change|edit|fix)\b",
    )
    return any(re.search(marker, normalized) for marker in action_markers)


def _has_filesystem_write_evidence(current_turn_messages: Iterable[dict[str, Any]]) -> bool:
    return _current_turn_called_any_tool(
        current_turn_messages,
        (
            "edit_file",
            "fs_write_file",
            "write_file",
            "apply_patch",
        ),
    )


def _looks_like_unexecuted_filesystem_write_answer(text: str) -> bool:
    normalized = str(text or "").lower()
    tool_reference = re.search(r"\b(?:edit_file|fs_write_file|write_file|apply_patch)\b", normalized)
    patterns = (
        r"\bi\s+(?:have\s+not|haven't)\s+(?:patched|edited|written|updated|changed|modified|fixed)\b",
        r"\bno\s+(?:successful\s+)?(?:edit|write|patch|update|change)\s+tool\s+call\s+(?:has\s+been\s+)?(?:run|made|performed)\b",
        r"\bno\s+(?:successful\s+)?(?:edit_file|fs_write_file|write_file|apply_patch)\b",
        r"\bnext\s+step\s+(?:needs|is)\s+(?:to\s+)?(?:be\s+)?(?:real\s+)?(?:edit|write|patch|update)\b",
        r"\bto\s+actually\s+finish\b.*\b(?:edit_file|fs_write_file|write_file|apply_patch)\b",
        r"\bneed\s+to\s+(?:use|call|run)\s+(?:the\s+)?(?:real\s+)?(?:edit_file|fs_write_file|write_file|apply_patch)\b",
        r"\b(?:edit|write)\s+tools?\s+(?:are\s+)?available\b.*\b(?:but|however)\b.*\bno\s+(?:successful\s+)?(?:edit|write|tool)\b",
        r"\b(?:cannot|can't)\s+(?:edit|write|patch|update|modify)\b.*\b(?:tool|shell|filesystem|file)\b",
    )
    if any(re.search(pattern, normalized, flags=re.DOTALL) for pattern in patterns):
        return True
    return bool(
        tool_reference
        and re.search(
            r"\b(?:next\s+step|need\s+to|would\s+need\s+to|still\s+need\s+to|hasn't\s+been|has\s+not\s+been)\b",
            normalized,
        )
    )


def _external_web_request_no_evidence_nudge(
    *,
    user_input: str,
    assistant_output: str,
    current_turn_messages: Iterable[dict[str, Any]],
) -> ResponseIntegrityResult | None:
    messages = list(current_turn_messages or [])
    if not _looks_like_external_web_research_request(user_input):
        return None
    if _has_external_web_evidence(messages):
        return None
    if not _looks_like_no_external_web_evidence_answer(assistant_output):
        return None
    return ResponseIntegrityResult(
        output=(
            "This request needs current external web evidence before a factual report. "
            "Use an available web evidence path now: web_search, web_fetch, http_request, "
            "or browser_start plus browser_navigate/browser_snapshot. If every relevant "
            "path fails, report the exact tool blocker and attempted fallback instead of "
            "answering from missing supplied evidence."
        ),
        revised=True,
        reasons=["requires_web_research"],
    )


def _looks_like_external_web_research_request(text: str) -> bool:
    normalized = str(text or "").lower()
    external_markers = (
        "2channel",
        "2ch",
        "5channel",
        "5ch",
        "5ちゃん",
        "board",
        "browser",
        "current",
        "forum",
        "headline",
        "http://",
        "https://",
        "internet",
        "latest",
        "live",
        "news",
        "online",
        "page",
        "recent",
        "site",
        "thread",
        "today",
        "url",
        "web",
    )
    research_markers = (
        "browse",
        "fetch",
        "find",
        "look up",
        "report",
        "research",
        "search",
        "summarize",
        "verify",
        "what people are talking about",
    )
    return (
        any(marker in normalized for marker in external_markers)
        and any(marker in normalized for marker in research_markers)
    )


def _has_external_web_evidence(current_turn_messages: Iterable[dict[str, Any]]) -> bool:
    return _current_turn_called_any_tool(
        current_turn_messages,
        (
            "web_search",
            "web_fetch",
            "http_request",
            "browser_start",
            "browser_navigate",
            "browser_snapshot",
        ),
    )


def _looks_like_no_external_web_evidence_answer(text: str) -> bool:
    normalized = str(text or "").lower()
    patterns = (
        r"\bno\s+(?:browser,\s*)?(?:search,\s*)?(?:or\s+)?fetch\s+(?:output|evidence|results?)\s+(?:was|were)\s+supplied\b",
        r"\bno\s+browser,\s*search,\s*or\s*fetch\s+(?:output|evidence|results?)\s+(?:was|were)\s+supplied\b",
        r"\bno\s+(?:browser|search|fetch|web|http|browsing)\s+(?:output|evidence|results?)\s+(?:was|were)\s+supplied\b",
        r"\b(?:browser|search|fetch|web|http|browsing)\s+(?:output|evidence|results?)\s+(?:was|were)\s+not\s+supplied\b",
        r"\bno\s+live\s+(?:browsing|web|search)\s+evidence\b",
        r"\breal\s+report\s+would\s+need\s+live\s+(?:browsing|search|web)\b",
        r"\bcan'?t\s+truthfully\s+name\b.*\b(?:verified|current|live)\b.*\b(?:threads?|links?|sources?)\b",
        r"\bcan'?t\s+(?:verify|confirm)\b.*\b(?:without|no)\s+(?:browser|search|fetch|web|browsing)\b",
    )
    return any(re.search(pattern, normalized, flags=re.DOTALL) for pattern in patterns)


def _looks_like_memory_denial_after_recall(text: str) -> bool:
    patterns = (
        r"\bi\s+only\s+retrieve\b",
        r"\bi\s+don'?t\s+(?:truly\s+)?remember\b",
        r"\bi\s+can'?t\s+say\s+i\s+remember\b",
        r"\bi\s+don'?t\s+have\s+continuous\s+(?:autobiographical\s+)?memory\b",
        r"\bnot\s+memories\s+i\s+experience\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _turn_scoped_no_evidence_nudge(text: str) -> ResponseIntegrityResult | None:
    if not _looks_like_turn_scoped_no_evidence(text):
        return None
    return ResponseIntegrityResult(
        output=(
            "Evidence lookup is required before answering. Use the relevant memory, "
            "autobiographical recall, runtime record, workflow record, filesystem, "
            "or artifact provenance source, then answer from the lookup result."
        ),
        revised=True,
        reasons=["turn_scoped_no_evidence"],
    )


def _looks_like_turn_scoped_no_evidence(text: str) -> bool:
    normalized = str(text or "").lower()
    patterns = (
        r"\b(?:don'?t|do not)\s+have\s+evidence\s+in\s+this\s+turn\b",
        r"\b(?:don'?t|do not)\s+have\s+(?:current\s+)?evidence\s+in\s+this\s+turn\b",
        r"\b(?:don'?t|do not)\s+have\s+(?:fresh\s+)?evidence\s+from\s+this\s+turn\b",
        r"\b(?:don'?t|do not)\s+have\s+(?:a\s+)?current\s+evidence-tool\s+result\s+in\s+this\s+turn\b",
        r"\b(?:don'?t|do not)\s+have\s+(?:a\s+)?fresh\s+opencas\s+evidence-tool\s+result\s+in\s+this\s+turn\b",
        r"\bno\s+evidence\s+(?:in|from)\s+this\s+turn\b",
        r"\bfresh\s+evidence\s+from\s+this\s+turn\b",
        r"\b(?:this|the)\s+turn\s+(?:doesn'?t|does not)\s+(?:contain|include)\s+evidence\b",
        r"\bcurrent\s+turn\s+(?:doesn'?t|does not)\s+(?:contain|include)\s+evidence\b",
        r"\bcurrent-turn\s+evidence\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _unnecessary_permission_stall_nudge(
    *,
    user_input: str,
    assistant_output: str,
) -> ResponseIntegrityResult | None:
    if not _looks_like_user_already_authorized_work(user_input):
        return None
    if not _looks_like_permission_stall(assistant_output):
        return None
    return ResponseIntegrityResult(
        output=(
            "The operator already authorized this low-risk next step. I should not "
            "ask whether to do it again; I should use the relevant available tools, "
            "gather the evidence or make the requested change, and then report what happened."
        ),
        revised=True,
        reasons=["unnecessary_permission_stall"],
    )


def _looks_like_user_already_authorized_work(text: str) -> bool:
    lowered = str(text or "").lower()
    patterns = (
        r"\b(?:continue|keep\s+working|finish|fix|start|do\s+it|go\s+ahead)\b",
        r"\b(?:work\s+on|look\s+at|look\s+up|check|read|inspect|examine|update|write|create|build|implement)\b",
        r"\b(?:please|can\s+you|could\s+you|would\s+you)\s+(?:fix|check|read|write|create|build|implement|continue)\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _looks_like_permission_stall(text: str) -> bool:
    lowered = str(text or "").lower()
    patterns = (
        r"\bshould\s+i\s+(?:do|start|check|read|look|inspect|begin)\b",
        r"\bshould\s+i\s+do\s+that\s+first\b",
        r"\bdo\s+you\s+want\s+me\s+to\b",
        r"\bwould\s+you\s+like\s+me\s+to\b",
        r"\bif\s+you(?:'d| would)\s+like\b",
        r"\bwhat\s+i\s+can\s+do\s+(?:right\s+)?now\b.*\bshould\s+i\b",
        r"\bto\s+actually\s+start\b.*\b(?:need|would\s+need)\s+to\b",
    )
    return any(re.search(pattern, lowered, flags=re.DOTALL) for pattern in patterns)


def _current_turn_autobiographical_packet(
    current_turn_messages: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    for message in current_turn_messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        if str(message.get("name") or "") != "recall_autobiography":
            continue
        packet = _load_json_object(message.get("content"))
        if not packet:
            continue
        scope = str(packet.get("evidence_scope") or "").lower()
        confidence = str(packet.get("confidence") or "").lower()
        evidence = packet.get("strongest_evidence")
        has_self_evidence = scope in {"autobiographical", "mixed"} and confidence in {
            "high",
            "medium",
            "low",
        }
        if isinstance(evidence, list):
            has_self_evidence = has_self_evidence or any(
                isinstance(item, dict)
                and str(item.get("kind") or "").lower()
                in {"autobiographical", "artifact", "procedural", "assistant_turn"}
                for item in evidence
            )
        if has_self_evidence:
            return packet
    return None


def _current_turn_has_raw_self_memory(
    current_turn_messages: Iterable[dict[str, Any]],
) -> bool:
    for message in current_turn_messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        name = str(message.get("name") or "")
        if name not in {"search_memories", "recall_concepts"}:
            continue
        content = str(message.get("content") or "")
        if re.search(r"\[(?:SELF|ACTION|ARTIFACT|PROCEDURAL)\]", content, re.IGNORECASE):
            return True
        if re.search(r"\[ACTION\]\[tool=", content, re.IGNORECASE):
            return True
    return False


def _grounded_self_state_context_nudge(
    *,
    user_input: str,
    assistant_output: str,
    current_turn_messages: Iterable[dict[str, Any]],
    capability_context: str | None = None,
) -> ResponseIntegrityResult | None:
    if not _looks_like_self_state_question(user_input):
        return None
    if not _looks_like_no_evidence_boilerplate(assistant_output):
        return None
    snapshot = _current_turn_somatic_snapshot(
        current_turn_messages,
        capability_context=capability_context,
    )
    if not snapshot:
        return None

    tag = snapshot.get("tag")
    descriptors: list[str] = []
    if tag:
        descriptors.append(str(tag).replace("_", " "))
    tension = _float_from_snapshot(snapshot, "tension")
    fatigue = _float_from_snapshot(snapshot, "fatigue")
    energy = _float_from_snapshot(snapshot, "energy")
    certainty = _float_from_snapshot(snapshot, "certainty")
    valence = _float_from_snapshot(snapshot, "valence")
    if tension is not None and tension <= 0.15:
        descriptors.append("low tension")
    if fatigue is not None and fatigue <= 0.15:
        descriptors.append("low fatigue")
    if energy is not None and energy >= 0.75:
        descriptors.append("high energy")
    if certainty is not None:
        if certainty >= 0.75:
            descriptors.append("high certainty")
        elif certainty >= 0.45:
            descriptors.append("moderate certainty")
    if valence is not None:
        if valence > 0.2:
            descriptors.append("positive valence")
        elif valence < -0.2:
            descriptors.append("negative valence")

    if descriptors:
        state_text = ", ".join(dict.fromkeys(descriptors))
    else:
        state_text = "available but not strongly valenced"
    return ResponseIntegrityResult(
        output=(
            "From my current OpenCAS somatic snapshot, I am in a "
            f"{state_text} state right now."
        ),
        revised=True,
        reasons=["grounded_self_state_context"],
    )


def _looks_like_self_state_question(text: str) -> bool:
    normalized = str(text or "").lower()
    return any(
        marker in normalized
        for marker in (
            "are you feeling",
            "how are you",
            "how do you feel",
            "what state are you",
            "your current state",
            "you daydream",
            "you dream",
            "outside this turn",
            "some kind of way",
        )
    )


def _looks_like_no_evidence_boilerplate(text: str) -> bool:
    normalized = str(text or "").lower()
    return (
        "no current evidence-tool result" in normalized
        or "do not have a current evidence-tool result" in normalized
        or "fresh opencas evidence-tool result" in normalized
        or "do not have a fresh opencas evidence-tool result" in normalized
        or "no evidence tool output" in normalized
        or "cannot truthfully claim a runtime check" in normalized
        or "can't truthfully claim a runtime check" in normalized
        or "should not claim a fresh opencas state check" in normalized
        or "cannot truthfully claim a current state packet" in normalized
        or "can't truthfully claim a current state packet" in normalized
        or "cannot honestly claim a fresh opencas state" in normalized
        or "can't honestly claim a fresh opencas state" in normalized
    )


def _current_turn_somatic_snapshot(
    current_turn_messages: Iterable[dict[str, Any]],
    *,
    capability_context: str | None = None,
) -> dict[str, str] | None:
    pattern = re.compile(r"Current somatic snapshot:\s*(?P<snapshot>[^\n]+)", re.IGNORECASE)
    if capability_context:
        snapshot = _parse_somatic_snapshot_text(capability_context, pattern=pattern)
        if snapshot:
            return snapshot
    for message in current_turn_messages:
        if not isinstance(message, dict):
            continue
        snapshot = _parse_somatic_snapshot_text(message.get("content"), pattern=pattern)
        if snapshot:
            return snapshot
    return None


def _parse_somatic_snapshot_text(
    text: Any,
    *,
    pattern: re.Pattern[str],
) -> dict[str, str] | None:
    match = pattern.search(str(text or ""))
    if not match:
        return None
    snapshot: dict[str, str] = {}
    for part in match.group("snapshot").split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            snapshot[key] = value
    return snapshot or None


def _float_from_snapshot(snapshot: dict[str, str], key: str) -> float | None:
    try:
        return float(snapshot[key])
    except (KeyError, TypeError, ValueError):
        return None


def _paragraph_overlap(left: str, right: str) -> float:
    left_words = {word for word in re.findall(r"\w+", left.lower()) if len(word) > 3}
    right_words = {word for word in re.findall(r"\w+", right.lower()) if len(word) > 3}
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / max(1, min(len(left_words), len(right_words)))


async def _review_revised_response(
    llm: Any,
    *,
    session_id: str,
    agent_name: str = "OpenCAS",
    user_input: str,
    assistant_output: str,
    history: Iterable[MessageEntry],
    capability_context: str | None = None,
    current_turn_messages: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Run one extra review pass on a proposed repair.

    The repair model cannot call tools, so a second pass catches cases where the
    repair itself becomes a new unsupported "I'll check now" promise.
    """

    review_messages = [
        {"role": "system", "content": _review_system_prompt(agent_name=agent_name)},
        {
            "role": "user",
            "content": _build_review_payload(
                user_input=user_input,
                assistant_output=assistant_output,
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
            source="response_integrity_revised",
        )
    except Exception:
        return None
    return _parse_review_json(raw_content)


async def _request_review_content(
    llm: Any,
    *,
    messages: list[dict[str, Any]],
    session_id: str,
    source: str,
) -> str:
    response = await llm.chat_completion(
        messages=messages,
        complexity="light",
        payload={"temperature": 0, "response_format": {"type": "json_object"}},
        source=source,
        session_id=session_id,
        generation_request=GenerationPolicyRequest(
            phase=GenerationPhase.VERIFY,
            domain=GenerationDomain.SAFETY,
            structured_output=True,
            risk_level="response_integrity",
            source=source,
        ),
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
        parts.extend(["", "Supplied tool/output evidence:", *current_turn_evidence])
    else:
        parts.extend(
            [
                "",
                "Supplied tool/output evidence:",
                "No tool/output evidence was supplied for this response.",
            ]
        )
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
            content = _compact_tool_content(name, message.get("content"), limit=1800)
            rendered.append(f"tool {name}: {content}")
            continue
        content_value = message.get("content")
        content_text = str(content_value or "")
        if role == "user" and "[Attached file:" in content_text:
            content = _compact_message_content(
                content_value,
                limit=4200,
                preserve_tail=True,
            )
        else:
            content = _compact_message_content(content_value, limit=900)
        if content:
            rendered.append(f"{role}: {content}")
    return rendered


def _compact_message_content(
    value: Any,
    *,
    limit: int,
    preserve_tail: bool = False,
) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        if preserve_tail and limit >= 24:
            head_len = max(1, (limit - 15) // 2)
            tail_len = max(1, limit - 15 - head_len)
            return (
                text[:head_len].rstrip()
                + " ...[truncated]... "
                + text[-tail_len:].lstrip()
            )
        return text[: limit - 3].rstrip() + "..."
    return text


def _compact_tool_content(name: str, value: Any, *, limit: int) -> str:
    if name == "self_inspection_query":
        summary = _compact_self_inspection_query_content(value, limit=limit)
        if summary:
            return summary
    return _compact_message_content(value, limit=limit)


def _compact_self_inspection_query_content(value: Any, *, limit: int) -> str:
    data = _load_json_object(value)
    if not data:
        return ""
    items = data.get("items")
    if not isinstance(items, list):
        return _compact_message_content(value, limit=limit)
    compact_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact_items.append(
            {
                "record_id": item.get("record_id"),
                "phase": item.get("phase"),
                "created_at": item.get("created_at"),
                "tool_use_inspections": _compact_tool_use_inspections(
                    item.get("tool_use_inspections")
                ),
                "tool_call_transits": _compact_tool_call_transits(
                    item.get("tool_call_transits")
                ),
                "tool_chain_summary": _compact_tool_chain_summary(
                    item.get("tool_chain_summary")
                ),
                "commitment_gaps": _compact_commitment_gaps(item.get("commitment_gaps")),
            }
        )
    summary = {
        "count": data.get("count", len(compact_items)),
        "items": compact_items,
    }
    return _compact_message_content(
        json.dumps(summary, separators=(",", ":"), sort_keys=True),
        limit=limit,
    )


def _load_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _compact_tool_use_inspections(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact = []
    for item in value:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "tool_name": item.get("tool_name"),
                "call_id": item.get("call_id"),
                "objective": item.get("objective"),
            }
        )
    return compact


def _compact_tool_call_transits(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact = []
    for item in value:
        if not isinstance(item, dict):
            continue
        result_shape = item.get("result_shape")
        if not isinstance(result_shape, dict):
            result_shape = {}
        compact.append(
            {
                "tool_name": item.get("tool_name"),
                "call_id": item.get("call_id"),
                "trust_context": item.get("trust_context"),
                "success": item.get("success"),
                "result_shape_tags": result_shape.get("tags") or [],
            }
        )
    return compact


def _compact_tool_chain_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "call_ids": value.get("call_ids") or [],
        "trust_context": value.get("trust_context"),
        "call_count": value.get("call_count"),
        "success_count": value.get("success_count"),
        "failure_count": value.get("failure_count"),
        "result_shape_counts": value.get("result_shape_counts") or {},
        "undercoupled_somatic": value.get("undercoupled_somatic"),
    }


def _compact_commitment_gaps(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact = []
    for item in value:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "gap_type": item.get("gap_type"),
                "status": item.get("status"),
                "commitment_id": item.get("commitment_id"),
            }
        )
    return compact


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
