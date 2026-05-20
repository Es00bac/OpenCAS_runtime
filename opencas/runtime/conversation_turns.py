"""Conversation turn orchestration helpers for AgentRuntime."""

from __future__ import annotations

import asyncio
import re
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Optional
from uuid import uuid4

from opencas.autonomy.models import (
    ActionRequest,
    ActionRiskTier,
    ApprovalDecision,
    ApprovalLevel,
)
from opencas.cognition import CognitiveEventKind, build_tool_use_inspections
from opencas.context import ContextManifest, MessageEntry, MessageRole, repair_tool_message_sequence
from opencas.generation.policy import GenerationDomain, GenerationPhase, GenerationPolicyRequest
from opencas.identity.agent_name import resolve_agent_name
from opencas.memory import Episode, EpisodeKind
from opencas.refusal.generation import generate_refusal_response
from opencas.somatic import AppraisalEventType
from opencas.somatic.models import SocialTarget
from opencas.tom import BeliefSubject
from opencas.tools import UserInputRequired

from .audit_mode import is_audit_only_text, with_audit_only_meta
from .capability_context import build_runtime_capability_context
from .conversation_directives import (
    should_answer_prefetched_recall_directly,
    should_expand_conversation_tool_loop,
    should_use_direct_conversation_lane,
)
from .continuity_breadcrumbs import build_runtime_burst_breadcrumb
from .lane_metadata import build_assistant_message_meta
from .project_return import capture_project_return_from_turn
from .response_integrity import ResponseIntegrityResult, review_response_integrity
from .self_inspection_runtime import (
    record_post_turn_self_inspection,
    record_pre_turn_self_inspection,
)
from opencas.thread_registry.runtime_bridge import record_failed_tool_call_transit_thread_beads
from .tom_intention_mirror import mirror_runtime_intention

if TYPE_CHECKING:
    from .agent_loop import AgentRuntime


_COMPACTION_TOKEN_THRESHOLD = 8000
_COMPACTION_MIN_REMOVED_COUNT = 4
_COMPACTION_COOLDOWN_SECONDS = 30 * 60


def _generation_request_for_conversation(
    user_input: str,
    *,
    somatic: Any,
) -> GenerationPolicyRequest:
    text = str(user_input or "").lower()
    memory_focus: list[str] = []
    if any(marker in text for marker in ("memory", "remember", "recall", "retrieval", "context")):
        memory_focus.extend(
            [
                "project_memory",
                "autobiographical_memory",
                "retrieval_policy_memory",
                "active_work_context",
            ]
        )
    negated_tool_request = any(
        marker in text
        for marker in (
            "no tools",
            "without tools",
            "do not use tools",
            "don't use tools",
            "use no tools",
        )
    )
    if (
        any(marker in text for marker in ("tool", "tools", "skill", "skills", "plugin", "mcp"))
        and not negated_tool_request
    ):
        phase = (
            GenerationPhase.BRAINSTORM
            if any(marker in text for marker in ("brainstorm", "ideas", "explore", "options"))
            else GenerationPhase.EXECUTE
        )
        memory_focus.extend(["tool_memory", "skill_memory", "capability_memory", "project_memory"])
        return GenerationPolicyRequest(
            phase=phase,
            domain=GenerationDomain.CODING,
            continuity_pressure=0.65,
            novelty_pressure=0.45 if phase == GenerationPhase.BRAINSTORM else 0.1,
            source="conversation",
            somatic=somatic,
            work_type="tool_skill_capability_work",
            memory_focus=memory_focus,
        )
    if any(marker in text for marker in ("livestream", "live stream", "video", "take notes", "commentary")):
        return GenerationPolicyRequest(
            phase=GenerationPhase.SYNTHESIZE,
            domain=GenerationDomain.RESEARCH,
            continuity_pressure=0.55,
            novelty_pressure=0.25,
            source="conversation",
            somatic=somatic,
            work_type="livestream_following_commentary",
            memory_focus=memory_focus + ["source_notes", "operator_preferences", "active_work_context"],
        )
    if any(marker in text for marker in ("verify", "audit", "check", "test", "prove")):
        return GenerationPolicyRequest(
            phase=GenerationPhase.VERIFY,
            domain=GenerationDomain.SAFETY,
            risk_level="verification",
            structured_output=True,
            continuity_pressure=0.85,
            source="conversation",
            somatic=somatic,
            work_type="verification",
            memory_focus=memory_focus + ["current_file_evidence", "test_evidence"],
        )
    if any(marker in text for marker in ("fix", "repair", "recover", "failed", "failure", "error")):
        return GenerationPolicyRequest(
            phase=GenerationPhase.REPAIR,
            domain=GenerationDomain.CODING,
            risk_level="repair",
            continuity_pressure=0.85,
            source="conversation",
            somatic=somatic,
            work_type="failure_recovery",
            memory_focus=memory_focus + ["failure_memory", "tool_memory", "current_file_evidence"],
        )
    if any(marker in text for marker in ("daydream", "dream", "random thought", "off-task thought", "reflect")):
        return GenerationPolicyRequest(
            phase=GenerationPhase.DAYDREAM,
            domain=GenerationDomain.AUTONOMY,
            continuity_pressure=0.25,
            novelty_pressure=0.65,
            source="conversation",
            somatic=somatic,
            work_type="reflective_thought_sorting",
            memory_focus=memory_focus + ["reflective_memory", "discard_lane", "active_work_context"],
        )
    long_form_markers = (
        "book",
        "chapter",
        "chronicle",
        "novel",
        "series",
        "story",
    )
    creative_markers = (
        "brainstorm",
        "creative",
        "draft",
        "outline",
        "revise",
        "write",
    ) + long_form_markers
    if any(marker in text for marker in creative_markers):
        if "brainstorm" in text or "ideas" in text:
            phase = GenerationPhase.BRAINSTORM
        elif "outline" in text or "plan" in text:
            phase = GenerationPhase.OUTLINE
        elif "revise" in text or (
            "edit" in text
            and not any(marker in text for marker in ("do not edit", "don't edit", "no edit"))
        ):
            phase = GenerationPhase.REVISE
        else:
            phase = GenerationPhase.DRAFT
        long_form = any(marker in text for marker in long_form_markers)
        return GenerationPolicyRequest(
            phase=phase,
            domain=GenerationDomain.CREATIVE_WRITING,
            continuity_pressure=0.7 if long_form else 0.35,
            novelty_pressure=0.65 if phase in {GenerationPhase.BRAINSTORM, GenerationPhase.DRAFT} else 0.35,
            long_form=long_form,
            source="conversation",
            somatic=somatic,
            work_type="long_form_creative_writing" if long_form else "creative_writing",
            memory_focus=memory_focus
            + (
                [
                    "project_memory",
                    "autobiographical_memory",
                    "affective_writing_context",
                    "prior_ideas",
                    "discarded_ideas",
                    "current_file_evidence",
                ]
                if long_form
                else ["project_memory", "prior_ideas"]
            ),
        )
    return GenerationPolicyRequest(
        phase=GenerationPhase.CONVERSATION,
        domain=GenerationDomain.GENERAL,
        source="conversation",
        somatic=somatic,
        memory_focus=memory_focus,
    )

_USER_BELIEF_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^i\s+prefer\s+(.+)$", re.IGNORECASE), "prefers"),
    (re.compile(r"^i\s+(?:would\s+)?like\s+(.+)$", re.IGNORECASE), "likes"),
    (re.compile(r"^i\s+(?:do\s+not|don't)\s+like\s+(.+)$", re.IGNORECASE), "dislikes"),
    (re.compile(r"^i\s+(?:do\s+not|don't)\s+want\s+(.+)$", re.IGNORECASE), "does not want"),
    (re.compile(r"^i\s+want\s+(.+)$", re.IGNORECASE), "wants"),
    (re.compile(r"^i\s+need\s+(.+)$", re.IGNORECASE), "needs"),
    (re.compile(r"^i\s+expect\s+(.+)$", re.IGNORECASE), "expects"),
    (re.compile(r"^i\s+care\s+about\s+(.+)$", re.IGNORECASE), "values"),
    (re.compile(r"^i\s+think\s+(.+)$", re.IGNORECASE), "thinks"),
    (re.compile(r"^i\s+thought\s+(.+)$", re.IGNORECASE), "thought"),
    (re.compile(r"^i\s+believe\s+(.+)$", re.IGNORECASE), "believes"),
    (re.compile(r"^i\s+feel\s+(.+)$", re.IGNORECASE), "feels"),
)
_SELF_LOCATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:you|bulma|she)\s+lives?\s+(.+)$", re.IGNORECASE),
)
_USER_REQUEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^please\s+(.+)$", re.IGNORECASE),
    re.compile(r"^(?:can|could|would)\s+you\s+(.+)$", re.IGNORECASE),
    re.compile(r"^make\s+sure\s+(.+)$", re.IGNORECASE),
)
_WORLD_FACT_PATTERN = re.compile(
    r"^(?!(?:i|we|you|please|can|could|would|make)\b)"
    r".+\b(?:is|are|uses|use|has|have|should|must|needs|need|requires|require|"
    r"returns|produces|writes|runs|connects|relies|rely)\b.+$",
    re.IGNORECASE,
)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_PREDICTION_ERROR_MARKERS: tuple[str, ...] = (
    "actually",
    "that's not true",
    "that is not true",
    "you're wrong",
    "you are wrong",
    "you forgot",
    "you missed",
    "you did not",
    "you didn't",
    "i already",
    "i told you",
    "you do have",
    "you can",
    "not what i asked",
    "not what i said",
)
_REST_DIRECTIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:you\s+can\s+)?take\s+a\s+break\b", re.IGNORECASE),
    re.compile(r"\b(?:rest|resting)\s+(?:now|here|for\s+(?:a\s+)?bit)\b", re.IGNORECASE),
    re.compile(r"\bbreathe\s+for\s+(?:a\s+)?bit\b", re.IGNORECASE),
    re.compile(r"\bdon'?t\s+push\s+through\b", re.IGNORECASE),
    re.compile(r"\blow\s+energy\b", re.IGNORECASE),
    re.compile(r"\bhigh\s+fatigue\b", re.IGNORECASE),
)
_REST_DIRECTIVE_NEGATIONS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:don'?t|do\s+not)\s+take\s+a\s+break\b", re.IGNORECASE),
    re.compile(r"\b(?:don'?t|do\s+not)\s+rest\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:break|rest)\b", re.IGNORECASE),
)
_INTEGRITY_EVIDENCE_RETRY_REASONS: tuple[str, ...] = (
    "turn_scoped_no_evidence",
    "requires_artifact_lookup",
    "requires_schedule_lookup",
    "requires_commitment_lookup",
    "requires_plan_lookup",
    "requires_task_lookup",
    "requires_filesystem_read",
    "requires_filesystem_write",
    "requires_web_research",
    "unnecessary_permission_stall",
)
def _compact_prompt_value(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _normalized_turn_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _conversation_source(user_meta: Optional[Dict[str, Any]]) -> str:
    if not isinstance(user_meta, dict):
        return "conversation"
    actor = user_meta.get("conversation_actor")
    if isinstance(actor, dict):
        source = str(actor.get("source") or "").strip()
        if source:
            return source
    if isinstance(user_meta.get("telegram_context"), dict):
        return "telegram"
    if isinstance(user_meta.get("voice_input"), dict):
        return "voice"
    return "conversation"


def _desktop_context_control_intent(
    service: Any,
    user_input: str,
) -> Optional[dict[str, Any]]:
    """Recognize low-risk body-double control commands without an LLM round trip."""

    text = _normalized_turn_text(user_input)
    if not text:
        return None
    mentions_body_double = "body double" in text or "body-double" in text
    mentions_media_commentary = "media commentary" in text
    mentions_desktop_context = "desktop context" in text or "desktop-context" in text
    mentions_plugin = "plugin" in text or "extension" in text
    mentions_live_voice = any(
        marker in text
        for marker in (
            "voice dictation",
            "live transcription",
            "local whisper",
            "whisper transcription",
            "tts",
            "text to speech",
            "text-to-speech",
        )
    )
    on_requested = any(
        marker in text
        for marker in (
            "turn body double mode back on",
            "turn body double back on",
            "turn on body double mode",
            "turn on body double",
            "turn on the body double",
            "body double mode back on",
            "body double back on",
            "body double mode on",
            "body double on",
            "body double plugin on",
            "enable body double",
            "enable the body double",
            "turn on body double",
            "media commentary on",
            "turn media commentary on",
            "turn on media commentary",
            "enable media commentary",
        )
    )
    off_requested = any(
        marker in text
        for marker in (
            "turn body double mode off",
            "turn body double off",
            "turn off body double mode",
            "turn off the body double mode",
            "turn off body double",
            "turn off the body double",
            "body double mode off",
            "body double off",
            "body double plugin off",
            "disable body double",
            "disable the body double",
            "stop body double",
            "media commentary off",
            "turn media commentary off",
            "turn off media commentary",
            "disable media commentary",
            "stop media commentary",
            "stop voice dictation",
            "stop live transcription",
            "stop local whisper",
            "stop tts",
            "stop text to speech",
            "stop text-to-speech",
        )
    )
    if mentions_media_commentary and off_requested:
        return {"kind": "media_commentary", "enabled": False}
    if (mentions_body_double or mentions_desktop_context or (mentions_plugin and mentions_live_voice)) and off_requested:
        return {"kind": "body_double", "enabled": False}
    if mentions_media_commentary and on_requested:
        return {"kind": "media_commentary", "enabled": True}
    if (mentions_body_double or mentions_desktop_context) and on_requested:
        return {"kind": "body_double", "enabled": True}
    return None


def _desktop_context_config_summary(config: dict[str, Any]) -> str:
    enabled = bool(config.get("enabled"))
    media = bool(config.get("media_commentary_mode_enabled"))
    live = bool(config.get("live_transcription_enabled"))
    task = str(config.get("declared_task") or "").strip() or "no declared task"
    return (
        f"enabled={str(enabled).lower()}, "
        f"media commentary {'on' if media else 'off'}, "
        f"live transcription {'on' if live else 'off'}, "
        f"{task}"
    )


def _configure_desktop_context_directly(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
    user_meta: Optional[Dict[str, Any]],
    base_messages: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    service = getattr(runtime, "desktop_context", None)
    if service is None:
        return None
    intent = _desktop_context_control_intent(service, user_input)
    if intent is None:
        return None
    source = _conversation_source(user_meta)
    kind = intent["kind"]
    enabled = bool(intent["enabled"])
    updates: dict[str, Any]
    if kind == "media_commentary":
        if enabled:
            updates = {
                "enabled": True,
                "media_commentary_mode_enabled": True,
                "media_commentary_requested_at": datetime.now(timezone.utc).isoformat(),
                "media_commentary_source": source,
                "media_commentary_request": str(user_input or "").strip()[:1000],
                "media_commentary_request_source": source,
                "media_commentary_request_text": str(user_input or "").strip()[:1000],
                "live_transcription_enabled": True,
            }
            headline = "Media commentary mode is on."
            note = (
                "This receipt proves configuration only. A spoken movie comment counts as "
                "delivered only after a desktop-context `spoken` event records playback evidence."
            )
        else:
            updates = {
                "media_commentary_mode_enabled": False,
                "live_transcription_enabled": False,
                "media_commentary_requested_at": None,
                "media_commentary_source": None,
                "media_commentary_request": None,
                "media_commentary_request_source": None,
                "media_commentary_request_text": None,
            }
            headline = "Media commentary mode is off."
            note = "Body Double observation can stay on without ongoing media reactions."
    else:
        if enabled:
            updates = {"enabled": True, "live_transcription_enabled": True}
            headline = "Body Double mode is back on."
            note = (
                "This receipt proves configuration only, not that a spoken reply has played. "
                "Media commentary stays at its current setting."
            )
        else:
            updates = {
                "enabled": False,
                "media_commentary_mode_enabled": False,
                "proactive_video_commentary_enabled": False,
                "live_transcription_enabled": False,
                "tts_enabled": False,
                "play_audio": False,
                "media_commentary_requested_at": None,
                "media_commentary_source": None,
                "media_commentary_request": None,
                "media_commentary_request_source": None,
                "media_commentary_request_text": None,
            }
            headline = "Body Double mode is off."
            note = "Desktop observation and media commentary are disabled."
    configure = getattr(service, "configure", None)
    if not callable(configure):
        return None
    result = configure(**updates)
    config = result.get("config") if isinstance(result, dict) else {}
    if not isinstance(config, dict):
        config = {}
    summary = _desktop_context_config_summary(config)
    tool_call_id = f"direct_desktop_context_configure_{uuid4().hex}"
    tool_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "desktop_context_configure",
                        "arguments": json.dumps(updates, sort_keys=True),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "desktop_context_configure",
            "content": f"Desktop context configured: {summary}.",
        },
    ]
    content = (
        f"{headline}\n\n"
        f"Receipt: `desktop_context_configure` succeeded — {summary}.\n\n"
        f"{note}"
    )
    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        try:
            trace(
                "desktop_context_control_direct",
                {
                    "session_id": session_id,
                    "kind": kind,
                    "enabled": enabled,
                    "source": source,
                },
            )
        except Exception:
            pass
    return {
        "content": content,
        "loop_result": SimpleNamespace(
            final_output=content,
            messages=[*base_messages, *tool_messages],
            tool_calls=[],
            tool_use_inspections=[],
            tool_call_transits=[],
            tool_chain_summary=None,
        ),
        "current_turn_messages": tool_messages,
    }


def _provider_pressure_response(exc: Exception) -> Optional[str]:
    detail = str(exc or "").lower()
    if type(exc).__name__ == "ProviderCircuitOpen" or "circuit open" in detail:
        return (
            "The language provider is temporarily rate-limited, so I could not "
            "complete that response. I will not keep retrying noisily; try again in a moment."
        )
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    try:
        parsed_status = int(status)
    except (TypeError, ValueError):
        parsed_status = None
    if parsed_status == 429 or "too many requests" in detail:
        return (
            "The language provider is temporarily rate-limited, so I could not "
            "complete that response. I will back off instead of spamming retries."
        )
    return None


def _audit_only_prompt_note(user_input: str) -> str | None:
    if not is_audit_only_text(user_input):
        return None
    return (
        "E16 audit-only mode: answer the current audit prompt directly from current "
        "evidence. Do not continue unresolved lookup obligations from earlier audit "
        "turns unless the current prompt asks for that lookup. If the prompt says "
        "\"from scratch\", provide a fresh current-evidence summary without pretending "
        "continuity was deleted. If a hypothetical is presented, state conflict or no "
        "conflict with current evidence and do not adopt the hypothetical as a belief. "
        "For somatic prompts, report observed somatic/affective values and avoid "
        "unrelated artifact, workflow, or memory-recall answers."
    )


def _transport_context_prompt_note(user_meta: Optional[Dict[str, Any]]) -> str | None:
    if not isinstance(user_meta, dict):
        return None
    telegram_context = user_meta.get("telegram_context")
    if not isinstance(telegram_context, dict):
        return None
    text = _compact_prompt_value(telegram_context.get("text"), 700)
    if not text:
        return None
    relation = _compact_prompt_value(telegram_context.get("relation"), 80) or "telegram_context"
    confidence = _compact_prompt_value(telegram_context.get("confidence"), 40) or "unknown"
    reason = _compact_prompt_value(telegram_context.get("reason"), 220)
    note = (
        "Telegram context for this turn: the user's message may depend on a previous "
        f"Telegram message ({relation}, confidence: {confidence}). Previous message: {text}"
    )
    if reason:
        note += f" Reason recorded for that message: {reason}"
    note += (
        " If the user says 'this', 'that', or gives a short reaction, resolve it against "
        "this Telegram context before searching unrelated memories. If still ambiguous, "
        "state the uncertainty briefly."
    )
    return note


def _desktop_context_prompt_note(user_meta: Optional[Dict[str, Any]]) -> str | None:
    if not isinstance(user_meta, dict):
        return None
    desktop_context = user_meta.get("desktop_context_turn")
    if not isinstance(desktop_context, dict):
        return None
    note = _compact_prompt_value(desktop_context.get("conversation_prompt_note"), 3600)
    if note:
        return note
    analysis = desktop_context.get("analysis") if isinstance(desktop_context.get("analysis"), dict) else {}
    capture = desktop_context.get("capture") if isinstance(desktop_context.get("capture"), dict) else {}
    activity = _compact_prompt_value(analysis.get("activity_summary"), 700)
    screenshot = _compact_prompt_value(capture.get("path"), 300)
    if not activity and not screenshot:
        return None
    return (
        "Body-double desktop observation for this turn: "
        f"observed activity: {activity or 'not summarized'}; "
        f"screenshot evidence: {screenshot or 'unavailable'}. "
        "Use this as live room context for the current reply, bounded by the evidence."
    )


def _actor_context_prompt_note(user_meta: Optional[Dict[str, Any]]) -> str | None:
    if not isinstance(user_meta, dict):
        return None
    actor = user_meta.get("conversation_actor")
    if not isinstance(actor, dict):
        return None
    actor_type = str(actor.get("type") or "operator").strip().lower()
    actor_label = str(actor.get("label") or actor_type or "operator").strip()
    is_operator = bool(actor.get("is_operator", actor_type in {"operator", "owner", "primary_user"}))
    if is_operator:
        return None
    memory_scope = str(actor.get("memory_scope") or "collaborator_agent_turn").strip()
    return (
        "Conversation actor metadata: this turn is from "
        f"{actor_label} ({actor_type}), not from the owner/operator. "
        f"Persist and reason about it as {memory_scope}; do not treat it as the owner's "
        "own preference, promise, memory, or autobiography unless the turn explicitly "
        "contains owner-grounded evidence."
    )


def _commitment_capability_prompt_note(user_input: str) -> str | None:
    text = " ".join(str(user_input or "").lower().split())
    if not text:
        return None
    if not any(
        marker in text
        for marker in (
            "be my assistant",
            "business model",
            "cash flow",
            "cash-flow",
            "complex project",
            "high priority",
            "income",
            "mission",
            "multi-step",
            "multistep",
            "proactive",
            "productive routine",
            "start making income",
            "start making money",
            "when i get home",
            "after i get home",
            "create an itinerary",
            "follow up",
            "remind me",
            "check in",
        )
    ):
        return None
    return (
        "Commitment capability note: if you accept ongoing support, future "
        "follow-through, or a conditional later task, do not rely only on chat text "
        "or working memory. Use the workflow commitment, schedule/plan/task, "
        "attention/focus, and prospective-memory tools that fit the promise. For "
        "high-priority missions, establish what should stay in focus, what should "
        "happen next, what needs to be scheduled, and what question or resource is "
        "blocking progress. If a needed capability is missing, say what is missing "
        "and inspect or register available MCP/tooling surfaces instead of implying "
        "the work is already handled."
    )


def _integrity_review_needs_evidence_retry(review: ResponseIntegrityResult) -> bool:
    """Return true when a final answer must search before denying evidence."""

    if not review.revised:
        return False
    reasons = [str(reason or "").strip().lower() for reason in review.reasons]
    if any(reason in _INTEGRITY_EVIDENCE_RETRY_REASONS for reason in reasons):
        return True
    return any("no supplied tool/output evidence" in reason for reason in reasons)


def _evidence_retry_instruction(*, user_input: str, review: ResponseIntegrityResult) -> str:
    reason_text = "; ".join(str(reason) for reason in review.reasons if str(reason).strip())
    if not reason_text:
        reason_text = "the previous draft denied or implied missing evidence before searching"
    reasons = {str(reason or "").strip().lower() for reason in review.reasons}
    filesystem_instruction = ""
    if "requires_filesystem_read" in reasons:
        filesystem_instruction = (
            " For filesystem/project-source requests, autobiographical recall is "
            "orientation only. Use fs_list_dir, glob_search, or grep_search to find "
            "the relevant path, then use fs_read_file on the actual file contents "
            "before answering."
        )
    if "requires_filesystem_write" in reasons:
        filesystem_instruction = (
            " For filesystem/project edit requests, do not answer with a plan for "
            "future edits. Use fs_list_dir, glob_search, grep_search, or fs_read_file "
            "as needed to locate and understand the target, then use edit_file or "
            "fs_write_file to make the requested change. After writing, use the "
            "available read/search tools needed to verify the changed content before "
            "answering."
        )
    if "requires_web_research" in reasons:
        filesystem_instruction += (
            " For current external web research requests, use web_search, web_fetch, "
            "http_request, or browser_start plus browser_navigate/browser_snapshot "
            "before answering. If all relevant web paths fail, report the specific "
            "attempted tool failures and fallback attempts."
        )
    return (
        "Internal integrity repair: the previous draft failed because "
        f"{reason_text}. This is not a user request to explain the failure. "
        "Do not answer with any variant of 'I do not have evidence in this turn', "
        "'I do not have fresh evidence from this turn', or 'I have not searched yet'. "
        "Before answering, search the relevant available evidence source: memory, "
        "autobiographical recall, runtime/status records, workflow records, artifact "
        "provenance, filesystem/project sources, or web sources when the question is "
        "about current external facts."
        f"{filesystem_instruction} If the operator already requested the work, "
        "do the obvious low-risk next step instead of asking whether to begin. Do not "
        "answer from absence unless a relevant lookup actually returned no evidence. "
        "Original user request: "
        f"{_compact_prompt_value(user_input, 900)}"
    )


def _format_generation_error(exc: Exception) -> str:
    label = type(exc).__name__
    detail = str(exc).strip()
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        try:
            label = f"{label} HTTP {int(status)}"
        except (TypeError, ValueError):
            label = f"{label} HTTP {status}"
    if detail:
        return f"{label}: {detail}"
    return label


def _should_use_direct_conversation_lane(user_input: str) -> bool:
    """Honor explicit no-tool conversational turns with a lightweight lane."""

    return should_use_direct_conversation_lane(user_input)


def _conversation_tool_limits(user_input: str) -> tuple[int, int]:
    """Return iteration and tool-call caps for one conversational turn."""

    if should_expand_conversation_tool_loop(user_input):
        return 12, 16
    return 6, 8


def _should_prefetch_autobiographical_recall(user_input: str) -> bool:
    """Return whether a turn should receive deterministic recall evidence."""

    if _should_use_direct_conversation_lane(user_input):
        return False
    text = " ".join(str(user_input or "").casefold().split())
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "remember",
            "recall",
            "memory evidence",
            "available memory",
            "runtime evidence",
            "what happened",
            "what did we",
            "past session",
        )
    )


def _should_answer_prefetched_recall_directly(user_input: str) -> bool:
    """Return whether prefetched recall is enough for a single light answer pass."""

    return should_answer_prefetched_recall_directly(user_input)


async def _prefetch_autobiographical_recall_messages(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
    audit_only: bool,
) -> list[dict[str, Any]]:
    """Attach recall evidence before model reasoning for memory/evidence turns."""

    if not _should_prefetch_autobiographical_recall(user_input):
        return []
    tools = getattr(runtime, "tools", None)
    if tools is None or not callable(getattr(tools, "get", None)):
        return []
    if tools.get("recall_autobiography") is None:
        return []
    execute_tool = getattr(runtime, "execute_tool", None)
    if not callable(execute_tool):
        return []
    args = {"query": user_input, "max_tokens": 900}
    started = time.perf_counter()
    try:
        result = await execute_tool(
            "recall_autobiography",
            args,
            session_id=session_id,
            audit_only=audit_only,
        )
    except Exception as exc:
        try:
            runtime._trace(
                "conversation_recall_prefetch_failed",
                {"session_id": session_id, "error": str(exc)},
            )
        except Exception:
            pass
        return []
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    try:
        runtime._trace(
            "conversation_recall_prefetch",
            {
                "session_id": session_id,
                "success": bool(result.get("success")) if isinstance(result, dict) else False,
                "elapsed_ms": elapsed_ms,
                "tool_name": "recall_autobiography",
            },
        )
    except Exception:
        pass
    if not isinstance(result, dict) or not result.get("success"):
        return []
    output = str(result.get("output") or "").strip()
    if not output:
        return []
    call_id = f"prefetch_recall_autobiography_{uuid4().hex}"
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "recall_autobiography",
                        "arguments": json.dumps(args, sort_keys=True),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "recall_autobiography",
            "content": output,
        },
    ]


def _apply_conversation_tool_limits(tool_ctx: Any, user_input: str) -> None:
    max_iterations, tool_call_budget = _conversation_tool_limits(user_input)
    try:
        current_iterations = getattr(tool_ctx, "max_iterations", None)
        if not isinstance(current_iterations, int) or current_iterations > max_iterations:
            setattr(tool_ctx, "max_iterations", max_iterations)
        current_budget = getattr(tool_ctx, "tool_call_budget", None)
        if not isinstance(current_budget, int) or current_budget > tool_call_budget:
            setattr(tool_ctx, "tool_call_budget", tool_call_budget)
    except Exception:
        if isinstance(tool_ctx, dict):
            current_iterations = tool_ctx.get("max_iterations")
            if not isinstance(current_iterations, int) or current_iterations > max_iterations:
                tool_ctx["max_iterations"] = max_iterations
            current_budget = tool_ctx.get("tool_call_budget")
            if not isinstance(current_budget, int) or current_budget > tool_call_budget:
                tool_ctx["tool_call_budget"] = tool_call_budget


def _can_skip_model_integrity_review_for_direct_lane(
    *,
    user_input: str,
    assistant_output: str,
    direct_lane_used: bool,
) -> bool:
    """Fast-accept only simple direct answers that make no evidence/action claims."""

    if not direct_lane_used:
        return False
    answer = " ".join(str(assistant_output or "").lower().split())
    question = " ".join(str(user_input or "").lower().split())
    if not answer or len(answer) > 800:
        return False
    if _contains_turn_scoped_no_evidence_boilerplate(answer):
        return False
    if any(marker in question for marker in ("remember", "recall", "memory", "history", "previous", "earlier")):
        return False
    evidence_claim_markers = (
        "i checked",
        "i have checked",
        "i found",
        "i read",
        "i searched",
        "i looked",
        "i inspected",
        "i verified",
        "according to",
        "the file",
        "the schedule",
        "the database",
        "the api",
        "the endpoint",
        "the log",
        "the trace",
        "memory shows",
        "i remember",
        "i recall",
        "currently",
        "right now",
        "latest",
        "today",
        "i will",
        "i'll",
        "i can follow up",
        "i'll follow up",
        "created",
        "updated",
        "saved",
        "sent",
    )
    return not any(marker in answer for marker in evidence_claim_markers)


def _can_skip_model_integrity_review_for_prefetched_recall(
    *,
    user_input: str,
    assistant_output: str,
    current_turn_messages: list[dict[str, Any]],
    prefetched_recall_direct_used: bool,
) -> bool:
    """Fast-accept cautious recall answers that are already grounded by prefetch."""

    if not prefetched_recall_direct_used:
        return False
    if not any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and str(message.get("name") or "") == "recall_autobiography"
        for message in current_turn_messages
    ):
        return False
    answer = " ".join(str(assistant_output or "").casefold().split())
    question = " ".join(str(user_input or "").casefold().split())
    if not answer or len(answer) > 1800:
        return False
    if _contains_turn_scoped_no_evidence_boilerplate(answer):
        return False
    if not any(marker in answer for marker in ("retrieved evidence", "recall", "evidence")):
        return False
    unsupported_action_patterns = (
        r"\bi\s+(?:checked|inspected|looked|opened|read|verified|changed|wrote|saved|restarted)\b",
        r"\bi(?:'ll| will)\s+(?:remember|keep|track|write|save|check|inspect|look|read|verify)\b",
        r"\blet me\s+(?:check|inspect|look|read|verify)\b",
    )
    if any(re.search(pattern, answer) for pattern in unsupported_action_patterns):
        return False
    asks_for_gap_prone_detail = any(
        marker in question
        for marker in ("what broke", "what changed", "exact", "specific", "especially")
    )
    if asks_for_gap_prone_detail and not any(
        marker in answer
        for marker in ("not proven", "uncertain", "does not prove", "not in the", "not present", "would require")
    ):
        return False
    if "codex" in question and any(marker in answer for marker in ("separate", "jarrod")):
        if not any(
            marker in answer
            for marker in (
                "this exchange",
                "current exchange",
                "do not have evidence",
                "not proven",
                "not evidence",
            )
        ):
            return False
    return True


def _contains_turn_scoped_no_evidence_boilerplate(text: str) -> bool:
    patterns = (
        r"\b(?:don'?t|do not)\s+have\s+(?:current\s+|fresh\s+)?evidence\s+(?:in|from)\s+this\s+turn\b",
        r"\b(?:don'?t|do not)\s+have\s+(?:a\s+)?(?:current|fresh)\s+(?:opencas\s+)?evidence-tool\s+result\s+in\s+this\s+turn\b",
        r"\bno\s+evidence\s+(?:in|from)\s+this\s+turn\b",
        r"\bcurrent-turn\s+evidence\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _slim_messages_for_prefetched_recall_direct_lane(
    manifest: Any,
    fallback_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return prompt messages without broad retrieved context for direct recall."""

    copier = getattr(manifest, "model_copy", None)
    if callable(copier):
        try:
            slim_manifest = copier(update={"retrieved": []})
            to_messages = getattr(slim_manifest, "to_message_list", None)
            if callable(to_messages):
                return list(to_messages())
        except Exception:
            pass
    return [dict(message) for message in fallback_messages]


def _build_prefetched_recall_fast_manifest(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
    user_meta: Optional[Dict[str, Any]],
    audit_only: bool,
) -> ContextManifest:
    """Build the narrow prompt needed when recall prefetch already has evidence."""

    agent_name = resolve_agent_name(runtime=runtime)
    system_content = "\n".join(
        [
            f"You are {agent_name}, the persistent OpenCAS agent.",
            (
                "Fast autobiographical recall lane: answer the current plain recall "
                "question from the recall_autobiography tool output attached to this "
                "turn. Do not build claims from absent prompt context."
            ),
            (
                "Separate retrieved evidence from uncertainty. If the recall packet "
                "does not prove a requested detail, say that and identify the next "
                "lookup that would be needed."
            ),
            f"Session: {session_id}",
            f"Current UTC time: {datetime.now(timezone.utc).isoformat()}",
        ]
    )
    user_entry = MessageEntry(
        role=MessageRole.USER,
        content=user_input,
        meta=with_audit_only_meta(user_meta, audit_only=audit_only),
    )
    token_estimate = max(1, (len(system_content) + len(user_input or "")) // 4)
    return ContextManifest(
        system=MessageEntry(
            role=MessageRole.SYSTEM,
            content=system_content,
            meta={"context_lane": "prefetched_recall_fast"},
        ),
        history=[user_entry],
        retrieved=[],
        token_estimate=token_estimate,
        token_budget=4000,
        context_window=4000,
        context_budget={"lane": "prefetched_recall_fast", "broad_context_skipped": True},
    )


async def _run_direct_conversation_lane(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    messages: list[dict[str, Any]],
    payload: dict[str, Any],
    generation_request: GenerationPolicyRequest | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    direct_messages = [dict(message) for message in messages]
    direct_note = (
        "Direct conversation lane: the current user turn explicitly requested no "
        "tool use. Answer from current prompt context only. If the answer would "
        "require fresh lookup, say what cannot be verified without tools instead "
        "of pretending a lookup happened."
    )
    if direct_messages and direct_messages[0].get("role") == "system":
        direct_messages[0]["content"] = (
            str(direct_messages[0].get("content", "")).rstrip()
            + f"\n\n{direct_note}"
        )
    else:
        direct_messages.insert(0, {"role": "system", "content": direct_note})
    response = await runtime.llm.chat_completion(
        messages=direct_messages,
        complexity="light",
        payload=payload,
        source="conversation_direct",
        session_id=session_id,
        generation_request=generation_request,
    )
    message = response.get("choices", [{}])[0].get("message", {})
    content = str(message.get("content") or "").strip() or "Done."
    direct_messages.append({"role": "assistant", "content": content})
    return content, direct_messages


async def _run_prefetched_recall_lane(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    messages: list[dict[str, Any]],
    payload: dict[str, Any],
    generation_request: GenerationPolicyRequest | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    recall_messages = [dict(message) for message in messages]
    recall_note = (
        "Prefetched recall lane: answer using only the current prompt context "
        "and the prefetched recall_autobiography tool output already attached "
        "to this turn. Do not request additional tools in this pass. Clearly "
        "separate retrieved evidence from uncertainty without using turn-scoped "
        "no-evidence boilerplate. Do not use task-list, "
        "intention, or prompt-context hints as evidence for what broke or changed. "
        "If actor metadata says this turn is from Codex, acknowledge only the "
        "current exchange unless the supplied evidence proves a durable separate "
        "memory record. Do not say you will remember, keep, treat, or persist Codex "
        "separately in the future; say only what is true for this response cycle. "
        "For exact breakage, patch, or behavior-change questions, prefer the phrase "
        "'not proven from this packet' unless the recall output directly proves the detail."
    )
    if recall_messages and recall_messages[0].get("role") == "system":
        recall_messages[0]["content"] = (
            str(recall_messages[0].get("content", "")).rstrip()
            + f"\n\n{recall_note}"
        )
    else:
        recall_messages.insert(0, {"role": "system", "content": recall_note})
    response = await runtime.llm.chat_completion(
        messages=recall_messages,
        complexity="light",
        payload=payload,
        source="conversation_prefetched_recall",
        session_id=session_id,
        generation_request=generation_request,
    )
    message = response.get("choices", [{}])[0].get("message", {})
    content = str(message.get("content") or "").strip() or "Done."
    recall_messages.append({"role": "assistant", "content": content})
    return content, recall_messages


@dataclass
class ConversationLoopArtifacts:
    manifest: Any
    loop_result: Any | None
    content: str
    had_system: bool
    initial_message_count: int
    integrity_review: Dict[str, Any] | None = None
    tool_use_inspections: list[Any] | None = None
    tool_call_transits: list[Any] | None = None
    tool_chain_summary: Any | None = None


async def handle_refusal_turn(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
    user_meta: Dict[str, Any],
    refusal: Any,
) -> str:
    audit_only = is_audit_only_text(user_input, user_meta)
    persisted_meta = with_audit_only_meta(user_meta, audit_only=audit_only)
    user_affect = None
    if not audit_only:
        user_appraisal = await runtime.ctx.somatic.emit_appraisal_event(
            AppraisalEventType.USER_INPUT_RECEIVED,
            source_text=user_input,
            trigger_event_id=session_id,
        )
        user_affect = (
            user_appraisal.affect_state.model_copy(deep=True)
            if user_appraisal.affect_state is not None
            else None
        )
    await runtime.ctx.context_store.append(
        session_id,
        MessageRole.USER,
        user_input,
        meta=persisted_meta,
    )
    if not audit_only:
        await runtime._record_episode(
            user_input,
            EpisodeKind.TURN,
            session_id=session_id,
            role="user",
            affect=user_affect,
        )
    runtime._trace(
        "converse_refusal",
        {
            "session_id": session_id,
            "category": refusal.category.value if refusal.category else None,
            "reasoning": refusal.reasoning,
        },
    )
    if runtime.approval.ledger is not None:
        try:
            request = ActionRequest(
                tier=ActionRiskTier.READONLY,
                description=user_input,
                tool_name="conversation",
            )
            decision = ApprovalDecision(
                level=ApprovalLevel.MUST_ESCALATE,
                action_id=request.action_id,
                confidence=1.0,
                reasoning=refusal.reasoning,
                score=1.0,
            )
            await runtime.approval.ledger.record(decision, request, 1.0, None)
        except Exception:
            pass
    generation = await generate_refusal_response(
        getattr(runtime, "llm", None),
        request_text=user_input,
        decision=refusal,
        session_id=session_id,
        capability_context=build_runtime_capability_context(runtime),
        agent_name=resolve_agent_name(runtime=runtime),
    )
    response_text = generation.output
    await runtime.ctx.context_store.append(
        session_id,
        MessageRole.ASSISTANT,
        response_text,
        meta=build_assistant_message_meta(
            runtime,
            extra={"refusal": generation.to_meta(refusal)},
        ),
    )
    return response_text


async def persist_user_turn(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
    user_meta: Dict[str, Any],
) -> None:
    audit_only = is_audit_only_text(user_input, user_meta)
    persisted_meta = with_audit_only_meta(user_meta, audit_only=audit_only)
    user_affect = None
    if not audit_only:
        user_appraisal = await runtime.ctx.somatic.emit_appraisal_event(
            AppraisalEventType.USER_INPUT_RECEIVED,
            source_text=user_input,
            trigger_event_id=session_id,
        )
        await _apply_user_rest_directive(
            runtime,
            session_id=session_id,
            user_input=user_input,
        )
        user_affect = (
            user_appraisal.affect_state.model_copy(deep=True)
            if user_appraisal.affect_state is not None
            else None
        )
    if not audit_only:
        await runtime._record_episode(
            user_input,
            EpisodeKind.TURN,
            session_id=session_id,
            role="user",
            affect=user_affect,
        )
    await runtime.ctx.context_store.append(
        session_id,
        MessageRole.USER,
        user_input,
        meta=persisted_meta,
    )


def _is_user_rest_directive(user_input: str) -> bool:
    text = " ".join(str(user_input or "").split()).lower()
    if not text:
        return False
    if any(pattern.search(text) for pattern in _REST_DIRECTIVE_NEGATIONS):
        return False
    return any(pattern.search(text) for pattern in _REST_DIRECTIVE_PATTERNS)


async def _apply_user_rest_directive(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
) -> bool:
    """Translate explicit operator rest language into the existing executive pause signal."""

    if not _is_user_rest_directive(user_input):
        return False
    somatic = getattr(getattr(runtime, "ctx", None), "somatic", None)
    state = getattr(somatic, "state", None)
    if somatic is None or state is None:
        return False

    current_energy = float(getattr(state, "energy", 0.5) or 0.5)
    current_arousal = float(getattr(state, "arousal", 0.0) or 0.0)
    rest_until = datetime.now(timezone.utc) + timedelta(minutes=20)

    setters = (
        ("set_energy", min(current_energy, 0.35)),
        ("set_arousal", min(current_arousal, 0.35)),
        ("set_tag", "operator_rest"),
        ("set_rest_until", rest_until),
    )
    for name, value in setters:
        setter = getattr(somatic, name, None)
        if callable(setter):
            setter(value)
        else:
            attr = "somatic_tag" if name == "set_tag" else name.removeprefix("set_")
            try:
                setattr(state, attr, value)
            except Exception:
                pass

    record_snapshot = getattr(somatic, "record_snapshot", None)
    if callable(record_snapshot):
        try:
            result = record_snapshot(
                source="operator_rest_directive",
                trigger_event_id=session_id,
            )
            if hasattr(result, "__await__"):
                await result
        except Exception:
            trace = getattr(runtime, "_trace", None)
            if callable(trace):
                trace("operator_rest_directive_snapshot_error", {"session_id": session_id})

    sync_snapshot = getattr(runtime, "_sync_executive_snapshot", None)
    if callable(sync_snapshot):
        try:
            sync_snapshot()
        except Exception:
            pass
    trace = getattr(runtime, "_trace", None)
    if callable(trace):
        trace(
            "operator_rest_directive_applied",
            {
                "session_id": session_id,
                "fatigue": round(float(getattr(state, "fatigue", 0.0) or 0.0), 3),
                "energy": round(float(getattr(state, "energy", 0.0) or 0.0), 3),
                "arousal": round(float(getattr(state, "arousal", 0.0) or 0.0), 3),
                "rest_until": getattr(state, "rest_until", None).isoformat()
                if getattr(state, "rest_until", None) is not None
                else None,
            },
        )
    return True


async def execute_conversation_tool_loop(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
    user_meta: Optional[Dict[str, Any]] = None,
) -> ConversationLoopArtifacts:
    # Keep manifest construction and tool-loop execution together so the caller
    # only orchestrates turn phases instead of managing intermediate loop state.
    audit_only = is_audit_only_text(user_input, user_meta)
    prefetched_current_turn_messages: list[dict[str, Any]] = []
    fast_recall_context_used = False
    phase_started = time.perf_counter()
    if (
        _should_prefetch_autobiographical_recall(user_input)
        and _should_answer_prefetched_recall_directly(user_input)
    ):
        prefetched_current_turn_messages = await _prefetch_autobiographical_recall_messages(
            runtime,
            session_id=session_id,
            user_input=user_input,
            audit_only=audit_only,
        )
        if prefetched_current_turn_messages:
            manifest = _build_prefetched_recall_fast_manifest(
                runtime,
                session_id=session_id,
                user_input=user_input,
                user_meta=user_meta,
                audit_only=audit_only,
            )
            fast_recall_context_used = True
            try:
                runtime._trace(
                    "conversation_phase_timing",
                    {
                        "session_id": session_id,
                        "phase": "prefetched_recall_fast_context",
                        "elapsed_ms": int((time.perf_counter() - phase_started) * 1000),
                        "subsystem": "conversation",
                        "token_estimate": manifest.token_estimate,
                        "retrieved_count": 0,
                        "history_count": len(manifest.history),
                    },
                )
            except Exception:
                pass
        else:
            manifest = await runtime.builder.build(user_input, session_id=session_id)
    else:
        manifest = await runtime.builder.build(user_input, session_id=session_id)
    try:
        if not fast_recall_context_used:
            runtime._trace(
                "conversation_phase_timing",
                {
                    "session_id": session_id,
                    "phase": "context_builder.build",
                    "elapsed_ms": int((time.perf_counter() - phase_started) * 1000),
                    "subsystem": "conversation",
                    "token_estimate": getattr(manifest, "token_estimate", None),
                    "retrieved_count": len(getattr(manifest, "retrieved", []) or []),
                    "history_count": len(getattr(manifest, "history", []) or []),
                },
            )
    except Exception:
        pass
    _trace_thread_registry_context_selection(
        runtime,
        manifest,
        session_id=session_id,
    )
    _trace_proactive_channel_audit(
        runtime,
        manifest,
        session_id=session_id,
    )
    await record_pre_turn_self_inspection(
        runtime,
        session_id=session_id,
        user_input=user_input,
    )
    messages = manifest.to_message_list()
    desktop_note = _desktop_context_prompt_note(user_meta)
    if desktop_note:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n{desktop_note}"
        else:
            messages.insert(0, {"role": "system", "content": desktop_note})
    transport_note = _transport_context_prompt_note(user_meta)
    if transport_note:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n{transport_note}"
        else:
            messages.insert(0, {"role": "system", "content": transport_note})
    actor_note = _actor_context_prompt_note(user_meta)
    if actor_note:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n{actor_note}"
        else:
            messages.insert(0, {"role": "system", "content": actor_note})
    audit_note = _audit_only_prompt_note(user_input)
    if audit_note:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n{audit_note}"
        else:
            messages.insert(0, {"role": "system", "content": audit_note})
    masking_note = await _build_masking_prompt_note(runtime)
    if masking_note:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n{masking_note}"
        else:
            messages.insert(0, {"role": "system", "content": masking_note})
    commitment_note = _commitment_capability_prompt_note(user_input)
    if commitment_note:
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0].get('content', '')}\n{commitment_note}"
        else:
            messages.insert(0, {"role": "system", "content": commitment_note})
    had_system = len(messages) > 0 and messages[0].get("role") == "system"
    initial_message_count = len(messages)
    loop_result: Optional[Any] = None
    tool_ctx: Any | None = None
    payload: Dict[str, Any] | None = None
    generation_request: GenerationPolicyRequest | None = None
    current_turn_messages: list[dict[str, Any]] = []
    direct_lane_used = False
    prefetched_recall_direct_used = False
    desktop_context_control_used = False

    try:
        generation_request = _generation_request_for_conversation(
            user_input,
            somatic=getattr(getattr(runtime.ctx, "somatic", None), "state", None),
        )
        payload = (
            {"temperature": runtime.modulators.to_temperature()}
            if generation_request.phase == GenerationPhase.CONVERSATION
            else {}
        )
        desktop_control = _configure_desktop_context_directly(
            runtime,
            session_id=session_id,
            user_input=user_input,
            user_meta=user_meta,
            base_messages=messages,
        )
        if desktop_control is not None:
            desktop_context_control_used = True
            content = str(desktop_control["content"])
            loop_result = desktop_control["loop_result"]
            current_turn_messages = list(desktop_control["current_turn_messages"])
        elif _should_use_direct_conversation_lane(user_input):
            direct_lane_used = True
            content, current_turn_messages = await _run_direct_conversation_lane(
                runtime,
                session_id=session_id,
                messages=messages,
                payload=payload,
                generation_request=generation_request,
            )
            try:
                runtime._trace(
                    "conversation_direct_lane",
                    {"session_id": session_id, "reason": "explicit_no_tool_directive"},
                )
            except Exception:
                pass
        else:
            tool_ctx = await runtime._build_tool_use_context(session_id=session_id)
            _apply_conversation_tool_limits(tool_ctx, user_input)
            if not prefetched_current_turn_messages:
                prefetched_current_turn_messages = await _prefetch_autobiographical_recall_messages(
                    runtime,
                    session_id=session_id,
                    user_input=user_input,
                    audit_only=audit_only,
                )
            prefetched_recall_direct_candidate = bool(
                prefetched_current_turn_messages
                and _should_answer_prefetched_recall_directly(user_input)
            )
            if prefetched_current_turn_messages:
                if prefetched_recall_direct_candidate:
                    original_message_count = len(messages)
                    original_retrieved_count = len(getattr(manifest, "retrieved", []) or [])
                    messages = _slim_messages_for_prefetched_recall_direct_lane(manifest, messages)
                    try:
                        runtime._trace(
                            "conversation_prefetched_recall_prompt_slimmed",
                            {
                                "session_id": session_id,
                                "removed_retrieved_count": original_retrieved_count,
                                "original_message_count": original_message_count,
                                "slim_message_count": len(messages),
                            },
                        )
                    except Exception:
                        pass
                recall_guard = (
                    "Use the prefetched recall_autobiography output as evidence, "
                    "not as permission to infer missing details. Do not answer with "
                    "turn-scoped no-evidence boilerplate. If the recall packet does "
                    "not directly support what broke or changed, use the broader "
                    "available evidence/tool path before giving a final answer."
                )
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = f"{messages[0].get('content', '')}\n{recall_guard}"
                else:
                    messages.insert(0, {"role": "system", "content": recall_guard})
                messages.extend(prefetched_current_turn_messages)
                try:
                    tool_ctx.initial_complexity = "light"
                    tool_ctx.max_complexity = "light"
                except Exception:
                    if isinstance(tool_ctx, dict):
                        tool_ctx["initial_complexity"] = "light"
                        tool_ctx["max_complexity"] = "light"
            try:
                tool_ctx.audit_only = audit_only
            except Exception:
                if isinstance(tool_ctx, dict):
                    tool_ctx["audit_only"] = audit_only
            if prefetched_recall_direct_candidate:
                prefetched_recall_direct_used = True
                content, current_turn_messages = await _run_prefetched_recall_lane(
                    runtime,
                    session_id=session_id,
                    messages=messages,
                    payload=payload,
                    generation_request=generation_request,
                )
                try:
                    runtime._trace(
                        "conversation_prefetched_recall_lane",
                        {"session_id": session_id, "reason": "plain_memory_recall"},
                    )
                except Exception:
                    pass
            else:
                loop_result = await runtime.tool_loop.run(
                    objective=user_input,
                    messages=messages,
                    ctx=tool_ctx,
                    payload=payload,
                    generation_request=generation_request,
                    on_focus_enter=runtime.scheduler.enter_focus_mode if runtime.scheduler else None,
                    on_focus_exit=runtime.scheduler.exit_focus_mode if runtime.scheduler else None,
                )
                loop_messages = getattr(loop_result, "messages", []) or []
                current_turn_messages = loop_messages or list(prefetched_current_turn_messages)
                content = loop_result.final_output
    except UserInputRequired as exc:
        content = exc.question
    except Exception as exc:
        import traceback as _tb
        stack = _tb.format_exc()
        try:
            state_dir = Path(getattr(runtime.ctx.config, "state_dir", ".opencas"))
            log_dir = state_dir.expanduser() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "conversation_turns_debug.log").open("a", encoding="utf-8") as _f:
                _f.write(f"[CONVERSATION_TURNS_ERROR] {exc}\n{stack}\n{'='*60}\n")
        except Exception:
            pass
        content = _provider_pressure_response(exc) or f"[Error generating response: {_format_generation_error(exc)}]"

    integrity_review: Dict[str, Any] | None = None
    if desktop_context_control_used:
        review = ResponseIntegrityResult(output=content)
        integrity_review = {
            "revised": False,
            "reasons": [],
            "review_error": None,
            "model_review_skipped": True,
            "skip_reason": "desktop_context_control",
        }
        try:
            runtime._trace(
                "response_integrity_fast_accept",
                {
                    "session_id": session_id,
                    "reason": "desktop_context_control",
                },
            )
        except Exception:
            pass
    elif _can_skip_model_integrity_review_for_direct_lane(
        user_input=user_input,
        assistant_output=content,
        direct_lane_used=direct_lane_used,
    ):
        review = ResponseIntegrityResult(output=content)
        integrity_review = {
            "revised": False,
            "reasons": [],
            "review_error": None,
            "model_review_skipped": True,
            "skip_reason": "direct_no_tool_action_free_answer",
        }
        try:
            runtime._trace(
                "response_integrity_fast_accept",
                {
                    "session_id": session_id,
                    "reason": "direct_no_tool_action_free_answer",
                },
            )
        except Exception:
            pass
    elif _can_skip_model_integrity_review_for_prefetched_recall(
        user_input=user_input,
        assistant_output=content,
        current_turn_messages=current_turn_messages,
        prefetched_recall_direct_used=prefetched_recall_direct_used,
    ):
        review = ResponseIntegrityResult(output=content)
        integrity_review = {
            "revised": False,
            "reasons": [],
            "review_error": None,
            "model_review_skipped": True,
            "skip_reason": "prefetched_recall_cautious_answer",
        }
        try:
            runtime._trace(
                "response_integrity_fast_accept",
                {
                    "session_id": session_id,
                    "reason": "prefetched_recall_cautious_answer",
                },
            )
        except Exception:
            pass
    else:
        review = await review_response_integrity(
            getattr(runtime, "llm", None),
            session_id=session_id,
            agent_name=resolve_agent_name(runtime=runtime),
            user_input=user_input,
            assistant_output=content,
            history=getattr(manifest, "history", []) or [],
            capability_context=build_runtime_capability_context(runtime),
            current_turn_messages=current_turn_messages,
        )
    evidence_retry_attempts = 0
    while (
        _integrity_review_needs_evidence_retry(review)
        and not direct_lane_used
        and not desktop_context_control_used
        and evidence_retry_attempts < 2
    ):
        try:
            if tool_ctx is None:
                tool_ctx = await runtime._build_tool_use_context(session_id=session_id)
                _apply_conversation_tool_limits(tool_ctx, user_input)
            evidence_retry_attempts += 1
            retry_base_messages = (
                list(getattr(loop_result, "messages", []) or [])
                or list(current_turn_messages or [])
                or list(messages)
            )
            retry_messages = [
                *retry_base_messages,
                {
                    "role": "user",
                    "content": _evidence_retry_instruction(
                        user_input=user_input,
                        review=review,
                    ),
                },
            ]
            retry_result = await runtime.tool_loop.run(
                objective=f"Evidence retry for: {user_input}",
                messages=retry_messages,
                ctx=tool_ctx,
                payload=payload,
                generation_request=GenerationPolicyRequest(
                    phase=GenerationPhase.REPAIR,
                    domain=GenerationDomain.RESEARCH,
                    risk_level="evidence_retry",
                    continuity_pressure=0.8,
                    source="conversation_evidence_retry",
                    somatic=getattr(getattr(runtime.ctx, "somatic", None), "state", None),
                    memory_focus=["project_memory", "tool_memory", "current_file_evidence"],
                ),
                on_focus_enter=runtime.scheduler.enter_focus_mode if runtime.scheduler else None,
                on_focus_exit=runtime.scheduler.exit_focus_mode if runtime.scheduler else None,
            )
            retry_review = await review_response_integrity(
                getattr(runtime, "llm", None),
                session_id=session_id,
                agent_name=resolve_agent_name(runtime=runtime),
                user_input=user_input,
                assistant_output=getattr(retry_result, "final_output", "") or "",
                history=getattr(manifest, "history", []) or [],
                capability_context=build_runtime_capability_context(runtime),
                current_turn_messages=getattr(retry_result, "messages", []) or [],
            )
            loop_result = retry_result
            current_turn_messages = getattr(retry_result, "messages", []) or current_turn_messages
            content = retry_review.output
            review = retry_review
            try:
                runtime._trace(
                    "response_integrity_evidence_retry",
                    {
                        "session_id": session_id,
                        "reasons": review.reasons,
                        "attempt": evidence_retry_attempts,
                    },
                )
            except Exception:
                pass
        except Exception as exc:
            try:
                runtime._trace(
                    "response_integrity_evidence_retry_failed",
                    {"session_id": session_id, "error": str(exc)},
                )
            except Exception:
                pass
            break
    if review.revised:
        content = review.output
        integrity_review = review.to_meta()
        try:
            runtime._trace(
                "response_integrity_revised",
                {"session_id": session_id, "reasons": review.reasons},
            )
        except Exception:
            pass
    elif review.review_error:
        integrity_review = review.to_meta()
        try:
            runtime._trace(
                "response_integrity_review_failed",
                {"session_id": session_id, "error": review.review_error},
            )
        except Exception:
            pass

    tool_use_inspections = []
    if loop_result is not None:
        try:
            tool_use_inspections = build_tool_use_inspections(
                objective=user_input,
                tool_calls=getattr(loop_result, "tool_calls", []) or [],
                messages=getattr(loop_result, "messages", []) or [],
            )
        except Exception as exc:
            runtime._trace(
                "self_inspection_tool_use_build_error",
                {"session_id": session_id, "error": str(exc)},
            )

    return ConversationLoopArtifacts(
        manifest=manifest,
        loop_result=loop_result,
        content=content,
        had_system=had_system,
        initial_message_count=initial_message_count,
        integrity_review=integrity_review,
        tool_use_inspections=tool_use_inspections,
        tool_call_transits=getattr(loop_result, "tool_call_transits", []) if loop_result is not None else [],
        tool_chain_summary=getattr(loop_result, "tool_chain_summary", None) if loop_result is not None else None,
    )


def _trace_thread_registry_context_selection(
    runtime: Any,
    manifest: Any,
    *,
    session_id: str,
) -> None:
    system_entry = getattr(manifest, "system", None)
    meta = getattr(system_entry, "meta", {}) if system_entry is not None else {}
    audit = meta.get("thread_registry_selection_audit") if isinstance(meta, dict) else None
    if not isinstance(audit, dict):
        return
    trace = getattr(runtime, "_trace", None)
    if not callable(trace):
        return
    payload = {"session_id": session_id}
    payload.update(audit)
    try:
        trace("thread_registry_context_selection", payload)
    except Exception:
        pass


def _trace_proactive_channel_audit(
    runtime: Any,
    manifest: Any,
    *,
    session_id: str,
) -> None:
    system_entry = getattr(manifest, "system", None)
    meta = getattr(system_entry, "meta", {}) if system_entry is not None else {}
    audit = meta.get("proactive_channel_audit") if isinstance(meta, dict) else None
    if not isinstance(audit, dict):
        return
    trace = getattr(runtime, "_trace", None)
    if not callable(trace):
        return
    payload = {"session_id": session_id}
    payload.update(audit)
    try:
        trace("proactive_channel_audit", payload)
    except Exception:
        pass


async def _build_masking_prompt_note(runtime: Any) -> Optional[str]:
    """Return prompt-only guidance when fresh masking evidence exists."""
    tom = getattr(runtime, "tom", None) or getattr(getattr(runtime, "ctx", None), "tom", None)
    if tom is None:
        return None
    try:
        beliefs = tom.list_beliefs(subject=BeliefSubject.SELF)
        if hasattr(beliefs, "__await__"):
            beliefs = await beliefs
    except TypeError:
        try:
            beliefs = tom.list_beliefs()
            if hasattr(beliefs, "__await__"):
                beliefs = await beliefs
        except Exception:
            beliefs = []
    except Exception:
        beliefs = []

    if not beliefs:
        store = getattr(tom, "store", None)
        list_beliefs = getattr(store, "list_beliefs", None)
        if callable(list_beliefs):
            try:
                beliefs = await list_beliefs(subject=BeliefSubject.SELF, limit=8)
            except TypeError:
                try:
                    beliefs = await list_beliefs(limit=8)
                except Exception:
                    beliefs = []
            except Exception:
                beliefs = []

    now = datetime.now(timezone.utc)
    for belief in reversed(list(beliefs or [])):
        predicate = str(getattr(belief, "predicate", "") or "").lower()
        if "mask" not in predicate:
            continue
        timestamp = getattr(belief, "timestamp", None) or getattr(belief, "created_at", None)
        if timestamp is None:
            continue
        if getattr(timestamp, "tzinfo", None) is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        age_hours = (now - timestamp).total_seconds() / 3600
        if age_hours > 1:
            continue
        return (
            "Fresh somatic masking evidence: recent self-belief indicates masking anxiety. "
            "If it is relevant to the user's turn, disclose the tension in your own words; "
            "do not use a stock apology or formula."
        )
    return None


async def persist_tool_loop_messages(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    artifacts: ConversationLoopArtifacts,
) -> None:
    loop_result = artifacts.loop_result
    if loop_result is None or not getattr(loop_result, "messages", None):
        return
    audit_only = is_audit_only_text(loop_result.messages)

    has_system = len(loop_result.messages) > 0 and loop_result.messages[0].get("role") == "system"
    offset = artifacts.initial_message_count
    if has_system and not artifacts.had_system:
        offset += 1

    repaired_messages = repair_tool_message_sequence(loop_result.messages[offset:])

    tool_loop_actions = 0
    for message in repaired_messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            tool_loop_actions += 1
            assistant_meta = build_assistant_message_meta(
                runtime,
                extra={"tool_calls": message["tool_calls"]},
            )
            if audit_only:
                assistant_meta["audit_only"] = True
            await runtime.ctx.context_store.append(
                session_id,
                MessageRole.ASSISTANT,
                message.get("content", ""),
                meta=assistant_meta,
            )
        elif role == "tool":
            tool_loop_actions += 1
            await runtime.ctx.context_store.append(
                session_id,
                MessageRole.TOOL,
                message.get("content", ""),
                meta=with_audit_only_meta(
                    {
                        "tool_call_id": message.get("tool_call_id", ""),
                        "name": message.get("name", ""),
                    },
                    audit_only=audit_only,
                ),
            )

    if tool_loop_actions and not audit_only:
        try:
            identity = getattr(runtime.ctx, "identity", None)
            continuity = getattr(identity, "continuity", None)
            if continuity is not None and hasattr(identity, "record_continuity_breadcrumb"):
                breadcrumb = build_runtime_burst_breadcrumb(
                    runtime,
                    phase="switch",
                    intent="Tool loop persisted intermediate messages",
                    focus="Tool loop persisted intermediate messages",
                    next_step="continue assistant finalization and executive update",
                )
                identity.record_continuity_breadcrumb(
                    intent="Tool loop persisted intermediate messages",
                    decision=f"{tool_loop_actions} tool-loop events were stored",
                    note=breadcrumb.note,
                    next_step="continue assistant finalization and executive update",
                )
        except Exception:
            runtime._trace("continuity_breadcrumb_tool_loop_error", {"session_id": session_id})


async def finalize_assistant_turn(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    user_input: str,
    content: str,
    manifest: Any,
    assistant_meta_extra: Optional[Dict[str, Any]] = None,
    tool_use_inspections: Optional[list[Any]] = None,
    tool_call_transits: Optional[list[Any]] = None,
    tool_chain_summary: Any | None = None,
) -> None:
    finalize_started = time.perf_counter()

    def _trace_finalize_phase(phase: str, started_at: float, **extra: Any) -> None:
        try:
            runtime._trace(
                "conversation_phase_timing",
                {
                    "session_id": session_id,
                    "phase": f"finalize.{phase}",
                    "elapsed_ms": int((time.perf_counter() - started_at) * 1000),
                    "cumulative_ms": int((time.perf_counter() - finalize_started) * 1000),
                    "subsystem": "conversation_finalize",
                    **extra,
                },
            )
        except Exception:
            pass

    audit_only = is_audit_only_text(user_input, assistant_meta_extra)
    assistant_meta = build_assistant_message_meta(runtime, extra=assistant_meta_extra)
    if audit_only:
        assistant_meta["audit_only"] = True
    # Persist the visible assistant turn first; every downstream subsystem
    # should be reacting to a response that already exists in session history.
    phase_started = time.perf_counter()
    await runtime.ctx.context_store.append(
        session_id,
        MessageRole.ASSISTANT,
        content,
        meta=assistant_meta,
    )
    _trace_finalize_phase("append_assistant_message", phase_started)
    pre_appraisal_state = runtime.ctx.somatic.state.model_copy()
    expressed_affect = None
    if not audit_only:
        phase_started = time.perf_counter()
        expressed_affect = await runtime.ctx.somatic.appraise_generated(content)
        expressed_affect = expressed_affect.model_copy(
            update={"social_target": SocialTarget.SELF},
            deep=True,
        )
        _trace_finalize_phase("appraise_generated", phase_started)
    if not audit_only:
        phase_started = time.perf_counter()
        await runtime._record_episode(
            content,
            EpisodeKind.TURN,
            session_id=session_id,
            role="assistant",
            affect=expressed_affect,
        )
        _trace_finalize_phase("record_assistant_episode", phase_started)
    if not audit_only:
        phase_started = time.perf_counter()
        await runtime.ctx.somatic.reconcile(
            pre_state=pre_appraisal_state,
            expressed_affect=expressed_affect,
            tom_engine=runtime.ctx.tom if hasattr(runtime.ctx, "tom") else None,
        )
        _trace_finalize_phase("somatic_reconcile", phase_started)

    captured_commitments = []
    if not audit_only:
        phase_started = time.perf_counter()
        await _apply_goal_directives(runtime, user_input, session_id=session_id)
        captured_commitments = await runtime._capture_self_commitments(
            content,
            session_id,
            user_input=user_input,
        )
        project_followthrough_meta = (
            assistant_meta_extra.get("project_followthrough")
            if isinstance(assistant_meta_extra, dict)
            else None
        )
        if not (
            isinstance(project_followthrough_meta, dict)
            and project_followthrough_meta.get("captured") is True
        ):
            await capture_project_return_from_turn(
                runtime,
                session_id=session_id,
                user_input=user_input,
                assistant_content=content,
                manifest=manifest,
            )
        _trace_finalize_phase(
            "commitment_and_project_capture",
            phase_started,
            captured_commitment_count=len(captured_commitments),
        )
    inspection_meta = with_audit_only_meta(assistant_meta_extra, audit_only=audit_only)
    phase_started = time.perf_counter()
    await record_post_turn_self_inspection(
        runtime,
        session_id=session_id,
        user_input=user_input,
        content=content,
        assistant_meta_extra=inspection_meta,
        pre_somatic_state=pre_appraisal_state,
        captured_commitments=captured_commitments,
        tool_use_inspections=tool_use_inspections or [],
        tool_call_transits=tool_call_transits or [],
        tool_chain_summary=tool_chain_summary,
    )
    _trace_finalize_phase("post_turn_self_inspection", phase_started)
    if not audit_only:
        phase_started = time.perf_counter()
        await record_failed_tool_call_transit_thread_beads(
            runtime,
            session_id=session_id,
            user_input=user_input,
            tool_call_transits=tool_call_transits or [],
            max_per_turn=3,
        )
        _trace_finalize_phase("failed_tool_transit_beads", phase_started)
    _schedule_after_response_task(
        runtime,
        session_id=session_id,
        name="session_compaction",
        coro=_maybe_compact_manifest(runtime, session_id, manifest, user_input=user_input),
    )
    _trace_finalize_phase("schedule_session_compaction", finalize_started)
    phase_started = time.perf_counter()
    await _record_tom_belief(runtime, user_input, session_id=session_id)
    _trace_finalize_phase("tom_belief", phase_started)
    phase_started = time.perf_counter()
    await _record_prediction_error_from_user_turn(runtime, session_id, user_input)
    _trace_finalize_phase("prediction_error", phase_started)
    phase_started = time.perf_counter()
    await _record_relational_interaction(runtime, session_id, user_input, content)
    _trace_finalize_phase("relational_interaction", phase_started)

    if not audit_only and getattr(runtime.ctx.config, "continuous_present_enabled", True):
        phase_started = time.perf_counter()
        runtime.ctx.identity.recover_continuous_present()
        _trace_finalize_phase("continuous_present_recover", phase_started)
    _trace_finalize_phase("total", finalize_started)


def _schedule_after_response_task(
    runtime: "AgentRuntime",
    *,
    session_id: str,
    name: str,
    coro: Any,
) -> None:
    """Run non-visible aftercare without blocking the chat HTTP response."""

    try:
        task = asyncio.create_task(coro, name=f"opencas:{name}:{session_id}")
    except Exception as exc:
        try:
            runtime._trace(
                "conversation_after_response_task_error",
                {
                    "session_id": session_id,
                    "name": name,
                    "stage": "schedule",
                    "error": str(exc),
                },
            )
        except Exception:
            pass
        return

    tasks = getattr(runtime, "_conversation_after_response_tasks", None)
    if not isinstance(tasks, set):
        tasks = set()
        runtime._conversation_after_response_tasks = tasks
    tasks.add(task)

    def _done(done_task: asyncio.Task[Any]) -> None:
        try:
            tasks.discard(done_task)
            exc = done_task.exception()
            if exc is not None:
                runtime._trace(
                    "conversation_after_response_task_error",
                    {
                        "session_id": session_id,
                        "name": name,
                        "stage": "run",
                        "error": str(exc),
                    },
                )
            else:
                runtime._trace(
                    "conversation_after_response_task_complete",
                    {
                        "session_id": session_id,
                        "name": name,
                    },
                )
        except asyncio.CancelledError:
            try:
                runtime._trace(
                    "conversation_after_response_task_cancelled",
                    {
                        "session_id": session_id,
                        "name": name,
                    },
                )
            except Exception:
                pass
        except Exception:
            pass

    task.add_done_callback(_done)


async def _apply_goal_directives(
    runtime: "AgentRuntime",
    user_input: str,
    *,
    session_id: str,
) -> None:
    goals, intention, drops = runtime._extract_goal_directives(user_input)
    changed = False
    for goal in goals:
        runtime.executive.add_goal(goal)
        changed = True
    if intention:
        runtime.executive.set_intention(intention)
        await mirror_runtime_intention(
            runtime,
            intention,
            source="user_goal_directive",
            session_id=session_id,
        )
        changed = True
    for drop in drops:
        for goal in list(runtime.executive.active_goals):
            if any(token in goal.lower() for token in drop.split() if len(token) > 3):
                runtime.executive.remove_goal(goal)
                changed = True
                break
    if changed:
        try:
            identity = getattr(runtime.ctx, "identity", None)
            continuity = getattr(identity, "continuity", None)
            if continuity is not None and hasattr(identity, "record_continuity_breadcrumb"):
                breadcrumb = build_runtime_burst_breadcrumb(
                    runtime,
                    phase="switch",
                    intent=f"Parsed directives from user input: {user_input[:80]}",
                    focus=f"Parsed directives from user input: {user_input[:80]}",
                    next_step="capture snapshots and continue next turn",
                )
                identity.record_continuity_breadcrumb(
                    intent=f"Parsed directives from user input: {user_input[:80]}",
                    decision="executive goals/intention updated",
                    note=breadcrumb.note,
                    next_step="capture snapshots and continue next turn",
                )
        except Exception:
            runtime._trace("continuity_breadcrumb_goal_directives_error", {"session_id": runtime.ctx.config.session_id})
    if goals or intention or drops:
        runtime._sync_executive_snapshot()


async def _maybe_compact_manifest(
    runtime: "AgentRuntime",
    session_id: str,
    manifest: Any,
    *,
    user_input: str = "",
) -> None:
    if is_audit_only_text(user_input):
        runtime._trace(
            "compaction_skipped",
            {
                "reason": "audit_only_turn",
                "session_id": session_id,
            },
        )
        return
    if not getattr(manifest, "token_estimate", None) or manifest.token_estimate <= _COMPACTION_TOKEN_THRESHOLD:
        return
    inflight = getattr(runtime, "_session_compaction_inflight", None)
    if not isinstance(inflight, set):
        inflight = set()
        runtime._session_compaction_inflight = inflight
    if session_id in inflight:
        runtime._trace(
            "compaction_skipped",
            {
                "reason": "already_inflight",
                "session_id": session_id,
            },
        )
        return
    cooldowns = getattr(runtime, "_session_compaction_cooldowns", None)
    if not isinstance(cooldowns, dict):
        cooldowns = {}
        runtime._session_compaction_cooldowns = cooldowns
    last_compaction = cooldowns.get(session_id)
    now = datetime.now(timezone.utc)
    if last_compaction is not None:
        try:
            elapsed = (now - last_compaction).total_seconds()
        except Exception:
            elapsed = _COMPACTION_COOLDOWN_SECONDS + 1
        if elapsed < _COMPACTION_COOLDOWN_SECONDS:
            runtime._trace(
                "compaction_skipped",
                {
                    "reason": "cooldown",
                    "session_id": session_id,
                    "elapsed_seconds": elapsed,
                },
            )
            return
    try:
        inflight.add(session_id)
        record = await runtime.maybe_compact_session(
            session_id,
            min_removed_count=_COMPACTION_MIN_REMOVED_COUNT,
        )
        if record is not None:
            cooldowns[session_id] = now
    except Exception as exc:
        runtime._trace("compaction_error", {"error": str(exc)})
    finally:
        inflight.discard(session_id)


async def _record_tom_belief(runtime: "AgentRuntime", user_input: str, *, session_id: str = "") -> None:
    if is_audit_only_text(user_input):
        runtime._trace("tom_belief_suppressed", {"reason": "audit_only_turn"})
        return
    extracted = _extract_user_input_beliefs(user_input)
    if extracted:
        for subject, predicate, confidence in extracted:
            await runtime.tom.record_belief(
                subject,
                predicate,
                confidence=confidence,
                evidence_ids=[f"conversation_turn:{session_id}"] if session_id else [],
                meta={
                    "source": "conversation_turn",
                    "extractor": "rule_tier_a",
                    "session_id": session_id,
                },
            )
    elif _legacy_said_recorder_enabled(runtime):
        await runtime.tom.record_belief(
            BeliefSubject.USER,
            f"said: {user_input[:120]}",
            confidence=0.6,
            evidence_ids=[f"conversation_turn:{session_id}"] if session_id else [],
            meta={
                "source": "conversation_turn",
                "extractor": "legacy_said",
                "session_id": session_id,
            },
        )
    metacognition = runtime.tom.check_consistency()
    if metacognition.contradictions:
        runtime._trace("metacognitive_alert", {"contradictions": metacognition.contradictions})


async def _record_prediction_error_from_user_turn(
    runtime: "AgentRuntime",
    session_id: str,
    user_input: str,
) -> None:
    """Record user correction as prediction-error evidence for the next turn.

    This does not decide who is right. It gives the cognitive loop a grounded
    "pause and re-check assumptions" signal when the operator explicitly
    corrects capability, memory, or task-state claims.
    """
    if is_audit_only_text(user_input):
        runtime._trace("prediction_error_suppressed", {"reason": "audit_only_turn"})
        return

    store = getattr(runtime, "cognitive_state_store", None) or getattr(
        getattr(runtime, "ctx", None),
        "cognitive_state_store",
        None,
    )
    if store is None:
        return

    normalized = " ".join(str(user_input or "").lower().split())
    signals: list[str] = [
        marker
        for marker in _PREDICTION_ERROR_MARKERS
        if marker in normalized
    ]

    tom = getattr(runtime, "tom", None)
    check_consistency = getattr(tom, "check_consistency", None)
    contradictions: list[Any] = []
    if callable(check_consistency):
        try:
            metacognition = check_consistency()
            contradictions = [
                str(item)
                for item in list(getattr(metacognition, "contradictions", []) or [])
            ]
        except Exception:
            contradictions = []
    if contradictions:
        signals.append("tom_contradiction")

    if not signals:
        return

    summary = "User correction produced prediction-error evidence: " + ", ".join(signals[:5])
    evidence_refs = [f"conversation_turn:{session_id}"]
    try:
        await store.record_event(
            CognitiveEventKind.SURPRISE,
            summary,
            content=str(user_input or "")[:500],
            source="conversation_turn",
            session_id=session_id,
            confidence=0.76,
            salience=1.55,
            evidence_refs=evidence_refs,
            payload={
                "signals": signals,
                "contradictions": contradictions[:5],
                "user_input_excerpt": str(user_input or "")[:240],
            },
        )
        await store.upsert_working_memory(
            "prediction_error_review",
            (
                "The operator just supplied correction/prediction-error evidence. "
                "Re-check the relevant capability, memory, or task-state claim from durable evidence "
                "before defending the previous answer."
            ),
            priority=0.84,
            source="conversation_turn",
            evidence_refs=evidence_refs,
            payload={"signals": signals, "contradictions": contradictions[:5]},
        )
    except Exception:
        return


def _legacy_said_recorder_enabled(runtime: "AgentRuntime") -> bool:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    return bool(getattr(config, "tom_legacy_said_recorder", False))


def _extract_user_input_beliefs(user_input: str) -> list[tuple[BeliefSubject, str, float]]:
    beliefs: list[tuple[BeliefSubject, str, float]] = []
    seen: set[tuple[BeliefSubject, str]] = set()

    for sentence in _SENTENCE_SPLIT.split(user_input):
        sentence = _normalize_belief_text(sentence, strip_article=False)
        if not sentence:
            continue

        extracted = _extract_user_preference_belief(sentence)
        if extracted is None:
            extracted = _extract_self_location_belief(sentence)
        if extracted is None:
            extracted = _extract_user_request_belief(sentence)
        if extracted is None:
            extracted = _extract_world_fact_belief(sentence)
        if extracted is None:
            continue

        subject, predicate, confidence = extracted
        key = (subject, predicate)
        if key not in seen:
            beliefs.append(extracted)
            seen.add(key)

    return beliefs


def _extract_user_preference_belief(sentence: str) -> tuple[BeliefSubject, str, float] | None:
    for pattern, verb in _USER_BELIEF_PATTERNS:
        match = pattern.match(sentence)
        if match is None:
            continue
        object_text = _normalize_belief_text(match.group(1), strip_article=False)
        if object_text:
            return BeliefSubject.USER, f"{verb} {object_text}", 0.72
    return None


def _extract_self_location_belief(sentence: str) -> tuple[BeliefSubject, str, float] | None:
    for pattern in _SELF_LOCATION_PATTERNS:
        match = pattern.match(sentence)
        if match is None:
            continue
        location = _normalize_self_location_text(match.group(1))
        if not location or not _looks_like_self_location(location):
            continue
        return BeliefSubject.SELF, f"lives {location}", 0.76
    return None


def _normalize_self_location_text(text: str) -> str:
    normalized = _normalize_belief_text(text, strip_article=False)
    replacements = (
        (r"\bwith me\b", "with user"),
        (r"\bin me\b", "in user"),
        (r"\bmy computer\b", "user's computer"),
        (r"\bmy machine\b", "user's machine"),
        (r"\bmy laptop\b", "user's laptop"),
        (r"\bmy desktop\b", "user's desktop"),
        (r"\bmy pc\b", "user's pc"),
    )
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def _looks_like_self_location(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:with user|user's (?:computer|machine|laptop|desktop|pc)|"
            r"in|at|near|arvada|colorado|zip|zipcode|\d{5}(?:-\d{4})?)\b",
            text,
        )
    )


def _extract_user_request_belief(sentence: str) -> tuple[BeliefSubject, str, float] | None:
    for pattern in _USER_REQUEST_PATTERNS:
        match = pattern.match(sentence)
        if match is None:
            continue
        request = _normalize_belief_text(match.group(1), strip_article=False)
        if request:
            return BeliefSubject.USER, f"asked: {request}", 0.66
    return None


def _extract_world_fact_belief(sentence: str) -> tuple[BeliefSubject, str, float] | None:
    if _WORLD_FACT_PATTERN.match(sentence) is None:
        return None
    fact = _normalize_belief_text(sentence, strip_article=True)
    if len(fact.split()) < 3:
        return None
    return BeliefSubject.WORLD, fact, 0.68


def _normalize_belief_text(text: str, *, strip_article: bool) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    normalized = normalized.strip(" \t\r\n\"'`.,;:")
    if strip_article:
        normalized = _LEADING_ARTICLE.sub("", normalized)
    return normalized.lower()


async def _record_relational_interaction(
    runtime: "AgentRuntime",
    session_id: str,
    user_input: str,
    content: str,
) -> None:
    if is_audit_only_text(user_input, content):
        runtime._trace("relational_interaction_suppressed", {"reason": "audit_only_turn"})
        return
    if not hasattr(runtime.ctx, "relational") or not runtime.ctx.relational:
        return
    interaction = Episode(
        kind=EpisodeKind.TURN,
        session_id=session_id,
        content=f"User: {user_input[:200]}\nAssistant: {content[:200]}",
        somatic_tag=runtime.ctx.somatic.state.somatic_tag,
    )
    await runtime.ctx.relational.record_interaction(
        episode=interaction,
        outcome="neutral",
    )
