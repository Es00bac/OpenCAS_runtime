"""Capture unfinished conversational projects as durable return points."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from opencas.autonomy.commitment import Commitment
from opencas.scheduling import (
    ScheduleAction,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleStatus,
)

_PROJECT_MARKERS = (
    "project",
    "book",
    "manuscript",
    "draft",
    "story",
    "chronicle",
    "creative task",
    "research task",
    "writing",
)
_RETURN_MARKERS = (
    "keep working",
    "continue",
    "follow through",
    "follow up",
    "return to",
    "resume",
    "until done",
    "until it feels complete",
    "until complete",
    "without approval",
    "shouldn't need my approval",
    "should not need my approval",
)
_UNFINISHED_MARKERS = (
    "next",
    "need to",
    "still need",
    "continue",
    "keep revising",
    "keep working",
    "return",
    "resume",
    "finish",
    "not complete",
    "remaining",
    "before i commit",
)
_NEXT_STEP_MARKERS = (
    "next",
    "i need to",
    "i still need",
    "need to",
    "continue",
    "keep revising",
    "keep working",
    "return to",
    "resume",
    "finish",
)
_PROJECT_INTENT_MARKERS = (
    "revise",
    "revision",
    "edit",
    "finish",
    "complete",
    "until it feels complete",
    "until complete",
    "happy with",
    "satisfied",
    "critique",
    "manuscript",
    "book",
    "novel",
)
_CREATIVE_WRITING_MARKERS = (
    "book",
    "manuscript",
    "draft",
    "story",
    "novel",
    "writing",
    "creative",
    "chronicle",
)
_CHRONICLE_RE = re.compile(r"\bChronicle\s+\d+\b", re.IGNORECASE)
_TITLE_RE = re.compile(
    r"\b(?:project|book|manuscript|draft|story)\s+"
    r"(?:called|named|titled)?\s*['\"]?(?P<title>[A-Z][A-Za-z0-9 _:-]{2,80})",
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NON_KEY_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class ProjectReturnCapture:
    """Result of capturing a project return point from a chat turn."""

    project_key: str
    project_title: str
    project_intent: str
    next_step: str
    commitment_id: str
    schedule_id: Optional[str]


async def capture_project_return_from_turn(
    runtime: Any,
    *,
    session_id: str,
    user_input: str,
    assistant_content: str,
    manifest: Any = None,
    now: Optional[datetime] = None,
) -> Optional[ProjectReturnCapture]:
    """Persist a future project return when a chat turn exposes unfinished work.

    This is not a script for any one project. It only captures grounded signals
    already present in the turn/context: a project-like object, an expectation of
    autonomous return/follow-through, and an unfinished next step.
    """
    now = _as_utc(now or datetime.now(timezone.utc))
    context_text = _context_text(
        user_input=user_input,
        assistant_content=assistant_content,
        manifest=manifest,
    )
    if not _should_capture_project_return(context_text, assistant_content):
        return None

    title = _infer_project_title(context_text)
    project_key = _project_key(title)
    next_step = _infer_next_step(assistant_content, user_input, context_text)
    project_intent = _infer_project_intent(context_text, title, next_step)
    commitment = await _upsert_project_return_commitment(
        runtime,
        project_key=project_key,
        title=title,
        project_intent=project_intent,
        next_step=next_step,
        session_id=session_id,
        user_input=user_input,
        assistant_content=assistant_content,
        now=now,
    )
    if commitment is None:
        return None

    schedule_id = await _upsert_project_return_schedule(
        runtime,
        project_key=project_key,
        title=title,
        project_intent=project_intent,
        next_step=next_step,
        session_id=session_id,
        commitment=commitment,
        now=now,
    )
    _trace(
        runtime,
        "project_return_captured",
        {
            "project_key": project_key,
            "project_title": title,
            "commitment_id": str(commitment.commitment_id),
            "schedule_id": schedule_id,
        },
    )
    return ProjectReturnCapture(
        project_key=project_key,
        project_title=title,
        project_intent=project_intent,
        next_step=next_step,
        commitment_id=str(commitment.commitment_id),
        schedule_id=schedule_id,
    )


def _should_capture_project_return(context_text: str, assistant_content: str) -> bool:
    text = context_text.lower()
    assistant = assistant_content.lower()
    return (
        any(marker in text for marker in _PROJECT_MARKERS)
        and any(marker in text for marker in _RETURN_MARKERS)
        and any(marker in assistant for marker in _UNFINISHED_MARKERS)
    )


def _context_text(*, user_input: str, assistant_content: str, manifest: Any) -> str:
    parts = [user_input or "", assistant_content or ""]
    to_messages = getattr(manifest, "to_message_list", None)
    if callable(to_messages):
        try:
            messages = to_messages()
        except Exception:
            messages = []
        for message in list(messages)[-12:]:
            role = str(message.get("role") or "")
            if role not in {"system", "user", "assistant"}:
                continue
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(
                    str(item.get("text"))
                    for item in content
                    if isinstance(item, dict) and item.get("text")
                )
    return "\n".join(part for part in parts if part)


def _infer_project_title(text: str) -> str:
    chronicle = _CHRONICLE_RE.search(text)
    if chronicle:
        return chronicle.group(0).title()
    title_match = _TITLE_RE.search(text)
    if title_match:
        return _clean_title(title_match.group("title"))
    return "Conversation Project"


def _infer_next_step(assistant_content: str, user_input: str, context_text: str) -> str:
    for sentence in _sentences(assistant_content):
        lowered = sentence.lower()
        if any(marker in lowered for marker in _NEXT_STEP_MARKERS):
            return _normalize_next_step(sentence)
    for sentence in _sentences(user_input):
        lowered = sentence.lower()
        if any(marker in lowered for marker in _RETURN_MARKERS):
            return _normalize_next_step(sentence)
    for sentence in _sentences(context_text):
        lowered = sentence.lower()
        if any(marker in lowered for marker in _RETURN_MARKERS):
            return _normalize_next_step(sentence)
    return "Review the project context and decide the next meaningful step."


def _infer_project_intent(context_text: str, title: str, next_step: str) -> str:
    text = context_text.lower()
    if any(marker in text for marker in _CREATIVE_WRITING_MARKERS) and any(
        marker in text for marker in _PROJECT_INTENT_MARKERS
    ):
        return (
            f"revise and finish {title} until the OpenCAS agent is satisfied with the manuscript, "
            "using user critique as input while continuing writing and revision autonomously."
        )
    return (
        f"Return to {title} as a continuing project, keeping the larger project objective ahead "
        f"of the immediate subtask: {next_step}"
    )


async def _upsert_project_return_commitment(
    runtime: Any,
    *,
    project_key: str,
    title: str,
    project_intent: str,
    next_step: str,
    session_id: str,
    user_input: str,
    assistant_content: str,
    now: datetime,
) -> Optional[Commitment]:
    store = getattr(runtime, "commitment_store", None)
    if store is None:
        return None
    existing = await _find_existing_commitment(store, project_key)
    if existing is None:
        existing = Commitment(
            content=f"Return to project: {title}",
            priority=7.5,
            tags=["project_return", "conversation", "self_directed"],
        )
    existing.updated_at = now
    existing.meta.update(
        {
            "source": "project_return_capture",
            "source_session_id": session_id,
            "project_key": project_key,
            "project_title": title,
            "project_intent": project_intent,
            "next_step": next_step,
            "source_user_turn": _excerpt(user_input),
            "source_assistant_turn": _excerpt(assistant_content),
            "return_policy": "scheduled_self_review",
            "captured_at": now.isoformat(),
        }
    )
    await store.save(existing)
    return existing


async def _upsert_project_return_schedule(
    runtime: Any,
    *,
    project_key: str,
    title: str,
    project_intent: str,
    next_step: str,
    session_id: str,
    commitment: Commitment,
    now: datetime,
) -> Optional[str]:
    service = getattr(runtime, "schedule_service", None)
    if service is None:
        return None
    store = getattr(getattr(runtime, "ctx", None), "schedule_store", None) or getattr(service, "store", None)
    existing = await _find_existing_schedule(store, project_key, str(commitment.commitment_id))
    objective = _build_schedule_objective(title, project_intent, next_step, session_id, str(commitment.commitment_id))
    start_at = now + timedelta(minutes=5)
    if existing is None:
        item = await service.create_schedule(
            kind=ScheduleKind.TASK,
            action=ScheduleAction.SUBMIT_BAA,
            title=f"Return to {title}",
            description=f"Autonomous project return point for {title}.",
            objective=objective,
            start_at=start_at,
            recurrence=ScheduleRecurrence.NONE,
            priority=max(7.0, float(commitment.priority)),
            tags=["project_return", "self_directed"],
            commitment_id=str(commitment.commitment_id),
            meta={
                "source": "project_return_capture",
                "project_key": project_key,
                "project_title": title,
                "project_intent": project_intent,
                "next_step": next_step,
                "source_session_id": session_id,
            },
        )
        return str(item.schedule_id)
    existing.objective = objective
    existing.description = f"Autonomous project return point for {title}."
    existing.priority = max(existing.priority, float(commitment.priority))
    existing.meta.update(
        {
            "source": "project_return_capture",
            "project_key": project_key,
            "project_title": title,
            "project_intent": project_intent,
            "next_step": next_step,
            "source_session_id": session_id,
            "refreshed_at": now.isoformat(),
        }
    )
    if existing.next_run_at and existing.next_run_at > start_at:
        existing.next_run_at = start_at
        existing.start_at = min(existing.start_at, start_at)
    existing.recurrence = ScheduleRecurrence.NONE
    existing.interval_hours = None
    existing.max_occurrences = None
    await store.save(existing)
    return str(existing.schedule_id)


async def _find_existing_commitment(store: Any, project_key: str) -> Optional[Commitment]:
    for commitment in await store.list_active(limit=200):
        if "project_return" not in set(commitment.tags):
            continue
        if str((commitment.meta or {}).get("project_key", "")) == project_key:
            return commitment
    return None


async def _find_existing_schedule(store: Any, project_key: str, commitment_id: str) -> Any:
    if store is None:
        return None
    items = await store.list_items(status=ScheduleStatus.ACTIVE, limit=500)
    for item in items:
        if "project_return" not in set(item.tags):
            continue
        if item.commitment_id != commitment_id:
            continue
        if str((item.meta or {}).get("project_key", "")) == project_key:
            return item
    return None


def _build_schedule_objective(
    title: str,
    project_intent: str,
    next_step: str,
    session_id: str,
    commitment_id: str,
) -> str:
    return (
        f'Return to project "{title}" from session {session_id}. '
        f"Book-level intent: {project_intent} "
        f"Immediate next step: {next_step} "
        "Review the latest project context and decide whether to continue, finish, "
        "or schedule another return. Treat research and naming work as support for the manuscript, "
        "not as a substitute for manuscript or character-bible revision. Continue with writing, "
        "critique, revision, or necessary research when meaningful progress is possible without "
        "asking the user for approval. If the project remains unfinished at the end of this run, "
        "use your OpenCAS calendar to choose and create the next return time that fits the work; "
        "do not default to tomorrow when sooner is right. "
        f"If the project is finished, mark commitment {commitment_id} complete."
    )


def _sentences(text: str) -> Iterable[str]:
    for sentence in _SENTENCE_RE.split(text or ""):
        cleaned = " ".join(sentence.split()).strip(" -:")
        if cleaned:
            yield cleaned


def _normalize_next_step(sentence: str) -> str:
    cleaned = " ".join(sentence.split()).strip(" -:")
    cleaned = re.sub(r"^(?:next[:,]?\s*)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:i\s+(?:still\s+)?need\s+to\s+)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:let\s+me\s+)", "", cleaned, flags=re.IGNORECASE)
    return _excerpt(cleaned, limit=220)


def _clean_title(title: str) -> str:
    title = re.split(r"[.!?\n]", title, maxsplit=1)[0]
    return " ".join(title.strip(" '\".,:;").split())[:80] or "Conversation Project"


def _project_key(title: str) -> str:
    normalized = _NON_KEY_RE.sub("-", title.lower()).strip("-")
    return normalized or "conversation-project"


def _excerpt(text: str, *, limit: int = 320) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _trace(runtime: Any, event: str, payload: dict[str, Any]) -> None:
    tracer = getattr(runtime, "_trace", None)
    if callable(tracer):
        try:
            tracer(event, payload)
        except Exception:
            pass
