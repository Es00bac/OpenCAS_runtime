"""Capture unfinished conversational projects as durable return points."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from opencas.autonomy.commitment import Commitment, CommitmentStatus
from opencas.projects.classifier import (
    PROJECT_TYPE_SOFTWARE,
    PROJECT_TYPE_WRITING,
    any_marker_in_text,
    classify_project_type,
)
from opencas.projects.workspace_registry import (
    WorkspacePathRequest,
    WorkspaceProjectCandidate,
    resolve_requested_workspace_path,
    resolve_recent_workspace_project,
    resolve_workspace_project,
    resolve_workspace_project_from_text,
)
from opencas.scheduling import (
    ScheduleAction,
    ScheduleKind,
    ScheduleRecurrence,
    ScheduleStatus,
)
from .audit_mode import is_audit_only_text

_PROJECT_MARKERS = (
    "project",
    "app",
    "application",
    "book",
    "client",
    "code",
    "program",
    "software",
    "tool",
    "manuscript",
    "draft",
    "story",
    "creative_writing",
    "creative task",
    "research task",
    "writing",
)
_RETURN_MARKERS = (
    "keep working",
    "continue",
    "follow through",
    "follow up",
    "finish",
    "finished",
    "not complete",
    "not finished",
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
    "not finished",
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
    "creative_writing",
)
_PROJECT_ID_RE = re.compile(r"\bWriting Project\s+\d+\b", re.IGNORECASE)
_BRANDED_IDENTIFIER_RE = re.compile(r"\b[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b")
_TITLE_RE = re.compile(
    r"\b(?:project|book|manuscript|draft|story|app|application|client|tool|program|software)\s+"
    r"(?:called|named|titled)?\s*['\"]?(?P<title>[A-Za-z][A-Za-z0-9 _:-]{1,80})",
)
_CONTEXTUAL_TITLE_RE = re.compile(
    r"\b(?:marketing\s+materials?|launch\s+site|website|social\s+media\s+posts?)\s+"
    r"(?:for|about)\s+['\"]?(?P<title>[A-Z][A-Za-z0-9]*(?:\s+[A-Z0-9][A-Za-z0-9]*){0,5})",
)
_EXPLICIT_TITLE_RE = re.compile(
    r"\b(?:project\s+codename|codename|working\s+title|title)\s*[:=]\s*['\"*_ ]*(?P<title>[A-Za-z][A-Za-z0-9 _:-]{1,80})",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NON_KEY_RE = re.compile(r"[^a-z0-9]+")
_WORD_COUNT_RE = re.compile(r"\b(?P<count>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?P<suffix>k)?\s*words?\b", re.IGNORECASE)
_WORD_COUNT_TARGET_PREFIXES = (
    "aim for",
    "at least",
    "complete",
    "completion",
    "finish",
    "goal",
    "minimum",
    "produce",
    "request",
    "target",
    "want",
    "write",
)
_WORD_COUNT_TARGET_SUFFIXES = (
    "book",
    "draft",
    "goal",
    "manuscript",
    "novel",
    "target",
)
_GENERIC_PROJECT_TITLES = {
    "app",
    "application",
    "book",
    "build",
    "client",
    "code",
    "conversation project",
    "dist",
    "draft",
    "manuscript",
    "node_modules",
    "out",
    "program",
    "project",
    "software",
    "software project",
    "story",
    "tool",
    "writing project",
}
_REST_DEFER_MARKERS = (
    "take a break",
    "resting now",
    "rest now",
    "breathe for a bit",
)
_NO_AUTONOMOUS_RETURN_MARKERS = (
    "i haven't saved",
    "i have not saved",
    "i won't claim",
    "i will not claim",
    "if you want me to resume later",
    "remind me where we were",
)
_COMPLETION_CHALLENGE_MARKERS = (
    "first draft",
    "not done",
    "not finished",
    "not complete",
    "wasn't done",
    "was not done",
    "isn't done",
    "is not done",
    "needs revision",
    "need to revise",
    "needs to revise",
    "need to edit",
    "needs to edit",
)
_IMMEDIATE_OPERATOR_WORK_MARKERS = (
    "do the work",
    "start working",
    "start the work",
    "start writing",
    "start a new story",
    "start a new project",
    "start over",
    "finish it",
    "finish the",
    "complete it",
    "complete the",
    "don't stop",
    "do not stop",
    "get rid of",
    "remove copied",
    "remove the copied",
    "keep going",
    "keep working",
    "return to",
    "resume",
    "without asking",
    "without waiting",
    "until complete",
    "until done",
)
_CONTEXTUAL_IMMEDIATE_WORK_MARKERS = (
    "begin implementing",
    "do it",
    "do the thing",
    "go ahead",
    "make it happen",
    "start it",
    "use the tools",
)
_CONTEXTUAL_PROJECT_WORK_MARKERS = (
    "build",
    "create",
    "implement",
    "launch site",
    "marketing material",
    "marketing materials",
    "social media",
    "website",
)
_PROJECT_CORRECTION_MARKERS = (
    "completely different",
    "different book",
    "different project",
    "different story",
    "get rid of",
    "not a revision",
    "not seeing anything change",
    "not the same",
    "remove copied",
    "still has the copied",
    "wrong book",
    "wrong files",
    "wrong project",
    "wrong story",
)
_PROJECT_WORKING_RESPONSE_BLOCKERS = (
    "can't help",
    "cannot help",
    "don't have evidence",
    "do not have evidence",
    "don't have fresh evidence",
    "do not have fresh evidence",
    "don't have current evidence",
    "do not have current evidence",
    "haven't completed",
    "have not completed",
    "haven't searched",
    "have not searched",
    "haven't yet",
    "have not yet",
    "i can help",
    "if you want",
    "no evidence",
    "not enough evidence",
    "not yet read",
    "tell me which",
    "waiting for",
)
_CLEAN_REVISION_MARKERS = (
    "clean manuscript",
    "clean revised manuscript",
    "publication-clean",
    "publishable",
    "revised clean manuscript",
    "revised manuscript",
    "revision lock",
    "second draft",
)
_NAME_ORIGINALITY_MARKERS = (
    "derivative naming",
    "final names",
    "name/place",
    "name and place",
    "name checks",
    "naming checks",
    "originality",
    "research/originality",
)
_WORKSPACE_CONTRACT_FILES = (
    "PROJECT.md",
    "notes/writing_workflow.md",
    "revision/revision_plan.md",
    "review/review_log.md",
    "research/name_place_research.md",
)


@dataclass(frozen=True)
class ProjectReturnCapture:
    """Result of capturing a project return point from a chat turn."""

    project_key: str
    project_title: str
    project_type: str
    project_intent: str
    next_step: str
    commitment_id: str
    schedule_id: Optional[str]
    workspace_rel_path: Optional[str] = None
    requested_workspace_rel_path: Optional[str] = None
    start_immediately: bool = False
    task_id: Optional[str] = None


@dataclass(frozen=True)
class ProjectFollowthroughResponse:
    """Pre-finalize project work capture plus optional visible progress update."""

    content: str
    capture: ProjectReturnCapture
    meta: dict[str, Any]


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
    current_turn_text = "\n".join(part for part in (user_input, assistant_content) if part)
    manifest_text = _manifest_context_text(manifest)
    context_text = _context_text(
        user_input=user_input,
        assistant_content=assistant_content,
        manifest_text=manifest_text,
    )
    if _is_audit_only_turn(current_turn_text):
        _trace(
            runtime,
            "project_return_suppressed",
            {
                "reason": "audit_only_turn",
                "session_id": session_id,
            },
        )
        return None
    if _is_rest_defer_turn(user_input, assistant_content):
        _trace(
            runtime,
            "project_return_suppressed",
            {
                "reason": "rest_defer_turn",
                "session_id": session_id,
            },
        )
        return None
    direct_immediate_work_request = _is_immediate_operator_work_request(user_input)
    contextual_immediate_work_request = False
    start_immediately = direct_immediate_work_request
    if not start_immediately:
        contextual_immediate_work_request = _is_contextual_immediate_operator_work_request(user_input, manifest_text)
        start_immediately = contextual_immediate_work_request
    if not start_immediately:
        start_immediately = _is_project_correction_work_request(user_input)
    if not _should_capture_project_return(
        current_turn_text,
        manifest_text,
        assistant_content,
        immediate_operator_work_request=start_immediately,
    ):
        return None

    project_type = _infer_project_type(current_turn_text, context_text)
    archived_project_keys = await _archived_project_return_keys(runtime)
    allow_archived_reopen = _explicitly_reopens_project(current_turn_text) or _explicitly_challenges_completion(
        current_turn_text
    )
    requested_workspace = (
        resolve_requested_workspace_path(runtime, current_turn_text)
        or (
            resolve_requested_workspace_path(runtime, manifest_text)
            if start_immediately
            else None
        )
    )
    title_source_text = (
        manifest_text
        if contextual_immediate_work_request and requested_workspace is None
        else user_input
        if start_immediately and requested_workspace is not None
        else current_turn_text
    )
    title = ""
    if start_immediately and requested_workspace is not None:
        title = _infer_explicit_project_title(manifest_text) or _infer_explicit_project_title(user_input)
    if not title:
        title = _infer_project_title(title_source_text, allow_default=False)
    raw_title_fragment = title or _infer_project_title_fragment(title_source_text)
    workspace_project = resolve_workspace_project_from_text(runtime, current_turn_text)
    if workspace_project is None and start_immediately and requested_workspace is None:
        workspace_project = resolve_workspace_project_from_text(runtime, manifest_text)
    if (
        workspace_project is None
        and requested_workspace is None
        and (not title or _is_generic_project_title(title) or _is_weak_project_title(title))
    ):
        workspace_project = resolve_recent_workspace_project(
            runtime,
            project_type=project_type,
            excluded_project_keys=() if allow_archived_reopen else archived_project_keys,
        )
    if (
        workspace_project is None
        and requested_workspace is not None
        and (not title or _is_generic_project_title(title) or _is_weak_project_title(title))
    ):
        title = _title_from_requested_workspace(requested_workspace, project_type=project_type, now=now)
    if not title and workspace_project is None:
        title = _infer_project_title(context_text, allow_default=False)
    explicit_context_title = _infer_explicit_project_title(context_text)
    if (
        workspace_project is not None
        and _is_project_correction_work_request(user_input)
        and explicit_context_title
    ):
        title = explicit_context_title
    elif (
        workspace_project is not None
        and _is_project_correction_work_request(user_input)
        and _turn_rejects_project_title(current_turn_text, workspace_project.project_title)
    ):
        title = _title_from_workspace_directory(workspace_project)
    elif workspace_project is not None:
        title = workspace_project.project_title
    quarantine_title = raw_title_fragment or title
    if not title or _is_generic_project_title(title) or _is_weak_project_title(title):
        await _quarantine_project_return_candidate(
            runtime,
            session_id=session_id,
            title=quarantine_title,
            user_input=user_input,
            assistant_content=assistant_content,
            reason="weak_project_identity",
            now=now,
        )
        _trace(
            runtime,
            "project_return_quarantined",
            {
                "reason": "weak_project_identity",
                "title": title,
                "session_id": session_id,
            },
        )
        return None
    if (
        workspace_project is None
        and requested_workspace is None
        and not _title_has_return_grounding(title, current_turn_text, manifest_text)
    ):
        await _quarantine_project_return_candidate(
            runtime,
            session_id=session_id,
            title=title,
            user_input=user_input,
            assistant_content=assistant_content,
            reason="ungrounded_project_identity",
            now=now,
        )
        _trace(
            runtime,
            "project_return_quarantined",
            {
                "reason": "ungrounded_project_identity",
                "title": title,
                "session_id": session_id,
            },
        )
        return None
    project_key = _project_key(title)
    if project_key in archived_project_keys and not allow_archived_reopen:
        await _quarantine_project_return_candidate(
            runtime,
            session_id=session_id,
            title=title,
            user_input=user_input,
            assistant_content=assistant_content,
            reason="archived_project_identity",
            now=now,
        )
        _trace(
            runtime,
            "project_return_quarantined",
            {
                "reason": "archived_project_identity",
                "title": title,
                "session_id": session_id,
            },
        )
        return None
    if workspace_project is None and requested_workspace is not None and start_immediately:
        workspace_project = _materialize_requested_workspace_project(
            runtime,
            requested_workspace=requested_workspace,
            project_key=project_key,
            title=title,
            project_type=project_type,
            session_id=session_id,
            user_input=user_input,
            now=now,
        )
    next_step = _infer_next_step(assistant_content, user_input, context_text)
    project_intent = _infer_project_intent(current_turn_text, context_text, title, next_step, project_type)
    if workspace_project is None:
        workspace_project = resolve_workspace_project(runtime, project_key=project_key, project_title=title)
    if workspace_project is not None:
        if (
            _is_project_correction_work_request(user_input)
            and explicit_context_title
        ):
            title = explicit_context_title
            project_key = _project_key(title)
        elif (
            _is_project_correction_work_request(user_input)
            and _turn_rejects_project_title(current_turn_text, workspace_project.project_title)
        ):
            title = _title_from_workspace_directory(workspace_project)
            project_key = _project_key(title)
        else:
            title = workspace_project.project_title
            project_key = workspace_project.project_key
    completion_contract = _creative_completion_contract(project_type, context_text, workspace_project=workspace_project)
    project_start_contract = _project_start_contract(
        title=title,
        requested_workspace=requested_workspace,
        workspace_project=workspace_project,
        context_text=context_text,
    )
    commitment = await _upsert_project_return_commitment(
        runtime,
        project_key=project_key,
        title=title,
        project_type=project_type,
        project_intent=project_intent,
        next_step=next_step,
        workspace_project=workspace_project,
        requested_workspace=requested_workspace,
        completion_contract=completion_contract,
        project_start_contract=project_start_contract,
        session_id=session_id,
        user_input=user_input,
        assistant_content=assistant_content,
        allow_reopen_archived=allow_archived_reopen,
        start_immediately=start_immediately,
        now=now,
    )
    if commitment is None:
        return None

    schedule_id, task_id = await _upsert_project_return_schedule(
        runtime,
        project_key=project_key,
        title=title,
        project_type=project_type,
        project_intent=project_intent,
        next_step=next_step,
        workspace_project=workspace_project,
        requested_workspace=requested_workspace,
        completion_contract=completion_contract,
        project_start_contract=project_start_contract,
        session_id=session_id,
        commitment=commitment,
        start_immediately=start_immediately,
        now=now,
    )
    _trace(
        runtime,
        "project_return_captured",
        {
            "project_key": project_key,
            "project_title": title,
            "project_type": project_type,
            "workspace_rel_path": str(workspace_project.workspace_rel_path) if workspace_project else "",
            "requested_workspace_rel_path": (
                str(requested_workspace.workspace_rel_path) if requested_workspace else ""
            ),
            "commitment_id": str(commitment.commitment_id),
            "schedule_id": schedule_id,
            "task_id": task_id,
            "start_immediately": start_immediately,
        },
    )
    return ProjectReturnCapture(
        project_key=project_key,
        project_title=title,
        project_type=project_type,
        project_intent=project_intent,
        next_step=next_step,
        commitment_id=str(commitment.commitment_id),
        schedule_id=schedule_id,
        workspace_rel_path=str(workspace_project.workspace_rel_path) if workspace_project else None,
        requested_workspace_rel_path=(
            str(requested_workspace.workspace_rel_path) if requested_workspace else None
        ),
        start_immediately=start_immediately,
        task_id=task_id,
    )


async def prepare_project_followthrough_response(
    runtime: Any,
    *,
    session_id: str,
    user_input: str,
    assistant_content: str,
    manifest: Any = None,
    now: Optional[datetime] = None,
) -> Optional[ProjectFollowthroughResponse]:
    """Capture project follow-through before the visible assistant turn is stored.

    This prevents a direct work request from ending as a conversational stall. The
    durable commitment/schedule is the real work-start signal; the optional
    replacement response is only a grounded progress update for the operator.
    """

    capture = await capture_project_return_from_turn(
        runtime,
        session_id=session_id,
        user_input=user_input,
        assistant_content=assistant_content,
        manifest=manifest,
        now=now,
    )
    if capture is None:
        return None
    replace_response = _should_use_project_working_response(
        user_input=user_input,
        assistant_content=assistant_content,
        capture=capture,
    )
    content = (
        _project_working_response(capture)
        if replace_response
        else assistant_content
    )
    return ProjectFollowthroughResponse(
        content=content,
        capture=capture,
        meta={
            "captured": True,
            "response_replaced": replace_response,
            "commitment_id": capture.commitment_id,
            "schedule_id": capture.schedule_id,
            "task_id": capture.task_id,
            "project_key": capture.project_key,
            "project_title": capture.project_title,
            "project_type": capture.project_type,
            "workspace_rel_path": capture.workspace_rel_path,
            "requested_workspace_rel_path": capture.requested_workspace_rel_path,
            "start_immediately": capture.start_immediately,
        },
    )


def _should_capture_project_return(
    current_turn_text: str,
    manifest_text: str,
    assistant_content: str,
    *,
    immediate_operator_work_request: bool = False,
) -> bool:
    current = current_turn_text.lower()
    manifest = manifest_text.lower()
    assistant = assistant_content.lower()
    has_project_context = (
        _has_project_return_markers(current)
        or _has_project_return_markers(manifest)
        or _has_contextual_project_work_context(manifest)
    )
    if immediate_operator_work_request and _is_project_correction_work_request(current_turn_text):
        return True
    if not has_project_context:
        return False
    return immediate_operator_work_request or any_marker_in_text(assistant, _UNFINISHED_MARKERS)


def _is_audit_only_turn(text: str) -> bool:
    return is_audit_only_text(text)


def _is_rest_defer_turn(user_input: str, assistant_content: str) -> bool:
    current = f"{user_input or ''}\n{assistant_content or ''}".lower()
    assistant = (assistant_content or "").lower()
    return (
        any_marker_in_text(current, _REST_DEFER_MARKERS)
        and any_marker_in_text(assistant, _NO_AUTONOMOUS_RETURN_MARKERS)
    )


def _has_project_return_markers(text: str) -> bool:
    return any_marker_in_text(text, _PROJECT_MARKERS) and any_marker_in_text(text, _RETURN_MARKERS)


def _has_contextual_project_work_context(text: str) -> bool:
    lowered = str(text or "").lower()
    return any_marker_in_text(lowered, _PROJECT_MARKERS) and any_marker_in_text(
        lowered,
        _CONTEXTUAL_PROJECT_WORK_MARKERS,
    )


def _is_immediate_operator_work_request(user_input: str) -> bool:
    lowered = str(user_input or "").lower()
    if not lowered:
        return False
    if not any_marker_in_text(lowered, _PROJECT_MARKERS):
        return False
    return any_marker_in_text(lowered, _IMMEDIATE_OPERATOR_WORK_MARKERS)


def _is_contextual_immediate_operator_work_request(user_input: str, manifest_text: str) -> bool:
    lowered = str(user_input or "").lower()
    if not any_marker_in_text(lowered, _CONTEXTUAL_IMMEDIATE_WORK_MARKERS):
        return False
    return _has_contextual_project_work_context(manifest_text)


def _is_project_correction_work_request(user_input: str) -> bool:
    lowered = str(user_input or "").lower()
    if not lowered:
        return False
    has_project_surface = any_marker_in_text(lowered, _PROJECT_MARKERS) or "/workspace/" in lowered or "workspace/" in lowered
    if not has_project_surface:
        return False
    return any_marker_in_text(lowered, _PROJECT_CORRECTION_MARKERS)


def _should_use_project_working_response(
    *,
    user_input: str,
    assistant_content: str,
    capture: ProjectReturnCapture,
) -> bool:
    if not capture.start_immediately:
        return False
    if _is_immediate_operator_work_request(user_input) or _is_project_correction_work_request(user_input):
        return True
    if any_marker_in_text(str(user_input or "").lower(), _CONTEXTUAL_IMMEDIATE_WORK_MARKERS):
        return True
    lowered = str(assistant_content or "").lower()
    return any_marker_in_text(lowered, _PROJECT_WORKING_RESPONSE_BLOCKERS)


def _project_working_response(capture: ProjectReturnCapture) -> str:
    evidence = [f"commitment {capture.commitment_id}"]
    if capture.schedule_id:
        evidence.append(f"schedule {capture.schedule_id}")
    if capture.workspace_rel_path:
        evidence.append(f"workspace {capture.workspace_rel_path}")
    if capture.requested_workspace_rel_path:
        evidence.append(f"requested workspace {capture.requested_workspace_rel_path}")
    if capture.task_id:
        evidence.append(f"task {capture.task_id}")
    lines = [
        "Working now. I recorded the return point and queued the background work instead of waiting for another prompt.",
        f"Evidence: {', '.join(evidence)}.",
    ]
    if capture.next_step:
        lines.append(f"Next step: {capture.next_step}")
    lines.append(
        "I will keep executing against the project completion criteria; if a real blocker appears, I will record the blocker and continue through the recovery path instead of silently waiting."
    )
    return "\n".join(lines)


def _title_has_return_grounding(title: str, current_turn_text: str, manifest_text: str) -> bool:
    title_text = str(title or "").lower().strip()
    if _is_weak_project_title(title_text):
        return False
    pattern = re.compile(rf"(?<![a-z0-9]){re.escape(title_text)}(?![a-z0-9])", re.IGNORECASE)
    scopes = (current_turn_text, manifest_text)
    return any(
        pattern.search(scope or "")
        and (
            _has_project_return_markers((scope or "").lower())
            or _has_contextual_project_work_context(scope or "")
        )
        for scope in scopes
    )


def _turn_rejects_project_title(text: str, title: str) -> bool:
    title_key = _project_key(title)
    if not title_key:
        return False
    lowered = str(text or "").lower()
    if "not " not in lowered and "wrong" not in lowered and "different" not in lowered:
        return False
    title_words = [word for word in re.findall(r"[a-z0-9]+", title.lower()) if len(word) >= 5]
    return any(word in lowered for word in title_words)


def _title_from_workspace_directory(workspace_project: WorkspaceProjectCandidate) -> str:
    path_name = workspace_project.path.name
    title = " ".join(part for part in re.split(r"[-_]+", path_name) if part).strip().title()
    return title or workspace_project.project_title


def _is_weak_project_title(title: str) -> bool:
    lowered = " ".join(str(title or "").lower().split()).strip()
    if not lowered:
        return True
    if len(lowered) < 4:
        return True
    words = re.findall(r"[a-z0-9]+", lowered)
    if len(words) == 1 and len(words[0]) < 5 and not re.search(r"\d", words[0]):
        return True
    if lowered in {
        "at",
        "build",
        "check",
        "complete",
        "edit",
        "evidence",
        "here",
        "produce",
        "read",
        "research",
        "result",
        "revise",
        "task",
        "use",
        "used",
        "work",
    }:
        return True
    if re.match(r"^(?:standard|standards)\b", lowered):
        return True
    if re.match(r"^(?:that\s+means|this\s+means|you\s+need)\b", lowered):
        return True
    if re.match(r"^(?:and|are|be|being|been|end\s+to\s+end|is|of|was|were)\b", lowered):
        return True
    if re.match(r"^(?:evidence|proof|context|artifact|artifacts?)\s+(?:here|to|that|for|in)\b", lowered):
        return True
    if re.search(r"\bto claim\b", lowered):
        return True
    if re.match(r"^(?:advances?|progress(?:es)?|moves?)\s+(?:toward|towards)\s+completion\b", lowered):
        return True
    if re.match(r"^(?:closure\s+and\s+later\s+clean\s+revision\s+evidence)\b", lowered):
        return True
    if re.match(r"^(?:state|current\s+state)\s+by\s+producing\b", lowered):
        return True
    if re.match(r"^(?:and\s+)?(?:keep\s+working|keep\s+revising|continue|resume|finish|return|work)\b", lowered):
        return True
    if any_marker_in_text(lowered, ("without asking", "without approval", "until complete", "until the book is complete")):
        return True
    if any_marker_in_text(lowered, _CONTEXTUAL_IMMEDIATE_WORK_MARKERS):
        return True
    return False


def _context_text(*, user_input: str, assistant_content: str, manifest_text: str) -> str:
    parts = [user_input or "", assistant_content or "", manifest_text or ""]
    return "\n".join(part for part in parts if part)


def _manifest_context_text(manifest: Any) -> str:
    parts: list[str] = []
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


def _infer_project_title(text: str, *, allow_default: bool = True) -> str:
    creative_writing = _PROJECT_ID_RE.search(text)
    if creative_writing:
        return creative_writing.group(0)
    explicit_title = _infer_explicit_project_title(text)
    if explicit_title:
        return explicit_title
    contextual_title = _infer_contextual_project_title(text)
    if contextual_title:
        return contextual_title
    title_candidates = []
    for match in _TITLE_RE.finditer(text or ""):
        title = _clean_title(match.group("title"))
        if _is_generic_project_title(title) or _is_weak_project_title(title):
            continue
        window = (text or "")[max(0, match.start() - 80): match.end() + 120].lower()
        score = 0
        if any_marker_in_text(window, ("created", "workspace", "project file", "draft", "bible", "review")):
            score += 4
        if any_marker_in_text(window, _RETURN_MARKERS):
            score += 2
        title_candidates.append((score, match.start(), title))
    if title_candidates:
        title_candidates.sort(key=lambda item: (-item[0], item[1]))
        return title_candidates[0][2]
    branded_identifier = _infer_branded_identifier(text)
    if branded_identifier:
        return branded_identifier
    return "Conversation Project" if allow_default else ""


def _infer_explicit_project_title(text: str) -> str:
    for match in _EXPLICIT_TITLE_RE.finditer(text or ""):
        title = _clean_title(match.group("title"))
        if title and not _is_generic_project_title(title) and not _is_weak_project_title(title):
            return title
    return ""


def _infer_contextual_project_title(text: str) -> str:
    for match in _CONTEXTUAL_TITLE_RE.finditer(text or ""):
        title = _clean_title(match.group("title"))
        if title and not _is_generic_project_title(title) and not _is_weak_project_title(title):
            return title
    return ""


def _infer_project_title_fragment(text: str) -> str:
    for match in _TITLE_RE.finditer(text or ""):
        fragment = _clean_title(match.group("title"), preserve_weak_fragment=True)
        if fragment:
            return fragment
    return ""


def _title_from_requested_workspace(
    requested_workspace: WorkspacePathRequest,
    *,
    project_type: str,
    now: datetime,
) -> str:
    path_name = requested_workspace.path.name
    path_title = " ".join(part for part in re.split(r"[-_]+", path_name) if part).strip().title()
    if (
        requested_workspace.kind == "target"
        and path_title
        and not _is_generic_project_title(path_title)
        and not _is_weak_project_title(path_title)
    ):
        return path_title
    prefix = "Writing Project" if project_type == PROJECT_TYPE_WRITING else "Project"
    return f"{prefix} {now.strftime('%Y%m%d-%H%M%S')}"


def _materialize_requested_workspace_project(
    runtime: Any,
    *,
    requested_workspace: WorkspacePathRequest,
    project_key: str,
    title: str,
    project_type: str,
    session_id: str,
    user_input: str,
    now: datetime,
) -> Optional[WorkspaceProjectCandidate]:
    root = _requested_project_root(requested_workspace, project_key=project_key)
    if root is None:
        return None
    try:
        root.mkdir(parents=True, exist_ok=True)
        if project_type == PROJECT_TYPE_WRITING:
            for dirname in ("drafts", "notes", "research", "review", "bible"):
                (root / dirname).mkdir(exist_ok=True)
        project_file = root / "PROJECT.md"
        if not project_file.exists():
            project_file.write_text(
                "\n".join(
                    [
                        f"# {title}",
                        "",
                        "## Origin",
                        "",
                        f"- Captured at: {now.isoformat()}",
                        f"- Source session: {session_id}",
                        f"- Requested workspace: {requested_workspace.path}",
                        f"- Requested workspace kind: {requested_workspace.kind}",
                        f"- Initial operator request: {_excerpt(user_input)}",
                        "",
                        "## Status",
                        "",
                        "Project root reserved by OpenCAS before autonomous execution so follow-through has a concrete target.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
    except OSError as exc:
        _trace(
            runtime,
            "project_return_workspace_materialization_failed",
            {
                "project_key": project_key,
                "requested_workspace": str(requested_workspace.path),
                "error": str(exc),
            },
        )
        return None
    resolved = resolve_workspace_project(runtime, project_key=project_key, project_title=title)
    if resolved is not None:
        return resolved
    evidence = tuple(_materialized_project_evidence(root))
    workspace_root = _agent_workspace_root(runtime)
    rel_path = _workspace_relative_for_materialized(runtime, workspace_root, root)
    return WorkspaceProjectCandidate(
        project_key=project_key,
        project_title=title,
        path=root.resolve(),
        workspace_rel_path=rel_path,
        confidence=1.0,
        evidence=evidence,
    )


def _requested_project_root(requested_workspace: WorkspacePathRequest, *, project_key: str) -> Optional[Path]:
    try:
        if requested_workspace.kind == "parent":
            root = requested_workspace.path / project_key
        else:
            root = requested_workspace.path
        resolved_root = root.expanduser().resolve(strict=False)
        requested_workspace.path.expanduser().resolve(strict=False)
        if requested_workspace.kind == "parent":
            resolved_root.relative_to(requested_workspace.path.expanduser().resolve(strict=False))
        return resolved_root
    except (OSError, ValueError):
        return None


def _materialized_project_evidence(root: Path) -> list[str]:
    evidence: list[str] = []
    for marker in ("PROJECT.md", "drafts", "notes", "research", "review", "bible"):
        if (root / marker).exists():
            evidence.append(marker)
    return evidence


def _agent_workspace_root(runtime: Any) -> Optional[Path]:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    getter = getattr(config, "agent_workspace_root", None)
    if not callable(getter):
        return None
    try:
        return Path(getter()).expanduser().resolve()
    except Exception:
        return None


def _primary_workspace_root(runtime: Any) -> Optional[Path]:
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    getter = getattr(config, "primary_workspace_root", None)
    if not callable(getter):
        return None
    try:
        return Path(getter()).expanduser().resolve()
    except Exception:
        return None


def _workspace_relative_for_materialized(
    runtime: Any,
    workspace_root: Optional[Path],
    project_root: Path,
) -> Path:
    primary_root = _primary_workspace_root(runtime)
    if primary_root is not None:
        try:
            return project_root.resolve().relative_to(primary_root)
        except ValueError:
            pass
    if workspace_root is not None:
        try:
            return Path(workspace_root.name) / project_root.resolve().relative_to(workspace_root)
        except ValueError:
            pass
    return Path(project_root.name)


def _infer_branded_identifier(text: str) -> str:
    """Infer product-style names like kPony without relying on canned project lists."""

    candidates = []
    for match in _BRANDED_IDENTIFIER_RE.finditer(text or ""):
        name = match.group(0)
        if name.lower() in {"openai", "opencas", "qtquick"}:
            continue
        window = (text or "")[max(0, match.start() - 80): match.end() + 80].lower()
        score = 0
        if any_marker_in_text(window, _PROJECT_MARKERS):
            score += 2
        if any_marker_in_text(window, ("build", "cmake", "compile", "qt", "qt6", "software")):
            score += 2
        if "/" in window or "workspace" in window:
            score += 1
        candidates.append((score, match.start(), name))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def _infer_project_type(current_turn_text: str, context_text: str) -> str:
    return classify_project_type(
        current_turn_text=current_turn_text,
        context_text=context_text,
    ).project_type


def _infer_next_step(assistant_content: str, user_input: str, context_text: str) -> str:
    for sentence in _sentences(assistant_content):
        lowered = sentence.lower()
        if any_marker_in_text(lowered, _NEXT_STEP_MARKERS):
            return _normalize_next_step(sentence)
    for sentence in _sentences(user_input):
        lowered = sentence.lower()
        if any_marker_in_text(lowered, _RETURN_MARKERS):
            return _normalize_next_step(sentence)
    for sentence in _sentences(context_text):
        lowered = sentence.lower()
        if any_marker_in_text(lowered, _RETURN_MARKERS):
            return _normalize_next_step(sentence)
    return "Review the project context and decide the next meaningful step."


def _infer_project_intent(
    current_turn_text: str,
    context_text: str,
    title: str,
    next_step: str,
    project_type: str,
) -> str:
    if project_type == PROJECT_TYPE_SOFTWARE:
        return (
            f"continue building {title} until it has concrete working project artifacts, "
            "verification evidence, honest proof, and any remaining defects recorded."
        )
    text = current_turn_text.lower() or context_text.lower()
    if (
        project_type == PROJECT_TYPE_WRITING
        and any_marker_in_text(text, _CREATIVE_WRITING_MARKERS)
        and any_marker_in_text(text, _PROJECT_INTENT_MARKERS)
    ):
        return (
            f"revise and finish {title} until the manuscript has explicit completion evidence, "
            "using user critique as input while continuing writing and revision autonomously."
        )
    return (
        f"Return to {title} as a continuing project, keeping the larger project objective ahead "
        f"of the immediate subtask: {next_step}"
    )


async def _quarantine_project_return_candidate(
    runtime: Any,
    *,
    session_id: str,
    title: str,
    user_input: str,
    assistant_content: str,
    reason: str,
    now: datetime,
) -> None:
    service = getattr(runtime, "thread_registry_service", None)
    ingest = getattr(service, "ingest_autonomous_artifact", None)
    if not callable(ingest):
        return
    source_ref = f"project_return_quarantine:{session_id or 'unknown'}"
    content = "\n".join(
        part
        for part in (
            f"Captured at: {now.isoformat()}",
            f"Reason: {reason}",
            f"Inferred title: {title or '(none)'}",
            f"User turn: {_excerpt(user_input)}",
            f"Assistant turn: {_excerpt(assistant_content)}",
            "Disposition: not promoted to a project-return commitment or schedule.",
        )
        if part
    )
    try:
        bead = await ingest(
            source_ref=source_ref,
            content=content,
            thread_title="Project return quarantine",
            title="Quarantined weak project-return signal",
            summary=(
                f"A project-return capture was quarantined for {reason} "
                f"({_human_reason(reason)}); inferred title: {title or '(none)'}. "
                "It remains inspectable without becoming an autonomous commitment."
            ),
            thread_kind="project_return_quarantine",
        )
        _trace(
            runtime,
            "project_return_quarantine_recorded",
            {
                "reason": reason,
                "session_id": session_id,
                "title": title,
                "bead_id": getattr(bead, "bead_id", ""),
            },
        )
    except Exception as exc:
        _trace(
            runtime,
            "project_return_quarantine_record_failed",
            {
                "reason": reason,
                "session_id": session_id,
                "title": title,
                "error": str(exc),
            },
        )


async def _upsert_project_return_commitment(
    runtime: Any,
    *,
    project_key: str,
    title: str,
    project_type: str,
    project_intent: str,
    next_step: str,
    workspace_project: Optional[WorkspaceProjectCandidate],
    requested_workspace: Optional[WorkspacePathRequest],
    completion_contract: dict[str, Any],
    project_start_contract: dict[str, Any],
    session_id: str,
    user_input: str,
    assistant_content: str,
    allow_reopen_archived: bool,
    start_immediately: bool,
    now: datetime,
) -> Optional[Commitment]:
    store = getattr(runtime, "commitment_store", None)
    if store is None:
        return None
    existing = await _find_existing_commitment(store, project_key, include_terminal=allow_reopen_archived)
    if existing is None:
        existing = Commitment(
            content=f"Return to project: {title}",
            priority=7.5,
            tags=["project_return", "conversation", "self_directed"],
        )
    elif getattr(existing, "status", None) != CommitmentStatus.ACTIVE:
        existing.status = CommitmentStatus.ACTIVE
        existing.meta["reopened_at"] = now.isoformat()
        existing.meta["reopened_reason"] = "operator indicated prior completion was incomplete"
        for key in ("completion_evidence", "completion_evidence_recorded_at", "completed_at"):
            existing.meta.pop(key, None)
    existing.updated_at = now
    existing.meta.update(
        {
            "source": "project_return_capture",
            "source_session_id": session_id,
            "project_key": project_key,
            "project_title": title,
            "project_type": project_type,
            "project_intent": project_intent,
            "next_step": next_step,
            **_workspace_project_meta(workspace_project),
            **_requested_workspace_meta(requested_workspace),
            **_completion_contract_meta(completion_contract),
            **_project_start_contract_meta(project_start_contract),
            "source_user_turn": _excerpt(user_input),
            "source_assistant_turn": _excerpt(assistant_content),
            "return_policy": (
                "immediate_autonomous_work"
                if start_immediately
                else "scheduled_self_review"
            ),
            "start_policy": (
                "immediate_operator_work_request"
                if start_immediately
                else "scheduled_self_review"
            ),
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
    project_type: str,
    project_intent: str,
    next_step: str,
    workspace_project: Optional[WorkspaceProjectCandidate],
    requested_workspace: Optional[WorkspacePathRequest],
    completion_contract: dict[str, Any],
    project_start_contract: dict[str, Any],
    session_id: str,
    commitment: Commitment,
    start_immediately: bool,
    now: datetime,
) -> tuple[Optional[str], Optional[str]]:
    service = getattr(runtime, "schedule_service", None)
    if service is None:
        return None, None
    store = getattr(getattr(runtime, "ctx", None), "schedule_store", None) or getattr(service, "store", None)
    existing = await _find_existing_schedule(store, project_key, str(commitment.commitment_id))
    objective = _build_schedule_objective(
        title,
        project_intent,
        next_step,
        session_id,
        str(commitment.commitment_id),
        project_type=project_type,
        workspace_project=workspace_project,
        requested_workspace=requested_workspace,
        completion_contract=completion_contract,
        project_start_contract=project_start_contract,
    )
    start_at = now if start_immediately else now + timedelta(minutes=5)
    start_policy = (
        "immediate_operator_work_request"
        if start_immediately
        else "scheduled_self_review"
    )
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
                "project_type": project_type,
                "project_intent": project_intent,
                "next_step": next_step,
                **_workspace_project_meta(workspace_project),
                **_requested_workspace_meta(requested_workspace),
                **_completion_contract_meta(completion_contract),
                **_project_start_contract_meta(project_start_contract),
                "source_session_id": session_id,
                "start_policy": start_policy,
            },
        )
        task_id = await _trigger_immediate_project_schedule(
            runtime,
            service=service,
            item=item,
            start_immediately=start_immediately,
            now=now,
        )
        return str(item.schedule_id), task_id
    existing.objective = objective
    existing.description = f"Autonomous project return point for {title}."
    existing.priority = max(existing.priority, float(commitment.priority))
    existing.meta.update(
        {
            "source": "project_return_capture",
            "project_key": project_key,
            "project_title": title,
            "project_type": project_type,
            "project_intent": project_intent,
            "next_step": next_step,
            **_workspace_project_meta(workspace_project),
            **_requested_workspace_meta(requested_workspace),
            **_completion_contract_meta(completion_contract),
            **_project_start_contract_meta(project_start_contract),
            "source_session_id": session_id,
            "refreshed_at": now.isoformat(),
            "start_policy": start_policy,
        }
    )
    if existing.next_run_at and existing.next_run_at > start_at:
        existing.next_run_at = start_at
        existing.start_at = min(existing.start_at, start_at)
    existing.recurrence = ScheduleRecurrence.NONE
    existing.interval_hours = None
    existing.max_occurrences = None
    await store.save(existing)
    task_id = await _trigger_immediate_project_schedule(
        runtime,
        service=service,
        item=existing,
        start_immediately=start_immediately,
        now=now,
    )
    return str(existing.schedule_id), task_id


async def _archived_project_return_keys(runtime: Any) -> set[str]:
    store = getattr(runtime, "commitment_store", None)
    if store is None:
        return set()
    keys: set[str] = set()
    for status in (CommitmentStatus.COMPLETED, CommitmentStatus.ABANDONED):
        try:
            commitments = await store.list_by_status(status, limit=500)
        except Exception:
            continue
        for commitment in commitments:
            if "project_return" not in set(getattr(commitment, "tags", []) or []):
                continue
            meta = getattr(commitment, "meta", {}) or {}
            for value in (meta.get("project_key"), meta.get("project_title")):
                key = _project_key(str(value or ""))
                if key:
                    keys.add(key)
    return keys


def _explicitly_reopens_project(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "reopen",
            "restart",
            "revive",
            "start it again",
            "new work on",
            "work on it again even though",
        )
    )


def _explicitly_challenges_completion(text: str) -> bool:
    lowered = str(text or "").lower()
    return any_marker_in_text(lowered, _COMPLETION_CHALLENGE_MARKERS)


async def _find_existing_commitment(
    store: Any,
    project_key: str,
    *,
    include_terminal: bool = False,
) -> Optional[Commitment]:
    for commitment in await store.list_active(limit=200):
        if "project_return" not in set(commitment.tags):
            continue
        if str((commitment.meta or {}).get("project_key", "")) == project_key:
            return commitment
    if not include_terminal:
        return None
    for status in (CommitmentStatus.COMPLETED, CommitmentStatus.ABANDONED):
        try:
            commitments = await store.list_by_status(status, limit=500)
        except Exception:
            continue
        for commitment in commitments:
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


async def _trigger_immediate_project_schedule(
    runtime: Any,
    *,
    service: Any,
    item: Any,
    start_immediately: bool,
    now: datetime,
) -> Optional[str]:
    if not start_immediately:
        return None
    if runtime is None or not getattr(runtime, "baa", None):
        return None
    trigger = getattr(service, "trigger", None)
    if not callable(trigger):
        return None
    try:
        run = await trigger(item, now=now, manual=False)
    except Exception as exc:
        _trace(
            runtime,
            "project_return_immediate_trigger_failed",
            {
                "schedule_id": str(getattr(item, "schedule_id", "")),
                "error": str(exc),
            },
        )
        return None
    task_id = str(getattr(run, "task_id", "") or "").strip()
    _trace(
        runtime,
        "project_return_immediate_triggered",
        {
            "schedule_id": str(getattr(item, "schedule_id", "")),
            "run_id": str(getattr(run, "run_id", "")),
            "status": str(getattr(getattr(run, "status", None), "value", getattr(run, "status", ""))),
            "task_id": task_id,
        },
    )
    return task_id or None


def _build_schedule_objective(
    title: str,
    project_intent: str,
    next_step: str,
    session_id: str,
    commitment_id: str,
    *,
    project_type: str = "general",
    workspace_project: Optional[WorkspaceProjectCandidate] = None,
    requested_workspace: Optional[WorkspacePathRequest] = None,
    completion_contract: Optional[dict[str, Any]] = None,
    project_start_contract: Optional[dict[str, Any]] = None,
) -> str:
    workspace_context = _workspace_project_objective_context(workspace_project)
    if not workspace_context:
        workspace_context = _requested_workspace_objective_context(requested_workspace)
    continuation_contract = (
        "Do not turn missing context or missing evidence into a final user-facing stop. "
        "Missing evidence means inspect the workspace, use available evidence/research tools, "
        "then continue the work. Stop only for explicit completion evidence, an external/safety/"
        "credential blocker recorded with provenance, or an automatic continuation schedule after "
        "concrete progress. "
    )
    if project_type == PROJECT_TYPE_SOFTWARE:
        start_contract_context = _project_start_objective_context(project_start_contract)
        return (
            f'Return to software project "{title}" from session {session_id}. '
            f"Software project intent: {project_intent} "
            f"Immediate next step: {next_step} "
            f"{workspace_context}"
            f"{start_contract_context}"
            f"{continuation_contract}"
            "Review the latest project context and continue implementation when meaningful progress "
            "is possible without asking the user for approval. Prioritize source correctness, build, "
            "test, run instructions, and proof. If the project remains unfinished at the end of this run, "
            "record the exact blocker and use your OpenCAS calendar to create the next return time. "
            "If the project is not worth continuing, use workflow_cancel_project to compost it with "
            "salvage and discard notes. Only mark the commitment complete after the requested proof "
            f"criteria are satisfied, and include commitment {commitment_id} in the completion evidence."
        )
    if project_type != PROJECT_TYPE_WRITING:
        start_contract_context = _project_start_objective_context(project_start_contract)
        return (
            f'Return to project "{title}" from session {session_id}. '
            f"Project intent: {project_intent} "
            f"Immediate next step: {next_step} "
            f"{workspace_context}"
            f"{start_contract_context}"
            f"{continuation_contract}"
            "Review the latest project context and continue only scoped meaningful work. If unfinished, "
            "record the blocker and schedule the next return. If the project should stop, use "
            "workflow_cancel_project to compost it. Only mark the commitment complete after the requested "
            f"proof criteria are satisfied, and include commitment {commitment_id} in the completion evidence."
        )
    contract_context = _creative_completion_objective_context(completion_contract)
    start_contract_context = _project_start_objective_context(project_start_contract)
    return (
        f'Return to project "{title}" from session {session_id}. '
        f"Book-level intent: {project_intent} "
        f"Immediate next step: {next_step} "
        f"{workspace_context}"
        f"{contract_context}"
        f"{start_contract_context}"
        f"{continuation_contract}"
        "Review the latest project context and decide whether to continue, finish, "
        "or schedule another return. Treat research and naming work as support for the manuscript, "
        "not as a substitute for manuscript or character-bible revision. Continue with writing, "
        "critique, revision, or necessary research when meaningful progress is possible without "
        "asking the user for approval. If the project remains unfinished at the end of this run, "
        "use your OpenCAS calendar to choose and create the next return time that fits the work; "
        "do not default to tomorrow when sooner is right. "
        f"If the project is finished, mark commitment {commitment_id} complete."
    )


def _creative_completion_contract(
    project_type: str,
    context_text: str,
    *,
    workspace_project: Optional[WorkspaceProjectCandidate] = None,
) -> dict[str, Any]:
    if project_type != PROJECT_TYPE_WRITING:
        return {}
    contract_context = _context_with_workspace_contract(context_text, workspace_project)
    target_word_count = _target_word_count(contract_context)
    if target_word_count is None:
        return {}
    requires_clean_revision = _requires_clean_revision_context(contract_context)
    requires_name_originality_research = _requires_name_originality_context(contract_context)
    required_artifacts = ["PROJECT.md", "drafts/", "review/"]
    criteria = [
        f"Manuscript draft artifacts reach at least {target_word_count:,} words.",
        "Project setup, research notes, and partial chapters do not satisfy completion by themselves.",
        "Completion evidence names the draft artifact path, current word count, and review/revision evidence.",
    ]
    if requires_clean_revision:
        required_artifacts.append("revision/")
        criteria.extend(
            [
                "A revised clean manuscript exists beyond the first/full draft.",
                "First/full draft completion alone does not satisfy the book completion contract.",
            ]
        )
    if requires_name_originality_research:
        required_artifacts.append("research/name_place_research.md")
        criteria.append("Name, place, and originality research has been applied before manuscript lock.")
    return {
        "target_word_count": target_word_count,
        "required_artifacts": required_artifacts,
        "completion_criteria": criteria,
        "requires_clean_revision": requires_clean_revision,
        "requires_name_originality_research": requires_name_originality_research,
    }


def _completion_contract_meta(completion_contract: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not completion_contract:
        return {}
    meta: dict[str, Any] = {"creative_completion_contract": dict(completion_contract)}
    target_word_count = completion_contract.get("target_word_count")
    if target_word_count is not None:
        meta["target_word_count"] = target_word_count
    return meta


def _project_start_contract(
    *,
    title: str,
    requested_workspace: Optional[WorkspacePathRequest],
    workspace_project: Optional[WorkspaceProjectCandidate],
    context_text: str,
) -> dict[str, Any]:
    if requested_workspace is None or requested_workspace.kind != "parent" or workspace_project is None:
        return {}
    target_root = workspace_project.path.resolve()
    parent = requested_workspace.path.resolve()
    forbidden_paths: list[str] = []
    forbidden_terms: list[str] = []
    try:
        for child in parent.iterdir():
            if not child.is_dir() or child.resolve() == target_root:
                continue
            if child.name.startswith(".") or child.name in {"_compost", "__pycache__", "node_modules"}:
                continue
            forbidden_paths.append(str(child.resolve()))
            project_file = child / "PROJECT.md"
            try:
                for line in project_file.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        forbidden_terms.append(stripped.lstrip("#").strip())
                        break
            except OSError:
                pass
            forbidden_terms.append(child.name)
    except OSError:
        pass
    for term in _source_reference_terms(context_text, title):
        if term not in forbidden_terms:
            forbidden_terms.append(term)
    return {
        "new_project": True,
        "source_copy_policy": "no_sibling_materialization",
        "allowed_source_use": "reference_research_premise_only",
        "target_title": title,
        "target_workspace_abs_path": str(target_root),
        "target_workspace_rel_path": str(workspace_project.workspace_rel_path),
        "requested_parent_abs_path": str(parent),
        "requested_parent_rel_path": str(requested_workspace.workspace_rel_path),
        "forbidden_source_paths": forbidden_paths[:50],
        "forbidden_source_terms": forbidden_terms[:100],
    }


def _project_start_contract_meta(project_start_contract: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not project_start_contract:
        return {}
    return {"project_start_contract": dict(project_start_contract)}


def _project_start_objective_context(project_start_contract: Optional[dict[str, Any]]) -> str:
    if not project_start_contract or project_start_contract.get("new_project") is not True:
        return ""
    return (
        "New-project contract: this is a distinct project root. Do not satisfy it by copying, "
        "moving, or materializing sibling project artifacts into the new root. If another project "
        "is used as inspiration, use it only as reference/premise context and produce new artifacts "
        "for this project. Completion evidence must not describe sibling-project materialization. "
    )


def _source_reference_terms(context_text: str, title: str) -> list[str]:
    text = str(context_text or "")
    terms: list[str] = []
    quoted = re.findall(r"['\"“”‘’]([A-Z][A-Za-z0-9][A-Za-z0-9 :'-]{2,80})['\"“”‘’]", text)
    heading_titles = re.findall(r"\b(?:codename|title|project)\s*[:=]\s*['\"*_ ]*([A-Z][A-Za-z0-9][A-Za-z0-9 :'-]{2,80})", text)
    explicit_source = re.findall(r"\b(?:same universe as|based on|premise as|from)\s+([A-Z][A-Za-z0-9][A-Za-z0-9 :'-]{2,80})", text)
    for candidate in quoted + heading_titles + explicit_source:
        cleaned = _clean_title(candidate)
        if cleaned and cleaned != title and cleaned not in terms:
            terms.append(cleaned)
    return terms


def _target_word_count(text: str) -> Optional[int]:
    candidates: list[tuple[int, int]] = []
    source = text or ""
    for match in _WORD_COUNT_RE.finditer(source):
        score = _word_count_target_score(source, match.start(), match.end())
        if score <= 0:
            continue
        raw = match.group("count").replace(",", "")
        value = float(raw)
        if match.group("suffix"):
            value *= 1000
        if value > 0:
            candidates.append((score, int(value)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    return candidates[0][1]


def _word_count_target_score(text: str, start: int, end: int) -> int:
    prefix = text[max(0, start - 80) : start].lower()
    local_prefix = prefix[-32:]
    suffix = text[end : end + 40].lower()
    score = 0
    if any(marker in prefix for marker in _WORD_COUNT_TARGET_PREFIXES):
        score += 2
    if any(marker in local_prefix for marker in _WORD_COUNT_TARGET_PREFIXES):
        score += 2
    if "target length" in prefix or "target length" in local_prefix:
        score += 3
    if any(marker in suffix for marker in _WORD_COUNT_TARGET_SUFFIXES):
        score += 1
    if any(marker in local_prefix for marker in ("current", "actual", "chapter", "drafted", "wrote")):
        score -= 2
    return score


def _creative_completion_objective_context(completion_contract: Optional[dict[str, Any]]) -> str:
    if not completion_contract:
        return ""
    target_word_count = completion_contract.get("target_word_count")
    if not target_word_count:
        return ""
    text = (
        f"Completion contract: this project has a {target_word_count:,}-word manuscript target; "
        "setup files, research notes, and a partial chapter do not satisfy completion. "
        "Before marking complete, include current manuscript word count, draft artifact path, "
        "and review or revision evidence. If the manuscript target is not met, schedule the next return. "
    )
    if completion_contract.get("requires_clean_revision"):
        text += (
            "This project's own standard requires a revised clean manuscript beyond the first/full draft; "
            "do not mark it complete from first-draft or word-count evidence alone. "
        )
    if completion_contract.get("requires_name_originality_research"):
        text += (
            "Name, place, and originality research must be applied to the manuscript/supporting bibles before final completion. "
        )
    return text


def _context_with_workspace_contract(
    context_text: str,
    workspace_project: Optional[WorkspaceProjectCandidate],
) -> str:
    workspace_text = _workspace_contract_text(workspace_project)
    return "\n".join(part for part in (context_text or "", workspace_text) if part)


def _workspace_contract_text(workspace_project: Optional[WorkspaceProjectCandidate]) -> str:
    if workspace_project is None:
        return ""
    parts: list[str] = []
    root = workspace_project.path
    for rel_path in _WORKSPACE_CONTRACT_FILES:
        path = root / rel_path
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        parts.append(content[:8000])
    return "\n".join(parts)


def _requires_clean_revision_context(text: str) -> bool:
    lowered = str(text or "").lower()
    if any_marker_in_text(lowered, _CLEAN_REVISION_MARKERS):
        return True
    return "first draft" in lowered and any_marker_in_text(lowered, ("revise", "revision", "edit"))


def _requires_name_originality_context(text: str) -> bool:
    return any_marker_in_text(str(text or "").lower(), _NAME_ORIGINALITY_MARKERS)


def _workspace_project_meta(workspace_project: Optional[WorkspaceProjectCandidate]) -> dict[str, Any]:
    if workspace_project is None:
        return {}
    return {
        "workspace_rel_path": str(workspace_project.workspace_rel_path),
        "workspace_abs_path": str(workspace_project.path),
        "workspace_project_confidence": workspace_project.confidence,
        "workspace_project_evidence": list(workspace_project.evidence),
    }


def _requested_workspace_meta(requested_workspace: Optional[WorkspacePathRequest]) -> dict[str, Any]:
    if requested_workspace is None:
        return {}
    return {
        "requested_workspace_rel_path": str(requested_workspace.workspace_rel_path),
        "requested_workspace_abs_path": str(requested_workspace.path),
        "requested_workspace_kind": requested_workspace.kind,
        "requested_workspace_raw_text": requested_workspace.raw_text,
    }


def _workspace_project_objective_context(workspace_project: Optional[WorkspaceProjectCandidate]) -> str:
    if workspace_project is None:
        return ""
    return (
        f"Canonical workspace project root: {workspace_project.path}. "
        f"Workspace-relative project root: {workspace_project.workspace_rel_path}. "
        "Start by inspecting this project root. Do not create a new scratch project when this path exists. "
    )


def _requested_workspace_objective_context(requested_workspace: Optional[WorkspacePathRequest]) -> str:
    if requested_workspace is None:
        return ""
    if requested_workspace.kind == "project_root":
        return (
            f"Requested workspace project root: {requested_workspace.path}. "
            f"Workspace-relative path: {requested_workspace.workspace_rel_path}. "
            "Start by inspecting this project root and continue there. "
        )
    if requested_workspace.kind == "target":
        return (
            f"Requested workspace target path: {requested_workspace.path}. "
            f"Workspace-relative path: {requested_workspace.workspace_rel_path}. "
            "Create or continue the project at this managed-workspace path. "
        )
    return (
        f"Requested workspace parent directory: {requested_workspace.path}. "
        f"Workspace-relative path: {requested_workspace.workspace_rel_path}. "
        "Create the new project under this managed-workspace directory and keep all generated artifacts there. "
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


def _clean_title(title: str, *, preserve_weak_fragment: bool = False) -> str:
    title = re.split(r"[.!?\n]", title, maxsplit=1)[0]
    title = re.split(r"\s+(?:in|with)\s+(?:the\s+)?(?:workspace|project\s+file)\b", title, maxsplit=1, flags=re.IGNORECASE)[0]
    title = re.split(
        r"\s+(?:and|but|then)\s+(?:keep\s+working|keep\s+revising|continue|resume|finish|return|work)\b",
        title,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    title = re.sub(
        r"^(?:and\s+)?(?:keep\s+working(?:\s+on)?|keep\s+revising|continue|resume|finish|return\s+to|work\s+on)\b\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(title.strip(" '*\".,:;").split())[:80]
    if cleaned:
        return cleaned
    return str(title or "").strip(" '*\".,:;")[:80] if preserve_weak_fragment else ""


def _human_reason(reason: str) -> str:
    return " ".join(str(reason or "").replace("_", " ").split()) or "unknown reason"


def _is_generic_project_title(title: str) -> bool:
    lowered = " ".join(str(title or "").lower().split())
    leading_phrase = re.split(r"[,;:]", lowered, maxsplit=1)[0].strip()
    if lowered in _GENERIC_PROJECT_TITLES or leading_phrase in _GENERIC_PROJECT_TITLES:
        return True
    return any(
        lowered.startswith(prefix)
        for prefix in (
            "without ",
            "with ",
            "for ",
            "in ",
            "toward ",
            "project not ",
            "project and ",
            "project or ",
            "project but ",
            "software project not ",
            "writing project not ",
        )
    )


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
