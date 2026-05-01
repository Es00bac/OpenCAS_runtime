"""Continuity breadcrumb helpers for burst lifecycle writes.

The helpers in this module keep burst-start, burst-complete, and burst-interrupt
updates on one narrow path so the runtime can record a short recovery cue even
when one durable store is unavailable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import unquote
from textwrap import shorten
from typing import Any, Optional


@dataclass(frozen=True)
class ContinuityBreadcrumb:
    """Rendered breadcrumb line plus the shorter note used in musubi history."""

    timestamp: str
    phase: str
    breadcrumb: str
    note: str
    comment: Optional[str]
    objective: str
    branch: str
    handoff: str
    intent: str
    focus: str
    next_step: str
    musubi: Optional[float]


def build_burst_breadcrumb(
    *,
    phase: str,
    intent: str,
    focus: str,
    branch: Optional[str] = None,
    next_step: str,
    note: Optional[str] = None,
    musubi: Optional[float] = None,
    timestamp: Optional[datetime] = None,
) -> ContinuityBreadcrumb:
    """Render a compact continuity breadcrumb line and structured recovery note."""
    stamp = (timestamp or datetime.now(timezone.utc)).isoformat()
    clean_phase = _compact(phase, 24).lower()
    clean_intent = _compact(intent, 72)
    clean_comment = _compact(note, 72) if note else None
    clean_last_action = _compact(branch or focus or intent or note or phase, 56)
    clean_focus = _compact(focus or clean_last_action, 72)
    clean_next_step = _compact(next_step, 64)
    musubi_label = _format_musubi(musubi)
    note = build_continuity_note(
        note=clean_last_action,
        objective=clean_intent,
        last_action=clean_last_action,
        resume_hint=clean_next_step,
        musubi=musubi,
        timestamp=stamp,
    )
    breadcrumb = (
        f"{stamp} | phase: {clean_phase} | intent: {clean_intent} | last_action: {clean_last_action}"
        f" | next_resume_point: {clean_next_step} | musubi: {musubi_label}"
    )
    if clean_comment:
        breadcrumb = f"{breadcrumb} | comment: {clean_comment}"
    return ContinuityBreadcrumb(
        timestamp=stamp,
        phase=clean_phase,
        breadcrumb=breadcrumb,
        note=note,
        comment=clean_comment,
        objective=clean_intent,
        branch=clean_last_action,
        handoff=clean_next_step,
        intent=clean_intent,
        focus=clean_focus,
        next_step=clean_next_step,
        musubi=musubi,
    )


def build_runtime_burst_breadcrumb(
    runtime: Any,
    *,
    phase: str,
    intent: str,
    focus: str,
    next_step: str,
    note: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> ContinuityBreadcrumb:
    """Build a burst breadcrumb using the musubi score currently on the runtime."""
    return build_burst_breadcrumb(
        phase=phase,
        intent=intent,
        focus=focus,
        branch=current_runtime_branch(runtime, focus),
        next_step=next_step,
        note=note,
        musubi=current_runtime_musubi(runtime),
        timestamp=timestamp,
    )


def current_runtime_musubi(runtime: Any) -> Optional[float]:
    """Return the current musubi score from the runtime's musubi mirrors."""
    relational = getattr(getattr(runtime, "ctx", None), "relational", None)
    state = getattr(relational, "state", None)
    musubi = getattr(state, "musubi", None)
    if isinstance(musubi, (int, float)):
        return float(musubi)
    somatic = getattr(getattr(runtime, "ctx", None), "somatic", None)
    state = getattr(somatic, "state", None)
    musubi = getattr(state, "musubi", None)
    if isinstance(musubi, (int, float)):
        return float(musubi)
    return None


def current_runtime_focus(runtime: Any, fallback: str) -> str:
    """Derive the current focus from executive state or a fallback label."""
    executive = getattr(runtime, "executive", None)
    if executive is not None:
        intention = str(getattr(executive, "intention", "") or "").strip()
        if intention:
            return intention
        task_queue = list(getattr(executive, "task_queue", []) or [])
        if task_queue:
            queued = str(getattr(task_queue[0], "content", "") or "").strip()
            if queued:
                return queued
    baa = getattr(runtime, "baa", None)
    if baa is not None and int(getattr(baa, "active_count", 0) or 0) > 0:
        return "active BAA work"
    return fallback


def current_runtime_branch(runtime: Any, fallback: str) -> str:
    """Derive the current work branch from the runtime's executive state."""
    return current_runtime_focus(runtime, fallback)


async def record_burst_continuity(
    runtime: Any,
    *,
    trigger: str,
    phase: str,
    intent: str,
    focus: str,
    next_step: str,
    note: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    episode_id: Optional[str] = None,
) -> Optional[str]:
    """Persist a burst breadcrumb in identity and musubi history."""
    identity = getattr(getattr(runtime, "ctx", None), "identity", None)
    relational = getattr(getattr(runtime, "ctx", None), "relational", None)
    rendered = build_runtime_burst_breadcrumb(
        runtime,
        phase=phase,
        intent=intent,
        focus=focus,
        next_step=next_step,
        note=note,
        timestamp=timestamp,
    )
    breadcrumb = rendered.breadcrumb
    note_text = rendered.note

    identity_breadcrumb: Optional[str] = None
    if identity is not None and hasattr(identity, "record_continuity_breadcrumb"):
        try:
            identity_breadcrumb = identity.record_continuity_breadcrumb(
                intent=f"{rendered.phase} burst: {rendered.intent}",
                decision=f"{rendered.phase} burst boundary captured",
                note=rendered.note,
                next_step=rendered.next_step,
            )
        except Exception:
            pass

    relational_breadcrumb: Optional[str] = None
    if relational is not None and hasattr(relational, "record_burst_event"):
        try:
            await relational.record_burst_event(
                trigger=trigger,
                continuity_breadcrumb=breadcrumb,
                state_continuity_breadcrumb=rendered.note,
                note=note_text,
                episode_id=episode_id,
            )
            relational_breadcrumb = breadcrumb
        except Exception:
            pass

    if identity_breadcrumb is not None:
        return identity_breadcrumb
    if relational_breadcrumb is not None:
        return relational_breadcrumb

    runtime_trace = getattr(runtime, "_trace", None)
    if callable(runtime_trace):
        runtime_trace(
            "continuity_breadcrumb_fallback",
            {
                "trigger": trigger,
                "breadcrumb": breadcrumb,
                "note": note_text,
                "episode_id": episode_id,
            },
        )
    return None


def recover_burst_continuity_context(
    breadcrumb: Optional[str],
    current_musubi: Optional[float],
    note: Optional[str] = None,
) -> Optional[str]:
    """Reconstruct the latest burst context from a breadcrumb and current musubi."""
    parsed = parse_burst_breadcrumb(breadcrumb)
    if parsed is None:
        return None
    note_context = _parse_burst_note(parsed.note) if parsed.note else None
    musubi_label = _format_musubi(current_musubi if current_musubi is not None else parsed.musubi)
    current_branch = (
        note_context.branch
        if note_context is not None
        else parsed.branch or parsed.focus or parsed.intent
    )
    parts = [
        f"{parsed.phase} burst",
        f"intent={_compact(parsed.intent, 72)}",
        f"branch={_compact(current_branch, 72)}",
        f"next_recovery_cue={_compact(note_context.next_step if note_context is not None else parsed.next_step, 72)}",
    ]
    if parsed.comment:
        parts.append(f"comment={_compact(parsed.comment, 72)}")
    recovered_note = note or parsed.note or parsed.decision
    parts.append(f"current musubi={musubi_label}")
    if recovered_note and _compact(recovered_note, 72) not in parts[2]:
        parts.append(f"note={_compact(recovered_note, 72)}")
    return "; ".join(parts)


def is_recoverable_burst_breadcrumb(
    breadcrumb: Optional[str],
    current_musubi: Optional[float] = None,
    note: Optional[str] = None,
) -> bool:
    """Return True when the breadcrumb can recover a burst intent and next step."""
    return recover_burst_continuity_context(breadcrumb, current_musubi, note=note) is not None


def build_continuity_note(
    *,
    note: str,
    objective: str,
    branch: Optional[str] = None,
    recovery_cue: Optional[str] = None,
    last_action: Optional[str] = None,
    resume_hint: Optional[str] = None,
    handoff: Optional[str] = None,
    musubi: Optional[float] = None,
    timestamp: Optional[str] = None,
    salvage_packet_id: Optional[str] = None,
    project_signature: Optional[str] = None,
) -> str:
    """Render the stable short note format shared across burst entry points.

    The emitted note stays compact and machine-recoverable:
    timestamp / intent / branch / next_recovery_cue / musubi / salvage_packet_id / project_signature
    """
    resolved_branch = branch or last_action or note or handoff or objective
    resolved_next_recovery_cue = recovery_cue or resume_hint or handoff or objective
    return _build_burst_note(
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        intent=objective,
        branch=resolved_branch,
        next_recovery_cue=resolved_next_recovery_cue,
        musubi=musubi,
        salvage_packet_id=salvage_packet_id,
        project_signature=project_signature,
    )


def build_work_burst_continuity_note(
    runtime: Any,
    *,
    intent: str,
    last_action: str,
    resume_hint: str,
    timestamp: Optional[str] = None,
) -> str:
    """Build the short musubi-aware note used at work-burst boundaries."""
    return build_continuity_note(
        note=last_action,
        objective=intent,
        last_action=last_action,
        resume_hint=resume_hint,
        musubi=current_runtime_musubi(runtime),
        timestamp=timestamp,
    )


@dataclass(frozen=True)
class ParsedBurstBreadcrumb:
    """Structured view over the rendered burst breadcrumb."""

    phase: str
    intent: str
    decision: Optional[str]
    note: Optional[str]
    comment: Optional[str]
    branch: str
    focus: str
    next_step: str
    musubi: Optional[float]


def parse_burst_breadcrumb(breadcrumb: Optional[str]) -> Optional[ParsedBurstBreadcrumb]:
    """Parse a rendered breadcrumb into a recovery-friendly structure."""
    if not breadcrumb:
        return None

    fields: dict[str, str] = {}
    for chunk in breadcrumb.split(" | "):
        if ": " not in chunk:
            continue
        key, value = chunk.split(": ", 1)
        fields[key.strip().lower()] = value.strip()

    note_fields: dict[str, str] = {}
    if "=" in breadcrumb and (
        ("last_action=" in breadcrumb and "next_resume_point=" in breadcrumb)
        or ("branch=" in breadcrumb and "next_recovery_cue=" in breadcrumb)
        or ("objective=" in breadcrumb and "handoff=" in breadcrumb)
        or ("scope=" in breadcrumb and "recovery=" in breadcrumb)
        or ("last_action=" in breadcrumb and "resume_hint=" in breadcrumb)
    ):
        for chunk in breadcrumb.split(";"):
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            note_fields[key.strip().lower()] = _decode_breadcrumb_value(value.strip())

    if note_fields and not fields:
        fields = note_fields

    phase = fields.get("phase")
    intent = fields.get("intent") or fields.get("objective")
    decision = fields.get("decision")
    note = fields.get("last_action") or fields.get("branch") or fields.get("note")
    comment = fields.get("comment")
    branch = fields.get("last_action") or fields.get("branch") or fields.get("focus")
    next_step = (
        fields.get("next_resume_point")
        or fields.get("next_recovery_cue")
        or fields.get("handoff")
        or fields.get("next_step")
    )
    if (not phase or not intent or not next_step) and (
        ("intent" in fields and ("last_action" in fields or "next_resume_point" in fields or "branch" in fields))
        or ("objective" in fields and "handoff" in fields)
    ):
        phase = phase or fields.get("phase") or fields.get("scope") or fields.get("last_action")
        intent = intent or fields.get("intent") or fields.get("objective")
        next_step = next_step or fields.get("next_resume_point") or fields.get("next_recovery_cue") or fields.get("handoff") or fields.get("recovery") or fields.get("resume_hint")
        note = note or fields.get("last_action") or fields.get("note") or fields.get("branch") or fields.get("scope") or breadcrumb
        branch = branch or fields.get("last_action") or fields.get("branch") or fields.get("focus") or intent
    if not phase or not intent or not next_step:
        if intent and next_step:
            inferred_phase, inferred_intent = _split_identity_burst_intent(intent)
            if inferred_phase and inferred_intent:
                phase = phase or inferred_phase
                intent = inferred_intent
                branch = branch or inferred_intent
        if not phase or not intent or not next_step:
            parsed_note = _parse_burst_note(breadcrumb)
            if parsed_note is None:
                return None
            return parsed_note

    musubi_value = fields.get("musubi")
    musubi = None
    if musubi_value:
        try:
            musubi = float(musubi_value)
        except ValueError:
            musubi = None

    return ParsedBurstBreadcrumb(
        phase=phase,
        intent=intent,
        decision=decision,
        note=note,
        comment=comment,
        branch=branch or intent,
        focus=branch or intent,
        next_step=next_step,
        musubi=musubi,
    )


def _parse_burst_note(note: str) -> Optional[ParsedBurstBreadcrumb]:
    """Parse the short note form stored alongside burst continuity state."""
    fields: dict[str, str] = {}
    for chunk in note.split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        fields[key.strip().lower()] = _decode_breadcrumb_value(value.strip())

    short_note = fields.get("last_action") or fields.get("branch") or fields.get("note") or fields.get("scope")
    objective = fields.get("intent") or fields.get("objective") or fields.get("scope") or short_note
    handoff = (
        fields.get("next_resume_point")
        or fields.get("next_recovery_cue")
        or fields.get("handoff")
        or fields.get("recovery")
        or fields.get("resume_hint")
        or fields.get("next_step")
    )
    if not short_note or not objective or not handoff:
        return None

    musubi_value = fields.get("musubi")
    musubi = None
    if musubi_value:
        try:
            musubi = float(musubi_value)
        except ValueError:
            musubi = None

    return ParsedBurstBreadcrumb(
        phase=fields.get("phase") or objective,
        intent=objective,
        decision=None,
        note=short_note,
        comment=fields.get("comment"),
        branch=fields.get("last_action") or fields.get("branch") or fields.get("focus") or objective or short_note,
        focus=fields.get("focus") or fields.get("last_action") or fields.get("branch") or objective or short_note,
        next_step=handoff,
        musubi=musubi,
    )


def _build_burst_note(
    *,
    timestamp: str,
    intent: str,
    branch: str,
    next_recovery_cue: str,
    musubi: Optional[float] = None,
    salvage_packet_id: Optional[str] = None,
    project_signature: Optional[str] = None,
) -> str:
    """Render the fixed note format stored alongside each burst boundary."""
    chunks = [
        f"timestamp={_encode_breadcrumb_value(_compact(timestamp, 32))}",
        f"intent={_encode_breadcrumb_value(_compact(intent, 56))}",
        f"branch={_encode_breadcrumb_value(_compact(branch, 56))}",
        f"next_recovery_cue={_encode_breadcrumb_value(_compact(next_recovery_cue, 56))}",
        f"musubi={_format_musubi(musubi)}",
    ]
    if salvage_packet_id:
        chunks.append(f"salvage_packet_id={_encode_breadcrumb_value(salvage_packet_id)}")
    if project_signature:
        chunks.append(f"project_signature={_encode_breadcrumb_value(project_signature)}")
    return ";".join(chunks)


def _split_identity_burst_intent(intent: str) -> tuple[Optional[str], Optional[str]]:
    """Split an identity continuity intent into burst phase and intent text."""
    match = re.match(r"^([A-Za-z_]+)\s+burst:\s+(.+)$", intent.strip())
    if not match:
        return None, None
    phase, burst_intent = match.group(1).strip().lower(), match.group(2).strip()
    if not phase or not burst_intent:
        return None, None
    return phase, burst_intent


def _compact(text: str, width: int) -> str:
    return shorten(" ".join(str(text).split()), width=width, placeholder="…")


def _encode_breadcrumb_value(text: str) -> str:
    return (
        text.replace("%", "%25")
        .replace(";", "%3B")
        .replace("=", "%3D")
        .replace("|", "%7C")
        .replace("\n", "%0A")
        .replace("\r", "%0D")
    )


def _decode_breadcrumb_value(text: str) -> str:
    return unquote(text)


def _format_musubi(musubi: Optional[float]) -> str:
    if musubi is None:
        return "unknown"
    return f"{musubi:+.2f}"

# Keep this module source-stable so Python recompiles cached bytecode when the
# continuity note contract changes.
