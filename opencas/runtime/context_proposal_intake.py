"""Promote reflective context proposals into reusable cognitive state.

This module deliberately does not execute proposed work.  It turns qualified
reflective proposals into attention, working-memory, and contained self-work
signals so prior observations can influence later judgment without bypassing
the executive/arbiter boundary.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from datetime import datetime, timezone
from typing import Any

from opencas.cognition import (
    CognitiveEventKind,
    CognitionGrounding,
    GroundingKind,
    GroundingSource,
)
from opencas.context import ContextLane, ProposalStatus
from opencas.daydream.signals import (
    ContactPosture,
    PossibilitySignal,
    PossibilitySignalRoute,
    SelfWorkKind,
)


_ACTIONLESS_PREFIXES = (
    "continue listening",
    "continue monitoring",
    "do not comment",
    "hold commentary",
    "hold the comment",
    "keep listening",
    "keep watching",
    "wait for",
    "watch for",
)


async def intake_context_proposals(
    runtime: Any,
    cognitive_store: Any,
    *,
    limit: int = 96,
    max_intakes: int = 8,
    max_self_work_signals: int = 3,
) -> dict[str, Any]:
    """Intake recent pending proposals into cognitive state.

    Pending proposals remain proposals.  The intake receipt is stored in the
    proposal validation payload so maintenance can run repeatedly without
    duplicating self-work artifacts.
    """

    proposal_store = _context_proposal_store(runtime)
    list_recent = getattr(proposal_store, "list_recent", None)
    save_proposal = getattr(proposal_store, "save", None)
    if not callable(list_recent):
        return {"available": False, "reason": "context_proposal_store_unavailable"}

    try:
        proposals = await _maybe_await(
            list_recent(status=ProposalStatus.PENDING, limit=max(1, min(100, int(limit))))
        )
    except TypeError:
        proposals = await _maybe_await(list_recent(limit=max(1, min(100, int(limit)))))
    except Exception as exc:
        return {"available": False, "reason": f"context_proposal_list_failed:{type(exc).__name__}"}

    result = {
        "available": True,
        "scanned": 0,
        "eligible": 0,
        "already_ingested": 0,
        "attention_updates": 0,
        "working_memory_updates": 0,
        "self_work_signals": 0,
        "max_intakes": max(1, int(max_intakes)),
        "max_self_work_signals": max(0, int(max_self_work_signals)),
        "saved_proposals": 0,
    }
    for proposal in list(proposals or []):
        result["scanned"] += 1
        if result["eligible"] >= result["max_intakes"]:
            break
        validation = dict(getattr(proposal, "validation", {}) or {})
        if validation.get("context_proposal_intake"):
            result["already_ingested"] += 1
            continue
        candidate = _candidate_from_proposal(proposal)
        if candidate is None:
            continue
        result["eligible"] += 1
        receipt: dict[str, Any] = {
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "source": "cognitive_maintenance",
            "actionable": candidate["actionable"],
            "priority": candidate["priority"],
        }

        attention = await _upsert_attention(cognitive_store, candidate, proposal)
        if attention:
            result["attention_updates"] += 1
            receipt["attention_target_id"] = str(getattr(attention, "target_id", "") or "")

        if candidate["actionable"]:
            working_memory = await _upsert_working_memory(cognitive_store, candidate, proposal)
            if working_memory:
                result["working_memory_updates"] += 1
                receipt["working_memory_item_id"] = str(getattr(working_memory, "item_id", "") or "")
            if result["self_work_signals"] < result["max_self_work_signals"]:
                routed = await _route_self_work_signal(runtime, candidate, proposal)
                if routed:
                    result["self_work_signals"] += 1
                    receipt["signal_id"] = routed.get("signal_id")
                    receipt["signal_route"] = routed.get("route")
                    receipt["signal_status"] = routed.get("status")
            else:
                receipt["signal_status"] = "deferred_by_intake_cap"

        validation["context_proposal_intake"] = receipt
        proposal.validation = validation
        if callable(save_proposal):
            try:
                await _maybe_await(save_proposal(proposal))
                result["saved_proposals"] += 1
            except Exception:
                pass

    if result["eligible"]:
        record_event = getattr(cognitive_store, "record_event", None)
        if callable(record_event):
            try:
                await _maybe_await(
                    record_event(
                        CognitiveEventKind.MAINTENANCE_OUTCOME,
                        "Context proposal intake promoted reflective proposals into cognitive state",
                        source="context_proposal_intake",
                        confidence=0.72,
                        salience=1.15,
                        payload=result,
                    )
                )
            except Exception:
                pass
    return result


def _candidate_from_proposal(proposal: Any) -> dict[str, Any] | None:
    status = _enum_or_text(getattr(proposal, "status", "")).lower()
    if status and status != ProposalStatus.PENDING.value:
        return None
    source_lane = _enum_or_text(getattr(proposal, "source_lane", "")).lower()
    if source_lane and source_lane != ContextLane.REFLECTIVE.value:
        return None
    content = _clean_text(getattr(proposal, "content", ""))
    if len(content) < 24:
        return None
    validation = dict(getattr(proposal, "validation", {}) or {})
    confidence = _coerce_float(getattr(proposal, "confidence", 0.0), default=0.0)
    salience = _coerce_float(validation.get("salience"), default=confidence)
    priority = max(confidence, salience)
    if priority < 0.62:
        return None
    next_step = _clean_text(validation.get("self_directed_next_step") or _next_step_from_content(content))
    actionable = bool(next_step) and not _actionless_next_step(next_step)
    summary = _proposal_summary(proposal, content, next_step=next_step)
    if not summary:
        return None
    return {
        "summary": summary,
        "content": content,
        "next_step": next_step,
        "actionable": actionable,
        "priority": priority,
        "confidence": confidence,
        "salience": salience,
        "novelty": _coerce_float(validation.get("novelty"), default=0.5),
        "kind": str(getattr(proposal, "proposal_kind", "") or "context_proposal"),
        "source": str(validation.get("source") or "context_proposal"),
        "work_relevant": bool(validation.get("work_relevant")),
        "self_directed": bool(validation.get("self_directed")),
    }


async def _upsert_attention(cognitive_store: Any, candidate: dict[str, Any], proposal: Any) -> Any | None:
    upsert = getattr(cognitive_store, "upsert_attention", None)
    if not callable(upsert):
        return None
    label = _truncate(f"Reflective proposal: {candidate['summary']}", 180)
    return await _maybe_await(
        upsert(
            label,
            strength=min(1.0, max(0.4, candidate["priority"])),
            source="context_proposal_intake",
            evidence_refs=_proposal_refs(proposal),
            payload=_candidate_payload(candidate, proposal),
        )
    )


async def _upsert_working_memory(cognitive_store: Any, candidate: dict[str, Any], proposal: Any) -> Any | None:
    upsert = getattr(cognitive_store, "upsert_working_memory", None)
    if not callable(upsert):
        return None
    proposal_id = str(getattr(proposal, "proposal_id", "") or _stable_key(candidate["summary"]))
    return await _maybe_await(
        upsert(
            f"context_proposal:{proposal_id}",
            _truncate(candidate["next_step"] or candidate["summary"], 520),
            priority=min(1.0, max(0.45, candidate["priority"])),
            source="context_proposal_intake",
            evidence_refs=_proposal_refs(proposal),
            payload=_candidate_payload(candidate, proposal),
        )
    )


async def _route_self_work_signal(runtime: Any, candidate: dict[str, Any], proposal: Any) -> dict[str, Any] | None:
    promotion = getattr(runtime, "daydream_promotion", None)
    signal_store = getattr(promotion, "signal_store", None)
    save_signal = getattr(signal_store, "save_signal", None)
    route_signal = getattr(promotion, "route_signal", None)
    if not callable(save_signal) or not callable(route_signal):
        return None
    route = _signal_route(candidate)
    signal = PossibilitySignal(
        signal_id=f"context-proposal-{_stable_key(str(getattr(proposal, 'proposal_id', '')))}",
        source_reflection_id=str(getattr(proposal, "proposal_id", "")),
        source_mode="context_proposal_intake",
        summary=candidate["summary"],
        practical_branch=candidate["next_step"],
        bridge="Reflective context proposal intake converted this into contained self-work.",
        novelty=max(0.0, min(1.0, candidate["novelty"])),
        usefulness=max(0.0, min(1.0, candidate["priority"])),
        confidence=max(0.0, min(1.0, candidate["confidence"])),
        risk=0.1,
        evidence_ids=_proposal_refs(proposal),
        grounding=[
            CognitionGrounding(
                kind=GroundingKind.OBSERVED,
                source=GroundingSource.RUNTIME,
                subject="context_proposal_intake",
                claim=candidate["summary"],
                confidence=max(0.0, min(1.0, candidate["confidence"])),
                evidence_ids=_proposal_refs(proposal),
                allowed_surface="internal",
                meta={"proposal_kind": candidate["kind"]},
            )
        ],
        suggested_route=route,
        contact_posture=ContactPosture.SILENT,
        self_work_kind=SelfWorkKind.RESEARCH if route == PossibilitySignalRoute.RESEARCH else SelfWorkKind.NOTE,
        self_work_intent=candidate["next_step"],
        route_reason="context proposal intake routed an actionable reflective next step",
        meta=_candidate_payload(candidate, proposal),
    )
    await _maybe_await(save_signal(signal))
    routed = await _maybe_await(route_signal(signal))
    return routed if isinstance(routed, dict) else {"signal_id": signal.signal_id, "route": route.value}


def _signal_route(candidate: dict[str, Any]) -> PossibilitySignalRoute:
    text = f"{candidate.get('summary', '')} {candidate.get('next_step', '')}".lower()
    if any(token in text for token in ("research", "look up", "investigate", "study")):
        return PossibilitySignalRoute.RESEARCH
    return PossibilitySignalRoute.SELF_NOTE


def _candidate_payload(candidate: dict[str, Any], proposal: Any) -> dict[str, Any]:
    return {
        "proposal_id": str(getattr(proposal, "proposal_id", "") or ""),
        "proposal_kind": candidate["kind"],
        "proposal_source": candidate["source"],
        "work_relevant": candidate["work_relevant"],
        "self_directed": candidate["self_directed"],
        "next_step": candidate["next_step"],
    }


def _proposal_refs(proposal: Any) -> list[str]:
    proposal_id = str(getattr(proposal, "proposal_id", "") or "").strip()
    refs = [f"context_proposal:{proposal_id}"] if proposal_id else []
    refs.extend(str(ref).strip() for ref in list(getattr(proposal, "evidence_refs", []) or []) if str(ref).strip())
    return list(dict.fromkeys(refs))


def _proposal_summary(proposal: Any, content: str, *, next_step: str) -> str:
    validation = dict(getattr(proposal, "validation", {}) or {})
    for key in ("agent_viewpoint", "connection_summary", "why_it_matters"):
        value = _clean_text(validation.get(key))
        if value:
            return _truncate(value, 220)
    for label in ("Agent viewpoint:", "Connection:", "Why it matters:"):
        value = _section_after_label(content, label)
        if value:
            return _truncate(value, 220)
    return _truncate(next_step or content, 220)


def _next_step_from_content(content: str) -> str:
    return _section_after_label(content, "Next step:")


def _section_after_label(content: str, label: str) -> str:
    pattern = re.escape(label) + r"\s*(.+?)(?=\n[A-Z][A-Za-z /-]{2,40}:|\Z)"
    match = re.search(pattern, content, flags=re.DOTALL)
    if not match:
        return ""
    return _clean_text(match.group(1))


def _actionless_next_step(next_step: str) -> bool:
    lowered = _clean_text(next_step).lower()
    return any(lowered.startswith(prefix) for prefix in _ACTIONLESS_PREFIXES)


def _context_proposal_store(runtime: Any) -> Any:
    return getattr(runtime, "context_proposals", None) or getattr(
        getattr(runtime, "ctx", None),
        "context_proposal_store",
        None,
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _enum_or_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _truncate(value: str, limit: int) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _coerce_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_key(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:24]
