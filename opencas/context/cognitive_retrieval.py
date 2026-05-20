"""Cognitive-state modulation for memory retrieval.

This module keeps the retriever's cognitive feedback loop explicit: persisted
attention, working memory, prospective memory, and recent cognitive events are
converted into bounded ranking signals instead of living only in prompt text.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Tuple

CandidateMap = Dict[Tuple[str, str], Dict[str, Any]]


COGNITIVE_SIGNAL_KEYS = (
    "cognitive_focus_score",
    "cognitive_working_memory_score",
    "cognitive_prospective_score",
    "cognitive_event_score",
)


async def apply_cognitive_state_to_candidates(
    store: Any,
    candidate_map: CandidateMap,
    *,
    query: str,
    session_id: str | None,
) -> None:
    """Apply persisted cognitive state as inspectable retrieval signals."""

    if store is None or not candidate_map:
        return
    try:
        attention = await store.list_attention(limit=8)
        working = await store.list_working_memory(limit=8)
        prospective = await store.list_prospective_memories(limit=8)
        events = await store.list_recent_events(session_id=session_id, status="active", limit=40)
        if not events and session_id is not None:
            events = await store.list_recent_events(status="active", limit=40)
    except Exception:
        return

    query_tokens = _tokens(query)
    focus_profiles = [
        _EvidenceProfile(
            tokens=_tokens(f"{item.label} {item.source} {' '.join(item.evidence_refs)}"),
            weight=float(getattr(item, "strength", 0.0) or 0.0),
            reason=str(getattr(item, "label", "") or ""),
        )
        for item in attention
        if str(getattr(item, "status", "active") or "active") == "active"
    ]
    working_profiles = [
        _EvidenceProfile(
            tokens=_tokens(f"{item.slot} {item.content} {item.source}"),
            weight=float(getattr(item, "priority", 0.0) or 0.0),
            reason=str(getattr(item, "slot", "") or ""),
        )
        for item in working
        if str(getattr(item, "status", "active") or "active") == "active"
    ]
    prospective_profiles = [
        _EvidenceProfile(
            tokens=_tokens(f"{item.action} {item.condition} {item.proof_ref}"),
            weight=float(getattr(item, "confidence", 0.0) or 0.0),
            reason=str(getattr(item, "action", "") or ""),
        )
        for item in prospective
        if str(getattr(item, "status", "active") or "active") == "active"
    ]
    event_profiles = [
        _EvidenceProfile(
            tokens=_tokens(
                f"{getattr(getattr(item, 'kind', ''), 'value', getattr(item, 'kind', ''))} "
                f"{item.summary} {item.content} {item.source}"
            ),
            weight=min(1.0, (float(getattr(item, "salience", 0.0) or 0.0) / 3.0))
            * float(getattr(item, "confidence", 0.0) or 0.0),
            reason=str(getattr(item, "summary", "") or ""),
        )
        for item in events
    ]

    for candidate in candidate_map.values():
        content_tokens = _tokens(_candidate_text(candidate))
        if not content_tokens:
            continue
        candidate["cognitive_focus_score"], candidate["cognitive_focus_reason"] = _score_profiles(
            content_tokens,
            query_tokens,
            focus_profiles,
        )
        (
            candidate["cognitive_working_memory_score"],
            candidate["cognitive_working_memory_reason"],
        ) = _score_profiles(content_tokens, query_tokens, working_profiles)
        (
            candidate["cognitive_prospective_score"],
            candidate["cognitive_prospective_reason"],
        ) = _score_profiles(content_tokens, query_tokens, prospective_profiles)
        candidate["cognitive_event_score"], candidate["cognitive_event_reason"] = _score_profiles(
            content_tokens,
            query_tokens,
            event_profiles,
        )


class _EvidenceProfile:
    def __init__(self, *, tokens: set[str], weight: float, reason: str) -> None:
        self.tokens = tokens
        self.weight = max(0.0, min(1.0, weight))
        self.reason = " ".join(reason.split())[:160]


def _score_profiles(
    content_tokens: set[str],
    query_tokens: set[str],
    profiles: Iterable[_EvidenceProfile],
) -> tuple[float, str]:
    best_score = 0.0
    best_reason = ""
    for profile in profiles:
        if not profile.tokens or profile.weight <= 0.0:
            continue
        overlap = len(content_tokens & profile.tokens)
        if not overlap:
            continue
        denominator = max(1, min(len(profile.tokens), len(content_tokens)))
        query_overlap = len(query_tokens & profile.tokens) / max(1, len(query_tokens)) if query_tokens else 0.0
        score = min(1.0, (overlap / denominator) * profile.weight + query_overlap * 0.25)
        if score > best_score:
            best_score = score
            best_reason = profile.reason
    return best_score, best_reason


def _candidate_text(candidate: Dict[str, Any]) -> str:
    result = candidate.get("result")
    parts = [str(getattr(result, "content", "") or "")]
    episode = candidate.get("episode")
    memory = candidate.get("memory")
    for obj in (episode, memory):
        if obj is None:
            continue
        parts.append(str(getattr(obj, "content", "") or ""))
        parts.append(" ".join(str(tag) for tag in (getattr(obj, "tags", []) or [])))
        payload = getattr(obj, "payload", {}) or {}
        if isinstance(payload, dict):
            parts.extend(str(value) for value in payload.values() if isinstance(value, (str, int, float)))
    return " ".join(part for part in parts if part)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_./:-]{3,}", str(text or "").lower())
        if token not in {"the", "and", "for", "that", "with", "this", "from"}
    }
