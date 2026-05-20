"""Recurring operator-priority detection from episodic evidence.

This module is a small classifier scaffold, not a script for one life event.
Pattern definitions supply seed vocabulary, while activation and resolution use
the same evidence rules: repeated user turns across days activate a priority,
and newer user turns with resolution markers resolve it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any

from opencas.cognition import CognitiveEventKind


@dataclass(frozen=True)
class PriorityPatternDefinition:
    key: str
    label: str
    support_focus: str
    activation_markers: tuple[str, ...]
    resolution_markers: tuple[str, ...]
    min_turns: int = 4
    min_days: int = 3


PRIORITY_PATTERNS: tuple[PriorityPatternDefinition, ...] = (
    PriorityPatternDefinition(
        key="financial_stability",
        label="Operator priority: financial stability, rent, income, and practical support",
        support_focus=(
            "financial stability, rent/bill pressure, unemployment or job-search needs, "
            "and practical income-path follow-through"
        ),
        activation_markers=(
            "behind on rent",
            "bill collector",
            "bill collectors",
            "business model",
            "cash flow",
            "cash-flow",
            "eviction",
            "food stamps",
            "get money",
            "income",
            "job application",
            "job search",
            "lost my job",
            "make money",
            "medicaid",
            "money flowing",
            "negative amount",
            "no job",
            "no money",
            "not having a job",
            "not having any money",
            "productive routine",
            "rent assistance",
            "side hustle",
            "start making income",
            "start making money",
            "unemployed",
            "unemployment",
        ),
        resolution_markers=(
            "do not need rental assistance anymore",
            "full-time job",
            "got a full-time job",
            "got a job",
            "income situation is resolved",
            "money situation is resolved",
            "new job",
            "no longer need rental assistance",
            "rent caught up",
            "rent is caught up",
            "stable income",
            "started a job",
        ),
    ),
)

_EXPLICIT_PRIORITY_RE = re.compile(
    r"\b(?:right now\s+)?(?:my\s+)?priorit(?:y|ies)\s+(?:is|are)\s+(?P<claim>[^.!?\n]+)",
    re.IGNORECASE,
)
_RESOLUTION_WORDS = {
    "complete",
    "completed",
    "done",
    "finished",
    "handled",
    "over",
    "resolved",
    "settled",
}
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "but",
    "can",
    "current",
    "for",
    "from",
    "have",
    "help",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "need",
    "needs",
    "now",
    "priority",
    "right",
    "situation",
    "that",
    "the",
    "this",
    "to",
    "with",
}


async def record_priority_patterns(runtime: Any, store: Any) -> dict[str, Any]:
    """Promote or resolve recurring operator priorities from recent user turns."""

    memory = getattr(runtime, "memory", None)
    list_recent = getattr(memory, "list_recent_episodes", None)
    if not callable(list_recent):
        return {"available": False, "reason": "memory_unavailable"}
    try:
        episodes = await list_recent(limit=20000)
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    user_turns = _user_turns(episodes)
    result = {
        "available": True,
        "patterns_checked": len(PRIORITY_PATTERNS),
        "promoted": 0,
        "resolved": 0,
        "dynamic_promoted": 0,
        "dynamic_resolved": 0,
        "skipped_existing": 0,
    }
    for pattern in PRIORITY_PATTERNS:
        activation = _activation_evidence(pattern, user_turns)
        resolution = _resolution_evidence(pattern, user_turns)
        if _resolution_supersedes_activation(resolution, activation):
            outcome = await _resolve_priority(store, pattern, resolution)
            result[outcome] = int(result.get(outcome, 0)) + 1
            continue
        if not _activation_threshold_met(pattern, activation):
            continue
        outcome = await _promote_priority(store, pattern, activation)
        result[outcome] = int(result.get(outcome, 0)) + 1
    dynamic = await _record_explicit_priority_claims(store, user_turns)
    for key, value in dynamic.items():
        result[key] = int(result.get(key, 0)) + int(value or 0)
    return result


def _user_turns(episodes: list[Any]) -> list[Any]:
    turns = []
    for episode in episodes:
        kind = str(getattr(getattr(episode, "kind", ""), "value", getattr(episode, "kind", "")))
        if kind != "turn":
            continue
        payload = getattr(episode, "payload", {}) or {}
        role = str(payload.get("role") or payload.get("message_role") or "").lower()
        if role and role != "user":
            continue
        content = str(getattr(episode, "content", "") or "").strip()
        if not content:
            continue
        lowered = content.lower()
        if lowered.startswith("codex here, not the owner/operator") or lowered.startswith(
            "internal maintenance correction from codex"
        ):
            continue
        turns.append(episode)
    return sorted(turns, key=lambda item: getattr(item, "created_at", datetime.min.replace(tzinfo=timezone.utc)))


def _activation_evidence(pattern: PriorityPatternDefinition, episodes: list[Any]) -> dict[str, Any]:
    matches = []
    term_counts: Counter[str] = Counter()
    for episode in episodes:
        text = str(getattr(episode, "content", "") or "").lower()
        if _contains_any(text, pattern.resolution_markers):
            continue
        terms = [marker for marker in pattern.activation_markers if marker in text]
        if not terms:
            continue
        term_counts.update(terms)
        matches.append((episode, terms))
    return _summarize_matches(matches, term_counts)


def _resolution_evidence(pattern: PriorityPatternDefinition, episodes: list[Any]) -> dict[str, Any]:
    matches = []
    term_counts: Counter[str] = Counter()
    for episode in episodes:
        text = str(getattr(episode, "content", "") or "").lower()
        terms = [marker for marker in pattern.resolution_markers if marker in text]
        if not terms:
            continue
        term_counts.update(terms)
        matches.append((episode, terms))
    return _summarize_matches(matches, term_counts)


def _summarize_matches(matches: list[tuple[Any, list[str]]], term_counts: Counter[str]) -> dict[str, Any]:
    days = {
        _created_at(episode).date().isoformat()
        for episode, _terms in matches
        if _created_at(episode) is not None
    }
    latest = max((_created_at(episode) for episode, _terms in matches), default=None)
    return {
        "turn_count": len(matches),
        "distinct_days": len(days),
        "latest_at": latest,
        "latest_at_iso": latest.isoformat() if latest else "",
        "evidence_refs": [f"episode:{getattr(episode, 'episode_id', '')}" for episode, _terms in matches[-8:]],
        "samples": [str(getattr(episode, "content", "") or "")[:240] for episode, _terms in matches[-5:]],
        "terms": [term for term, _count in term_counts.most_common(12)],
    }


def _activation_threshold_met(
    pattern: PriorityPatternDefinition,
    evidence: dict[str, Any],
) -> bool:
    return (
        int(evidence.get("turn_count") or 0) >= pattern.min_turns
        and int(evidence.get("distinct_days") or 0) >= pattern.min_days
    )


def _resolution_supersedes_activation(resolution: dict[str, Any], activation: dict[str, Any]) -> bool:
    if not resolution.get("latest_at"):
        return False
    if not activation.get("latest_at"):
        return True
    return resolution["latest_at"] >= activation["latest_at"]


async def _promote_priority(
    store: Any,
    pattern: PriorityPatternDefinition,
    evidence: dict[str, Any],
) -> str:
    if await _pattern_event_exists(store, pattern.key, "active", evidence.get("latest_at_iso", "")):
        return "skipped_existing"
    confidence = min(
        0.94,
        0.48 + int(evidence.get("turn_count") or 0) * 0.05 + int(evidence.get("distinct_days") or 0) * 0.035,
    )
    payload = {
        "pattern_key": pattern.key,
        "status": "active",
        "schedule_priority": 9.2,
        "schedule_tags": ["life_priority", "operator_priority", pattern.key],
        **_event_payload(evidence),
    }
    await store.record_event(
        CognitiveEventKind.LIFE_PRIORITY,
        f"Recurring operator priority detected: {pattern.label}",
        content=(
            f"Detected {evidence['turn_count']} user turns across {evidence['distinct_days']} days "
            f"about {pattern.support_focus}. This priority should stay active until newer user "
            "context or completion evidence resolves it."
        ),
        source="episodic_priority_pattern",
        confidence=confidence,
        salience=2.1 + min(1.2, int(evidence.get("turn_count") or 0) * 0.08),
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    await store.upsert_attention(
        pattern.label,
        strength=max(0.88, confidence),
        decay_rate=0.03,
        source="episodic_priority_pattern",
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    await store.upsert_working_memory(
        f"operator_priority:{pattern.key}",
        (
            f"Recurring operator priority from {evidence['turn_count']} user turns over "
            f"{evidence['distinct_days']} days: {pattern.support_focus}. Treat this as a "
            "current high-priority support lane, use planning/schedules/tools/research when "
            "safe, and retire it when newer user evidence shows the situation has changed."
        ),
        priority=max(0.86, confidence),
        source="episodic_priority_pattern",
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    action, condition = _prospective_text(pattern)
    await store.upsert_prospective_memory(
        action,
        trigger_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        condition=condition,
        confidence=max(0.82, confidence - 0.03),
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    return "promoted"


async def _resolve_priority(
    store: Any,
    pattern: PriorityPatternDefinition,
    evidence: dict[str, Any],
) -> str:
    if await _pattern_event_exists(store, pattern.key, "resolved", evidence.get("latest_at_iso", "")):
        return "skipped_existing"
    payload = {"pattern_key": pattern.key, "status": "resolved", **_event_payload(evidence)}
    await store.record_event(
        CognitiveEventKind.LIFE_PRIORITY,
        f"Operator priority resolved or changed: {pattern.label}",
        content=(
            "Newer user context indicates this priority should no longer be treated as the "
            f"active situation: {pattern.support_focus}."
        ),
        source="episodic_priority_pattern",
        confidence=0.82,
        salience=1.4,
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    await store.upsert_attention(
        pattern.label,
        strength=0.2,
        decay_rate=0.2,
        source="episodic_priority_pattern",
        status="resolved",
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    await store.upsert_working_memory(
        f"operator_priority:{pattern.key}",
        (
            f"Resolved or changed operator priority: {pattern.support_focus}. Do not keep "
            "treating this as active unless newer evidence reactivates it."
        ),
        priority=0.2,
        source="episodic_priority_pattern",
        status="resolved",
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    action, condition = _prospective_text(pattern)
    await store.upsert_prospective_memory(
        action,
        trigger_at=None,
        condition=condition,
        status="resolved",
        proof_ref=evidence.get("evidence_refs", [""])[-1] if evidence.get("evidence_refs") else "",
        confidence=0.82,
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    return "resolved"


async def _pattern_event_exists(store: Any, pattern_key: str, status: str, latest_at_iso: str) -> bool:
    list_recent = getattr(store, "list_recent_events", None)
    if not callable(list_recent):
        return False
    try:
        events = await list_recent(kind=CognitiveEventKind.LIFE_PRIORITY, limit=40)
    except Exception:
        return False
    for event in events:
        payload = dict(getattr(event, "payload", {}) or {})
        if (
            payload.get("pattern_key") == pattern_key
            and payload.get("status") == status
            and (
                payload.get("latest_at") == latest_at_iso
                or payload.get("latest_at_iso") == latest_at_iso
            )
        ):
            return True
    return False


async def _record_explicit_priority_claims(store: Any, episodes: list[Any]) -> dict[str, int]:
    result = {"dynamic_promoted": 0, "dynamic_resolved": 0, "skipped_existing": 0}
    active_dynamic = await _active_dynamic_priorities(store)
    for episode in episodes:
        text = str(getattr(episode, "content", "") or "")
        if _looks_like_resolution(text):
            resolved = await _resolve_matching_dynamic_priorities(store, active_dynamic, episode)
            result["dynamic_resolved"] += resolved
            continue
        claim = _extract_explicit_priority_claim(text)
        if not claim:
            continue
        evidence = _dynamic_claim_evidence(claim, episode)
        if await _pattern_event_exists(store, evidence["pattern_key"], "active", evidence["latest_at_iso"]):
            result["skipped_existing"] += 1
            continue
        await _promote_dynamic_priority(store, evidence)
        result["dynamic_promoted"] += 1
        active_dynamic[evidence["pattern_key"]] = evidence
    return result


async def _active_dynamic_priorities(store: Any) -> dict[str, dict[str, Any]]:
    list_recent = getattr(store, "list_recent_events", None)
    if not callable(list_recent):
        return {}
    try:
        events = await list_recent(kind=CognitiveEventKind.LIFE_PRIORITY, limit=100)
    except Exception:
        return {}
    active: dict[str, dict[str, Any]] = {}
    for event in events:
        payload = dict(getattr(event, "payload", {}) or {})
        key = str(payload.get("pattern_key") or "")
        if not key.startswith("explicit:"):
            continue
        if payload.get("status") == "resolved":
            active.pop(key, None)
        elif payload.get("status") == "active":
            active[key] = {
                "pattern_key": key,
                "claim": str(payload.get("claim") or ""),
                "topic_tokens": list(payload.get("topic_tokens") or []),
            }
    return active


async def _resolve_matching_dynamic_priorities(
    store: Any,
    active_dynamic: dict[str, dict[str, Any]],
    episode: Any,
) -> int:
    text = str(getattr(episode, "content", "") or "")
    resolution_tokens = set(_topic_tokens(text))
    count = 0
    for key, evidence in list(active_dynamic.items()):
        topic_tokens = set(str(token) for token in evidence.get("topic_tokens") or [])
        if len(topic_tokens & resolution_tokens) < 2:
            continue
        resolved_evidence = dict(evidence)
        resolved_evidence.update(
            {
                "latest_at_iso": _created_at(episode).isoformat(),
                "evidence_refs": [f"episode:{getattr(episode, 'episode_id', '')}"],
                "samples": [text[:240]],
                "status": "resolved",
            }
        )
        if await _pattern_event_exists(store, key, "resolved", resolved_evidence["latest_at_iso"]):
            continue
        await _resolve_dynamic_priority(store, resolved_evidence)
        active_dynamic.pop(key, None)
        count += 1
    return count


async def _promote_dynamic_priority(store: Any, evidence: dict[str, Any]) -> None:
    payload = {
        "status": "active",
        "schedule_priority": 8.8,
        "schedule_tags": ["life_priority", "operator_priority", "explicit_priority"],
        **evidence,
    }
    label = f"Operator priority: {evidence['claim']}"
    await store.record_event(
        CognitiveEventKind.LIFE_PRIORITY,
        f"Explicit operator priority detected: {evidence['claim']}",
        content=(
            "The user explicitly identified this as a current priority. Treat it as active "
            "until newer user context or completion evidence resolves it."
        ),
        source="explicit_priority_claim",
        confidence=0.86,
        salience=2.0,
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    await store.upsert_attention(
        label,
        strength=0.88,
        decay_rate=0.04,
        source="explicit_priority_claim",
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    await store.upsert_working_memory(
        f"operator_priority:{evidence['pattern_key']}",
        (
            f"Explicit current operator priority: {evidence['claim']}. Use planning, "
            "schedules, tools, research, and follow-through when safe; resolve it when "
            "newer user evidence says the situation is complete or no longer active."
        ),
        priority=0.88,
        source="explicit_priority_claim",
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    action, condition = _dynamic_prospective_text(evidence)
    await store.upsert_prospective_memory(
        action,
        trigger_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        condition=condition,
        confidence=0.84,
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )


async def _resolve_dynamic_priority(store: Any, evidence: dict[str, Any]) -> None:
    payload = {"status": "resolved", **evidence}
    label = f"Operator priority: {evidence['claim']}"
    await store.record_event(
        CognitiveEventKind.LIFE_PRIORITY,
        f"Explicit operator priority resolved or changed: {evidence['claim']}",
        source="explicit_priority_resolution",
        confidence=0.8,
        salience=1.4,
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    await store.upsert_attention(
        label,
        strength=0.2,
        decay_rate=0.2,
        source="explicit_priority_resolution",
        status="resolved",
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    await store.upsert_working_memory(
        f"operator_priority:{evidence['pattern_key']}",
        (
            f"Resolved or changed explicit operator priority: {evidence['claim']}. Do not "
            "keep treating it as active unless newer evidence reactivates it."
        ),
        priority=0.2,
        source="explicit_priority_resolution",
        status="resolved",
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )
    action, condition = _dynamic_prospective_text(evidence)
    await store.upsert_prospective_memory(
        action,
        trigger_at=None,
        condition=condition,
        status="resolved",
        proof_ref=evidence.get("evidence_refs", [""])[-1] if evidence.get("evidence_refs") else "",
        confidence=0.8,
        evidence_refs=list(evidence.get("evidence_refs") or []),
        payload=payload,
    )


def _extract_explicit_priority_claim(text: str) -> str:
    match = _EXPLICIT_PRIORITY_RE.search(text)
    if not match:
        return ""
    claim = " ".join(match.group("claim").strip(" :;,.").split())
    return claim[:180]


def _dynamic_claim_evidence(claim: str, episode: Any) -> dict[str, Any]:
    tokens = _topic_tokens(claim)
    signature = "-".join(tokens[:8]) or "priority"
    return {
        "pattern_key": f"explicit:{signature}",
        "claim": claim,
        "topic_tokens": tokens,
        "latest_at_iso": _created_at(episode).isoformat(),
        "evidence_refs": [f"episode:{getattr(episode, 'episode_id', '')}"],
        "samples": [str(getattr(episode, "content", "") or "")[:240]],
    }


def _topic_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower())
    return [token for token in tokens if token not in _STOPWORDS][:12]


def _looks_like_resolution(text: str) -> bool:
    lowered = text.lower()
    if not any(word in lowered for word in _RESOLUTION_WORDS):
        return False
    return "priority" in lowered or "situation" in lowered or "no longer" in lowered


def _dynamic_prospective_text(evidence: dict[str, Any]) -> tuple[str, str]:
    claim = str(evidence.get("claim") or "explicit operator priority")
    return (
        f"Proactively support current operator priority: {claim}",
        (
            "When reflection, planning, scheduling, or daydreaming can safely advance this "
            "explicit operator priority, use available tools or ask the next needed question."
        ),
    )


def _event_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_count": int(evidence.get("turn_count") or 0),
        "distinct_days": int(evidence.get("distinct_days") or 0),
        "latest_at": evidence.get("latest_at_iso", ""),
        "terms": list(evidence.get("terms") or []),
        "samples": list(evidence.get("samples") or []),
    }


def _prospective_text(pattern: PriorityPatternDefinition) -> tuple[str, str]:
    return (
        f"Proactively support current operator priority: {pattern.support_focus}",
        (
            "When reflection, planning, scheduling, or daydreaming can safely advance this "
            "current operator priority, use available tools or ask the next needed question."
        ),
    )


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _created_at(episode: Any) -> datetime:
    value = getattr(episode, "created_at", None)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)
