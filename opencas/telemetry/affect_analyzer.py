"""Affect signal analyzer: extracts emotional dimensions from developer artifacts.

Instruments CI/CD pipelines by analyzing commit messages, PR tone, incident
response patterns, and other textual signals to produce AffectSnapshots.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from .affect_models import AffectDimension, AffectSnapshot, MoodSignalSource


# Simple lexicon-based sentiment analysis
# In production, this would use a proper NLP model
_POSITIVE_INDICATORS = {
    "love", "awesome", "great", "excellent", "clean", "elegant", "beautiful",
    "smooth", "perfect", "solid", "robust", "graceful", "delight", "happy",
    "excited", "proud", "confident", "clear", "simple", "refined", "polished",
    "improve", "better", "optimize", "enhance", "upgrade", "fix", "resolve",
    "thanks", "thank", "appreciate", "grateful", "nice", "good", "well",
}

_NEGATIVE_INDICATORS = {
    "hate", "terrible", "awful", "mess", "ugly", "broken", "fragile", "hack",
    "workaround", "kludge", "dirty", "quick", "temporary", "temp", "band-aid",
    "panic", "stress", "frustrated", "annoying", "pain", "suffer", "struggle",
    "confused", "unclear", "complicated", "complex", "bloated", "legacy",
    "debt", "todo", "fixme", "hacky", " brittle", "fragile", "worried",
    "concern", "afraid", "scared", "nervous", "anxious", "urgent", "asap",
    "emergency", "critical", "blocker", "disaster", "nightmare", "hell",
}

_AROUSAL_INDICATORS = {
    "urgent", "asap", "emergency", "critical", "blocker", "panic", "rush",
    "hurry", "quick", "fast", "immediately", "now", "deadline", "pressure",
    "excited", "thrilled", "amazing", "incredible", "wow", "blast", "fire",
    "deploy", "ship", "launch", "release", "merge", "push",
}

_CERTAINTY_POSITIVE = {
    "sure", "certain", "confident", "clear", "obvious", "definitely",
    "absolutely", "verified", "tested", "proven", "validated", "confirmed",
}

_CERTAINTY_NEGATIVE = {
    "maybe", "perhaps", "possibly", "might", "could", "unclear", "unsure",
    "tentative", "experimental", "wip", "draft", "rough", "untested",
    "unknown", "guess", "assume", "hope", "wish",
}

_COHERENCE_POSITIVE = {
    "clean", "clear", "organized", "structured", "consistent", "coherent",
    "logical", "sensible", "straightforward", "documented", "explained",
}

_COHERENCE_NEGATIVE = {
    "messy", "confusing", "inconsistent", "scattered", "disorganized",
    "chaotic", "random", "unclear", "vague", "ambiguous", "mysterious",
}

_URGENCY_INDICATORS = {
    "urgent", "asap", "emergency", "critical", "blocker", "deadline",
    "hotfix", "patch", "incident", "outage", "down", "broken", "fail",
    "must", "need", "required", "mandatory", "essential", "vital",
}


def _hash_signal(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _count_matches(text: str, indicators: set) -> int:
    text_lower = text.lower()
    return sum(1 for word in indicators if word in text_lower)


def _normalize_score(positive: int, negative: int, total_words: int) -> float:
    """Normalize to [-1, 1] range based on positive vs negative matches."""
    if total_words == 0:
        return 0.0
    raw = (positive - negative) / max(1, total_words * 0.1)
    return round(max(-1.0, min(1.0, raw)), 4)


def _extract_commit_metadata(text: str) -> Dict[str, Any]:
    """Extract structured metadata from a commit message."""
    lines = text.strip().split("\n")
    subject = lines[0] if lines else ""
    body = "\n".join(lines[1:]).strip()

    # Detect conventional commit type
    cc_match = re.match(r"^(\w+)(\(.+\))?!?:\s*(.+)$", subject)
    commit_type = None
    scope = None
    if cc_match:
        commit_type = cc_match.group(1)
        scope = cc_match.group(2).strip("()") if cc_match.group(2) else None

    # Detect issue references
    issue_refs = re.findall(r"#(\d+)", text)

    # Detect breaking changes
    breaking = "BREAKING CHANGE" in text or "!" in subject

    return {
        "subject": subject,
        "body": body,
        "commit_type": commit_type,
        "scope": scope,
        "issue_refs": issue_refs,
        "breaking": breaking,
        "line_count": len(lines),
    }


def analyze_commit_message(
    message: str,
    session_id: str,
    artifact_id: Optional[str] = None,
    actor: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> AffectSnapshot:
    """Analyze a git commit message for emotional tone."""
    metadata = _extract_commit_metadata(message)
    text = message.lower()
    words = text.split()
    total_words = len(words)

    # Valence from positive/negative indicators
    pos = _count_matches(text, _POSITIVE_INDICATORS)
    neg = _count_matches(text, _NEGATIVE_INDICATORS)
    valence = _normalize_score(pos, neg, total_words)

    # Adjust valence based on commit type
    if metadata.get("commit_type") in {"fix", "hotfix", "revert", "rollback"}:
        valence = max(-1.0, valence - 0.2)
    elif metadata.get("commit_type") in {"feat", "feature", "refactor", "perf"}:
        valence = min(1.0, valence + 0.1)

    # Arousal
    arousal_matches = _count_matches(text, _AROUSAL_INDICATORS)
    arousal = min(1.0, arousal_matches / max(1, total_words * 0.05))
    if metadata.get("breaking"):
        arousal = min(1.0, arousal + 0.3)

    # Certainty
    cert_pos = _count_matches(text, _CERTAINTY_POSITIVE)
    cert_neg = _count_matches(text, _CERTAINTY_NEGATIVE)
    certainty = _normalize_score(cert_pos, cert_neg, total_words)

    # Coherence
    coh_pos = _count_matches(text, _COHERENCE_POSITIVE)
    coh_neg = _count_matches(text, _COHERENCE_NEGATIVE)
    coherence = _normalize_score(coh_pos, coh_neg, total_words)
    # Longer, well-structured commits suggest higher coherence
    if metadata.get("line_count", 1) > 3 and metadata.get("body"):
        coherence = min(1.0, coherence + 0.1)

    # Urgency
    urgency_matches = _count_matches(text, _URGENCY_INDICATORS)
    urgency = min(1.0, urgency_matches / max(1, total_words * 0.05))
    if metadata.get("commit_type") in {"hotfix", "revert", "rollback"}:
        urgency = min(1.0, urgency + 0.4)

    dimensions = {
        AffectDimension.VALENCE.value: round(valence, 4),
        AffectDimension.AROUSAL.value: round(arousal, 4),
        AffectDimension.CERTAINTY.value: round(certainty, 4),
        AffectDimension.COHERENCE.value: round(coherence, 4),
        AffectDimension.URGENCY.value: round(urgency, 4),
    }

    snapshot = AffectSnapshot(
        session_id=session_id,
        artifact_id=artifact_id,
        actor=actor,
        dimensions=dimensions,
        source=MoodSignalSource.COMMIT_MESSAGE,
        source_payload=metadata,
        raw_signal=message[:4096],
        signal_hash=_hash_signal(message),
        confidence=0.6,  # Lexicon-based is moderately confident
    )

    if timestamp:
        from datetime import datetime, timezone
        try:
            snapshot.timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            pass

    return snapshot


def analyze_pr_content(
    title: str,
    description: str,
    session_id: str,
    artifact_id: Optional[str] = None,
    actor: Optional[str] = None,
    review_comments: Optional[List[str]] = None,
) -> AffectSnapshot:
    """Analyze PR title + description + optional review comments for tone."""
    full_text = f"{title}\n\n{description}"
    if review_comments:
        full_text += "\n\n" + "\n".join(review_comments)

    words = full_text.lower().split()
    total_words = len(words)

    pos = _count_matches(full_text.lower(), _POSITIVE_INDICATORS)
    neg = _count_matches(full_text.lower(), _NEGATIVE_INDICATORS)
    valence = _normalize_score(pos, neg, total_words)

    # PRs with many review comments may indicate friction
    if review_comments and len(review_comments) > 5:
        valence = max(-1.0, valence - 0.15)

    arousal = min(1.0, _count_matches(full_text.lower(), _AROUSAL_INDICATORS) / max(1, total_words * 0.05))
    certainty = _normalize_score(
        _count_matches(full_text.lower(), _CERTAINTY_POSITIVE),
        _count_matches(full_text.lower(), _CERTAINTY_NEGATIVE),
        total_words,
    )
    coherence = _normalize_score(
        _count_matches(full_text.lower(), _COHERENCE_POSITIVE),
        _count_matches(full_text.lower(), _COHERENCE_NEGATIVE),
        total_words,
    )
    urgency = min(1.0, _count_matches(full_text.lower(), _URGENCY_INDICATORS) / max(1, total_words * 0.05))

    dimensions = {
        AffectDimension.VALENCE.value: round(valence, 4),
        AffectDimension.AROUSAL.value: round(arousal, 4),
        AffectDimension.CERTAINTY.value: round(certainty, 4),
        AffectDimension.COHERENCE.value: round(coherence, 4),
        AffectDimension.URGENCY.value: round(urgency, 4),
    }

    return AffectSnapshot(
        session_id=session_id,
        artifact_id=artifact_id,
        actor=actor,
        dimensions=dimensions,
        source=MoodSignalSource.PR_DESCRIPTION,
        source_payload={
            "title": title,
            "description_length": len(description),
            "review_comment_count": len(review_comments) if review_comments else 0,
        },
        raw_signal=full_text[:4096],
        signal_hash=_hash_signal(full_text),
        confidence=0.55,
    )


def analyze_incident_response(
    message: str,
    session_id: str,
    artifact_id: Optional[str] = None,
    actor: Optional[str] = None,
    severity_level: Optional[str] = None,
) -> AffectSnapshot:
    """Analyze incident response messages for stress and urgency signals."""
    text = message.lower()
    words = text.split()
    total_words = len(words)

    # Incident responses are typically high-stress contexts
    pos = _count_matches(text, _POSITIVE_INDICATORS)
    neg = _count_matches(text, _NEGATIVE_INDICATORS)
    valence = _normalize_score(pos, neg, total_words)

    # Base arousal is higher for incidents
    arousal = 0.5 + min(0.5, _count_matches(text, _AROUSAL_INDICATORS) / max(1, total_words * 0.05))
    urgency = 0.6 + min(0.4, _count_matches(text, _URGENCY_INDICATORS) / max(1, total_words * 0.05))

    certainty = _normalize_score(
        _count_matches(text, _CERTAINTY_POSITIVE),
        _count_matches(text, _CERTAINTY_NEGATIVE),
        total_words,
    )
    coherence = _normalize_score(
        _count_matches(text, _COHERENCE_POSITIVE),
        _count_matches(text, _COHERENCE_NEGATIVE),
        total_words,
    )

    # Severity adjustments
    if severity_level:
        sev_lower = severity_level.lower()
        if sev_lower in {"critical", "p1", "sev1"}:
            arousal = min(1.0, arousal + 0.3)
            urgency = min(1.0, urgency + 0.3)
            valence = max(-1.0, valence - 0.2)
        elif sev_lower in {"high", "p2", "sev2"}:
            arousal = min(1.0, arousal + 0.15)
            urgency = min(1.0, urgency + 0.15)

    dimensions = {
        AffectDimension.VALENCE.value: round(valence, 4),
        AffectDimension.AROUSAL.value: round(arousal, 4),
        AffectDimension.CERTAINTY.value: round(certainty, 4),
        AffectDimension.COHERENCE.value: round(coherence, 4),
        AffectDimension.URGENCY.value: round(urgency, 4),
    }

    return AffectSnapshot(
        session_id=session_id,
        artifact_id=artifact_id,
        actor=actor,
        dimensions=dimensions,
        source=MoodSignalSource.INCIDENT_RESPONSE,
        source_payload={"severity_level": severity_level},
        raw_signal=message[:4096],
        signal_hash=_hash_signal(message),
        confidence=0.65,  # Incident context provides clearer signals
    )


def compute_trajectory_from_snapshots(
    snapshots: List[AffectSnapshot],
    session_id: str,
    artifact_id: Optional[str] = None,
    actor: Optional[str] = None,
) -> "AffectTrajectory":
    """Build an AffectTrajectory from a chronologically sorted list of snapshots."""
    from .affect_models import AffectTrajectory

    snapshots = sorted(snapshots, key=lambda s: s.timestamp)
    traj = AffectTrajectory(
        session_id=session_id,
        artifact_id=artifact_id,
        actor=actor,
        snapshots=snapshots,
    )
    return traj
