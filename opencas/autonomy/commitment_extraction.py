"""Helpers for extracting durable self-commitments from conversational text."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TEMPORAL_TAIL_RE = re.compile(
    r"\s*(?:,?\s+)?(?:later|soon|tomorrow|tonight|after\b.*|when\b.*|once\b.*|in\s+a(?:\s+little)?\s+while)\s*$",
    re.IGNORECASE,
)
_LEADING_CONTEXT_RE = re.compile(
    r"^(?:"
    r"next(?:\s+up)?[:,-]?\s*|"
    r"for\s+now[:,]?\s*|"
    r"right\s+now[:,]?\s*|"
    r"the\s+main\s+thing\s+left\s+is\s+|"
    r"the\s+next\s+step\s+is\s+|"
    r"what'?s\s+left\s+is\s+|"
    r"i\s+(?:need|still\s+need|want|should|plan)\s+to\s+|"
    r"we\s+need\s+to\s+"
    r")",
    re.IGNORECASE,
)
_VERB_PREFIX_RE = re.compile(
    r"^(?:fix|finish|write|review|ship|debug|continue|return\s+to|follow\s+up\s+on|resume|work\s+on|build|implement|draft|refactor|investigate|test)\b",
    re.IGNORECASE,
)
_PRONOUN_OBJECTS = {"this", "it", "that", "this one", "that one", "it again", "this again"}


@dataclass(frozen=True)
class SelfCommitmentCandidate:
    """A normalized durable commitment extracted from assistant text."""

    content: str
    trigger: str
    source_sentence: str
    confidence: float
    normalization_source: str


@dataclass(frozen=True)
class _CommitmentPattern:
    action: str
    template: str
    pattern: re.Pattern[str]


_COMMITMENT_PATTERNS: tuple[_CommitmentPattern, ...] = (
    _CommitmentPattern(
        action="return",
        template="Return to {target}",
        pattern=re.compile(
            r"\b(?:i will|i'll)\s+(?P<trigger>come\s+back\s+to|get\s+back\s+to|return\s+to|pick\s+(?:this|it|that)\s+up|pick\s+up|resume|continue|take\s+this\s+up|follow\s+up\s+on)\s+(?P<object>.+)",
            re.IGNORECASE,
        ),
    ),
    _CommitmentPattern(
        action="finish",
        template="Finish {target}",
        pattern=re.compile(
            r"\b(?:i will|i'll)\s+(?P<trigger>finish)\s+(?P<object>.+)",
            re.IGNORECASE,
        ),
    ),
    _CommitmentPattern(
        action="return",
        template="Return to {target}",
        pattern=re.compile(
            r"\b(?:let\s+me|i'll)\s+(?:pause|rest|stop)\s+(?:here|for\s+now)?\s*(?:and\s+then\s+|and\s+)?(?P<trigger>come\s+back\s+to|return\s+to|resume|continue)\s+(?P<object>.+)",
            re.IGNORECASE,
        ),
    ),
)


def extract_self_commitments(text: str) -> List[SelfCommitmentCandidate]:
    """Extract compact self-commitments from assistant conversational text."""
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    commitments: List[SelfCommitmentCandidate] = []
    for index, sentence in enumerate(sentences):
        candidate = _extract_from_sentence(sentence, previous_sentence=sentences[index - 1] if index > 0 else None)
        if candidate is not None:
            commitments.append(candidate)
    return commitments


def _extract_from_sentence(
    sentence: str,
    previous_sentence: Optional[str],
) -> Optional[SelfCommitmentCandidate]:
    for rule in _COMMITMENT_PATTERNS:
        match = rule.pattern.search(sentence)
        if not match:
            continue
        raw_object = _strip_temporal_tail(match.group("object"))
        if not _has_deferral_cue(sentence):
            continue
        if not raw_object:
            continue

        normalized_target = _normalize_direct_target(raw_object)
        normalization_source = "direct_object"
        confidence = 0.9

        if normalized_target is None:
            normalized_target = _normalize_context_target(previous_sentence)
            normalization_source = "prior_sentence_context"
            confidence = 0.72

        if not normalized_target:
            return None

        if normalization_source == "prior_sentence_context" and _VERB_PREFIX_RE.match(normalized_target):
            content = _capitalize_first(normalized_target)
        else:
            content = rule.template.format(target=normalized_target)

        return SelfCommitmentCandidate(
            content=_clean_commitment_content(content),
            trigger=match.group("trigger").strip().lower(),
            source_sentence=sentence.strip(),
            confidence=confidence,
            normalization_source=normalization_source,
        )
    return None


def _strip_temporal_tail(value: str) -> str:
    return _TEMPORAL_TAIL_RE.sub("", value.strip(" .,!?:;"))


def _normalize_direct_target(value: str) -> Optional[str]:
    candidate = value.strip(" .,!?:;")
    candidate = re.sub(r"^(?:to\s+)?(?:the\s+)?same\s+", "", candidate, flags=re.IGNORECASE)
    if not candidate:
        return None
    if candidate.lower() in _PRONOUN_OBJECTS:
        return None
    return candidate


def _normalize_context_target(previous_sentence: Optional[str]) -> Optional[str]:
    if not previous_sentence:
        return None
    candidate = previous_sentence.strip(" .,!?:;")
    if not candidate:
        return None
    candidate = _LEADING_CONTEXT_RE.sub("", candidate).strip(" .,!?:;")
    if not candidate:
        return None
    return candidate


def _clean_commitment_content(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" .")
    return _capitalize_first(cleaned)


def _capitalize_first(value: str) -> str:
    if not value:
        return value
    return value[0].upper() + value[1:]


def _has_deferral_cue(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(
        cue in lowered
        for cue in ("later", "soon", "tomorrow", "tonight", "after ", "when ", "once ", "in a while")
    )
