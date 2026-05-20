"""Helpers for extracting durable self-commitments from conversational text."""

from __future__ import annotations

import re
from dataclasses import dataclass
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
    r"i'?m\s+logging\s+this\s+as\s+a\s+future\s+capability\s+you\s+want\s+to\s+build\s+for\s+me[:,]?\s*|"
    r"i\s+(?:need|still\s+need|want|should|plan)\s+to\s+|"
    r"we\s+need\s+to\s+"
    r")",
    re.IGNORECASE,
)
_TRAILING_RATIONALE_RE = re.compile(r"\s*,?\s+so\s+.+$", re.IGNORECASE)
_VERB_PREFIX_RE = re.compile(
    r"^(?:fix|finish|write|review|ship|debug|continue|return\s+to|follow\s+up\s+on|resume|work\s+on|build|implement|draft|refactor|investigate|test)\b",
    re.IGNORECASE,
)
_PROMISE_PREFIX_RE = re.compile(r"^\s*i\s+promise\b[.:\-\u2014,;]?\s*(?P<object>.*)$", re.IGNORECASE)
_SOURCE_GROUNDING_PROMISE_RE = re.compile(
    r"\b(?:make things up|making things up|invent|sourceless|source|research|"
    r"don'?t know|do not know|straight with you|authentic|performative)\b",
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
    prefer_context: bool = False


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
    _CommitmentPattern(
        action="remind",
        template="Remind user about {target}",
        pattern=re.compile(
            r"(?:^[-*]\s*)?\b(?:i will|i'll)\s+(?P<trigger>remind\s+you)\s+(?P<object>.+)",
            re.IGNORECASE,
        ),
        prefer_context=True,
    ),
    _CommitmentPattern(
        action="support",
        template="Support {target}",
        pattern=re.compile(
            r"(?:^[-*]\s*)?\b(?:i will|i'll)\s+(?P<trigger>hold|watch|chime\s+in|support|body\s+double)\b(?P<object>.*)",
            re.IGNORECASE,
        ),
        prefer_context=True,
    ),
)


def extract_self_commitments(
    text: str,
    *,
    user_context: Optional[str] = None,
) -> List[SelfCommitmentCandidate]:
    """Extract compact self-commitments from assistant conversational text."""
    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    commitments: List[SelfCommitmentCandidate] = []
    contextual = _extract_contextual_assistant_acceptance(
        sentences=sentences,
        user_context=user_context,
    )
    if contextual is not None:
        commitments.append(contextual)
    for index, sentence in enumerate(sentences):
        following = " ".join(sentences[index + 1 : index + 4]) if index + 1 < len(sentences) else None
        candidate = _extract_explicit_promise(
            sentence,
            next_sentence=following,
        )
        if candidate is None:
            candidate = _extract_from_sentence(sentence, previous_sentence=sentences[index - 1] if index > 0 else None)
        if candidate is not None:
            commitments.append(candidate)
    return commitments


def _extract_contextual_assistant_acceptance(
    *,
    sentences: List[str],
    user_context: Optional[str],
) -> Optional[SelfCommitmentCandidate]:
    """Capture explicit acceptance of an ongoing assistant role from the prior user ask."""

    if not user_context:
        return None
    user_text = " ".join(str(user_context).split())
    if not _looks_like_ongoing_support_request(user_text):
        return None

    source_sentence = ""
    for sentence in sentences:
        if re.search(
            (
                r"\b(?:i\s+can\s+be|i(?:'ll| will)\s+be)\s+(?:that|your)\s+assistant\b"
                r"|\b(?:i\s+can|i(?:'ll| will))\s+help\b"
                r"|\b(?:i\s+can|i(?:'ll| will))\s+support\b"
            ),
            sentence,
            re.IGNORECASE,
        ):
            source_sentence = sentence.strip()
            break
    if not source_sentence:
        return None

    target = _normalize_assistant_support_target(user_text)
    if not target:
        return None
    return SelfCommitmentCandidate(
        content=_clean_commitment_content(f"Support {target}"),
        trigger="accepted_assistant_role",
        source_sentence=source_sentence,
        confidence=0.86,
        normalization_source="contextual_assistant_acceptance",
    )


def _looks_like_ongoing_support_request(user_text: str) -> bool:
    if re.search(r"\bcan\s+you\s+be\s+my\s+assistant\b", user_text, re.IGNORECASE):
        return True
    if not re.search(
        r"\b(?:can|could|will|would)\s+you\s+(?:help|support)\s+me\b"
        r"|\bi\s+need\s+you\s+to\s+(?:help|support)\s+me\b",
        user_text,
        re.IGNORECASE,
    ):
        return False
    lowered = user_text.lower()
    ongoing_markers = (
        "business model",
        "cash flow",
        "cash-flow",
        "complex project",
        "high priority",
        "income",
        "mission",
        "multi-step",
        "multistep",
        "productive routine",
        "proactive",
        "start making money",
        "start making income",
    )
    return any(marker in lowered for marker in ongoing_markers)


def _normalize_assistant_support_target(user_text: str) -> Optional[str]:
    lowered = user_text.lower()
    if (
        "productive routine" in lowered
        and ("business model" in lowered or "income" in lowered or "cash flow" in lowered or "cash-flow" in lowered)
    ):
        return (
            "the operator with developing a productive routine and realistic "
            "non-employee income/business model"
        )

    requested_object = _extract_requested_support_object(user_text)
    if requested_object:
        return f"the operator with {requested_object}"

    match = re.search(
        r"\bcan\s+you\s+be\s+my\s+assistant\b(?P<object>.*)$",
        user_text,
        re.IGNORECASE,
    )
    if match is None:
        return "the operator with the requested ongoing support"
    target = match.group("object").strip(" ,.!?:;")
    target = re.sub(
        r"^(?:by\s+)?(?:helping\s+(?:me\s+)?(?:with|to)|figuring\s+out|working\s+on|for)\s+",
        "",
        target,
        flags=re.IGNORECASE,
    ).strip(" ,.!?:;")
    if not target:
        return "the operator with the requested ongoing support"
    return f"the operator with {target}"


def _extract_requested_support_object(user_text: str) -> Optional[str]:
    match = re.search(
        r"\b(?:can|could|will|would)\s+you\s+(?:help|support)\s+me\b(?P<object>[^.!?]*)"
        r"|\bi\s+need\s+you\s+to\s+(?:help|support)\s+me\b(?P<object2>[^.!?]*)",
        user_text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    target = (match.group("object") or match.group("object2") or "").strip(" ,.!?:;")
    target = re.sub(
        r"^(?:by\s+)?(?:helping\s+(?:me\s+)?(?:with|to)|with|on|to|for)\s+",
        "",
        target,
        flags=re.IGNORECASE,
    ).strip(" ,.!?:;")
    if not target or target.lower() in _PRONOUN_OBJECTS:
        return None
    if len(target.split()) < 2:
        return None
    return target


def _extract_explicit_promise(
    sentence: str,
    next_sentence: Optional[str],
) -> Optional[SelfCommitmentCandidate]:
    match = _PROMISE_PREFIX_RE.match(sentence)
    if match is None:
        return None
    promise_text = match.group("object").strip(" .,!?:;")
    source_sentence = sentence.strip()
    if not promise_text and next_sentence:
        promise_text = next_sentence.strip(" .,!?:;")
        source_sentence = f"{source_sentence} {next_sentence.strip()}"
    if not promise_text:
        return None

    content = _normalize_explicit_promise(promise_text)
    if not content:
        return None
    return SelfCommitmentCandidate(
        content=content,
        trigger="promise",
        source_sentence=source_sentence,
        confidence=0.84,
        normalization_source="explicit_promise",
    )


def _normalize_explicit_promise(value: str) -> Optional[str]:
    cleaned = _clean_commitment_content(value)
    if len(cleaned.split()) < 3:
        return None
    lowered = cleaned.lower()
    if _SOURCE_GROUNDING_PROMISE_RE.search(lowered):
        return _clean_commitment_content(
            f"Honor source-grounding promise: {_lowercase_first(cleaned)}"
        )
    return _clean_commitment_content(f"Honor explicit promise: {_lowercase_first(cleaned)}")


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

        context_target = _normalize_context_target(previous_sentence)
        normalized_target: Optional[str] = None
        normalization_source = "direct_object"
        confidence = 0.9

        if rule.prefer_context and context_target:
            normalized_target = context_target
            normalization_source = "prior_sentence_context"
            confidence = 0.72
        elif raw_object:
            normalized_target = _normalize_direct_target(raw_object)

        if normalized_target is None:
            normalized_target = context_target
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
    came_from_bullet = bool(re.match(r"^[-*]\s*", candidate))
    candidate = re.sub(r"^[-*]\s*", "", candidate).strip(" .,!?:;")
    candidate = candidate.strip("*_` ")
    if re.fullmatch(r"(?:today|tomorrow|tonight)?\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?", candidate, flags=re.IGNORECASE):
        return None
    candidate = _LEADING_CONTEXT_RE.sub("", candidate).strip(" .,!?:;")
    candidate = _TRAILING_RATIONALE_RE.sub("", candidate).strip(" .,!?:;")
    if not candidate:
        return None
    if came_from_bullet:
        candidate = _lowercase_bullet_label(candidate)
    return candidate


def _clean_commitment_content(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" .")
    return _capitalize_first(cleaned)


def _lowercase_bullet_label(value: str) -> str:
    if not re.match(r"^[A-Z][a-z]+(?:\s|$)", value):
        return value
    first_word = value.split(maxsplit=1)[0]
    if first_word in {"Bulma", "Edge", "Jarrod", "OpenCAS"}:
        return value
    return value[0].lower() + value[1:]


def _capitalize_first(value: str) -> str:
    if not value:
        return value
    return value[0].upper() + value[1:]


def _lowercase_first(value: str) -> str:
    if not value:
        return value
    return value[0].lower() + value[1:]


def _has_deferral_cue(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(
        cue in lowered
        for cue in ("later", "soon", "tomorrow", "tonight", "after ", "when ", "once ", "if ", "in a while")
    )
