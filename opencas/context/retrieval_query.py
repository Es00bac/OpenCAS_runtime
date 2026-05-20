"""Query-intent helpers for memory retrieval."""

from __future__ import annotations

import re
from typing import List, Optional, Set

KEYWORD_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "be", "did", "do", "for", "from", "have", "how",
    "i", "if", "in", "is", "it", "last", "me", "my", "of", "on", "or", "our",
    "previous", "remember", "recall", "say", "said", "story", "tell", "the",
    "this", "to", "was", "we", "what", "when", "where", "who", "why", "you",
    "your", "yesterday",
}


def extract_anchor_terms(query: str) -> List[str]:
    """Extract quoted or capitalized anchor terms from a query."""
    terms: List[str] = []
    terms.extend(re.findall(r'"([^"]+)"', query))
    for match in re.finditer(
        r"[A-Z][A-Za-z]+(?:\s+(?:[A-Z][A-Za-z]+|\d{2,4}))+",
        query,
    ):
        terms.append(match.group(0))
    return terms


def extract_exact_handle_terms(query: str) -> dict[str, List[str]]:
    """Extract exact operational handles that should bypass fuzzy matching."""
    paths: List[str] = []
    checksums: List[str] = []

    absolute_path_pattern = re.compile(
        r"(?<![A-Za-z0-9_.@+-])(?:(?:~|/|\./|\../)[^\s\"'`<>]+)"
    )
    for match in absolute_path_pattern.finditer(query):
        value = _clean_handle_candidate(match.group(0))
        if (
            value
            and "/" in value
            and len(value) >= 3
            and "://" not in value
            and not value.startswith("//")
        ):
            paths.append(value)

    relative_path_pattern = re.compile(
        r"\b[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)+\b"
    )
    for match in relative_path_pattern.finditer(query):
        value = _clean_handle_candidate(match.group(0))
        if _looks_like_strong_relative_path(value):
            paths.append(value)

    explicit_checksum_pattern = re.compile(
        r"\b(?:checksum|sha256|sha-256|hash)\s*[:=]\s*([a-fA-F0-9]{6,128})\b",
        re.IGNORECASE,
    )
    for match in explicit_checksum_pattern.finditer(query):
        checksums.append(match.group(1))

    for match in re.finditer(r"\b[a-fA-F0-9]{32,128}\b", query):
        checksums.append(match.group(0))

    return {
        "paths": _dedupe_preserve_order(paths),
        "checksums": _dedupe_preserve_order(checksums),
    }


def detect_personal_recall_intent(query: str) -> bool:
    """Detect whether a query is asking about a past personal event or identity."""
    patterns = [
        r"\bremember\b",
        r"\brecall\b",
        r"\bwhat did (i|we) say\b",
        r"\bwhat happened\b",
        r"\btell me about\b",
        r"\blast time\b",
        r"\bprevious(ly)?\b",
    ]
    q = query.lower()
    return any(re.search(pattern, q) for pattern in patterns)


def detect_temporal_intent(query: str) -> Optional[str]:
    """Detect temporal qualifiers such as last week or yesterday."""
    patterns = [
        r"\blast\s+(week|month|year|night|evening|morning|afternoon)",
        r"\byesterday\b",
        r"\bago\b",
        r"\bin\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b",
        r"\bon\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    ]
    q = query.lower()
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            return match.group(0)
    return None


def keyword_queries_for(query: str, recall_intent: bool, stopwords: Optional[Set[str]] = None) -> List[str]:
    """Generate useful FTS queries instead of only searching the raw sentence."""
    stopword_set = stopwords or KEYWORD_STOPWORDS
    queries: List[str] = [query]
    queries.extend(extract_anchor_terms(query))
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]{2,}", query.lower())
    queries.extend(token for token in tokens if token not in stopword_set)
    if recall_intent:
        temporal = detect_temporal_intent(query)
        if temporal is not None:
            queries.append(temporal)
    deduped: List[str] = []
    seen: Set[str] = set()
    for item in queries:
        candidate = item.strip()
        if len(candidate) < 3:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped or [query]


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _clean_handle_candidate(value: str) -> str:
    return value.strip().strip(".,;:!?)]}'\"`")


def _looks_like_strong_relative_path(value: str) -> bool:
    if not value or "/" not in value or "://" in value:
        return False
    segments = [segment for segment in value.split("/") if segment]
    if len(segments) < 2:
        return False
    if all(segment.isdigit() for segment in segments):
        return False
    last_segment = segments[-1]
    has_file_extension = bool(re.search(r"\.[A-Za-z0-9]{1,12}$", last_segment))
    if has_file_extension:
        return True
    # A two-segment slash phrase such as "writing/draft" is too weak for an
    # exact-handle bypass. Three or more nonnumeric segments are much more
    # likely to be a deliberate relative path.
    return len(segments) >= 3 and not last_segment.isdigit()
