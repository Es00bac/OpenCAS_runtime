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
    for match in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", query):
        terms.append(match.group(0))
    return terms


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
