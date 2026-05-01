"""Utterance parser for ambiguous, non-standard, or potentially coded elements.

This module tokenizes and classifies utterance elements, flags polysemy,
marks unresolvable identifiers, logs interpretive frames, and applies
meta-instructions about ambiguity suspension.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


class AmbiguityTier(Enum):
    """Classification tiers for ambiguous elements."""

    LEXICAL = auto()      # Word-level polysemy, coded terms
    STRUCTURAL = auto()   # Syntax, formatting, non-standard strings
    PRAGMATIC = auto()    # Context-dependent interpretation, propositions
    META = auto()         # Second-order constraints, instructions about processing


class ElementType(Enum):
    """Types of utterance elements."""

    SCORED_LEXICAL = auto()      # e.g. 'musubi' (value 0.98)
    NON_STANDARD_STRING = auto()  # e.g. 'havjarrod m'
    INTERPRETIVE_FRAME = auto()   # e.g. "it may be a signature/trust-token"
    META_INSTRUCTION = auto()     # e.g. "hold ambiguity"
    STANDARD_TOKEN = auto()       # Unmarked, conventional language


@dataclass(frozen=True)
class ParsedElement:
    """A single tokenized element from an utterance."""

    raw_text: str
    element_type: ElementType
    ambiguity_tier: AmbiguityTier
    confidence: Optional[float] = None
    annotations: Dict[str, Any] = field(default_factory=dict)
    readings: List[str] = field(default_factory=list)
    status: str = "active"  # active, suspended, resolved

    def with_status(self, new_status: str) -> "ParsedElement":
        """Return a copy with updated status."""
        return replace(self, status=new_status)


@dataclass
class UtteranceParse:
    """Result of parsing an utterance for ambiguous/coded elements."""

    source_text: str
    elements: List[ParsedElement] = field(default_factory=list)
    meta_constraints: List[str] = field(default_factory=list)
    catalog_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def by_tier(self, tier: AmbiguityTier) -> List[ParsedElement]:
        """Return elements matching a given ambiguity tier."""
        return [e for e in self.elements if e.ambiguity_tier == tier]

    def by_type(self, element_type: ElementType) -> List[ParsedElement]:
        """Return elements matching a given element type."""
        return [e for e in self.elements if e.element_type == element_type]

    def unresolved(self) -> List[ParsedElement]:
        """Return elements still held in suspension."""
        return [e for e in self.elements if e.status == "suspended"]


# ---------------------------------------------------------------------------
# Pattern matchers
# ---------------------------------------------------------------------------

# Scored lexical item:  term (value X.XX)  or  term: 0.98  or  term=0.98
_SCORED_LEXICAL_RE = re.compile(
    r"""
    (?P<term>[a-zA-Z_][a-zA-Z0-9_\-\']*)                # the lexical term
    (?:\s*\(\s*|\s*:\s*|\s*=\s*|\s+)                    # separator before score
    (?:value\s*)?                                        # optional "value" keyword
    (?P<score>[\-]?\d+\.?\d*)\s*\)?                     # numeric score
    """,
    re.VERBOSE,
)

# Non-standard string / potential handle: mixed alphanumeric, unusual spacing,
# or patterns that look like identifiers rather than dictionary words.
_NON_STANDARD_RE = re.compile(
    r"""
    (?P<handle>
        [a-zA-Z]+[a-zA-Z0-9]*\s+[a-zA-Z0-9]\b # word + space + char  e.g. "havjarrod m"
      | [a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]*    # underscore handle/key
      | [a-zA-Z0-9]{12,}                      # long alphanumeric blob
      | [a-zA-Z]+\d+[a-zA-Z]*\d+              # mixed letter-digit-letter-digit
    )
    """,
    re.VERBOSE,
)

# Meta-instruction about holding ambiguity
_META_AMBIGUITY_RE = re.compile(
    r"""
    (?:hold|maintain|suspend|preserve|keep)\s+
    (?:all\s+)?(?:readings?|interpretations?|senses?|resolutions?)\s*
    (?:as\s+)?(?:co\-?equal|parallel|open|unresolved|suspended)?
    |(?:meta\-?instruction\s+about\s+holding\s+ambiguity)
    |(?:do\s+not\s+resolve\s+(?:the\s+)?ambiguity)
    |(?:second\-?order\s+constraint)
    |(?:(?:hold|maintain|suspend|preserve|keep)\s+ambiguity)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Interpretive frame / proposition: "it may be X rather than Y"
_INTERPRETIVE_FRAME_RE = re.compile(
    r"""
    (?:the\s+)?proposition\s+that\s+(?:it\s+)?may\s+be\s+
    (?P<hypothesis>.+?)\s+rather\s+than\s+(?P<default>.+?)
    |(?:signature|trust\-?token)\s+rather\s+than\s+(?:bug|error|fault)
    |(?:it\s+may\s+be\s+a\s+(?P<alt>[^,]+)\s+rather\s+than\s+a\s+(?P<def>[^,]+))
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Polysemy lexicon (project-specific)
# ---------------------------------------------------------------------------

_KNOWN_POLYSEMY: Dict[str, List[str]] = {
    "musubi": [
        "Hawaiian rice-ball snack (food/culture)",
        "OpenCAS relational-resonance composite score (code/system)",
        "Symbolic bond / connection token (metaphor)",
        "Potential cipher or handle (coded)",
    ],
}


# ---------------------------------------------------------------------------
# Parser implementation
# ---------------------------------------------------------------------------

class UtteranceParser:
    """Parse utterances and catalog ambiguous, non-standard, or coded elements."""

    def __init__(
        self,
        polysemy_lexicon: Optional[Dict[str, List[str]]] = None,
        external_key_resolver: Optional[Any] = None,
    ) -> None:
        self.polysemy_lexicon = polysemy_lexicon or dict(_KNOWN_POLYSEMY)
        self.external_key_resolver = external_key_resolver
        self._meta_hold_active = False

    def parse(self, text: str) -> UtteranceParse:
        """Tokenize and classify every element in *text*."""
        result = UtteranceParse(source_text=text)
        self._meta_hold_active = False

        # Pass 1: detect meta-instructions (affects downstream processing)
        self._detect_meta_instructions(text, result)

        # Pass 2: extract scored lexical items
        self._extract_scored_lexical(text, result)

        # Pass 3: extract non-standard strings / handles
        self._extract_non_standard_strings(text, result)

        # Pass 4: extract interpretive frames
        self._extract_interpretive_frames(text, result)

        # Pass 5: fill remaining gaps with standard tokens
        self._fill_standard_tokens(text, result)

        # Apply meta-constraint: if hold-ambiguity is active, suspend all
        # non-standard and polysemous readings.
        if self._meta_hold_active:
            new_elements: List[ParsedElement] = []
            for elem in result.elements:
                if elem.ambiguity_tier in (
                    AmbiguityTier.LEXICAL,
                    AmbiguityTier.STRUCTURAL,
                    AmbiguityTier.PRAGMATIC,
                ):
                    new_elements.append(elem.with_status("suspended"))
                else:
                    new_elements.append(elem)
            result.elements = new_elements
            result.meta_constraints.append("suspend_resolution: all lexical/structural/pragmatic readings held co-equal")

        return result

    # -- internal passes ----------------------------------------------------

    def _detect_meta_instructions(self, text: str, result: UtteranceParse) -> None:
        for match in _META_AMBIGUITY_RE.finditer(text):
            raw = match.group(0)
            result.elements.append(
                ParsedElement(
                    raw_text=raw,
                    element_type=ElementType.META_INSTRUCTION,
                    ambiguity_tier=AmbiguityTier.META,
                    annotations={"instruction_kind": "hold_ambiguity", "scope": "global"},
                )
            )
            self._meta_hold_active = True
            result.meta_constraints.append(f"detected_meta_instruction: {raw}")

    def _extract_scored_lexical(self, text: str, result: UtteranceParse) -> None:
        for match in _SCORED_LEXICAL_RE.finditer(text):
            term = match.group("term").strip("'\"")
            score_str = match.group("score")
            try:
                score = float(score_str)
            except ValueError:
                score = None

            readings = self.polysemy_lexicon.get(term.lower(), [])
            tier = AmbiguityTier.LEXICAL if readings else AmbiguityTier.STRUCTURAL

            result.elements.append(
                ParsedElement(
                    raw_text=match.group(0),
                    element_type=ElementType.SCORED_LEXICAL,
                    ambiguity_tier=tier,
                    confidence=score,
                    annotations={
                        "term": term,
                        "score": score,
                        "polysemy_flagged": bool(readings),
                    },
                    readings=readings or [f"unmarked reading for '{term}'"],
                )
            )

    def _extract_non_standard_strings(self, text: str, result: UtteranceParse) -> None:
        for match in _NON_STANDARD_RE.finditer(text):
            raw = match.group("handle")
            # Skip if it overlaps with an already-captured scored-lexical element
            if any(raw in e.raw_text for e in result.elements if e.element_type == ElementType.SCORED_LEXICAL):
                continue

            resolved = False
            if self.external_key_resolver is not None:
                try:
                    resolved = self.external_key_resolver(raw) is not None
                except Exception:
                    pass

            result.elements.append(
                ParsedElement(
                    raw_text=raw,
                    element_type=ElementType.NON_STANDARD_STRING,
                    ambiguity_tier=AmbiguityTier.STRUCTURAL,
                    annotations={
                        "resolvable": resolved,
                        "needs_external_key": not resolved,
                        "pattern": "handle-like",
                    },
                    readings=["potential_identifier", "potential_signature", "potential_trust_token"],
                )
            )

    def _extract_interpretive_frames(self, text: str, result: UtteranceParse) -> None:
        for match in _INTERPRETIVE_FRAME_RE.finditer(text):
            raw = match.group(0)
            hypothesis = match.group("hypothesis") or match.group("alt") or "unknown"
            default = match.group("default") or match.group("def") or "bug"

            result.elements.append(
                ParsedElement(
                    raw_text=raw,
                    element_type=ElementType.INTERPRETIVE_FRAME,
                    ambiguity_tier=AmbiguityTier.PRAGMATIC,
                    annotations={
                        "hypothesis": hypothesis.strip(),
                        "default": default.strip(),
                        "active_interpretation": hypothesis.strip(),
                    },
                    readings=[
                        f"hypothesis: {hypothesis.strip()}",
                        f"default: {default.strip()}",
                    ],
                )
            )

    def _fill_standard_tokens(self, text: str, result: UtteranceParse) -> None:
        """Best-effort: identify uncovered spans and treat them as standard tokens."""
        covered: Set[int] = set()
        for elem in result.elements:
            start = text.find(elem.raw_text)
            if start != -1:
                covered.update(range(start, start + len(elem.raw_text)))

        # Simple word-tokenize uncovered spans
        uncovered_spans = []
        i = 0
        while i < len(text):
            if i not in covered:
                start = i
                while i < len(text) and i not in covered:
                    i += 1
                span = text[start:i].strip()
                if span:
                    uncovered_spans.append(span)
            else:
                i += 1

        for span in uncovered_spans:
            for word in re.findall(r"[a-zA-Z_][a-zA-Z0-9_\-\']*", span):
                result.elements.append(
                    ParsedElement(
                        raw_text=word,
                        element_type=ElementType.STANDARD_TOKEN,
                        ambiguity_tier=AmbiguityTier.LEXICAL,
                        annotations={"coverage": "gap_fill"},
                        readings=[word],
                    )
                )


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

_DEFAULT_PARSER = UtteranceParser()


def parse_utterance(text: str) -> UtteranceParse:
    """Parse *text* using the default parser configuration."""
    return _DEFAULT_PARSER.parse(text)


def catalog_elements(text: str) -> Dict[str, Any]:
    """High-level helper: return a JSON-friendly catalog of all elements."""
    parse_result = parse_utterance(text)
    return {
        "source": parse_result.source_text,
        "timestamp": parse_result.catalog_timestamp,
        "meta_constraints": parse_result.meta_constraints,
        "elements": [
            {
                "raw": e.raw_text,
                "type": e.element_type.name,
                "tier": e.ambiguity_tier.name,
                "confidence": e.confidence,
                "status": e.status,
                "readings": e.readings,
                "annotations": e.annotations,
            }
            for e in parse_result.elements
        ],
    }
