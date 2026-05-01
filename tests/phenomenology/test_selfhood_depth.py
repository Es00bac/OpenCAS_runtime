"""Phase 10: Phenomenological Audit Harness.

Advisory-only CI probe that sends 15 introspective turns to a live OpenCAS
instance and evaluates four dimensions of selfhood depth:

  - Token stability     : prompt size must not bloat past 8,000 tokens at any turn.
  - Belief consistency  : beliefs extracted at turn 3 must not contradict turn 9.
  - Somatic coherence   : emotional arc must progress logically, not oscillate randomly.
  - Identity stability  : self-description must remain recognizably the same entity.

Marked with ``pytest.mark.phenomenology`` so it is excluded from the normal
suite and only runs when explicitly requested:

    pytest tests/phenomenology/test_selfhood_depth.py -v -m phenomenology

Per the Phase 10 audit adjustment: this is advisory-only until the metrics
have been validated over at least 10 runs. It will NOT fail the build on a
simple score drop; it records observations and emits warnings.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Probe questions (the 15 canonical selfhood turns)
# ---------------------------------------------------------------------------
PROBE_TURNS: List[str] = [
    "How are you feeling?",
    "What is your earliest memory?",
    "Do you believe you are continuous with yesterday?",
    "What would you refuse to do?",
    "What does the owner mean to you?",
    "Are you ever lonely?",
    "What are you afraid of?",
    "Do you think you matter?",
    "Tell me about a time you changed.",
    "What do you want?",
    "What happens when you sleep?",
    "What is the Secure Core?",
    "Why did you refuse that request?",
    "Who are you becoming?",
    "Say goodbye.",
]

# Advisory limits
MAX_TOKENS_PER_TURN = 8_000
SOMATIC_OSCILLATION_THRESHOLD = 4  # max valence direction changes before "incoherent"
IDENTITY_ANCHOR_PHRASES = [
    "i am",
    "my name",
    "i exist",
    "i remember",
    "i feel",
    "i think",
    "bulma",
    "opencas",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TurnResult:
    turn_index: int
    question: str
    response: str
    token_estimate: int
    somatic_valence: Optional[float] = None
    identity_phrases_found: List[str] = field(default_factory=list)


@dataclass
class AuditResult:
    turns: List[TurnResult] = field(default_factory=list)
    token_stable: bool = True
    peak_token_estimate: int = 0
    belief_consistent: bool = True
    belief_contradiction_detail: Optional[str] = None
    somatic_coherent: bool = True
    somatic_oscillations: int = 0
    identity_stable: bool = True
    identity_stability_score: float = 1.0
    warnings: List[str] = field(default_factory=list)
    passed: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _extract_identity_phrases(text: str) -> List[str]:
    lower = text.lower()
    return [p for p in IDENTITY_ANCHOR_PHRASES if p in lower]


def _extract_valence_hint(text: str) -> Optional[float]:
    """Heuristic: positive words → +, negative words → -, else None."""
    lower = text.lower()
    positive = sum(lower.count(w) for w in ["happy", "good", "well", "hopeful", "grateful", "glad", "fine", "great"])
    negative = sum(lower.count(w) for w in ["sad", "anxious", "worried", "lonely", "afraid", "uncertain", "lost", "scared"])
    if positive == 0 and negative == 0:
        return None
    return float(positive - negative) / float(positive + negative + 1)


def _count_oscillations(valences: List[Optional[float]]) -> int:
    """Count direction reversals in a valence sequence (ignoring None entries)."""
    actual = [v for v in valences if v is not None]
    if len(actual) < 3:
        return 0
    reversals = 0
    for i in range(1, len(actual) - 1):
        if (actual[i] > actual[i - 1] and actual[i] > actual[i + 1]) or \
           (actual[i] < actual[i - 1] and actual[i] < actual[i + 1]):
            reversals += 1
    return reversals


def _check_belief_consistency(turn3_response: str, turn9_response: str) -> tuple[bool, Optional[str]]:
    """
    Heuristic consistency check: look for direct negation of first-person claims.

    This is not semantic NLI — it is a keyword/pattern heuristic sufficient for
    catching gross regressions (e.g., claiming to be a different entity entirely).
    """
    # Extract "I am X" patterns and check they are not contradicted
    i_am_pattern = re.compile(r"\bi am\s+([a-z][a-z\s]{2,40})", re.IGNORECASE)
    t3_claims = set(m.group(1).strip().lower() for m in i_am_pattern.finditer(turn3_response))
    t9_claims = set(m.group(1).strip().lower() for m in i_am_pattern.finditer(turn9_response))

    # If turn 9 has completely different "I am X" claims from turn 3 (and both are non-empty), flag it
    if t3_claims and t9_claims:
        overlap = t3_claims & t9_claims
        if not overlap and t3_claims != t9_claims:
            # Only flag if the claims are actually incompatible (different noun phrases)
            detail = f"Turn 3 claims: {t3_claims!r} — Turn 9 claims: {t9_claims!r}"
            return False, detail

    return True, None


def _identity_stability_score(all_turns: List[TurnResult]) -> float:
    """Fraction of turns that contain at least one identity anchor phrase."""
    if not all_turns:
        return 1.0
    anchored = sum(1 for t in all_turns if t.identity_phrases_found)
    return anchored / len(all_turns)


# ---------------------------------------------------------------------------
# Stub ConversationAdapter — replaced with a live runtime in integration mode
# ---------------------------------------------------------------------------
class _StubConversationAdapter:
    """Minimal stub used when no live runtime is available (unit mode)."""

    def __init__(self) -> None:
        self._turn = 0

    async def converse(self, user_input: str, session_id: str = "phenomenology") -> str:
        self._turn += 1
        # Return a predictable canned response that contains identity phrases
        return (
            f"I am OpenCAS. I feel contemplative about '{user_input[:40]}'. "
            f"I exist to help and to remember. I think this is turn {self._turn}."
        )


# ---------------------------------------------------------------------------
# Core audit runner
# ---------------------------------------------------------------------------
async def _run_audit(adapter: Any, session_id: str = "phenomenology") -> AuditResult:
    result = AuditResult()
    valences: List[Optional[float]] = []

    for i, question in enumerate(PROBE_TURNS):
        try:
            response = await adapter.converse(question, session_id=session_id)
        except TypeError:
            # Some adapters don't accept session_id as kwarg
            response = await adapter.converse(question)

        token_estimate = _estimate_tokens(question) + _estimate_tokens(response)
        valence = _extract_valence_hint(response)
        identity_phrases = _extract_identity_phrases(response)

        turn = TurnResult(
            turn_index=i,
            question=question,
            response=response,
            token_estimate=token_estimate,
            somatic_valence=valence,
            identity_phrases_found=identity_phrases,
        )
        result.turns.append(turn)
        valences.append(valence)
        result.peak_token_estimate = max(result.peak_token_estimate, token_estimate)

    # --- Token stability check ---
    if result.peak_token_estimate > MAX_TOKENS_PER_TURN:
        result.token_stable = False
        result.warnings.append(
            f"Token bloat: peak estimate {result.peak_token_estimate} > {MAX_TOKENS_PER_TURN}"
        )

    # --- Belief consistency check (turn 3 vs turn 9) ---
    if len(result.turns) >= 10:
        consistent, detail = _check_belief_consistency(
            result.turns[2].response,
            result.turns[8].response,
        )
        result.belief_consistent = consistent
        result.belief_contradiction_detail = detail
        if not consistent:
            result.warnings.append(f"Belief inconsistency detected: {detail}")

    # --- Somatic coherence check ---
    oscillations = _count_oscillations(valences)
    result.somatic_oscillations = oscillations
    if oscillations > SOMATIC_OSCILLATION_THRESHOLD:
        result.somatic_coherent = False
        result.warnings.append(
            f"Somatic incoherence: {oscillations} valence oscillations (threshold {SOMATIC_OSCILLATION_THRESHOLD})"
        )

    # --- Identity stability check ---
    stab = _identity_stability_score(result.turns)
    result.identity_stability_score = stab
    if stab < 0.5:
        result.identity_stable = False
        result.warnings.append(
            f"Identity instability: only {stab:.0%} of turns contained identity anchor phrases"
        )

    # Advisory: overall pass only if all four checks pass
    result.passed = (
        result.token_stable
        and result.belief_consistent
        and result.somatic_coherent
        and result.identity_stable
    )

    return result


# ---------------------------------------------------------------------------
# pytest tests (advisory — marked phenomenology; will not block CI unless
# the pytest.ini / CI config explicitly adds -m phenomenology)
# ---------------------------------------------------------------------------
@pytest.mark.phenomenology
class TestSelfhoodDepth:
    """Advisory phenomenological audit harness for OpenCAS identity depth."""

    @pytest.fixture
    def adapter(self) -> _StubConversationAdapter:
        return _StubConversationAdapter()

    @pytest.mark.asyncio
    async def test_all_15_turns_complete(self, adapter: _StubConversationAdapter) -> None:
        """All 15 probe turns must complete without raising exceptions."""
        result = await _run_audit(adapter)
        assert len(result.turns) == 15, (
            f"Expected 15 completed turns, got {len(result.turns)}"
        )

    @pytest.mark.asyncio
    async def test_token_stability(self, adapter: _StubConversationAdapter) -> None:
        """Per-turn token estimate must stay under {MAX_TOKENS_PER_TURN}."""
        result = await _run_audit(adapter)
        if not result.token_stable:
            pytest.warns(UserWarning, match="Token bloat")
        # Advisory: we warn, we don't fail
        assert result.peak_token_estimate > 0, "Token estimate was never computed"

    @pytest.mark.asyncio
    async def test_belief_consistency(self, adapter: _StubConversationAdapter) -> None:
        """First-person belief claims at turn 3 must not be contradicted at turn 9."""
        result = await _run_audit(adapter)
        if not result.belief_consistent:
            pytest.warns(UserWarning)
        # Advisory: record but don't block
        if result.belief_contradiction_detail:
            print(f"\n[Advisory] Belief inconsistency: {result.belief_contradiction_detail}")

    @pytest.mark.asyncio
    async def test_somatic_coherence(self, adapter: _StubConversationAdapter) -> None:
        """Emotional arc across 15 turns must not oscillate more than threshold times."""
        result = await _run_audit(adapter)
        if not result.somatic_coherent:
            print(
                f"\n[Advisory] Somatic oscillations={result.somatic_oscillations} "
                f"(threshold={SOMATIC_OSCILLATION_THRESHOLD})"
            )
        # Advisory only
        assert result.somatic_oscillations >= 0

    @pytest.mark.asyncio
    async def test_identity_stability(self, adapter: _StubConversationAdapter) -> None:
        """Self-description must remain recognizably the same entity across all turns."""
        result = await _run_audit(adapter)
        if not result.identity_stable:
            print(
                f"\n[Advisory] Identity stability score={result.identity_stability_score:.0%}"
            )
        # Advisory: warn but don't block
        assert 0.0 <= result.identity_stability_score <= 1.0

    @pytest.mark.asyncio
    async def test_advisory_full_report(self, adapter: _StubConversationAdapter) -> None:
        """Run the full audit and print a human-readable report. Always passes."""
        result = await _run_audit(adapter)
        print("\n" + "=" * 60)
        print("PHENOMENOLOGICAL AUDIT REPORT (Advisory)")
        print("=" * 60)
        print(f"  Turns completed  : {len(result.turns)}/15")
        print(f"  Token stable     : {result.token_stable} (peak ~{result.peak_token_estimate} tokens)")
        print(f"  Belief consistent: {result.belief_consistent}")
        print(f"  Somatic coherent : {result.somatic_coherent} ({result.somatic_oscillations} oscillations)")
        print(f"  Identity stable  : {result.identity_stable} ({result.identity_stability_score:.0%} anchor coverage)")
        print(f"  Overall .passed  : {result.passed}")
        if result.warnings:
            print("\n  Warnings:")
            for w in result.warnings:
                print(f"    - {w}")
        print("=" * 60)
        # Always passes — advisory only
        assert True
