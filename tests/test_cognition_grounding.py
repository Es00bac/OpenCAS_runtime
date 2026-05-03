"""Tests for grounded cognition evidence primitives."""

from opencas.cognition import CognitionGrounding, GroundingKind, GroundingSource


def test_grounding_records_claim_source_and_confidence() -> None:
    grounding = CognitionGrounding(
        kind=GroundingKind.LEARNED,
        source=GroundingSource.MEMORY,
        claim="Writing-project returns work better as BAA tasks than reminders.",
        subject="tool_use",
        confidence=0.82,
        evidence_ids=["episode-1", "outcome-2"],
        allowed_surface="internal",
    )

    payload = grounding.model_dump(mode="json")

    assert payload["kind"] == "learned"
    assert payload["source"] == "memory"
    assert payload["claim"] == (
        "Writing-project returns work better as BAA tasks than reminders."
    )
    assert payload["subject"] == "tool_use"
    assert payload["confidence"] == 0.82
    assert payload["evidence_ids"] == ["episode-1", "outcome-2"]
    assert payload["allowed_surface"] == "internal"


def test_grounding_confidence_is_validated() -> None:
    grounding = CognitionGrounding(
        kind=GroundingKind.SOMATIC_METRIC,
        source=GroundingSource.SOMATIC,
        claim="Tension metric is elevated.",
        confidence=1.7,
    )

    assert grounding.confidence == 1.0
