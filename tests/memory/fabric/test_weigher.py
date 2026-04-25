"""Tests for EdgeWeigher."""

import pytest

from opencas.memory import EdgeKind
from opencas.memory.fabric.weigher import ContextProfile, EdgeWeigher


@pytest.mark.parametrize(
    "profile,scores,expected_kind",
    [
        (
            ContextProfile.CONSOLIDATION,
            {"conceptual": 0.9, "emotional": 0.2, "relational": 0.1, "temporal": 0.1, "causal": 0.1},
            EdgeKind.CONCEPTUAL,
        ),
        (
            ContextProfile.RETRIEVAL,
            {"conceptual": 0.1, "emotional": 0.9, "relational": 0.1, "temporal": 0.1, "causal": 0.1},
            EdgeKind.EMOTIONAL,
        ),
        (
            ContextProfile.CAUSAL_INFERENCE,
            {"conceptual": 0.1, "emotional": 0.1, "relational": 0.1, "temporal": 0.1, "causal": 0.9},
            EdgeKind.CAUSAL,
        ),
        (
            ContextProfile.CONSOLIDATION,
            {"conceptual": 0.1, "emotional": 0.1, "relational": 0.9, "temporal": 0.1, "causal": 0.1},
            EdgeKind.RELATIONAL,
        ),
        (
            ContextProfile.CONSOLIDATION,
            {"conceptual": 0.1, "emotional": 0.1, "relational": 0.1, "temporal": 0.9, "causal": 0.1},
            EdgeKind.TEMPORAL,
        ),
    ],
)
def test_fuse_elects_expected_kind(profile, scores, expected_kind):
    weigher = EdgeWeigher(profile=profile)
    result = weigher.fuse(scores)
    assert result["kind"] == expected_kind
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["weights"].keys()) == {"conceptual", "emotional", "relational", "temporal", "causal"}


def test_fuse_returns_semantic_on_tie():
    # When all scores are equal, max() returns the first key encountered.
    # In Python 3.7+ dicts preserve insertion order, so "conceptual" wins
    weigher = EdgeWeigher(profile=ContextProfile.CONSOLIDATION)
    result = weigher.fuse(
        {"conceptual": 0.5, "emotional": 0.5, "relational": 0.5, "temporal": 0.5, "causal": 0.5}
    )
    assert result["kind"] == EdgeKind.CONCEPTUAL


def test_different_profiles_produce_different_confidences():
    scores = {"conceptual": 0.9, "emotional": 0.1, "relational": 0.1, "temporal": 0.1, "causal": 0.1}
    retrieval = EdgeWeigher(ContextProfile.RETRIEVAL).fuse(scores)
    consolidation = EdgeWeigher(ContextProfile.CONSOLIDATION).fuse(scores)
    assert retrieval["confidence"] != consolidation["confidence"]
