"""Tests for daydream reflection evaluator."""

import pytest

from opencas.daydream import DaydreamReflection, ReflectionEvaluator
from opencas.identity import IdentityManager, IdentityStore


@pytest.fixture
def evaluator():
    return ReflectionEvaluator()


@pytest.fixture
def identity(tmp_path):
    store = IdentityStore(tmp_path)
    im = IdentityManager(store)
    im.seed_defaults()
    return im


def test_score_alignment_no_identity(evaluator):
    reflection = DaydreamReflection(spark_content="test spark")
    score = evaluator.score_alignment(reflection, None)
    assert score == 0.5


def test_score_alignment_with_identity(evaluator, identity):
    reflection = DaydreamReflection(
        spark_content="I value clarity and growth",
        synthesis="growth",
        interpretation="clarity",
    )
    score = evaluator.score_alignment(reflection, identity)
    assert score > 0.2
    assert reflection.alignment_score == score


def test_score_novelty_empty_history(evaluator):
    reflection = DaydreamReflection(spark_content="new idea")
    score = evaluator.score_novelty(reflection, [])
    assert score == 1.0
    assert reflection.novelty_score == 1.0


def test_score_novelty_penalizes_overlap(evaluator):
    reflection = DaydreamReflection(spark_content="the quick brown fox")
    recent = ["the quick brown dog"]
    score = evaluator.score_novelty(reflection, recent)
    assert score < 1.0
    assert reflection.novelty_score == score


def test_decide_keeper_passes(evaluator):
    reflection = DaydreamReflection(spark_content="spark")
    reflection.alignment_score = 0.5
    reflection.novelty_score = 0.5
    assert evaluator.decide_keeper(reflection) is True
    assert reflection.keeper is True


def test_decide_keeper_fails_alignment(evaluator):
    reflection = DaydreamReflection(spark_content="spark")
    reflection.alignment_score = 0.1
    reflection.novelty_score = 0.5
    assert evaluator.decide_keeper(reflection) is False
    assert reflection.keeper is False


def test_decide_keeper_fails_novelty(evaluator):
    reflection = DaydreamReflection(spark_content="spark")
    reflection.alignment_score = 0.5
    reflection.novelty_score = 0.1
    assert evaluator.decide_keeper(reflection) is False
    assert reflection.keeper is False


def test_detect_conflicts_obligation_vs_curiosity(evaluator):
    reflection = DaydreamReflection(
        spark_content="I should study but I want to play",
    )
    conflicts = evaluator.detect_conflicts(reflection)
    kinds = [c[0] for c in conflicts]
    assert "obligation_vs_curiosity" in kinds


def test_detect_conflicts_closeness_vs_distance(evaluator):
    reflection = DaydreamReflection(
        spark_content="I feel close yet so alone and distant",
    )
    conflicts = evaluator.detect_conflicts(reflection)
    kinds = [c[0] for c in conflicts]
    assert "closeness_vs_distance" in kinds


def test_detect_conflicts_energy_vs_ambition(evaluator):
    reflection = DaydreamReflection(
        spark_content="I am tired but I still want to do it",
    )
    conflicts = evaluator.detect_conflicts(reflection)
    kinds = [c[0] for c in conflicts]
    assert "energy_vs_ambition" in kinds


def test_detect_conflicts_action_vs_avoidance(evaluator):
    reflection = DaydreamReflection(
        spark_content="I want to avoid this but I must face it",
    )
    conflicts = evaluator.detect_conflicts(reflection)
    kinds = [c[0] for c in conflicts]
    assert "action_vs_avoidance" in kinds


def test_detect_conflicts_none(evaluator):
    reflection = DaydreamReflection(spark_content="Everything is fine today.")
    conflicts = evaluator.detect_conflicts(reflection)
    assert conflicts == []
