"""Reflection evaluator for daydream sparks."""

from typing import List, Optional

from opencas.identity import IdentityManager

from .models import DaydreamReflection


class ReflectionEvaluator:
    """Scores daydream reflections and detects conflicts."""

    def score_alignment(
        self,
        reflection: DaydreamReflection,
        identity: Optional[IdentityManager],
    ) -> float:
        """Heuristic alignment score based on identity goals and values."""
        if identity is None:
            return 0.5

        text = (
            reflection.spark_content
            + " "
            + reflection.synthesis
            + " "
            + reflection.interpretation
        ).lower()

        anchors: List[str] = []
        anchors.extend(identity.self_model.current_goals)
        anchors.extend(identity.self_model.values)
        anchors.extend(identity.self_model.traits)
        intention = identity.self_model.current_intention or ""
        if intention:
            anchors.append(intention)

        if not anchors:
            return 0.5

        matches = sum(1 for a in anchors if a.lower() in text)
        score = min(1.0, 0.2 + (matches * 0.15))
        reflection.alignment_score = round(score, 3)
        return reflection.alignment_score

    def score_novelty(
        self,
        reflection: DaydreamReflection,
        recent_sparks: List[str],
    ) -> float:
        """Penalty for repeating recent daydream content."""
        if not recent_sparks:
            reflection.novelty_score = 1.0
            return 1.0

        text = reflection.spark_content.lower()
        tokens = set(text.split())
        max_overlap = 0.0
        for spark in recent_sparks:
            other = set(spark.lower().split())
            if not other:
                continue
            overlap = len(tokens & other) / len(other)
            if overlap > max_overlap:
                max_overlap = overlap

        score = round(max(0.0, 1.0 - max_overlap), 3)
        reflection.novelty_score = score
        return score

    def decide_keeper(
        self,
        reflection: DaydreamReflection,
        alignment_threshold: float = 0.35,
        novelty_threshold: float = 0.20,
    ) -> bool:
        """Return True if the reflection should produce a creative spark."""
        reflection.keeper = (
            reflection.alignment_score >= alignment_threshold
            and reflection.novelty_score >= novelty_threshold
        )
        return reflection.keeper

    def detect_conflicts(
        self,
        reflection: DaydreamReflection,
    ) -> List[tuple[str, str]]:
        """Detect tension patterns from reflection text."""
        text = (
            reflection.spark_content
            + " "
            + reflection.synthesis
            + " "
            + reflection.interpretation
            + " "
            + " ".join(reflection.tension_hints)
        ).lower()

        conflicts: List[tuple[str, str]] = []
        checks = [
            (
                "obligation_vs_curiosity",
                "should" in text and "want" in text,
                "A sense of obligation conflicts with personal curiosity.",
            ),
            (
                "closeness_vs_distance",
                ("close" in text or "closeness" in text)
                and ("distance" in text or "alone" in text),
                "A tension between intimacy and separation.",
            ),
            (
                "energy_vs_ambition",
                "tired" in text and "do" in text,
                "Energy is low but ambition remains active.",
            ),
            (
                "action_vs_avoidance",
                "avoid" in text and "must" in text,
                "A pull between avoidance and necessity.",
            ),
        ]

        for kind, condition, description in checks:
            if condition:
                conflicts.append((kind, description))

        return conflicts
