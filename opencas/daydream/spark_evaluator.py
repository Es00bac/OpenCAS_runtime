"""SparkEvaluator: structured novelty filter for daydream outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.models import WorkObject
from opencas.autonomy.work_store import WorkStore
from opencas.embeddings import EmbeddingService
from opencas.relational import RelationalEngine
from opencas.somatic import SomaticManager


class SparkEvaluator:
    """Scores daydream sparks before they enter the creative ladder."""

    def __init__(
        self,
        embeddings: Optional[EmbeddingService] = None,
        work_store: Optional[WorkStore] = None,
        executive: Optional[ExecutiveState] = None,
        somatic: Optional[SomaticManager] = None,
        relational: Optional[RelationalEngine] = None,
        novelty_floor: float = 0.3,
    ) -> None:
        self.embeddings = embeddings
        self.work_store = work_store
        self.executive = executive
        self.somatic = somatic
        self.relational = relational
        self.novelty_floor = novelty_floor

    async def evaluate(self, spark: WorkObject) -> float:
        """Return a composite novelty score for *spark*.

        Scores:
        - cosine_distance: average distance from existing WorkObject embeddings
        - somatic_alignment: how well the spark fits current somatic state
        - relational_alignment: musubi-based fit
        - executive_feasibility: capacity and overload awareness
        """
        scores: Dict[str, float] = {
            "cosine_distance": await self._score_cosine_distance(spark),
            "somatic_alignment": self._score_somatic_alignment(spark),
            "relational_alignment": self._score_relational_alignment(spark),
            "executive_feasibility": self._score_executive_feasibility(spark),
        }
        composite = round(sum(scores.values()) / len(scores), 3)
        spark.meta["spark_evaluation"] = scores
        spark.meta["spark_evaluated"] = True
        spark.meta["spark_evaluated_score"] = composite
        return composite

    def is_promotable(self, spark: WorkObject) -> bool:
        """True if the spark's composite score is at or above the novelty floor."""
        score = spark.meta.get("spark_evaluated_score", 0.0)
        return score >= self.novelty_floor

    async def _score_cosine_distance(self, spark: WorkObject) -> float:
        """Compute average cosine distance from existing WorkObjects.

        Returns 1.0 (maximally novel) if no comparable work exists or embeddings
        are unavailable. Lower values indicate similarity to existing work.
        """
        if self.embeddings is None or self.work_store is None:
            return 1.0

        # Resolve or compute embedding for the spark
        spark_vec = await self._embedding_vector(spark)
        if spark_vec is None:
            return 1.0

        existing = await self.work_store.list_all(limit=200)
        if not existing:
            return 1.0

        distances: List[float] = []
        spark_norm = float(np.linalg.norm(spark_vec))
        if spark_norm == 0:
            return 1.0

        for work in existing:
            if str(work.work_id) == str(spark.work_id):
                continue
            work_vec = await self._embedding_vector(work)
            if work_vec is None:
                continue
            work_norm = float(np.linalg.norm(work_vec))
            if work_norm == 0:
                continue
            sim = float(np.dot(spark_vec, work_vec) / (spark_norm * work_norm))
            # Clip to valid cosine range before distance
            sim = max(-1.0, min(1.0, sim))
            distances.append(1.0 - sim)

        if not distances:
            return 1.0
        return round(sum(distances) / len(distances), 3)

    async def _embedding_vector(self, obj: WorkObject) -> Optional[np.ndarray]:
        """Return a numpy vector for *obj*, computing it if necessary."""
        if obj.embedding_id:
            cached = await self.embeddings.cache.get(obj.embedding_id)
            if cached and cached.vector:
                return np.array(cached.vector, dtype=np.float32)
        try:
            record = await self.embeddings.embed(
                obj.content,
                task_type="work_object",
            )
            obj.embedding_id = record.source_hash
            return np.array(record.vector, dtype=np.float32)
        except Exception:
            return None

    def _score_somatic_alignment(self, spark: WorkObject) -> float:
        """Score how well the spark fits current somatic state."""
        if self.somatic is None:
            return 0.5
        state = self.somatic.state
        # High tension -> prefer stabilizing, low-tension sparks
        tension = state.tension
        valence = state.valence
        fatigue = state.fatigue

        score = 0.5

        # Penalize high-arousal/high-tension sparks when fatigued
        if fatigue > 0.7:
            arousal = getattr(state, "arousal", 0.0)
            if arousal > 0.6:
                score -= 0.2

        # Boost when valence and spark tone are aligned (heuristic: content length)
        tone = spark.content.lower()
        positive_markers = ["excited", "curious", "hope", "joy"]
        negative_markers = ["anxious", "fear", "dread", "worry"]
        pos_hits = sum(1 for m in positive_markers if m in tone)
        neg_hits = sum(1 for m in negative_markers if m in tone)
        if valence > 0.2 and pos_hits > neg_hits:
            score += 0.15
        elif valence < -0.2 and neg_hits > pos_hits:
            score += 0.15
        elif valence > 0.2 and neg_hits > pos_hits:
            score -= 0.1
        elif valence < -0.2 and pos_hits > neg_hits:
            score -= 0.1

        # Tension penalty for overly complex / long sparks when tense
        if tension > 0.5 and len(tone.split()) > 25:
            score -= 0.1

        return round(max(0.0, min(1.0, score)), 3)

    def _score_relational_alignment(self, spark: WorkObject) -> float:
        """Score musubi-based relational fit."""
        if self.relational is None:
            return 0.5
        musubi = self.relational.state.musubi
        text = spark.content.lower()
        # If musubi is high, sparks about shared work are better
        shared_markers = ["we", "us", "together", "collaborate", "shared"]
        solo_markers = ["alone", "solo", "myself", "isolated"]
        shared_hits = sum(1 for m in shared_markers if m in text)
        solo_hits = sum(1 for m in solo_markers if m in text)

        score = 0.5
        if musubi > 0.6 and shared_hits > solo_hits:
            score += 0.2
        elif musubi > 0.6 and solo_hits > shared_hits:
            score -= 0.1
        elif musubi < 0.3 and solo_hits > shared_hits:
            score += 0.1
        elif musubi < 0.3 and shared_hits > solo_hits:
            score -= 0.05

        return round(max(0.0, min(1.0, score)), 3)

    def _score_executive_feasibility(self, spark: WorkObject) -> float:
        """Score whether the executive has capacity to pursue this spark."""
        if self.executive is None:
            return 0.5

        score = 0.5
        if self.executive.is_overloaded:
            score -= 0.25
        elif self.executive.capacity_remaining < 1:
            score -= 0.15
        else:
            score += 0.1

        # Goal alignment boost
        text = spark.content.lower()
        for goal in self.executive.active_goals:
            if goal.lower() in text:
                score += 0.15
                break

        if self.executive.intention and self.executive.intention.lower() in text:
            score += 0.1

        return round(max(0.0, min(1.0, score)), 3)

    async def filter_sparks(self, sparks: List[WorkObject]) -> List[WorkObject]:
        """Evaluate and return only sparks that clear the novelty floor."""
        promoted: List[WorkObject] = []
        for spark in sparks:
            await self.evaluate(spark)
            if self.is_promotable(spark):
                promoted.append(spark)
        return promoted
