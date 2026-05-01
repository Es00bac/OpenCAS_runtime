"""Shared candidate-map assembly and fusion helpers for ``MemoryRetriever``."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from opencas.context.models import RetrievalResult
from opencas.context.resonance import (
    compute_emotional_resonance,
    compute_reliability_score,
    compute_temporal_echo,
)
from opencas.context.retrieval_ranking import apply_temporal_decay
from opencas.context.retrieval_search import populate_graph_candidates, seed_graph_relational_scores

if TYPE_CHECKING:
    from opencas.context.retriever import MemoryRetriever


CandidateMap = Dict[Tuple[str, str], Dict[str, Any]]

_BASE_SIGNALS = (
    "semantic_score",
    "keyword_score",
    "recency_score",
    "salience_score",
    "emotional_resonance",
    "temporal_echo",
    "reliability",
    "relational_score",
)


def build_candidate_map(
    semantic_results: List[RetrievalResult],
    keyword_results: List[RetrievalResult],
    *,
    now: datetime,
    query_affect: Any,
) -> CandidateMap:
    """Create the initial candidate map from semantic and keyword seeds."""
    candidate_map: CandidateMap = {}

    def add_results(results: List[RetrievalResult], score_key: str) -> None:
        for result in results:
            key = (result.source_type, result.source_id)
            if key not in candidate_map:
                candidate_map[key] = {
                    "result": result,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "recency_score": 0.0,
                    "salience_score": 0.0,
                    "graph_score": 0.0,
                    "emotional_resonance": 0.0,
                    "temporal_echo": 0.0,
                    "reliability": 0.8,
                    "relational_score": 0.0,
                    "affective_pressure_score": 0.0,
                    "affective_pressure_reason": "",
                    "affective_action_pressure": "",
                }
            candidate_map[key][score_key] = result.score

            episode = getattr(result, "episode", None)
            memory = getattr(result, "memory", None)
            obj = episode or memory
            if obj is None:
                continue

            created = getattr(obj, "created_at", now)
            age_days = max(0.0, (now - created).total_seconds() / 86400.0)
            half_life = 180.0 if getattr(obj, "identity_core", False) else 60.0
            candidate_map[key]["recency_score"] = apply_temporal_decay(
                1.0,
                age_days,
                half_life_days=half_life,
            )
            salience = getattr(obj, "salience", 1.0)
            candidate_map[key]["salience_score"] = min(1.0, salience / 10.0)
            candidate_map[key]["episode"] = episode
            candidate_map[key]["memory"] = memory
            candidate_map[key]["embedding"] = result.embedding
            candidate_map[key]["emotional_resonance"] = compute_emotional_resonance(
                query_affect,
                getattr(episode, "affect", None),
            )
            candidate_map[key]["temporal_echo"] = compute_temporal_echo(now, created)
            candidate_map[key]["reliability"] = compute_reliability_score(
                getattr(obj, "used_successfully", 0),
                getattr(obj, "used_unsuccessfully", 0),
            )

    add_results(semantic_results, "semantic_score")
    add_results(keyword_results, "keyword_score")
    return candidate_map


def normalize_candidate_signals(candidate_map: CandidateMap, *, include_graph: bool = False) -> List[Tuple[str, str]]:
    """Normalize rank signals to ``[0, 1]`` and return ordered candidate keys."""
    keys = list(candidate_map.keys())
    signals = list(_BASE_SIGNALS)
    if include_graph:
        signals.append("graph_score")
    for signal in signals:
        values = _normalize([candidate_map[key][signal] for key in keys])
        for key, value in zip(keys, values):
            candidate_map[key][signal] = value
    return keys


async def expand_candidate_graph(
    retriever: "MemoryRetriever",
    candidate_map: CandidateMap,
    *,
    now: datetime,
) -> List[Tuple[str, str]]:
    """Expand graph neighbors and normalize graph-aware signals."""
    keys = list(candidate_map.keys())
    seed_results = [
        candidate_map[key]["result"]
        for key in keys
        if candidate_map[key]["graph_score"] == 0.0
    ]
    graph_results = await retriever._expand_graph(seed_results, decay=0.8)
    populate_graph_candidates(
        retriever,
        candidate_map,
        graph_results,
        now,
        seed_keys={(result.source_type, result.source_id) for result in seed_results},
    )
    return normalize_candidate_signals(candidate_map, include_graph=True)


def resolve_fusion_weights(
    defaults: Dict[str, float],
    overrides: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Merge operator-provided weight overrides onto the retriever defaults."""
    resolved = dict(defaults)
    for key, value in (overrides or {}).items():
        if key in resolved:
            resolved[key] = float(value)
    return resolved


def seed_relational_scores(retriever: "MemoryRetriever", candidate_map: CandidateMap) -> None:
    """Populate relational seed scores before signal normalization."""
    seed_graph_relational_scores(retriever, candidate_map)


def fuse_candidates(
    retriever: "MemoryRetriever",
    candidate_map: CandidateMap,
    *,
    keys: List[Tuple[str, str]],
    now: datetime,
    weights: Dict[str, float],
    adjustment: Any,
    min_confidence: float,
) -> tuple[List[RetrievalResult], List[Dict[str, Any]]]:
    """Fuse normalized candidate signals into ranked retrieval results."""
    fused: List[RetrievalResult] = []
    candidate_debug: List[Dict[str, Any]] = []

    for key in keys:
        candidate = candidate_map[key]
        base_score = sum(weights[name] * candidate.get(name, 0.0) for name in weights)

        somatic_bonus = 0.0
        if adjustment is not None:
            somatic_bonus = (
                candidate["recency_score"] * adjustment.recency_bonus
                + candidate["salience_score"] * adjustment.salience_bonus
                + candidate["emotional_resonance"] * adjustment.emotional_resonance_bonus
                + candidate["temporal_echo"] * adjustment.temporal_echo_bonus
                + candidate["graph_score"] * adjustment.graph_bonus
            )
        score = base_score + somatic_bonus

        reliability_multiplier = 0.7 + 0.3 * candidate["reliability"]
        score *= reliability_multiplier

        obj = candidate.get("episode") or candidate.get("memory")
        relational_multiplier = 1.0
        if retriever.relational_engine is not None and obj is not None:
            tags = getattr(obj, "tags", []) or []
            musubi_mod = retriever.relational_engine.to_memory_salience_modifier(
                has_user_collab_tag=("collab" in tags)
            )
            relational_multiplier = 1.0 + musubi_mod
            score *= relational_multiplier

        episode_obj = candidate.get("episode")
        confidence_multiplier = 1.0
        if episode_obj is not None:
            confidence = getattr(episode_obj, "confidence_score", 0.8)
            if confidence < 0.5:
                confidence_multiplier = 0.6
            elif confidence <= 0.8:
                confidence_multiplier = 0.8
            score *= confidence_multiplier

        created = getattr(obj, "created_at", now) if obj is not None else now
        age_days = max(0.0, (now - created).total_seconds() / 86400.0)
        affect = getattr(getattr(candidate.get("episode"), "affect", None), "primary_emotion", None)
        embedding = candidate.get("embedding")
        embedding_dimension = len(embedding) if isinstance(embedding, list) else None
        candidate_debug.append(
            {
                "source_type": key[0],
                "source_id": key[1],
                "content": candidate["result"].content,
                "kind": getattr(getattr(candidate.get("episode"), "kind", None), "value", "memory"),
                "session_id": getattr(candidate.get("episode"), "session_id", None),
                "created_at": created.isoformat(),
                "age_days": round(age_days, 3),
                "salience": getattr(obj, "salience", None),
                "confidence_score": getattr(candidate.get("episode"), "confidence_score", None),
                "used_successfully": getattr(obj, "used_successfully", None),
                "used_unsuccessfully": getattr(obj, "used_unsuccessfully", None),
                "somatic_tag": getattr(candidate.get("episode"), "somatic_tag", None),
                "primary_emotion": getattr(affect, "value", None),
                "embedding_dimension": embedding_dimension,
                "semantic_score": round(float(candidate["semantic_score"]), 6),
                "keyword_score": round(float(candidate["keyword_score"]), 6),
                "recency_score": round(float(candidate["recency_score"]), 6),
                "salience_score": round(float(candidate["salience_score"]), 6),
                "graph_score": round(float(candidate["graph_score"]), 6),
                "emotional_resonance": round(float(candidate.get("emotional_resonance", 0.0)), 6),
                "temporal_echo": round(float(candidate.get("temporal_echo", 0.0)), 6),
                "reliability": round(float(candidate.get("reliability", 0.0)), 6),
                "relational_score": round(float(candidate.get("relational_score", 0.0)), 6),
                "affective_pressure_score": round(float(candidate.get("affective_pressure_score", 0.0)), 6),
                "affective_action_pressure": candidate.get("affective_action_pressure", ""),
                "affective_pressure_reason": candidate.get("affective_pressure_reason", ""),
                "base_score": round(float(base_score), 6),
                "somatic_bonus": round(float(somatic_bonus), 6),
                "reliability_multiplier": round(float(reliability_multiplier), 6),
                "relational_multiplier": round(float(relational_multiplier), 6),
                "confidence_multiplier": round(float(confidence_multiplier), 6),
                "final_score": round(float(score), 6),
                "passed_min_confidence": bool(score >= min_confidence),
            }
        )

        if score < min_confidence:
            continue

        result = candidate["result"]
        fused.append(
            RetrievalResult(
                source_type=result.source_type,
                source_id=result.source_id,
                content=result.content,
                score=score,
                episode=candidate.get("episode"),
                memory=candidate.get("memory"),
                embedding=candidate.get("embedding"),
            )
        )

    return fused, candidate_debug


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return values
    min_value = min(values)
    max_value = max(values)
    span = max_value - min_value
    if span == 0.0:
        return [0.5] * len(values)
    return [(value - min_value) / span for value in values]
