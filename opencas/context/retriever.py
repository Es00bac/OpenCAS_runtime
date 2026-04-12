"""Hybrid memory retrieval: multi-signal fusion with MMR and temporal decay."""

from __future__ import annotations

import math
import re
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from typing import TYPE_CHECKING

from opencas.embeddings import EmbeddingService
from opencas.memory import MemoryStore
from opencas.somatic.models import AffectState

from .models import RetrievalResult
from .resonance import (
    compute_edge_strength,
    compute_emotional_resonance,
    compute_reliability_score,
    compute_temporal_echo,
)

if TYPE_CHECKING:
    from opencas.memory.fabric.graph import EpisodeGraph
    from opencas.relational import RelationalEngine
    from opencas.somatic import SomaticManager


class MemoryRetriever:
    """Retrieve relevant context by fusing semantic, keyword, recency, salience, and graph signals."""

    DEFAULT_FUSION_WEIGHTS: Dict[str, float] = {
        "semantic_score": 0.30,
        "keyword_score": 0.20,
        "recency_score": 0.15,
        "salience_score": 0.10,
        "graph_score": 0.10,
        "emotional_resonance": 0.08,
        "temporal_echo": 0.04,
        "reliability": 0.03,
    }
    _KEYWORD_STOPWORDS = {
        "a", "an", "and", "are", "be", "did", "do", "for", "from", "have", "how",
        "i", "if", "in", "is", "it", "last", "me", "my", "of", "on", "or", "our",
        "previous", "remember", "recall", "say", "said", "story", "tell", "the",
        "this", "to", "was", "we", "what", "when", "where", "who", "why", "you",
        "your", "yesterday",
    }

    def __init__(
        self,
        memory: MemoryStore,
        embeddings: EmbeddingService,
        rrf_k: int = 60,
        episode_graph: Optional["EpisodeGraph"] = None,
        somatic_manager: Optional["SomaticManager"] = None,
        relational_engine: Optional["RelationalEngine"] = None,
    ) -> None:
        self.memory = memory
        self.embeddings = embeddings
        self.rrf_k = rrf_k
        self.episode_graph = episode_graph
        self.somatic_manager = somatic_manager
        self.relational_engine = relational_engine

    @staticmethod
    def extract_anchor_terms(query: str) -> List[str]:
        """Extract quoted or capitalized anchor terms from the query."""
        terms: List[str] = []
        # Quoted terms
        terms.extend(re.findall(r'"([^"]+)"', query))
        # Capitalized phrases (potential named entities)
        for match in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", query):
            terms.append(match.group(0))
        return terms

    @staticmethod
    def detect_personal_recall_intent(query: str) -> bool:
        """Detect if the user is asking about a past personal event or their identity."""
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
        return any(re.search(p, q) for p in patterns)

    @staticmethod
    def detect_temporal_intent(query: str) -> Optional[str]:
        """Detect temporal qualifiers like 'last week', 'yesterday', 'in January'."""
        patterns = [
            r"\blast\s+(week|month|year|night|evening|morning|afternoon)",
            r"\byesterday\b",
            r"\bago\b",
            r"\bin\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b",
            r"\bon\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        ]
        q = query.lower()
        for p in patterns:
            m = re.search(p, q)
            if m:
                return m.group(0)
        return None

    async def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        expand_graph: bool = True,
        affect_query: Optional[AffectState] = None,
        affect_weight: float = 0.25,
        emotion_boost_tag: Optional[str] = None,
        emotion_boost_value: float = 0.0,
        min_confidence: float = 0.15,
        lambda_param: float = 0.5,
    ) -> List[RetrievalResult]:
        """Return top-k relevant memories/episodes using multi-signal fusion."""
        inspection = await self.inspect(
            query=query,
            session_id=session_id,
            limit=limit,
            expand_graph=expand_graph,
            affect_query=affect_query,
            affect_weight=affect_weight,
            emotion_boost_tag=emotion_boost_tag,
            emotion_boost_value=emotion_boost_value,
            min_confidence=min_confidence,
            lambda_param=lambda_param,
        )
        return inspection["results"]

    async def inspect(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        expand_graph: bool = True,
        affect_query: Optional[AffectState] = None,
        affect_weight: float = 0.25,
        emotion_boost_tag: Optional[str] = None,
        emotion_boost_value: float = 0.0,
        min_confidence: float = 0.15,
        lambda_param: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Return retrieval results plus score breakdowns for operator inspection."""
        semantic_results = await self._semantic_search(
            query, limit=limit * 3, affect_query=affect_query, affect_weight=affect_weight
        )
        keyword_results = await self._keyword_search(query, limit=limit * 3)

        # Build unified candidate map
        candidate_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        now = datetime.now(timezone.utc)

        # Resolve live somatic state for query-time adjustments
        adj = None
        query_affect = affect_query
        if self.somatic_manager is not None:
            from opencas.somatic.modulators import SomaticModulators
            modulators = SomaticModulators(self.somatic_manager.state)
            adj = modulators.to_retrieval_adjustment()
            if query_affect is None:
                query_affect = AffectState(
                    primary_emotion=modulators._infer_primary_emotion(),
                    valence=self.somatic_manager.state.valence,
                    arousal=self.somatic_manager.state.arousal,
                    intensity=self.somatic_manager.state.certainty,
                )

        def _add_results(results: List[RetrievalResult], score_key: str) -> None:
            for r in results:
                key = (r.source_type, r.source_id)
                if key not in candidate_map:
                    candidate_map[key] = {
                        "result": r,
                        "semantic_score": 0.0,
                        "keyword_score": 0.0,
                        "recency_score": 0.0,
                        "salience_score": 0.0,
                        "graph_score": 0.0,
                        "emotional_resonance": 0.0,
                        "temporal_echo": 0.0,
                        "reliability": 0.8,
                    }
                candidate_map[key][score_key] = r.score
                # Derive recency and salience from attached objects
                ep = getattr(r, "episode", None)
                mem = getattr(r, "memory", None)
                obj = ep or mem
                if obj is not None:
                    created = getattr(obj, "created_at", now)
                    age_days = max(0.0, (now - created).total_seconds() / 86400.0)
                    half_life = 180.0 if getattr(obj, "identity_core", False) else 60.0
                    candidate_map[key]["recency_score"] = self.apply_temporal_decay(
                        1.0, age_days, half_life_days=half_life
                    )
                    salience = getattr(obj, "salience", 1.0)
                    candidate_map[key]["salience_score"] = min(1.0, salience / 10.0)
                    candidate_map[key]["episode"] = ep
                    candidate_map[key]["memory"] = mem
                    candidate_map[key]["embedding"] = r.embedding
                    # Compute emotional resonance
                    ep_affect = getattr(ep, "affect", None)
                    candidate_map[key]["emotional_resonance"] = compute_emotional_resonance(
                        query_affect, ep_affect
                    )
                    # Compute temporal echo
                    candidate_map[key]["temporal_echo"] = compute_temporal_echo(now, created)
                    # Compute reliability from usage feedback
                    us = getattr(obj, "used_successfully", 0)
                    uu = getattr(obj, "used_unsuccessfully", 0)
                    candidate_map[key]["reliability"] = compute_reliability_score(us, uu)

        _add_results(semantic_results, "semantic_score")
        _add_results(keyword_results, "keyword_score")

        # Normalize each signal to [0, 1] across candidates
        def _normalize(values: List[float]) -> List[float]:
            if not values:
                return values
            max_v = max(values)
            if max_v <= 0.0:
                return [0.0] * len(values)
            return [v / max_v for v in values]

        keys = list(candidate_map.keys())
        for sig in (
            "semantic_score",
            "keyword_score",
            "recency_score",
            "salience_score",
            "emotional_resonance",
            "temporal_echo",
            "reliability",
        ):
            vals = _normalize([candidate_map[k][sig] for k in keys])
            for key, val in zip(keys, vals):
                candidate_map[key][sig] = val

        # Graph expansion
        if expand_graph:
            seed_results = [
                candidate_map[k]["result"] for k in keys
                if candidate_map[k]["graph_score"] == 0.0
            ]
            graph_results = await self._expand_graph(seed_results, decay=0.8)
            for r in graph_results:
                key = (r.source_type, r.source_id)
                if key in candidate_map:
                    candidate_map[key]["graph_score"] = r.score
                else:
                    ep = getattr(r, "episode", None)
                    mem = getattr(r, "memory", None)
                    obj = ep or mem
                    created = getattr(obj, "created_at", now) if obj else now
                    age_days = max(0.0, (now - created).total_seconds() / 86400.0) if obj else 0.0
                    half_life = 180.0 if (obj and getattr(obj, "identity_core", False)) else 60.0
                    candidate_map[key] = {
                        "result": r,
                        "semantic_score": 0.0,
                        "keyword_score": 0.0,
                        "recency_score": self.apply_temporal_decay(
                            1.0, age_days, half_life_days=half_life
                        ),
                        "salience_score": min(1.0, getattr(obj, "salience", 1.0) / 10.0) if obj else 0.1,
                        "graph_score": r.score,
                        "episode": ep,
                        "memory": mem,
                        "embedding": getattr(r, "embedding", None),
                        "emotional_resonance": 0.0,
                        "temporal_echo": compute_temporal_echo(now, created) if obj else 0.0,
                        "reliability": compute_reliability_score(
                            getattr(obj, "used_successfully", 0),
                            getattr(obj, "used_unsuccessfully", 0),
                        ) if obj else 0.8,
                    }
            # Normalize graph scores
            keys = list(candidate_map.keys())
            graph_vals = _normalize([candidate_map[k]["graph_score"] for k in keys])
            for key, val in zip(keys, graph_vals):
                candidate_map[key]["graph_score"] = val

        resolved_weights = dict(self.DEFAULT_FUSION_WEIGHTS)
        for key, value in (weights or {}).items():
            if key in resolved_weights:
                resolved_weights[key] = float(value)

        # Weighted fusion
        fused: List[RetrievalResult] = []
        candidate_debug: List[Dict[str, Any]] = []
        for key in keys:
            c = candidate_map[key]
            base_score = sum(
                resolved_weights[name] * c[name]
                for name in resolved_weights
            )

            # Apply somatic query-time adjustments
            somatic_bonus = 0.0
            if adj is not None:
                somatic_bonus = (
                    c["recency_score"] * adj.recency_bonus
                    + c["salience_score"] * adj.salience_bonus
                    + c["emotional_resonance"] * adj.emotional_resonance_bonus
                    + c["temporal_echo"] * adj.temporal_echo_bonus
                    + c["graph_score"] * adj.graph_bonus
                )
            score = base_score + somatic_bonus

            # Apply reliability multiplier
            reliability_multiplier = 0.7 + 0.3 * c["reliability"]
            score = score * reliability_multiplier

            # Apply relational modifier at query time
            obj = c.get("episode") or c.get("memory")
            relational_multiplier = 1.0
            if self.relational_engine is not None and obj is not None:
                tags = getattr(obj, "tags", []) or []
                musubi_mod = self.relational_engine.to_memory_salience_modifier(
                    has_user_collab_tag=("collab" in tags)
                )
                relational_multiplier = 1.0 + musubi_mod
                score = score * relational_multiplier

            # Tiered confidence reweighting
            episode_obj = c.get("episode")
            confidence_multiplier = 1.0
            if episode_obj is not None:
                conf = getattr(episode_obj, "confidence_score", 0.8)
                if conf < 0.5:
                    confidence_multiplier = 0.6
                elif conf <= 0.8:
                    confidence_multiplier = 0.8
                score *= confidence_multiplier

            created = getattr(obj, "created_at", now) if obj is not None else now
            age_days = max(0.0, (now - created).total_seconds() / 86400.0)
            affect = getattr(getattr(c.get("episode"), "affect", None), "primary_emotion", None)
            embedding = c.get("embedding")
            embedding_dimension = len(embedding) if isinstance(embedding, list) else None
            candidate_debug.append(
                {
                    "source_type": key[0],
                    "source_id": key[1],
                    "content": c["result"].content,
                    "kind": getattr(getattr(c.get("episode"), "kind", None), "value", "memory"),
                    "session_id": getattr(c.get("episode"), "session_id", None),
                    "created_at": created.isoformat(),
                    "age_days": round(age_days, 3),
                    "salience": getattr(obj, "salience", None),
                    "confidence_score": getattr(c.get("episode"), "confidence_score", None),
                    "used_successfully": getattr(obj, "used_successfully", None),
                    "used_unsuccessfully": getattr(obj, "used_unsuccessfully", None),
                    "somatic_tag": getattr(c.get("episode"), "somatic_tag", None),
                    "primary_emotion": getattr(affect, "value", None),
                    "embedding_dimension": embedding_dimension,
                    "semantic_score": round(float(c["semantic_score"]), 6),
                    "keyword_score": round(float(c["keyword_score"]), 6),
                    "recency_score": round(float(c["recency_score"]), 6),
                    "salience_score": round(float(c["salience_score"]), 6),
                    "graph_score": round(float(c["graph_score"]), 6),
                    "emotional_resonance": round(float(c["emotional_resonance"]), 6),
                    "temporal_echo": round(float(c["temporal_echo"]), 6),
                    "reliability": round(float(c["reliability"]), 6),
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

            r = c["result"]
            fused.append(
                RetrievalResult(
                    source_type=r.source_type,
                    source_id=r.source_id,
                    content=r.content,
                    score=score,
                    episode=c.get("episode"),
                    memory=c.get("memory"),
                    embedding=c.get("embedding"),
                )
            )

        fused.sort(key=lambda r: r.score, reverse=True)

        if emotion_boost_tag and emotion_boost_value:
            fused = self._apply_emotion_boost(
                fused, emotion_boost_tag, emotion_boost_value, query_affect
            )

        # Diversity penalty
        fused = self._apply_diversity_penalty(fused, window=5, penalty=0.05)

        # MMR rerank
        fused = await self._mmr_rerank(fused, lambda_param=lambda_param, limit=limit * 2)
        fused = fused[:limit]
        selected_ids = {(item.source_type, item.source_id) for item in fused}
        for item in candidate_debug:
            item["selected"] = (item["source_type"], item["source_id"]) in selected_ids

        candidate_debug.sort(key=lambda item: item["final_score"], reverse=True)
        return {
            "results": fused,
            "candidates": candidate_debug,
            "weights": resolved_weights,
            "meta": {
                "query": query,
                "expand_graph": expand_graph,
                "limit": limit,
                "min_confidence": min_confidence,
                "lambda_param": lambda_param,
                "emotion_boost_tag": emotion_boost_tag,
                "emotion_boost_value": emotion_boost_value,
                "semantic_seed_count": len(semantic_results),
                "keyword_seed_count": len(keyword_results),
            },
        }

    @staticmethod
    def apply_temporal_decay(
        score: float, age_days: float, half_life_days: float = 60.0
    ) -> float:
        """Apply exponential temporal decay to a score."""
        return score * math.exp(-math.log(2) * age_days / half_life_days)

    async def _mmr_rerank(
        self,
        results: List[RetrievalResult],
        lambda_param: float = 0.5,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """Rerank results using Maximal Marginal Relevance."""
        if not results:
            return results

        # Resolve embedding vectors
        vectors: List[Optional[np.ndarray]] = []
        for r in results:
            vec = None
            if r.embedding is not None:
                vec = np.array(r.embedding, dtype=np.float32)
            else:
                # Fallback: fetch from cache
                ep = getattr(r, "episode", None)
                mem = getattr(r, "memory", None)
                embed_id = None
                if ep is not None:
                    embed_id = getattr(ep, "embedding_id", None)
                elif mem is not None:
                    embed_id = getattr(mem, "embedding_id", None)
                if embed_id is not None:
                    record = await self.embeddings.cache.get(embed_id)
                    if record is not None and record.vector:
                        vec = np.array(record.vector, dtype=np.float32)
            vectors.append(vec)

        def _sim(i: int, j: int) -> float:
            vi = vectors[i]
            vj = vectors[j]
            if vi is None or vj is None:
                return 0.0
            if vi.shape != vj.shape:
                return 0.0
            ni = float(np.linalg.norm(vi))
            nj = float(np.linalg.norm(vj))
            if ni == 0.0 or nj == 0.0:
                return 0.0
            return float(np.dot(vi, vj) / (ni * nj))

        selected_indices: List[int] = []
        remaining = set(range(len(results)))

        while remaining and len(selected_indices) < limit:
            best_idx: Optional[int] = None
            best_score = -float("inf")
            for idx in remaining:
                relevance = results[idx].score
                max_sim = 0.0
                for sel in selected_indices:
                    max_sim = max(max_sim, _sim(idx, sel))
                mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            if best_idx is None:
                break
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return [results[i] for i in selected_indices]

    def _apply_diversity_penalty(
        self,
        results: List[RetrievalResult],
        window: int = 5,
        penalty: float = 0.05,
    ) -> List[RetrievalResult]:
        """Penalize duplicate primary emotions and source families in a sliding window."""
        if not results:
            return results

        def _emotion_of(r: RetrievalResult) -> Optional[str]:
            ep = getattr(r, "episode", None)
            if ep is not None and getattr(ep, "affect", None) is not None:
                return str(ep.affect.primary_emotion.value)
            return None

        def _family_of(r: RetrievalResult) -> Optional[str]:
            ep = getattr(r, "episode", None)
            if ep is not None:
                return str(ep.kind.value)
            return "memory"

        penalized: List[RetrievalResult] = []
        recent_emotions: deque[Optional[str]] = deque(maxlen=window)
        recent_families: deque[Optional[str]] = deque(maxlen=window)

        for r in results:
            emotion = _emotion_of(r)
            family = _family_of(r)
            deductions = 0
            if emotion is not None and emotion in recent_emotions:
                deductions += 1
            if family in recent_families:
                deductions += 1
            new_score = max(0.0, r.score - penalty * deductions)
            penalized.append(r.model_copy(update={"score": new_score}))
            recent_emotions.append(emotion)
            recent_families.append(family)

        penalized.sort(key=lambda r: r.score, reverse=True)
        return penalized

    async def _semantic_search(
        self,
        query: str,
        limit: int,
        affect_query: Optional[AffectState] = None,
        affect_weight: float = 0.25,
    ) -> List[RetrievalResult]:
        """Embedding-based similarity search over Memory records."""
        query_embed = await self.embeddings.embed(query, task_type="retrieval_query")
        similar = await self.embeddings.cache.search_similar(
            query_embed.vector, limit=limit, model_id=self.embeddings.model_id, query_text=query
        )
        if not similar:
            return []

        affect_vector = None
        if affect_query is not None:
            affect_text = (
                f"Affective query: primary emotion {affect_query.primary_emotion.value}, "
                f"valence {affect_query.valence:.2f}, arousal {affect_query.arousal:.2f}, "
                f"intensity {affect_query.intensity:.2f}."
            )
            affect_embed = await self.embeddings.embed(
                affect_text,
                task_type="affect_query",
            )
            affect_vector = np.array(affect_embed.vector, dtype=np.float32)

        source_hashes = [record.source_hash for record, _ in similar]
        memories = await self.memory.list_memories_by_embedding_ids(source_hashes)
        episodes = await self.memory.list_episodes_by_embedding_ids(source_hashes)
        score_map = {record.source_hash: sim for record, sim in similar}
        vector_map = {
            record.source_hash: np.array(record.vector, dtype=np.float32)
            for record, _ in similar
        }

        results: List[RetrievalResult] = []
        seen_ids: set[Tuple[str, str]] = set()

        def _add_result(source_type: str, source_id: str, content: str, semantic_score: float, embedding_id: Optional[str], mem=None, ep=None) -> None:
            key = (source_type, source_id)
            if key in seen_ids:
                return
            seen_ids.add(key)
            final_score = semantic_score
            if affect_vector is not None:
                cand_vec = vector_map.get(embedding_id) if embedding_id else None
                if cand_vec is not None and cand_vec.shape == affect_vector.shape:
                    c_norm = float(np.linalg.norm(cand_vec))
                    a_norm = float(np.linalg.norm(affect_vector))
                    if c_norm > 0 and a_norm > 0:
                        affect_sim = float(
                            np.dot(cand_vec, affect_vector) / (c_norm * a_norm)
                        )
                        final_score = (1 - affect_weight) * semantic_score + affect_weight * affect_sim
            results.append(
                RetrievalResult(
                    source_type=source_type,
                    source_id=source_id,
                    content=content,
                    score=final_score,
                    memory=mem,
                    episode=ep,
                    embedding=vector_map.get(embedding_id, []).tolist() if embedding_id in vector_map else None,
                )
            )

        for mem in memories:
            if mem.embedding_id and mem.embedding_id in score_map:
                _add_result(
                    "memory",
                    str(mem.memory_id),
                    mem.content,
                    score_map[mem.embedding_id],
                    mem.embedding_id,
                    mem=mem,
                )
        for ep in episodes:
            if ep.embedding_id and ep.embedding_id in score_map:
                _add_result(
                    "episode",
                    str(ep.episode_id),
                    ep.content,
                    score_map[ep.embedding_id],
                    ep.embedding_id,
                    ep=ep,
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    async def _keyword_search(
        self,
        query: str,
        limit: int,
    ) -> List[RetrievalResult]:
        """FTS keyword search over Episode records."""
        recall_intent = self.detect_personal_recall_intent(query)
        search_terms = self._keyword_queries_for(query, recall_intent=recall_intent)
        merged: Dict[str, RetrievalResult] = {}
        for term_index, term in enumerate(search_terms):
            episodes = await self.memory.search_episodes_by_content(term, limit=limit)
            base = 1.0 if term_index == 0 else 0.8
            for rank, ep in enumerate(episodes):
                score = max(0.2, base - (0.05 * rank))
                key = str(ep.episode_id)
                existing = merged.get(key)
                if existing is None or score > existing.score:
                    merged[key] = RetrievalResult(
                        source_type="episode",
                        source_id=key,
                        content=ep.content,
                        score=score,
                        episode=ep,
                    )
        results = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return results[:limit]

    def _keyword_queries_for(self, query: str, recall_intent: bool) -> List[str]:
        """Generate useful FTS queries instead of only searching the raw sentence."""
        queries: List[str] = [query]
        anchor_terms = self.extract_anchor_terms(query)
        queries.extend(anchor_terms)
        # Always include non-stopword tokens for better FTS coverage
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]{2,}", query.lower())
        queries.extend(
            token
            for token in tokens
            if token not in self._KEYWORD_STOPWORDS
        )
        if recall_intent:
            temporal = self.detect_temporal_intent(query)
            if temporal is not None:
                queries.append(temporal)
        deduped: List[str] = []
        seen: set[str] = set()
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

    async def _expand_graph(
        self,
        seed_results: List[RetrievalResult],
        decay: float = 0.8,
        edge_limit: int = 12,
    ) -> List[RetrievalResult]:
        """Pull in graph-neighbor episodes for seed episode results."""
        seed_episodes = [r for r in seed_results if r.source_type == "episode"]
        if not seed_episodes:
            return seed_results

        neighbor_ids: set[str] = set()
        edge_map: Dict[str, List[Any]] = {}
        for seed in seed_episodes:
            if self.episode_graph is not None:
                edges = await self.episode_graph.get_neighbors(
                    seed.source_id, min_confidence=0.0, limit=edge_limit
                )
            else:
                edges = await self.memory.get_edges_for(
                    seed.source_id, min_confidence=0.0, limit=edge_limit
                )
            edge_map[seed.source_id] = edges
            for edge in edges:
                neighbor_id = (
                    edge.target_id
                    if edge.source_id == seed.source_id
                    else edge.source_id
                )
                if neighbor_id != seed.source_id:
                    neighbor_ids.add(neighbor_id)

        if not neighbor_ids:
            return seed_results

        neighbors = await self.memory.get_episodes_by_ids(list(neighbor_ids))
        neighbor_by_id = {str(ep.episode_id): ep for ep in neighbors}

        result_by_key: Dict[Tuple[str, str], RetrievalResult] = {}
        for r in seed_results:
            result_by_key[(r.source_type, r.source_id)] = r

        for seed in seed_episodes:
            seed_score = seed.score or 0.0
            for edge in edge_map.get(seed.source_id, []):
                neighbor_id = (
                    edge.target_id
                    if edge.source_id == seed.source_id
                    else edge.source_id
                )
                ep = neighbor_by_id.get(neighbor_id)
                if ep is None:
                    continue
                key = ("episode", neighbor_id)
                strength = compute_edge_strength(edge)
                neighbor_score = seed_score * decay * strength
                if key in result_by_key:
                    existing = result_by_key[key]
                    result_by_key[key] = RetrievalResult(
                        source_type=existing.source_type,
                        source_id=existing.source_id,
                        content=existing.content,
                        score=max(existing.score or 0.0, neighbor_score),
                        episode=existing.episode,
                        embedding=existing.embedding,
                    )
                else:
                    result_by_key[key] = RetrievalResult(
                        source_type="episode",
                        source_id=neighbor_id,
                        content=ep.content,
                        score=neighbor_score,
                        episode=ep,
                    )

        expanded = list(result_by_key.values())
        expanded.sort(key=lambda r: r.score, reverse=True)
        return expanded

    def _apply_emotion_boost(
        self,
        results: List[RetrievalResult],
        tag: str,
        boost: float,
        query_affect: Optional[AffectState] = None,
    ) -> List[RetrievalResult]:
        """Blend emotion_boost_tag into emotional_resonance rather than doing substring matching."""
        if not results:
            return results
        boosted: List[RetrievalResult] = []
        for r in results:
            new_score = r.score
            ep = getattr(r, "episode", None)
            if ep is not None and getattr(ep, "affect", None) is not None:
                # Add a small bonus if primary emotion matches the tag
                if ep.affect.primary_emotion.value.lower() == tag.lower():
                    new_score = r.score + boost
                # Also blend from query_affect if available
                if query_affect is not None:
                    resonance = compute_emotional_resonance(query_affect, ep.affect)
                    new_score = max(new_score, r.score + resonance * boost)
            boosted.append(r.model_copy(update={"score": new_score}))
        boosted.sort(key=lambda r: r.score, reverse=True)
        return boosted

    def _reciprocal_rank_fusion(
        self,
        result_lists: List[Tuple[str, List[RetrievalResult]]],
    ) -> List[RetrievalResult]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion (legacy compatibility)."""
        scores: Dict[Tuple[str, str], float] = {}
        for _name, results in result_lists:
            for rank, result in enumerate(results, start=1):
                key = (result.source_type, result.source_id)
                scores[key] = scores.get(key, 0.0) + (1.0 / (self.rrf_k + rank))

        best: Dict[Tuple[str, str], RetrievalResult] = {}
        for _name, results in result_lists:
            for result in results:
                key = (result.source_type, result.source_id)
                if key not in best:
                    best[key] = result

        fused = [
            RetrievalResult(
                source_type=key[0],
                source_id=key[1],
                content=best[key].content,
                score=score,
            )
            for key, score in scores.items()
        ]
        fused.sort(key=lambda r: r.score, reverse=True)
        return fused
