"""FabricBuilder orchestrates the nightly memory-fabric edge rebuild."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from opencas.memory import EdgeKind, Episode, EpisodeEdge, MemoryStore
from opencas.telemetry import Tracer

from .indexer import MemoryIndexer
from .scorers import (
    CausalScorer,
    ConceptualScorer,
    EmotionalScorer,
    RelationalScorer,
    TemporalScorer,
)
from .weigher import ContextProfile, EdgeWeigher


class FabricBuilder:
    """Build typed episode edges using embedding-first indexing and scorer fusion."""

    def __init__(
        self,
        store: MemoryStore,
        indexer: MemoryIndexer,
        weigher: EdgeWeigher,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self.store = store
        self.indexer = indexer
        self.weigher = weigher
        self.tracer = tracer
        self.conceptual = ConceptualScorer(indexer.embeddings)
        self.emotional = EmotionalScorer()
        self.relational = RelationalScorer()
        self.temporal = TemporalScorer()
        self.causal = CausalScorer()

    @staticmethod
    def compute_bridge_affinity(ep_a: Episode, ep_b: Episode) -> float:
        """Compute bridge affinity between two episodes."""
        cross_source_echo = 0.3 if ep_a.kind != ep_b.kind else 0.0
        days_diff = abs((ep_a.created_at - ep_b.created_at).total_seconds()) / 86_400
        temporal_path_affinity = 1.0 / (1.0 + days_diff)
        return (cross_source_echo + temporal_path_affinity) / 2.0

    async def rebuild(
        self,
        episodes: List[Episode],
        decay: float = 0.95,
        existing_boost: float = 0.03,
        min_confidence: float = 0.15,
        prune_threshold: float = 0.05,
    ) -> int:
        """Rebuild episode edges for the given candidates."""
        # 1. Global decay
        await self.store.decay_all_edges(decay)

        # Ensure every candidate has an embedding_id so the indexer can match
        for ep in episodes:
            if not ep.embedding_id:
                record = await self.indexer.embeddings.embed(
                    ep.content,
                    task_type="memory_episode",
                )
                ep.embedding_id = record.source_hash

        # Map by embedding_id because indexer.candidates returns source_hash values
        episode_map: Dict[str, Episode] = {
            e.embedding_id: e for e in episodes if e.embedding_id
        }
        edges_to_save: List[EpisodeEdge] = []
        edges_created = 0
        processed_pairs: set[frozenset[str]] = set()

        for ep in episodes:
            candidates = await self.indexer.candidates(ep)
            if not candidates:
                continue

            for cand in candidates:
                if cand.episode_id not in episode_map:
                    continue
                ep_b = episode_map[cand.episode_id]
                source_id = str(ep.episode_id)
                target_id = str(ep_b.episode_id)
                if source_id == target_id:
                    continue

                pair = frozenset({source_id, target_id})
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                scores: Dict[str, float] = {
                    "conceptual": await self.conceptual.score(ep, ep_b),
                    "emotional": await self.emotional.score(ep, ep_b),
                    "relational": await self.relational.score(ep, ep_b),
                    "temporal": await self.temporal.score(ep, ep_b),
                    "causal": await self.causal.score(ep, ep_b),
                }
                fusion = self.weigher.fuse(scores)
                if fusion["confidence"] < min_confidence:
                    continue

                if await self.store.edge_exists(source_id, target_id):
                    await self.store.boost_edge_confidence(
                        source_id, existing_boost
                    )
                    edges_created += 1
                    continue

                salience_weight = min(ep.salience, ep_b.salience) / 10.0
                causal_weight = round(scores.get("causal", 0.0), 4)
                edge = EpisodeEdge(
                    source_id=source_id,
                    target_id=target_id,
                    kind=fusion["kind"],
                    semantic_weight=round(scores.get("conceptual", 0.0), 4),
                    emotional_weight=round(scores.get("emotional", 0.0), 4),
                    recency_weight=round(scores.get("temporal", 0.0), 4),
                    structural_weight=round(scores.get("relational", 0.0), 4),
                    salience_weight=round(salience_weight, 4),
                    causal_weight=causal_weight,
                    confidence=fusion["confidence"],
                )
                edges_to_save.append(edge)
                edges_created += 1

                # Batch flush to bound memory usage
                if len(edges_to_save) >= 256:
                    await self.store.save_edges_batch(edges_to_save)
                    edges_to_save.clear()

        # Bridge pass
        bridge_weigher = EdgeWeigher(profile=ContextProfile.BRIDGE)
        for ep in episodes:
            candidates = await self.indexer.candidates(ep)
            if not candidates:
                continue

            for cand in candidates:
                if cand.episode_id not in episode_map:
                    continue
                ep_b = episode_map[cand.episode_id]
                source_id = str(ep.episode_id)
                target_id = str(ep_b.episode_id)
                if source_id == target_id:
                    continue

                pair = frozenset({source_id, target_id})
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                bridge_affinity = self.compute_bridge_affinity(ep, ep_b)
                if bridge_affinity <= 0.3:
                    continue

                if await self.store.edge_exists(source_id, target_id):
                    await self.store.boost_edge_confidence(source_id, existing_boost)
                    edges_created += 1
                    continue

                scores = {
                    "conceptual": await self.conceptual.score(ep, ep_b),
                    "emotional": await self.emotional.score(ep, ep_b),
                    "relational": await self.relational.score(ep, ep_b),
                    "temporal": await self.temporal.score(ep, ep_b),
                    "causal": await self.causal.score(ep, ep_b),
                }
                fusion = bridge_weigher.fuse(scores)
                if fusion["confidence"] < 0.08:
                    continue

                salience_weight = min(ep.salience, ep_b.salience) / 10.0
                causal_weight = round(scores.get("causal", 0.0), 4)
                edge = EpisodeEdge(
                    source_id=source_id,
                    target_id=target_id,
                    kind=EdgeKind.SEMANTIC,
                    semantic_weight=round(scores.get("conceptual", 0.0), 4),
                    emotional_weight=round(scores.get("emotional", 0.0), 4),
                    recency_weight=round(scores.get("temporal", 0.0), 4),
                    structural_weight=round(scores.get("relational", 0.0), 4),
                    salience_weight=round(salience_weight, 4),
                    causal_weight=causal_weight,
                    confidence=0.1,
                )
                edges_to_save.append(edge)
                edges_created += 1

                if len(edges_to_save) >= 256:
                    await self.store.save_edges_batch(edges_to_save)
                    edges_to_save.clear()

        if edges_to_save:
            await self.store.save_edges_batch(edges_to_save)

        # 5. Prune weak edges
        await self.store.prune_weak_edges(prune_threshold)
        return edges_created
