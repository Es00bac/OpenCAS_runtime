"""Nightly consolidation engine for OpenCAS."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from opencas.api import LLMClient
from opencas.embeddings import EmbeddingService
from opencas.identity import IdentityManager
from opencas.memory import Memory, MemoryStore
from opencas.telemetry import EventKind, Tracer

from opencas.memory.fabric.builder import FabricBuilder
from opencas.memory.fabric.indexer import MemoryIndexer
from opencas.memory.fabric.weigher import ContextProfile, EdgeWeigher

from .models import ConsolidationResult, SalienceUpdate
from .signal_ranker import EpisodeSignalRanker, SignalScore


class NightlyConsolidationEngine:
    """Deep-memory cycle: cluster, summarize, reweight, prune, update identity."""

    def __init__(
        self,
        memory: MemoryStore,
        embeddings: EmbeddingService,
        llm: LLMClient,
        identity: IdentityManager,
        tracer: Optional[Tracer] = None,
        curation_store: Optional[Any] = None,
        fabric_builder: Optional[FabricBuilder] = None,
        signal_ranker: Optional[EpisodeSignalRanker] = None,
        tom_store: Optional[Any] = None,
    ) -> None:
        self.memory = memory
        self.embeddings = embeddings
        self.llm = llm
        self.identity = identity
        self.tracer = tracer
        self.curation_store = curation_store
        self.tom_store = tom_store
        if fabric_builder is None:
            indexer = MemoryIndexer(embeddings=self.embeddings, top_k=24)
            weigher = EdgeWeigher(profile=ContextProfile.CONSOLIDATION)
            fabric_builder = FabricBuilder(
                store=self.memory,
                indexer=indexer,
                weigher=weigher,
                tracer=self.tracer,
            )
        self.fabric_builder = fabric_builder
        if signal_ranker is None:
            signal_ranker = EpisodeSignalRanker(memory=self.memory)
        self.signal_ranker = signal_ranker

    async def run(
        self,
        salience_threshold: float = 0.5,
        similarity_threshold: float = 0.75,
        max_candidates: int = 500,
        signal_threshold: float = 0.65,
    ) -> ConsolidationResult:
        """Execute a full consolidation cycle."""
        result = ConsolidationResult()

        # 1. Gather candidate episodes
        candidates = await self.memory.list_non_compacted_episodes(limit=max_candidates)
        result.candidate_episodes = len(candidates)

        clusters: List[List[Any]] = []
        cluster_consumed_ids: set[str] = set()
        promoted_ids: set[str] = set()
        score_map: Dict[str, SignalScore] = {}
        if candidates:
            # 2. Cluster candidates by embedding similarity
            clusters = await self._cluster_episodes(candidates, similarity_threshold)
            result.clusters_formed = len(clusters)

            # 3. Summarize each cluster into a Memory
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                cluster_hash = self._cluster_hash(cluster)
                if self.curation_store is not None:
                    if await self.curation_store.is_rejected(cluster_hash):
                        result.merges_rejected += 1
                        continue
                summary = await self._summarize_cluster(cluster)
                if not summary:
                    if self.curation_store is not None:
                        await self.curation_store.record_rejection(
                            cluster_hash,
                            [str(e.episode_id) for e in cluster],
                            reason="empty_summary",
                        )
                    result.merges_rejected += 1
                    continue
                # Compute embedding for the summary
                embed_record = await self.embeddings.embed(
                    summary,
                    task_type="consolidation_summary",
                )
                memory = Memory(
                    content=summary,
                    source_episode_ids=[str(e.episode_id) for e in cluster],
                    tags=["consolidation"],
                    embedding_id=embed_record.source_hash,
                )
                await self.memory.save_memory(memory)
                result.memories_created += 1
                for ep in cluster:
                    cluster_consumed_ids.add(str(ep.episode_id))

            # 3b. Promote strong individual signals to Memory
            signal_scores = await self.signal_ranker.rank_episodes_with_degrees(
                candidates, identity=self.identity
            )
            score_map = {s.episode_id: s for s in signal_scores}
            promoted_ids = await self._promote_strong_signals(
                candidates,
                score_map,
                cluster_consumed_ids,
                signal_threshold,
            )
            result.signals_promoted = len(promoted_ids)

        # 4. Rebuild episode edges, recover orphans, and promote identity core
        if candidates:
            edges_created = await self.fabric_builder.rebuild(candidates)
            result.edges_created = edges_created
            orphans_recovered = await self._recover_orphans(candidates)
            result.orphans_recovered = orphans_recovered
            core_promotions = await self._promote_identity_core(candidates, score_map)
            result.identity_core_promotions = core_promotions

        # 5. Reweight episode salience
        episode_salience_updates = await self._reweight_episode_salience(
            candidates, cluster_consumed_ids, promoted_ids
        )
        result.episode_salience_updates = len(episode_salience_updates)

        # 6. Reweight memory salience
        salience_updates = await self._reweight_salience()
        result.salience_updates = salience_updates
        result.memories_updated = len(salience_updates)

        # 7. Prune low-salience episodes
        pruned = await self.memory.prune_episodes_by_salience(salience_threshold)
        result.episodes_pruned = pruned

        # 8. Belief consistency sweep: decay high-confidence ToM beliefs lacking recent corroboration
        beliefs_decayed = await self._sweep_belief_consistency()
        result.beliefs_decayed = beliefs_decayed

        # 9. Update identity anchors
        identity_updates = await self._update_identity(clusters)
        result.identity_updates = identity_updates

        if self.tracer:
            self.tracer.log(
                EventKind.CONSOLIDATION_RUN,
                "Nightly consolidation completed",
                result.model_dump(mode="json"),
            )

        return result

    async def _cluster_episodes(
        self,
        episodes: List[Any],
        threshold: float,
    ) -> List[List[Any]]:
        """Greedy clustering of episodes by cosine similarity of embeddings."""
        # Embed each episode if not already cached
        vectors: List[Optional[np.ndarray]] = []
        for ep in episodes:
            if ep.embedding_id:
                cached = await self.embeddings.cache.get(ep.embedding_id)
                if cached:
                    vectors.append(np.array(cached.vector, dtype=np.float32))
                    continue
            # Fallback: compute via embed service
            record = await self.embeddings.embed(
                ep.content,
                task_type="memory_episode",
            )
            vectors.append(np.array(record.vector, dtype=np.float32))

        clusters: List[List[Any]] = []
        used = set()
        for i, vi in enumerate(vectors):
            if i in used or vi is None:
                continue
            cluster = [episodes[i]]
            used.add(i)
            norm_i = np.linalg.norm(vi)
            if norm_i == 0:
                norm_i = 1.0
            for j, vj in enumerate(vectors[i + 1 :], start=i + 1):
                if j in used or vj is None:
                    continue
                norm_j = np.linalg.norm(vj)
                if norm_j == 0:
                    norm_j = 1.0
                sim = float(np.dot(vi, vj) / (norm_i * norm_j))
                if sim >= threshold:
                    cluster.append(episodes[j])
                    used.add(j)
            clusters.append(cluster)
        return clusters

    async def _summarize_cluster(self, episodes: List[Any]) -> str:
        """Ask the LLM to synthesize a cluster summary."""
        lines = [f"- {ep.content}" for ep in episodes]
        prompt = (
            "Synthesize the following related episodes into a single concise memory. "
            "Capture the core insight or factual takeaway.\n\n"
            + "\n".join(lines)
        )
        messages = [
            {"role": "system", "content": "You are a consolidation assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.llm.chat_completion(
                messages,
                source="consolidation",
            )
            choices = response.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
        except Exception:
            pass
        return episodes[0].content[:400]

    @staticmethod
    def _has_obsolete_system_reference(obj: Any) -> bool:
        """Detect references to superseded systems/workspace paths."""
        text = str(getattr(obj, "content", "")).lower()
        payload = getattr(obj, "payload", {}) or {}
        source = str(payload.get("legacy_agent_source", "")).lower()
        markers = ["legacy_agent-v4", "legacy_agent-v3", "legacy_agent-v2", "openclaw"]
        combined = f"{text} {source}"
        return any(marker in combined for marker in markers)

    async def _reweight_salience(self) -> List[SalienceUpdate]:
        """Boost frequently accessed memories, decay stale ones."""
        memories = await self.memory.list_memories(limit=1000)
        now = datetime.now(timezone.utc)
        updates: List[SalienceUpdate] = []
        for mem in memories:
            new_salience = mem.salience
            # Boost for access
            if mem.access_count > 5:
                new_salience += 0.2
            # Decay if never accessed and old
            if mem.access_count == 0 and mem.updated_at < now - timedelta(days=7):
                new_salience -= 0.3
            # Decay if stale access
            if mem.last_accessed and mem.last_accessed < now - timedelta(days=14):
                new_salience -= 0.2
            # Decay obsolete system/workspace references
            if not getattr(mem, "identity_core", False) and self._has_obsolete_system_reference(mem):
                new_salience -= 0.3
            new_salience = round(max(0.0, min(10.0, new_salience)), 3)
            if new_salience != mem.salience:
                await self.memory.update_memory_salience(str(mem.memory_id), new_salience)
                updates.append(
                    SalienceUpdate(
                        memory_id=str(mem.memory_id),
                        old_salience=mem.salience,
                        new_salience=new_salience,
                    )
                )
        return updates

    @staticmethod
    def _cluster_hash(cluster: List[Any]) -> str:
        ids = sorted({str(e.episode_id) for e in cluster})
        import hashlib
        return hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()[:32]

    async def _promote_strong_signals(
        self,
        candidates: List[Any],
        score_map: Dict[str, SignalScore],
        cluster_consumed_ids: set[str],
        threshold: float,
    ) -> set[str]:
        """Promote high-scoring individual episodes to Memory."""
        promoted: set[str] = set()
        for ep in candidates:
            ep_id = str(ep.episode_id)
            if ep_id in cluster_consumed_ids:
                continue
            score = score_map.get(ep_id)
            if not score or score.signal_score < threshold:
                continue

            # Resolve or compute embedding
            embed_record = None
            if ep.embedding_id:
                embed_record = await self.embeddings.cache.get(ep.embedding_id)
            if embed_record is None:
                try:
                    embed_record = await self.embeddings.embed(
                        ep.content,
                        task_type="memory_episode",
                    )
                except Exception:
                    pass

            # Deduplicate against existing memories via embedding similarity
            if embed_record is not None:
                try:
                    similar = await self.embeddings.cache.search_similar(
                        embed_record.vector, limit=5
                    )
                    if similar:
                        top_hit, top_sim = similar[0]
                        if top_sim > 0.92 and top_hit.source_hash != embed_record.source_hash:
                            continue
                except Exception:
                    pass

            memory = Memory(
                content=ep.content,
                source_episode_ids=[ep_id],
                tags=["consolidation", "strong_signal"],
                embedding_id=embed_record.source_hash if embed_record else None,
                salience=round(ep.salience, 3),
            )
            await self.memory.save_memory(memory)
            promoted.add(ep_id)
        return promoted

    async def _reweight_episode_salience(
        self,
        candidates: List[Any],
        cluster_consumed_ids: set[str],
        signal_promoted_ids: set[str],
    ) -> List[SalienceUpdate]:
        """Boost or decay episode salience based on promotion and graph state."""
        now = datetime.now(timezone.utc)
        updates: List[SalienceUpdate] = []
        promoted_all = cluster_consumed_ids | signal_promoted_ids
        for ep in candidates:
            ep_id = str(ep.episode_id)
            new_salience = ep.salience

            # Boost if promoted to memory
            if ep_id in promoted_all:
                new_salience += 0.3

            # Boost for graph reinforcement
            edges = await self.memory.get_edges_for(ep_id, min_confidence=0.0, limit=100)
            new_salience += (len(edges) // 5) * 0.1

            # Decay if old and never promoted
            if ep_id not in promoted_all and ep.created_at < now - timedelta(days=7):
                new_salience -= 0.2

            # Decay obsolete system/workspace references (preserve identity core)
            if not ep.identity_core and self._has_obsolete_system_reference(ep):
                new_salience -= 0.3

            new_salience = round(max(0.0, min(10.0, new_salience)), 3)
            if new_salience != ep.salience:
                old_salience = ep.salience
                ep.salience = new_salience
                await self.memory.save_episode(ep)
                updates.append(
                    SalienceUpdate(
                        memory_id=ep_id,
                        old_salience=old_salience,
                        new_salience=new_salience,
                    )
                )
        return updates

    async def _promote_identity_core(self, candidates: List[Any], score_map: Optional[Dict[str, SignalScore]] = None) -> int:
        """Promote high-degree or high-signal episodes to identity_core."""
        candidate_ids = [str(ep.episode_id) for ep in candidates]
        promoted = 0
        for ep_id in candidate_ids:
            edges = await self.memory.get_edges_for(ep_id, min_confidence=0.0, limit=100)
            degree = len(edges)
            signal_score = 0.0
            if score_map:
                s = score_map.get(ep_id)
                if s:
                    signal_score = s.signal_score
            if degree >= 5 or signal_score >= 0.80:
                ep = await self.memory.get_episode(ep_id)
                if ep and (ep.affect is None or ep.affect.valence >= 0.0):
                    if not ep.identity_core:
                        ep.identity_core = True
                        await self.memory.save_episode(ep)
                        promoted += 1
        return promoted

    async def _recover_orphans(self, candidates: List[Any]) -> int:
        """Find episodes with no edges and bridge them to their nearest neighbor."""
        if not candidates:
            return 0
        episode_map = {str(ep.episode_id): ep for ep in candidates}
        recovered = 0
        for ep in candidates:
            ep_id = str(ep.episode_id)
            edges = await self.memory.get_edges_for(ep_id, min_confidence=0.0, limit=1)
            if edges:
                continue
            # Find nearest neighbor by embedding cosine similarity
            best_neighbor: Optional[str] = None
            best_sim = -1.0
            if not ep.embedding_id:
                continue
            rec = await self.embeddings.cache.get(ep.embedding_id)
            if rec is None:
                continue
            vec = np.array(rec.vector, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm == 0:
                continue
            for other in candidates:
                other_id = str(other.episode_id)
                if other_id == ep_id:
                    continue
                if not other.embedding_id:
                    continue
                other_rec = await self.embeddings.cache.get(other.embedding_id)
                if other_rec is None:
                    continue
                other_vec = np.array(other_rec.vector, dtype=np.float32)
                other_norm = float(np.linalg.norm(other_vec))
                if other_norm == 0:
                    continue
                sim = float(np.dot(vec, other_vec) / (norm * other_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_neighbor = other_id
            if best_neighbor is not None:
                from opencas.memory import EpisodeEdge, EdgeKind
                edge = EpisodeEdge(
                    source_id=ep_id,
                    target_id=best_neighbor,
                    kind=EdgeKind.SEMANTIC,
                    confidence=0.1,
                )
                await self.memory.save_edge(edge)
                recovered += 1
        return recovered

    async def _sweep_belief_consistency(
        self,
        confidence_threshold: float = 0.7,
        decay_factor: float = 0.8,
        lookback_days: int = 7,
    ) -> int:
        """Decay high-confidence ToM beliefs that lack corroborating episodes in the last N days."""
        if self.tom_store is None:
            return 0
        decayed = 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        try:
            beliefs = await self.tom_store.list_beliefs(limit=1000)
            for belief in beliefs:
                if belief.confidence < confidence_threshold:
                    continue
                # Check whether any evidence episode is recent enough
                has_recent = False
                if belief.evidence_ids:
                    episodes = await self.memory.get_episodes_by_ids(belief.evidence_ids)
                    for ep in episodes:
                        if ep.created_at >= cutoff:
                            has_recent = True
                            break
                if not has_recent:
                    new_confidence = round(max(0.0, belief.confidence * decay_factor), 3)
                    new_revision_score = round(belief.belief_revision_score - 0.1, 3)
                    await self.tom_store.update_belief_confidence(
                        str(belief.belief_id),
                        confidence=new_confidence,
                        belief_revision_score=new_revision_score,
                    )
                    decayed += 1
                    if self.tracer:
                        self.tracer.log(
                            EventKind.CONSOLIDATION_RUN,
                            "NightlyConsolidationEngine: belief_decayed",
                            {
                                "belief_id": str(belief.belief_id),
                                "old_confidence": belief.confidence,
                                "new_confidence": new_confidence,
                                "revision_score": new_revision_score,
                            },
                        )
        except Exception:
            pass
        return decayed

    async def _update_identity(
        self,
        clusters: List[List[Any]],
    ) -> Dict[str, Any]:
        """Push high-level themes into identity self-beliefs."""
        if not clusters:
            return {}
        # Simple heuristic: belief keyed by consolidation date
        themes = [f"Cluster of {len(c)} episodes" for c in clusters[:3]]
        key = f"consolidation_themes_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        self.identity.update_self_belief(key, themes)
        return {key: themes}
