"""Nightly consolidation engine for OpenCAS."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from opencas.api import LLMClient
from opencas.autonomy.commitment import CommitmentStatus
from opencas.autonomy.commitment_store import CommitmentStore
from opencas.autonomy.work_store import WorkStore
from opencas.embeddings import EmbeddingService
from opencas.identity import IdentityManager
from opencas.memory import Memory, MemoryStore
from opencas.telemetry import EventKind, Tracer

from opencas.memory.fabric.builder import FabricBuilder
from opencas.memory.fabric.indexer import MemoryIndexer
from opencas.memory.fabric.weigher import ContextProfile, EdgeWeigher

from .commitment_cleanup import (
    CommitmentRecoveryCandidate,
    are_conservative_duplicate_contents as commitment_duplicate_contents,
    cluster_has_obvious_duplicate as commitment_cluster_has_obvious_duplicate,
    collect_commitment_recovery_candidates as collect_recovery_candidates,
    commitment_key as normalize_commitment_key,
    consolidate_commitments as consolidate_commitments_impl,
    extract_commitments_from_chat_logs as extract_commitments_from_chat_logs_impl,
    heuristic_survivor_index as commitment_heuristic_survivor_index,
    llm_pick_commitment_survivor as llm_pick_commitment_survivor_impl,
    merged_commitment_status as merged_commitment_status_impl,
    pick_commitment_survivor as pick_commitment_survivor_impl,
    refine_commitment_cluster as refine_commitment_cluster_impl,
)
from .models import ConsolidationResult, SalienceUpdate
from .signal_maintenance import (
    cluster_hash as cluster_hash_impl,
    has_obsolete_system_reference as has_obsolete_system_reference_impl,
    promote_identity_core as promote_identity_core_impl,
    promote_strong_signals as promote_strong_signals_impl,
    recover_orphans as recover_orphans_impl,
    reweight_episode_salience as reweight_episode_salience_impl,
    reweight_salience as reweight_salience_impl,
)
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
        commitment_store: Optional[CommitmentStore] = None,
        work_store: Optional[WorkStore] = None,
    ) -> None:
        self.memory = memory
        self.embeddings = embeddings
        self.llm = llm
        self.identity = identity
        self.tracer = tracer
        self.curation_store = curation_store
        self.tom_store = tom_store
        self.commitment_store = commitment_store
        self.work_store = work_store
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

        # 4b. Consolidate commitments: deduplicate via embeddings + LLM, extract from chat logs
        if self.commitment_store:
            commitment_result = await self._consolidate_commitments(
                similarity_threshold=similarity_threshold
            )
            result.commitments_consolidated = commitment_result.get("commitments_merged", 0)
            result.commitment_clusters_formed = commitment_result.get("clusters_formed", 0)
            result.commitment_work_objects_created = commitment_result.get("work_objects_created", 0)
            chat_extracted = await self._extract_commitments_from_chat_logs()
            result.commitments_extracted_from_chat = chat_extracted

        # 5. Rebuild episode edges, recover orphans, and promote identity core
        if candidates:
            edges_created = await self.fabric_builder.rebuild(candidates)
            result.edges_created = edges_created
            orphans_recovered = await self._recover_orphans(candidates)
            result.orphans_recovered = orphans_recovered
            core_promotions = await self._promote_identity_core(candidates, score_map)
            result.identity_core_promotions = core_promotions

            # Mark consumed/promoted episodes as compacted so health checks stay clean
            consumed_ids = list(cluster_consumed_ids | promoted_ids)
            if consumed_ids:
                await self.memory.mark_compacted(consumed_ids)

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
                complexity="standard",
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
        return has_obsolete_system_reference_impl(obj)

    async def _reweight_salience(self) -> List[SalienceUpdate]:
        return await reweight_salience_impl(self)

    @staticmethod
    def _cluster_hash(cluster: List[Any]) -> str:
        return cluster_hash_impl(cluster)

    async def _promote_strong_signals(
        self,
        candidates: List[Any],
        score_map: Dict[str, SignalScore],
        cluster_consumed_ids: set[str],
        threshold: float,
    ) -> set[str]:
        return await promote_strong_signals_impl(
            self,
            candidates,
            score_map,
            cluster_consumed_ids,
            threshold,
        )

    async def _reweight_episode_salience(
        self,
        candidates: List[Any],
        cluster_consumed_ids: set[str],
        signal_promoted_ids: set[str],
    ) -> List[SalienceUpdate]:
        return await reweight_episode_salience_impl(
            self,
            candidates,
            cluster_consumed_ids,
            signal_promoted_ids,
        )

    async def _promote_identity_core(self, candidates: List[Any], score_map: Optional[Dict[str, SignalScore]] = None) -> int:
        return await promote_identity_core_impl(self, candidates, score_map=score_map)

    async def _recover_orphans(self, candidates: List[Any]) -> int:
        return await recover_orphans_impl(self, candidates)

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

    async def _consolidate_commitments(
        self,
        similarity_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        return await consolidate_commitments_impl(self, similarity_threshold=similarity_threshold)


    async def _llm_pick_commitment_survivor(
        self, cluster: List[Any]
    ) -> Optional[int]:
        return await llm_pick_commitment_survivor_impl(self, cluster)


    async def _extract_commitments_from_chat_logs(self) -> int:
        return await extract_commitments_from_chat_logs_impl(self)


    @staticmethod
    def _merged_commitment_status(cluster: List[Any]) -> CommitmentStatus:
        return merged_commitment_status_impl(cluster)


    def _collect_commitment_recovery_candidates(
        self,
        episodes: List[Any],
        cutoff: datetime,
    ) -> List[CommitmentRecoveryCandidate]:
        return collect_recovery_candidates(episodes, cutoff=cutoff)


    async def _pick_commitment_survivor(
        self,
        cluster: List[Any],
    ) -> Tuple[int, str]:
        return await pick_commitment_survivor_impl(self, cluster)


    def _refine_commitment_cluster(
        self,
        cluster: List[Any],
    ) -> List[List[Any]]:
        return refine_commitment_cluster_impl(cluster)


    def _cluster_has_obvious_duplicate(self, cluster: List[Any]) -> bool:
        return commitment_cluster_has_obvious_duplicate(cluster)


    def _heuristic_survivor_index(self, cluster: List[Any]) -> int:
        return commitment_heuristic_survivor_index(cluster)


    def _are_conservative_duplicates(self, left: Any, right: Any) -> bool:
        return commitment_duplicate_contents(left.content, right.content)


    def _are_conservative_duplicate_contents(self, left_content: str, right_content: str) -> bool:
        return commitment_duplicate_contents(left_content, right_content)


    @staticmethod
    def _commitment_key(content: str) -> str:
        return normalize_commitment_key(content)

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
