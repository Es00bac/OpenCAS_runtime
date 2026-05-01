"""Signal promotion and salience maintenance helpers for nightly consolidation."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

import numpy as np

from opencas.memory import EdgeKind, EpisodeEdge, Memory

from .models import SalienceUpdate
from .signal_ranker import SignalScore


def has_obsolete_system_reference(obj: Any) -> bool:
    """Detect references to superseded systems/workspace paths."""
    text = str(getattr(obj, "content", "")).lower()
    payload = getattr(obj, "payload", {}) or {}
    source = str(payload.get("bulma_source", "")).lower()
    markers = ["openbulma-v4", "openbulma-v3", "openbulma-v2", "openclaw"]
    combined = f"{text} {source}"
    return any(marker in combined for marker in markers)


def cluster_hash(cluster: List[Any]) -> str:
    ids = sorted({str(episode.episode_id) for episode in cluster})
    return hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest()[:32]


async def reweight_salience(engine) -> List[SalienceUpdate]:
    """Boost frequently accessed memories and decay stale ones."""
    memories = await engine.memory.list_memories(limit=1000)
    now = datetime.now(timezone.utc)
    updates: List[SalienceUpdate] = []
    for mem in memories:
        new_salience = mem.salience
        if mem.access_count > 5:
            new_salience += 0.2
        if mem.access_count == 0 and mem.updated_at < now - timedelta(days=7):
            new_salience -= 0.3
        if mem.last_accessed and mem.last_accessed < now - timedelta(days=14):
            new_salience -= 0.2
        if not getattr(mem, "identity_core", False) and has_obsolete_system_reference(mem):
            new_salience -= 0.3
        new_salience = round(max(0.0, min(10.0, new_salience)), 3)
        if new_salience != mem.salience:
            await engine.memory.update_memory_salience(str(mem.memory_id), new_salience)
            updates.append(
                SalienceUpdate(
                    memory_id=str(mem.memory_id),
                    old_salience=mem.salience,
                    new_salience=new_salience,
                )
            )
    return updates


async def promote_strong_signals(
    engine,
    candidates: List[Any],
    score_map: Dict[str, SignalScore],
    cluster_consumed_ids: Set[str],
    threshold: float,
) -> Set[str]:
    """Promote high-scoring individual episodes to Memory."""
    promoted: Set[str] = set()
    for ep in candidates:
        ep_id = str(ep.episode_id)
        if ep_id in cluster_consumed_ids:
            continue
        score = score_map.get(ep_id)
        if not score or score.signal_score < threshold:
            continue

        embed_record = None
        if ep.embedding_id:
            embed_record = await engine.embeddings.cache.get(ep.embedding_id)
        if embed_record is None:
            try:
                embed_record = await engine.embeddings.embed(
                    ep.content,
                    task_type="memory_episode",
                )
            except Exception:
                pass

        if embed_record is not None:
            try:
                similar = await engine.embeddings.cache.search_similar(
                    embed_record.vector, limit=5
                )
                if similar:
                    top_hit, top_sim = similar[0]
                    if top_sim > 0.92 and top_hit.source_hash != embed_record.source_hash:
                        continue
            except Exception as exc:
                if engine.tracer:
                    engine.tracer.log(
                        EventKind.TOOL_CALL,
                        "signal_maintenance_search_similar_failed",
                        {"error": str(exc), "episode_id": ep_id},
                    )

        memory = Memory(
            content=ep.content,
            source_episode_ids=[ep_id],
            tags=["consolidation", "strong_signal"],
            embedding_id=embed_record.source_hash if embed_record else None,
            salience=round(ep.salience, 3),
        )
        await engine.memory.save_memory(memory)
        promoted.add(ep_id)
    return promoted


async def reweight_episode_salience(
    engine,
    candidates: List[Any],
    cluster_consumed_ids: Set[str],
    signal_promoted_ids: Set[str],
) -> List[SalienceUpdate]:
    """Boost or decay episode salience based on promotion and graph state."""
    now = datetime.now(timezone.utc)
    updates: List[SalienceUpdate] = []
    promoted_all = cluster_consumed_ids | signal_promoted_ids
    for ep in candidates:
        ep_id = str(ep.episode_id)
        new_salience = ep.salience

        if ep_id in promoted_all:
            new_salience += 0.3

        edges = await engine.memory.get_edges_for(ep_id, min_confidence=0.0, limit=100)
        new_salience += (len(edges) // 5) * 0.1

        if ep_id not in promoted_all and ep.created_at < now - timedelta(days=7):
            new_salience -= 0.2

        if not ep.identity_core and has_obsolete_system_reference(ep):
            new_salience -= 0.3

        new_salience = round(max(0.0, min(10.0, new_salience)), 3)
        if new_salience != ep.salience:
            old_salience = ep.salience
            ep.salience = new_salience
            await engine.memory.save_episode(ep)
            updates.append(
                SalienceUpdate(
                    memory_id=ep_id,
                    old_salience=old_salience,
                    new_salience=new_salience,
                )
            )
    return updates


async def promote_identity_core(
    engine,
    candidates: List[Any],
    score_map: Optional[Dict[str, SignalScore]] = None,
) -> int:
    """Promote high-degree or high-signal episodes to identity core."""
    candidate_ids = [str(ep.episode_id) for ep in candidates]
    promoted = 0
    for ep_id in candidate_ids:
        edges = await engine.memory.get_edges_for(ep_id, min_confidence=0.0, limit=100)
        degree = len(edges)
        signal_score = 0.0
        if score_map:
            score = score_map.get(ep_id)
            if score:
                signal_score = score.signal_score
        if degree >= 5 or signal_score >= 0.80:
            ep = await engine.memory.get_episode(ep_id)
            if ep and (ep.affect is None or ep.affect.valence >= 0.0):
                if not ep.identity_core:
                    ep.identity_core = True
                    await engine.memory.save_episode(ep)
                    promoted += 1
    return promoted


async def recover_orphans(engine, candidates: List[Any]) -> int:
    """Find episodes with no edges and bridge them to their nearest neighbor."""
    if not candidates:
        return 0
    recovered = 0
    for ep in candidates:
        ep_id = str(ep.episode_id)
        edges = await engine.memory.get_edges_for(ep_id, min_confidence=0.0, limit=1)
        if edges or not ep.embedding_id:
            continue
        best_neighbor: Optional[str] = None
        best_sim = -1.0
        rec = await engine.embeddings.cache.get(ep.embedding_id)
        if rec is None:
            continue
        vec = np.array(rec.vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0:
            continue
        for other in candidates:
            other_id = str(other.episode_id)
            if other_id == ep_id or not other.embedding_id:
                continue
            other_rec = await engine.embeddings.cache.get(other.embedding_id)
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
            edge = EpisodeEdge(
                source_id=ep_id,
                target_id=best_neighbor,
                kind=EdgeKind.SEMANTIC,
                confidence=0.1,
            )
            await engine.memory.save_edge(edge)
            recovered += 1
    return recovered
