"""Search, graph expansion, and reranking helpers for memory retrieval."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from opencas.memory.store_serialization import row_to_episode
from opencas.somatic.models import AffectState, SocialTarget

from .models import RetrievalResult
from .resonance import (
    compute_edge_strength,
    compute_emotional_resonance,
    compute_reliability_score,
    compute_temporal_echo,
)
from .retrieval_query import extract_exact_handle_terms
from .retrieval_ranking import apply_temporal_decay


async def semantic_search(
    retriever,
    query: str,
    limit: int,
    affect_query: Optional[AffectState] = None,
    affect_weight: float = 0.25,
) -> List[RetrievalResult]:
    """Embedding-based similarity search over Memory records."""
    query_embed = await retriever.embeddings.embed(query, task_type="retrieval_query")
    similar = await retriever.embeddings.cache.search_similar(
        query_embed.vector, limit=limit, model_id=retriever.embeddings.model_id, query_text=query
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
        affect_embed = await retriever.embeddings.embed(
            affect_text,
            task_type="affect_query",
        )
        affect_vector = np.array(affect_embed.vector, dtype=np.float32)

    source_hashes = [record.source_hash for record, _ in similar]
    memories = await retriever.memory.list_memories_by_embedding_ids(source_hashes)
    episodes = await retriever.memory.list_episodes_by_embedding_ids(source_hashes)
    score_map = {record.source_hash: sim for record, sim in similar}
    vector_map = {
        record.source_hash: np.array(record.vector, dtype=np.float32)
        for record, _ in similar
    }

    results: List[RetrievalResult] = []
    seen_ids: set[Tuple[str, str]] = set()

    def add_result(
        source_type: str,
        source_id: str,
        content: str,
        semantic_score: float,
        embedding_id: Optional[str],
        mem=None,
        ep=None,
    ) -> None:
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
                    affect_sim = float(np.dot(cand_vec, affect_vector) / (c_norm * a_norm))
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
            add_result("memory", str(mem.memory_id), mem.content, score_map[mem.embedding_id], mem.embedding_id, mem=mem)
    for ep in episodes:
        if ep.embedding_id and ep.embedding_id in score_map:
            add_result("episode", str(ep.episode_id), ep.content, score_map[ep.embedding_id], ep.embedding_id, ep=ep)
    results.sort(key=lambda r: r.score, reverse=True)
    return results


async def keyword_search(retriever, query: str, limit: int) -> List[RetrievalResult]:
    """FTS keyword search over Episode records."""
    recall_intent = retriever.detect_personal_recall_intent(query)
    search_terms = retriever._keyword_queries_for(query, recall_intent=recall_intent)
    merged: Dict[Tuple[str, str], RetrievalResult] = {}
    for term_index, term in enumerate(search_terms):
        episodes = await retriever.memory.search_episodes_by_content(term, limit=limit)
        memories = await retriever.memory.search_memories_by_content(term, limit=limit)
        base = 1.0 if term_index == 0 else 0.8
        for rank, ep in enumerate(episodes):
            score = max(0.2, base - (0.05 * rank))
            key = ("episode", str(ep.episode_id))
            existing = merged.get(key)
            if existing is None or score > existing.score:
                merged[key] = RetrievalResult(
                    source_type="episode",
                    source_id=key[1],
                    content=ep.content,
                    score=score,
                    episode=ep,
                )
        for rank, mem in enumerate(memories):
            score = max(0.2, base - (0.05 * rank))
            key = ("memory", str(mem.memory_id))
            existing = merged.get(key)
            if existing is None or score > existing.score:
                merged[key] = RetrievalResult(
                    source_type="memory",
                    source_id=key[1],
                    content=mem.content,
                    score=score,
                    memory=mem,
                )
    results = sorted(merged.values(), key=lambda item: item.score, reverse=True)
    return results[:limit]


async def exact_handle_search(retriever, query: str, limit: int) -> List[RetrievalResult]:
    """Search exact path/checksum handles in episode content and payload JSON."""
    handles = extract_exact_handle_terms(query)
    needles = handles["paths"] + handles["checksums"]
    if not needles:
        return []

    db = getattr(retriever.memory, "_db", None)
    if db is None:
        return []

    clauses: List[str] = []
    params: List[str] = []
    for needle in needles:
        clauses.extend(["content LIKE ?", "payload LIKE ?"])
        like = f"%{needle}%"
        params.extend([like, like])

    cursor = await db.execute(
        f"""
        SELECT *
        FROM episodes
        WHERE {" OR ".join(clauses)}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        tuple(params + [limit]),
    )
    rows = await cursor.fetchall()

    results: List[RetrievalResult] = []
    seen: set[str] = set()
    for row in rows:
        episode = row_to_episode(row)
        episode_id = str(episode.episode_id)
        if episode_id in seen:
            continue
        seen.add(episode_id)
        content = _exact_handle_content(episode, handles)
        results.append(
            RetrievalResult(
                source_type="episode",
                source_id=episode_id,
                content=content,
                score=1.0,
                episode=episode,
            )
        )
    return results


def _exact_handle_content(episode, handles: dict[str, List[str]]) -> str:
    payload = getattr(episode, "payload", {}) or {}
    if not isinstance(payload, dict):
        payload = {}
    haystack = f"{getattr(episode, 'content', '')} {json.dumps(payload, default=str)}"
    parts = ["exact_handle_match"]
    tool_name = payload.get("tool_name")
    if tool_name:
        parts.append(f"tool={tool_name}")
    for path in handles["paths"]:
        if path in haystack:
            parts.append(f"path={path}")
    for checksum in handles["checksums"]:
        if checksum in haystack:
            parts.append(f"checksum={checksum}")
    return f"[{'; '.join(parts)}] {episode.content}"


async def expand_graph(
    retriever,
    seed_results: List[RetrievalResult],
    decay: float = 0.8,
    edge_limit: int = 12,
) -> List[RetrievalResult]:
    """Pull in graph-neighbor episodes for seed episode results."""
    seed_episodes = [r for r in seed_results if r.source_type == "episode"]
    if not seed_episodes:
        return seed_results

    neighbor_ids: set[str] = set()
    edge_map: Dict[str, List[object]] = {}
    for seed in seed_episodes:
        if retriever.episode_graph is not None:
            edges = await retriever.episode_graph.get_neighbors(
                seed.source_id, min_confidence=0.0, limit=edge_limit
            )
        else:
            edges = await retriever.memory.get_edges_for(
                seed.source_id, min_confidence=0.0, limit=edge_limit
            )
        edge_map[seed.source_id] = edges
        for edge in edges:
            neighbor_id = edge.target_id if edge.source_id == seed.source_id else edge.source_id
            if neighbor_id != seed.source_id:
                neighbor_ids.add(neighbor_id)

    if not neighbor_ids:
        return seed_results

    neighbors = await retriever.memory.get_episodes_by_ids(list(neighbor_ids))
    neighbor_by_id = {str(ep.episode_id): ep for ep in neighbors}

    result_by_key: Dict[Tuple[str, str], RetrievalResult] = {}
    for result in seed_results:
        result_by_key[(result.source_type, result.source_id)] = result

    for seed in seed_episodes:
        seed_score = seed.score or 0.0
        for edge in edge_map.get(seed.source_id, []):
            neighbor_id = edge.target_id if edge.source_id == seed.source_id else edge.source_id
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
    expanded.sort(key=lambda result: result.score, reverse=True)
    return expanded


def emotion_boost(
    results: List[RetrievalResult],
    tag: str,
    boost: float,
    query_affect: Optional[AffectState] = None,
) -> List[RetrievalResult]:
    """Blend emotion_boost_tag into emotional resonance rather than substring matching."""
    if not results:
        return results
    boosted: List[RetrievalResult] = []
    for result in results:
        new_score = result.score
        ep = getattr(result, "episode", None)
        if ep is not None and getattr(ep, "affect", None) is not None:
            if ep.affect.primary_emotion.value.lower() == tag.lower():
                new_score = result.score + boost
            if query_affect is not None:
                resonance = compute_emotional_resonance(query_affect, ep.affect)
                new_score = max(new_score, result.score + resonance * boost)
        boosted.append(result.model_copy(update={"score": new_score}))
    boosted.sort(key=lambda result: result.score, reverse=True)
    return boosted


def reciprocal_rank_fusion(
    rrf_k: int,
    result_lists: List[Tuple[str, List[RetrievalResult]]],
) -> List[RetrievalResult]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion."""
    scores: Dict[Tuple[str, str], float] = {}
    for _name, results in result_lists:
        for rank, result in enumerate(results, start=1):
            key = (result.source_type, result.source_id)
            scores[key] = scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

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
    fused.sort(key=lambda result: result.score, reverse=True)
    return fused


def seed_graph_relational_scores(retriever, candidate_map: Dict[Tuple[str, str], Dict[str, object]]) -> None:
    """Populate relational score for existing candidates before graph expansion."""
    musubi_base = 0.0
    if retriever.relational_engine is not None:
        musubi_base = max(0.0, min(1.0, 0.5 + retriever.relational_engine.state.musubi * 0.5))

    keys = list(candidate_map.keys())
    for key in keys:
        candidate = candidate_map[key]
        ep = candidate.get("episode")
        if ep is not None and getattr(ep, "affect", None) is not None:
            if ep.affect.social_target == SocialTarget.USER:
                candidate["relational_score"] = musubi_base


def populate_graph_candidates(
    retriever,
    candidate_map: Dict[Tuple[str, str], Dict[str, object]],
    graph_results: List[RetrievalResult],
    now: datetime,
    *,
    seed_keys: Optional[set[Tuple[str, str]]] = None,
) -> None:
    """Merge expanded graph results into an existing candidate map."""
    seed_keys = seed_keys or set()
    for result in graph_results:
        key = (result.source_type, result.source_id)
        if key in candidate_map:
            # Search-seeded items already earned their direct relevance score.
            # Reserve graph_score for true neighbor expansion so the seed node
            # does not crowd out its weakest connected neighbor after normalization.
            if key not in seed_keys:
                candidate_map[key]["graph_score"] = max(
                    float(candidate_map[key].get("graph_score", 0.0)),
                    float(result.score or 0.0),
                )
            continue
        ep = getattr(result, "episode", None)
        mem = getattr(result, "memory", None)
        obj = ep or mem
        created = getattr(obj, "created_at", now) if obj else now
        age_days = max(0.0, (now - created).total_seconds() / 86400.0) if obj else 0.0
        half_life = 180.0 if (obj and getattr(obj, "identity_core", False)) else 60.0
        candidate_map[key] = {
            "result": result,
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "recency_score": apply_temporal_decay(1.0, age_days, half_life_days=half_life),
            "salience_score": min(1.0, getattr(obj, "salience", 1.0) / 10.0) if obj else 0.1,
            "graph_score": result.score,
            "episode": ep,
            "memory": mem,
            "embedding": getattr(result, "embedding", None),
            "emotional_resonance": 0.0,
            "temporal_echo": compute_temporal_echo(now, created) if obj else 0.0,
            "reliability": compute_reliability_score(
                getattr(obj, "used_successfully", 0),
                getattr(obj, "used_unsuccessfully", 0),
            ) if obj else 0.8,
            "relational_score": 0.0,
        }

    musubi_base = 0.0
    if retriever.relational_engine is not None:
        musubi_base = max(0.0, min(1.0, 0.5 + retriever.relational_engine.state.musubi * 0.5))
    for result in graph_results:
        key = (result.source_type, result.source_id)
        candidate = candidate_map[key]
        ep = candidate.get("episode")
        if candidate["relational_score"] == 0.0 and ep is not None and getattr(ep, "affect", None) is not None:
            if ep.affect.social_target == SocialTarget.USER:
                candidate["relational_score"] = musubi_base
