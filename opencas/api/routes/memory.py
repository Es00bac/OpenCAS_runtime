"""Memory API routes for the OpenCAS dashboard."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


from opencas.api.memory_projection import (
    build_memory_retriever,
    collect_embedding_points,
    enrich_nodes_with_embeddings,
    get_total_episode_count,
)
from opencas.context.resonance import compute_edge_strength
from opencas.api.memory_serialization import (
    affect_to_dict,
    edge_signal_summary,
    edge_to_dict,
    episode_to_dict,
    memory_to_dict,
    truncate_memory_text,
)
from opencas.context.retriever import MemoryRetriever
from opencas.memory import EdgeKind

router = APIRouter(tags=["memory"])


class EpisodesResponse(BaseModel):
    episodes: List[Dict[str, Any]]
    total: int


class GraphResponse(BaseModel):
    episode_id: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class MemoryStatsResponse(BaseModel):
    episode_count: int
    memory_count: int
    edge_count: int
    compacted_count: int
    identity_core_count: int
    avg_salience: float
    affect_distribution: Dict[str, int]


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]


class ProjectionResponse(BaseModel):
    points: List[Dict[str, Any]]
    method: str
    groups: List[Dict[str, Any]] = Field(default_factory=list)


class MemoryLandscapeResponse(BaseModel):
    stats: Dict[str, Any]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    projection: Dict[str, Any]


class RetrievalInspectResponse(BaseModel):
    query: str
    weights: Dict[str, float]
    candidates: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    meta: Dict[str, Any]


class NodeDetailResponse(BaseModel):
    node: Dict[str, Any]
    neighbors: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any]


def build_memory_router(runtime: Any) -> APIRouter:
    """Build memory routes wired to *runtime*."""
    r = APIRouter(prefix="/api/memory", tags=["memory"])

    @r.get("/episodes", response_model=EpisodesResponse)
    async def list_episodes(limit: int = 50, offset: int = 0) -> EpisodesResponse:
        mem = runtime.memory
        episodes = await mem.list_episodes(limit=limit, offset=offset)
        total = await get_total_episode_count(mem)
        return EpisodesResponse(
            episodes=[episode_to_dict(ep) for ep in episodes],
            total=total,
        )

    @r.get("/graph", response_model=GraphResponse)
    async def get_graph(episode_id: str, limit: int = 24) -> GraphResponse:
        graph = runtime.episode_graph
        edges = await graph.get_neighbors(episode_id, limit=limit)
        seen = {episode_id}
        node_ids = [episode_id]
        for edge in edges:
            other = edge.target_id if edge.source_id == episode_id else edge.source_id
            if other not in seen:
                seen.add(other)
                node_ids.append(other)
        episodes = await runtime.memory.get_episodes_by_ids(node_ids)
        return GraphResponse(
            episode_id=episode_id,
            nodes=[episode_to_dict(ep) for ep in episodes],
            edges=[edge_to_dict(e) for e in edges],
        )

    @r.get("/stats", response_model=MemoryStatsResponse)
    async def get_memory_stats() -> MemoryStatsResponse:
        stats = await runtime.memory.get_stats()
        return MemoryStatsResponse(
            episode_count=stats["episode_count"],
            memory_count=stats["memory_count"],
            edge_count=stats["edge_count"],
            compacted_count=stats["compacted_count"],
            identity_core_count=stats["identity_core_count"],
            avg_salience=stats["avg_salience"],
            affect_distribution=stats["affect_distribution"],
        )

    @r.get("/search", response_model=SearchResponse)
    async def search_memories(query: str, limit: int = 20) -> SearchResponse:
        episodes = await runtime.memory.search_episodes_by_content(query, limit=limit)
        return SearchResponse(
            query=query,
            results=[episode_to_dict(ep) for ep in episodes],
        )

    @r.get("/embedding-projection", response_model=ProjectionResponse)
    async def embedding_projection(
        limit: int = 500,
        method: str = "auto",
    ) -> ProjectionResponse:
        mem = runtime.memory
        episodes = await mem.list_episodes(limit=limit)
        points_data = await collect_embedding_points(
            runtime.ctx.embeddings,
            [
                (
                    "episode",
                    {
                        "node_id": f"episode:{ep.episode_id}",
                        "episode_id": str(ep.episode_id),
                        "salience": ep.salience,
                        "kind": ep.kind.value,
                    },
                    ep.embedding_id,
                )
                for ep in episodes
            ],
            requested_method=method,
        )

        points: List[Dict[str, Any]] = []
        for ep in episodes:
            point = points_data["points"].get(f"episode:{ep.episode_id}")
            if point is None:
                continue
            points.append(
                {
                    "episode_id": str(ep.episode_id),
                    "x": point["x"],
                    "y": point["y"],
                    "salience": ep.salience,
                    "kind": ep.kind.value,
                    "projection_group": point["projection_group"],
                    "projection_dimension": point["projection_dimension"],
                    "embedding_model_id": point["embedding_model_id"],
                }
            )

        return ProjectionResponse(
            points=points,
            method=points_data["method"],
            groups=points_data["groups"],
        )

    @r.get("/landscape", response_model=MemoryLandscapeResponse)
    async def memory_landscape(
        limit: int = 140,
        query: Optional[str] = None,
        session_id: Optional[str] = None,
        kind: Optional[str] = None,
        emotion: Optional[str] = None,
        max_age_days: Optional[float] = None,
        min_edge_confidence: float = 0.18,
        edge_kind: Optional[str] = None,
        include_memories: bool = True,
        method: str = "auto",
    ) -> MemoryLandscapeResponse:
        memory = runtime.memory
        now = datetime.now(timezone.utc)
        if query:
            episodes = await memory.search_episodes_by_content(query, limit=limit)
        else:
            episodes = await memory.list_episodes(limit=limit)

        filtered_episodes = []
        for episode in episodes:
            if session_id and episode.session_id != session_id:
                continue
            if kind and episode.kind.value != kind:
                continue
            episode_age_days = max(0.0, (now - episode.created_at).total_seconds() / 86400.0)
            if max_age_days is not None and episode_age_days > max_age_days:
                continue
            primary_emotion = getattr(getattr(episode, "affect", None), "primary_emotion", None)
            if emotion and getattr(primary_emotion, "value", None) != emotion:
                continue
            filtered_episodes.append(episode)

        memory_items = await memory.list_memories(limit=max(25, min(limit, 120))) if include_memories else []
        memories = []
        node_map: Dict[str, Dict[str, Any]] = {}
        episode_lookup: Dict[str, Any] = {}

        for episode in filtered_episodes:
            age_days = max(0.0, (now - episode.created_at).total_seconds() / 86400.0)
            node_id = f"episode:{episode.episode_id}"
            node_map[node_id] = {
                "node_id": node_id,
                "node_type": "episode",
                "episode_id": str(episode.episode_id),
                "label": truncate_memory_text(episode.content, 72),
                "content": episode.content,
                "created_at": episode.created_at.isoformat(),
                "age_days": round(age_days, 3),
                "kind": episode.kind.value,
                "session_id": episode.session_id,
                "salience": episode.salience,
                "confidence_score": episode.confidence_score,
                "somatic_tag": episode.somatic_tag,
                "compacted": episode.compacted,
                "identity_core": episode.identity_core,
                "used_successfully": episode.used_successfully,
                "used_unsuccessfully": episode.used_unsuccessfully,
                "affect": affect_to_dict(episode.affect),
                "embedding_id": episode.embedding_id,
                "connection_count": 0,
            }
            episode_lookup[str(episode.episode_id)] = episode

        for memory_item in memory_items:
            age_days = max(0.0, (now - memory_item.updated_at).total_seconds() / 86400.0)
            if max_age_days is not None and age_days > max_age_days:
                continue
            memories.append(memory_item)
            node_id = f"memory:{memory_item.memory_id}"
            node_map[node_id] = {
                "node_id": node_id,
                "node_type": "memory",
                "memory_id": str(memory_item.memory_id),
                "label": truncate_memory_text(memory_item.content, 72),
                "content": memory_item.content,
                "created_at": memory_item.created_at.isoformat(),
                "updated_at": memory_item.updated_at.isoformat(),
                "age_days": round(age_days, 3),
                "kind": "memory",
                "session_id": None,
                "salience": memory_item.salience,
                "confidence_score": None,
                "somatic_tag": None,
                "compacted": False,
                "identity_core": False,
                "used_successfully": None,
                "used_unsuccessfully": None,
                "affect": None,
                "embedding_id": memory_item.embedding_id,
                "tags": list(memory_item.tags),
                "access_count": memory_item.access_count,
                "last_accessed": memory_item.last_accessed.isoformat() if memory_item.last_accessed else None,
                "source_episode_ids": list(memory_item.source_episode_ids),
                "connection_count": 0,
            }

        projection = await collect_embedding_points(
            runtime.ctx.embeddings,
            [
                (node["node_type"], node, node.get("embedding_id"))
                for node in node_map.values()
            ],
            requested_method=method,
        )
        for node_id, point in projection["points"].items():
            node_map[node_id].update(
                {
                    "x": point["x"],
                    "y": point["y"],
                    "projection_group": point["projection_group"],
                    "projection_dimension": point["projection_dimension"],
                    "projection_method": point["projection_method"],
                    "embedding_model_id": point["embedding_model_id"],
                }
            )

        edges_out: List[Dict[str, Any]] = []
        episode_ids = list(episode_lookup.keys())
        if episode_ids:
            edges = await memory.get_edges_for_batch(
                episode_ids,
                min_confidence=min_edge_confidence,
            )
            for edge in edges:
                source_node = f"episode:{edge.source_id}"
                target_node = f"episode:{edge.target_id}"
                if source_node not in node_map or target_node not in node_map:
                    continue
                if edge_kind and edge.kind.value != edge_kind:
                    continue
                source_episode = episode_lookup.get(edge.source_id)
                target_episode = episode_lookup.get(edge.target_id)
                time_distance_days = None
                if source_episode is not None and target_episode is not None:
                    delta = abs(source_episode.created_at - target_episode.created_at)
                    time_distance_days = round(delta.total_seconds() / 86400.0, 3)
                edges_out.append(
                    {
                        "edge_id": str(edge.edge_id),
                        "source_node_id": source_node,
                        "target_node_id": target_node,
                        "kind": edge.kind.value,
                        "confidence": round(float(edge.confidence), 6),
                        "strength": round(float(compute_edge_strength(edge)), 6),
                        "created_at": edge.created_at.isoformat(),
                        "time_distance_days": time_distance_days,
                        "weights": {
                            "semantic": edge.semantic_weight,
                            "emotional": edge.emotional_weight,
                            "recency": edge.recency_weight,
                            "structural": edge.structural_weight,
                            "salience": edge.salience_weight,
                            "causal": edge.causal_weight,
                            "verification": edge.verification_weight,
                            "actor_affinity": edge.actor_affinity_weight,
                        },
                    }
                )
                node_map[source_node]["connection_count"] += 1
                node_map[target_node]["connection_count"] += 1

        if include_memories:
            for memory_item in memories:
                memory_node_id = f"memory:{memory_item.memory_id}"
                for source_episode_id in memory_item.source_episode_ids:
                    source_node_id = f"episode:{source_episode_id}"
                    if source_node_id not in node_map:
                        continue
                    strength = min(1.0, 0.35 + float(memory_item.salience) / 12.0)
                    if edge_kind and edge_kind != "distilled_from":
                        continue
                    edges_out.append(
                        {
                            "edge_id": f"memory-link:{memory_item.memory_id}:{source_episode_id}",
                            "source_node_id": source_node_id,
                            "target_node_id": memory_node_id,
                            "kind": "distilled_from",
                            "confidence": round(strength, 6),
                            "strength": round(strength, 6),
                            "created_at": memory_item.updated_at.isoformat(),
                            "time_distance_days": None,
                            "weights": {},
                            "synthetic": True,
                        }
                    )
                    node_map[source_node_id]["connection_count"] += 1
                    node_map[memory_node_id]["connection_count"] += 1

        emotion_distribution = Counter(
            node.get("affect", {}).get("primary_emotion")
            for node in node_map.values()
            if node.get("affect")
        )
        kind_distribution = Counter(node.get("kind") for node in node_map.values())
        node_type_distribution = Counter(node.get("node_type") for node in node_map.values())
        edge_kind_distribution = Counter(edge.get("kind") for edge in edges_out)
        model_distribution = Counter(
            node.get("embedding_model_id")
            for node in node_map.values()
            if node.get("embedding_model_id")
        )
        age_values = [node["age_days"] for node in node_map.values()]
        edge_strength_values = [float(edge.get("strength", 0.0)) for edge in edges_out]

        stats = {
            "query": query,
            "session_id": session_id,
            "visible_episode_count": len(filtered_episodes),
            "visible_memory_count": len(memories) if include_memories else 0,
            "visible_edge_count": len(edges_out),
            "projection_method": projection["method"],
            "projection_groups": projection["groups"],
            "embeddingless_node_count": projection["missing_count"],
            "kind_distribution": dict(kind_distribution),
            "node_type_distribution": dict(node_type_distribution),
            "edge_kind_distribution": dict(edge_kind_distribution),
            "emotion_distribution": {k: v for k, v in emotion_distribution.items() if k},
            "embedding_model_distribution": dict(model_distribution),
            "time_span_days": round(max(age_values) if age_values else 0.0, 3),
            "freshest_visible_age_days": round(min(age_values) if age_values else 0.0, 3),
            "average_edge_strength": round(sum(edge_strength_values) / len(edge_strength_values), 4) if edge_strength_values else 0.0,
            "max_age_days": max_age_days,
            "min_edge_confidence": min_edge_confidence,
            "edge_kind": edge_kind,
        }

        nodes = sorted(
            node_map.values(),
            key=lambda item: (
                item["node_type"] != "episode",
                item.get("age_days", 0.0),
            ),
        )
        return MemoryLandscapeResponse(
            stats=stats,
            nodes=nodes,
            edges=edges_out,
            projection={
                "method": projection["method"],
                "groups": projection["groups"],
            },
        )

    @r.get("/node-detail", response_model=NodeDetailResponse)
    async def node_detail(
        node_id: str,
        limit: int = 18,
        min_confidence: float = 0.12,
        edge_kind: Optional[str] = None,
    ) -> NodeDetailResponse:
        memory = runtime.memory
        now = datetime.now(timezone.utc)
        kind_filter = None
        if edge_kind and edge_kind != "distilled_from":
            try:
                kind_filter = EdgeKind(edge_kind)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Unsupported edge kind: {edge_kind}") from exc

        if ":" not in node_id:
            raise HTTPException(status_code=400, detail="node_id must be prefixed with episode: or memory:")
        node_type, raw_id = node_id.split(":", 1)

        node_payload: Optional[Dict[str, Any]] = None
        neighbors: List[Dict[str, Any]] = []
        edge_payloads: List[Dict[str, Any]] = []

        if node_type == "episode":
            episode = await memory.get_episode(raw_id)
            if episode is None:
                raise HTTPException(status_code=404, detail=f"Unknown episode node: {raw_id}")
            age_days = max(0.0, (now - episode.created_at).total_seconds() / 86400.0)
            node_payload = {
                "node_id": node_id,
                "node_type": "episode",
                **episode_to_dict(episode),
                "label": truncate_memory_text(episode.content, 72),
                "age_days": round(age_days, 3),
            }

            episode_edges = await memory.get_edges_for(
                raw_id,
                min_confidence=min_confidence,
                limit=limit,
                kind=kind_filter,
            )
            neighbor_ids = []
            for edge in episode_edges:
                other_id = edge.target_id if edge.source_id == raw_id else edge.source_id
                neighbor_ids.append(other_id)
            episode_lookup = {
                str(item.episode_id): item
                for item in await memory.get_episodes_by_ids(list(dict.fromkeys(neighbor_ids)))
            }

            for edge in episode_edges:
                other_id = edge.target_id if edge.source_id == raw_id else edge.source_id
                other_episode = episode_lookup.get(other_id)
                if other_episode is None:
                    continue
                other_age_days = max(0.0, (now - other_episode.created_at).total_seconds() / 86400.0)
                neighbor_payload = {
                    "node_id": f"episode:{other_id}",
                    "node_type": "episode",
                    **episode_to_dict(other_episode),
                    "label": truncate_memory_text(other_episode.content, 72),
                    "age_days": round(other_age_days, 3),
                }
                neighbors.append(neighbor_payload)
                edge_dict = edge_to_dict(edge)
                edge_dict.update(
                    {
                        "source_node_id": f"episode:{edge.source_id}",
                        "target_node_id": f"episode:{edge.target_id}",
                        "other_node_id": f"episode:{other_id}",
                        "time_distance_days": round(
                            abs((other_episode.created_at - episode.created_at).total_seconds()) / 86400.0,
                            3,
                        ),
                    }
                )
                edge_dict.update(edge_signal_summary(edge_dict))
                edge_payloads.append(edge_dict)

            if edge_kind in (None, "", "distilled_from"):
                memory_items = await memory.list_memories(limit=max(limit * 4, 80))
                connected_memories = [
                    memory_item
                    for memory_item in memory_items
                    if raw_id in list(memory_item.source_episode_ids)
                ][:limit]
                for memory_item in connected_memories:
                    memory_age_days = max(0.0, (now - memory_item.updated_at).total_seconds() / 86400.0)
                    neighbor_payload = {
                        "node_id": f"memory:{memory_item.memory_id}",
                        "node_type": "memory",
                        **memory_to_dict(memory_item),
                        "label": truncate_memory_text(memory_item.content, 72),
                        "age_days": round(memory_age_days, 3),
                    }
                    neighbors.append(neighbor_payload)
                    strength = min(1.0, 0.35 + float(memory_item.salience) / 12.0)
                    edge_payloads.append(
                        {
                            "edge_id": f"memory-link:{memory_item.memory_id}:{raw_id}",
                            "source_node_id": node_id,
                            "target_node_id": neighbor_payload["node_id"],
                            "other_node_id": neighbor_payload["node_id"],
                            "kind": "distilled_from",
                            "confidence": round(strength, 6),
                            "strength": round(strength, 6),
                            "created_at": memory_item.updated_at.isoformat(),
                            "time_distance_days": None,
                            "strongest_signal": "salience",
                            "strongest_signal_weight": round(float(memory_item.salience), 6),
                            "signal_weights": {"salience": float(memory_item.salience)},
                        }
                    )
        elif node_type == "memory":
            memory_item = await memory.get_memory(raw_id)
            if memory_item is None:
                raise HTTPException(status_code=404, detail=f"Unknown memory node: {raw_id}")
            age_days = max(0.0, (now - memory_item.updated_at).total_seconds() / 86400.0)
            node_payload = {
                "node_id": node_id,
                "node_type": "memory",
                **memory_to_dict(memory_item),
                "label": truncate_memory_text(memory_item.content, 72),
                "age_days": round(age_days, 3),
            }

            source_episodes = await memory.get_episodes_by_ids(list(memory_item.source_episode_ids))
            for episode in source_episodes[:limit]:
                other_age_days = max(0.0, (now - episode.created_at).total_seconds() / 86400.0)
                neighbor_payload = {
                    "node_id": f"episode:{episode.episode_id}",
                    "node_type": "episode",
                    **episode_to_dict(episode),
                    "label": truncate_memory_text(episode.content, 72),
                    "age_days": round(other_age_days, 3),
                }
                neighbors.append(neighbor_payload)
                strength = min(1.0, 0.35 + float(memory_item.salience) / 12.0)
                edge_payloads.append(
                    {
                        "edge_id": f"memory-link:{memory_item.memory_id}:{episode.episode_id}",
                        "source_node_id": neighbor_payload["node_id"],
                        "target_node_id": node_id,
                        "other_node_id": neighbor_payload["node_id"],
                        "kind": "distilled_from",
                        "confidence": round(strength, 6),
                        "strength": round(strength, 6),
                        "created_at": memory_item.updated_at.isoformat(),
                        "time_distance_days": round(
                            abs((memory_item.updated_at - episode.created_at).total_seconds()) / 86400.0,
                            3,
                        ),
                        "strongest_signal": "salience",
                        "strongest_signal_weight": round(float(memory_item.salience), 6),
                        "signal_weights": {"salience": float(memory_item.salience)},
                    }
                )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported node type: {node_type}")

        unique_neighbors: Dict[str, Dict[str, Any]] = {}
        for neighbor in neighbors:
            unique_neighbors[neighbor["node_id"]] = neighbor
        neighbors = list(unique_neighbors.values())

        await enrich_nodes_with_embeddings(
            runtime.ctx.embeddings,
            [node_payload, *neighbors],
        )

        edge_kind_distribution = Counter(edge.get("kind") for edge in edge_payloads)
        neighbor_type_distribution = Counter(node.get("node_type") for node in neighbors)
        return NodeDetailResponse(
            node=node_payload,
            neighbors=sorted(
                neighbors,
                key=lambda item: (
                    item.get("node_type") != "episode",
                    item.get("age_days", 0.0),
                ),
            ),
            edges=sorted(edge_payloads, key=lambda item: float(item.get("strength", 0.0)), reverse=True),
            stats={
                "neighbor_count": len(neighbors),
                "edge_count": len(edge_payloads),
                "edge_kind_distribution": dict(edge_kind_distribution),
                "neighbor_type_distribution": dict(neighbor_type_distribution),
            },
        )

    @r.get("/retrieval-inspect", response_model=RetrievalInspectResponse)
    async def retrieval_inspect(
        query: str,
        limit: int = 12,
        session_id: Optional[str] = None,
        min_confidence: float = 0.15,
        lambda_param: float = 0.5,
        expand_graph: bool = True,
        semantic_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["semantic_score"],
        keyword_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["keyword_score"],
        recency_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["recency_score"],
        salience_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["salience_score"],
        graph_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["graph_score"],
        emotional_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["emotional_resonance"],
        temporal_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["temporal_echo"],
        reliability_weight: float = MemoryRetriever.DEFAULT_FUSION_WEIGHTS["reliability"],
    ) -> RetrievalInspectResponse:
        retriever = _build_memory_retriever(runtime)
        inspection = await retriever.inspect(
            query=query,
            session_id=session_id,
            limit=limit,
            expand_graph=expand_graph,
            min_confidence=min_confidence,
            lambda_param=lambda_param,
            weights={
                "semantic_score": semantic_weight,
                "keyword_score": keyword_weight,
                "recency_score": recency_weight,
                "salience_score": salience_weight,
                "graph_score": graph_weight,
                "emotional_resonance": emotional_weight,
                "temporal_echo": temporal_weight,
                "reliability": reliability_weight,
            },
        )

        results = []
        for item in inspection["results"]:
            payload = {
                "source_type": item.source_type,
                "source_id": item.source_id,
                "content": item.content,
                "content_preview": truncate_memory_text(item.content, 200),
                "score": round(float(item.score), 6),
            }
            if item.episode is not None:
                payload["episode"] = episode_to_dict(item.episode)
                payload["node_id"] = f"episode:{item.source_id}"
            if item.memory is not None:
                payload["memory"] = memory_to_dict(item.memory)
                payload["node_id"] = f"memory:{item.source_id}"
            results.append(payload)

        return RetrievalInspectResponse(
            query=query,
            weights=inspection["weights"],
            candidates=inspection["candidates"],
            results=results,
            meta=inspection["meta"],
        )

    return r
