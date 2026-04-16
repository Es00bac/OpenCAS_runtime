"""Shared memory API projection and retrieval helpers.

The dashboard uses these helpers from multiple route handlers, so keeping them
here avoids repeating embedding cache, projection, and retriever bootstrap
logic across route files.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from opencas.context.retriever import MemoryRetriever


def normalize_projection(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0, keepdims=True)
    scale = float(np.max(np.abs(centered))) if centered.size else 1.0
    if scale <= 0.0:
        scale = 1.0
    return centered / scale


def project_embedding_matrix(vectors: np.ndarray, requested_method: str = "auto") -> tuple[np.ndarray, str]:
    if len(vectors) == 1:
        return np.array([[0.0, 0.0]], dtype=np.float32), "single-point"

    methods = ["umap", "pca", "random"] if requested_method == "auto" else [requested_method, "random"]
    last_error: Optional[Exception] = None
    for method in methods:
        try:
            if method == "umap":
                from umap import UMAP

                reducer = UMAP(n_components=2, random_state=42)
                return reducer.fit_transform(vectors), "umap"
            if method == "pca":
                from sklearn.decomposition import PCA

                reducer = PCA(n_components=2)
                return reducer.fit_transform(vectors), "pca"
            if method == "random":
                rng = np.random.default_rng(seed=42)
                projection = vectors @ rng.standard_normal(size=(vectors.shape[1], 2))
                return projection, "random"
        except Exception as exc:  # pragma: no cover - fallback path depends on env extras
            last_error = exc
            continue
    raise RuntimeError(f"projection failed: {last_error}")


async def collect_embedding_points(
    embeddings: Any,
    items: Iterable[tuple[str, Dict[str, Any], Optional[str]]],
    requested_method: str = "auto",
) -> Dict[str, Any]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    missing: List[Dict[str, Any]] = []

    for node_type, node, embedding_id in items:
        if not embedding_id:
            missing.append({"node_type": node_type, "node_id": node["node_id"]})
            continue
        try:
            record = await embeddings.cache.get(embedding_id)
        except Exception:
            record = None
        if record is None or not record.vector:
            missing.append({"node_type": node_type, "node_id": node["node_id"]})
            continue
        node["embedding_model_id"] = record.model_id
        node["embedding_dimension"] = record.dimension
        grouped[int(record.dimension)].append(
            {
                "node_type": node_type,
                "node_id": node["node_id"],
                "vector": np.array(record.vector, dtype=np.float32),
                "model_id": record.model_id,
            }
        )

    point_map: Dict[str, Dict[str, Any]] = {}
    group_summaries: List[Dict[str, Any]] = []
    method_labels: List[str] = []
    x_offset = 0.0

    for dimension, group_items in sorted(grouped.items(), key=lambda item: (-len(item[1]), -item[0])):
        vectors = np.stack([entry["vector"] for entry in group_items])
        projection, method = project_embedding_matrix(vectors, requested_method=requested_method)
        normalized = normalize_projection(np.asarray(projection, dtype=np.float32))
        method_labels.append(method)
        for entry, coords in zip(group_items, normalized):
            point_map[entry["node_id"]] = {
                "x": round(float(coords[0] + x_offset), 6),
                "y": round(float(coords[1]), 6),
                "projection_group": f"{dimension}d",
                "projection_dimension": dimension,
                "projection_method": method,
                "embedding_model_id": entry["model_id"],
            }
        group_summaries.append(
            {
                "dimension": dimension,
                "count": len(group_items),
                "method": method,
                "model_ids": sorted({entry["model_id"] for entry in group_items}),
                "x_offset": round(x_offset, 3),
            }
        )
        x_offset += 3.0

    if missing:
        for index, entry in enumerate(missing):
            point_map[entry["node_id"]] = {
                "x": round(float(x_offset), 6),
                "y": round(float(-1.25 + index * 0.14), 6),
                "projection_group": "no-embedding",
                "projection_dimension": None,
                "projection_method": "none",
                "embedding_model_id": None,
            }
        group_summaries.append(
            {
                "dimension": None,
                "count": len(missing),
                "method": "none",
                "model_ids": [],
                "x_offset": round(x_offset, 3),
            }
        )
        method_labels.append("none")

    method_label = method_labels[0] if len(set(method_labels)) == 1 else "mixed"
    return {
        "points": point_map,
        "method": method_label,
        "groups": group_summaries,
        "missing_count": len(missing),
    }


async def enrich_nodes_with_embeddings(
    embeddings: Any,
    nodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    projection = await collect_embedding_points(
        embeddings,
        [
            (node.get("node_type", "node"), node, node.get("embedding_id"))
            for node in nodes
        ],
        requested_method="auto",
    )
    for node in nodes:
        point = projection["points"].get(node["node_id"])
        if point is None:
            continue
        node.update(
            {
                "x": point["x"],
                "y": point["y"],
                "projection_group": point["projection_group"],
                "projection_dimension": point["projection_dimension"],
                "projection_method": point["projection_method"],
                "embedding_model_id": point["embedding_model_id"],
            }
        )
    return projection


def build_memory_retriever(runtime: Any) -> MemoryRetriever:
    retriever = getattr(runtime, "retriever", None)
    if retriever is not None:
        return retriever
    return MemoryRetriever(
        memory=runtime.memory,
        embeddings=runtime.ctx.embeddings,
        episode_graph=getattr(runtime, "episode_graph", None),
        somatic_manager=getattr(runtime.ctx, "somatic", None),
        relational_engine=getattr(runtime, "relational", None),
    )


async def get_total_episode_count(memory: Any) -> int:
    assert memory._db is not None
    cursor = await memory._db.execute("SELECT COUNT(*) FROM episodes")
    row = await cursor.fetchone()
    return row[0] if row else 0
