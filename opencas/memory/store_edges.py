"""Episode-edge persistence and query helpers for MemoryStore."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from .models import EdgeKind, EpisodeEdge
from .store_serialization import edge_db_params, row_to_edge as deserialize_edge

if TYPE_CHECKING:
    from .store import MemoryStore


EDGE_UPSERT_SQL = """
INSERT INTO episode_edges (
    edge_id, source_id, target_id, kind, semantic_weight,
    emotional_weight, recency_weight, structural_weight,
    salience_weight, causal_weight, verification_weight, actor_affinity_weight,
    confidence, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(source_id, target_id) DO UPDATE SET
    kind = excluded.kind,
    semantic_weight = excluded.semantic_weight,
    emotional_weight = excluded.emotional_weight,
    recency_weight = excluded.recency_weight,
    structural_weight = excluded.structural_weight,
    salience_weight = excluded.salience_weight,
    causal_weight = excluded.causal_weight,
    verification_weight = excluded.verification_weight,
    actor_affinity_weight = excluded.actor_affinity_weight,
    confidence = excluded.confidence
"""


async def edge_exists(store: "MemoryStore", source_id: str, target_id: str) -> bool:
    """Check whether an edge already exists between two episodes."""
    assert store._db is not None
    cursor = await store._db.execute(
        "SELECT 1 FROM episode_edges WHERE source_id = ? AND target_id = ? LIMIT 1",
        (source_id, target_id),
    )
    row = await cursor.fetchone()
    return row is not None


async def save_edge(store: "MemoryStore", edge: EpisodeEdge) -> None:
    """Upsert an episode edge."""
    assert store._db is not None
    await store._db.execute(EDGE_UPSERT_SQL, edge_db_params(edge))
    await store._db.commit()


async def save_edges_batch(store: "MemoryStore", edges: List[EpisodeEdge]) -> None:
    """Batch upsert episode edges."""
    assert store._db is not None
    params = [edge_db_params(edge) for edge in edges]
    await store._db.executemany(EDGE_UPSERT_SQL, params)
    await store._db.commit()


async def get_edges_for(
    store: "MemoryStore",
    episode_id: str,
    min_confidence: float = 0.0,
    limit: int = 24,
    kind: Optional[EdgeKind] = None,
) -> List[EpisodeEdge]:
    """Return edges connected to an episode, ordered by confidence desc."""
    assert store._db is not None
    if kind is not None:
        cursor = await store._db.execute(
            """
            SELECT * FROM episode_edges
            WHERE (source_id = ? OR target_id = ?)
              AND confidence >= ?
              AND kind = ?
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            """,
            (episode_id, episode_id, min_confidence, kind.value, limit),
        )
    else:
        cursor = await store._db.execute(
            """
            SELECT * FROM episode_edges
            WHERE (source_id = ? OR target_id = ?)
              AND confidence >= ?
            ORDER BY confidence DESC, created_at DESC
            LIMIT ?
            """,
            (episode_id, episode_id, min_confidence, limit),
        )
    rows = await cursor.fetchall()
    return [deserialize_edge(row) for row in rows]


async def get_edges_for_batch(
    store: "MemoryStore",
    episode_ids: List[str],
    min_confidence: float = 0.0,
    kind: Optional[EdgeKind] = None,
) -> List[EpisodeEdge]:
    """Return edges connected to a set of episodes, ordered by confidence desc."""
    if not episode_ids:
        return []
    assert store._db is not None
    placeholders = ", ".join("?" for _ in episode_ids)
    if kind is not None:
        cursor = await store._db.execute(
            f"""
            SELECT * FROM episode_edges
            WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders}))
              AND confidence >= ?
              AND kind = ?
            ORDER BY confidence DESC, created_at DESC
            """,
            (*episode_ids, *episode_ids, min_confidence, kind.value),
        )
    else:
        cursor = await store._db.execute(
            f"""
            SELECT * FROM episode_edges
            WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders}))
              AND confidence >= ?
            ORDER BY confidence DESC, created_at DESC
            """,
            (*episode_ids, *episode_ids, min_confidence),
        )
    rows = await cursor.fetchall()
    return [deserialize_edge(row) for row in rows]


async def delete_edges_for(store: "MemoryStore", episode_id: str) -> None:
    """Delete all edges referencing an episode."""
    assert store._db is not None
    await store._db.execute(
        "DELETE FROM episode_edges WHERE source_id = ? OR target_id = ?",
        (episode_id, episode_id),
    )
    await store._db.commit()


async def boost_edge_confidence(
    store: "MemoryStore",
    episode_id: str,
    boost: float = 0.05,
    max_confidence: float = 1.0,
) -> int:
    """Boost confidence of all edges connected to *episode_id*."""
    assert store._db is not None
    cursor = await store._db.execute(
        """
        UPDATE episode_edges
        SET confidence = min(?, confidence + ?)
        WHERE source_id = ? OR target_id = ?
        """,
        (max_confidence, boost, episode_id, episode_id),
    )
    await store._db.commit()
    return cursor.rowcount


async def decay_all_edges(store: "MemoryStore", decay: float = 0.95) -> int:
    """Decay confidence of all edges by *decay* factor."""
    assert store._db is not None
    cursor = await store._db.execute(
        "UPDATE episode_edges SET confidence = confidence * ?",
        (decay,),
    )
    await store._db.commit()
    return cursor.rowcount


async def prune_weak_edges(store: "MemoryStore", min_confidence: float = 0.05) -> int:
    """Remove edges whose confidence fell below *min_confidence*."""
    assert store._db is not None
    cursor = await store._db.execute(
        "DELETE FROM episode_edges WHERE confidence < ?",
        (min_confidence,),
    )
    await store._db.commit()
    return cursor.rowcount
