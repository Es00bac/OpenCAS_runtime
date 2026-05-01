"""Episode persistence and query helpers for MemoryStore."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional

from .models import Episode
from .store_serialization import affect_db_params, episode_db_params, row_to_episode as deserialize_episode

if TYPE_CHECKING:
    from .store import MemoryStore


EPISODE_UPSERT_SQL = """
INSERT INTO episodes (
    episode_id, created_at, kind, session_id, content,
    embedding_id, somatic_tag, affect_primary, affect_valence,
    affect_arousal, affect_certainty, affect_intensity,
    affect_social_target, affect_tags, salience, compacted,
    identity_core, confidence_score, access_count, last_accessed,
    used_successfully, used_unsuccessfully, identity_mutagen, payload
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(episode_id) DO UPDATE SET
    created_at = excluded.created_at,
    kind = excluded.kind,
    session_id = excluded.session_id,
    content = excluded.content,
    embedding_id = excluded.embedding_id,
    somatic_tag = excluded.somatic_tag,
    affect_primary = excluded.affect_primary,
    affect_valence = excluded.affect_valence,
    affect_arousal = excluded.affect_arousal,
    affect_certainty = excluded.affect_certainty,
    affect_intensity = excluded.affect_intensity,
    affect_social_target = excluded.affect_social_target,
    affect_tags = excluded.affect_tags,
    salience = excluded.salience,
    compacted = excluded.compacted,
    identity_core = excluded.identity_core,
    confidence_score = excluded.confidence_score,
    access_count = excluded.access_count,
    last_accessed = excluded.last_accessed,
    used_successfully = excluded.used_successfully,
    used_unsuccessfully = excluded.used_unsuccessfully,
    identity_mutagen = excluded.identity_mutagen,
    payload = excluded.payload
"""


async def update_episode_embedding(store: "MemoryStore", episode_id: str, embedding_id: str) -> None:
    """Update only the embedding_id for a specific episode."""
    assert store._db is not None
    await store._db.execute(
        "UPDATE episodes SET embedding_id = ? WHERE episode_id = ?",
        (embedding_id, episode_id),
    )
    await store._db.commit()


async def save_episodes_batch(store: "MemoryStore", episodes: List[Episode]) -> None:
    """Batch insert or update episodes in a single transaction."""
    assert store._db is not None
    params = [episode_db_params(episode) for episode in episodes]
    await store._db.executemany(EPISODE_UPSERT_SQL, params)
    await store._db.commit()


async def save_episode(store: "MemoryStore", episode: Episode) -> None:
    assert store._db is not None
    await store._db.execute(EPISODE_UPSERT_SQL, episode_db_params(episode))
    await store._db.commit()


async def get_episode(store: "MemoryStore", episode_id: str) -> Optional[Episode]:
    cursor = await store._execute(
        "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return deserialize_episode(row)


async def touch_episode(store: "MemoryStore", episode_id: str) -> None:
    """Increment episodic access counters when retrieval surfaces an episode."""
    now = datetime.now(timezone.utc).isoformat()
    await store._execute(
        "UPDATE episodes SET access_count = access_count + 1, last_accessed = ? WHERE episode_id = ?",
        (now, episode_id),
    )
    assert store._db is not None
    await store._db.commit()


async def list_episodes(
    store: "MemoryStore",
    session_id: Optional[str] = None,
    compacted: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> List[Episode]:
    conditions: List[str] = []
    params: List[Any] = []
    if session_id is not None:
        conditions.append("session_id = ?")
        params.append(session_id)
    if compacted is not None:
        conditions.append("compacted = ?")
        params.append(int(compacted))
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    cursor = await store._execute(
        f"""
        SELECT * FROM episodes
        {where}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (*params, limit, offset),
    )
    rows = await cursor.fetchall()
    return [deserialize_episode(row) for row in rows]


async def list_artifact_episodes(store: "MemoryStore", artifact_path: str) -> List[Episode]:
    """Return artifact-backed episodes for a specific relative artifact path."""
    assert store._db is not None
    cursor = await store._db.execute(
        """
        SELECT * FROM episodes
        WHERE json_extract(payload, '$.payload.artifact.path') = ?
        ORDER BY CAST(COALESCE(json_extract(payload, '$.payload.artifact.chunk_index'), 0) AS INTEGER) ASC
        """,
        (artifact_path,),
    )
    rows = await cursor.fetchall()
    return [deserialize_episode(row) for row in rows]


async def delete_episodes(store: "MemoryStore", episode_ids: List[str]) -> int:
    """Delete a set of episodes by id."""
    if not episode_ids:
        return 0
    assert store._db is not None
    placeholders = ", ".join("?" for _ in episode_ids)
    cursor = await store._db.execute(
        f"DELETE FROM episodes WHERE episode_id IN ({placeholders})",
        tuple(episode_ids),
    )
    await store._db.commit()
    return int(cursor.rowcount or 0)


async def mark_compacted(store: "MemoryStore", episode_ids: List[str]) -> None:
    if not episode_ids:
        return
    placeholders = ", ".join("?" for _ in episode_ids)
    await store._execute(
        f"UPDATE episodes SET compacted = 1 WHERE episode_id IN ({placeholders})",
        tuple(episode_ids),
    )
    assert store._db is not None
    await store._db.commit()


async def search_episodes_by_content(
    store: "MemoryStore",
    query: str,
    limit: int = 20,
) -> List[Episode]:
    """Search episodes using FTS5 over content."""
    assert store._db is not None
    escaped_query = query.replace('"', '""')
    safe_query = f'"{escaped_query}"'
    cursor = await store._db.execute(
        """
        SELECT e.* FROM episodes e
        JOIN episodes_fts fts ON e.rowid = fts.rowid
        WHERE episodes_fts MATCH ?
        ORDER BY fts.rank
        LIMIT ?
        """,
        (safe_query, limit),
    )
    rows = await cursor.fetchall()
    return [deserialize_episode(row) for row in rows]


async def list_recent_episodes(
    store: "MemoryStore",
    session_id: Optional[str] = None,
    limit: int = 20,
) -> List[Episode]:
    """Return the most recent episodes, optionally filtered by session."""
    return await store.list_episodes(session_id=session_id, limit=limit)


async def get_episodes_by_ids(store: "MemoryStore", episode_ids: List[str]) -> List[Episode]:
    """Fetch multiple episodes by their IDs."""
    if not episode_ids:
        return []
    placeholders = ", ".join("?" for _ in episode_ids)
    cursor = await store._execute(
        f"SELECT * FROM episodes WHERE episode_id IN ({placeholders})",
        tuple(episode_ids),
    )
    rows = await cursor.fetchall()
    return [deserialize_episode(row) for row in rows]


async def update_episode_affect(store: "MemoryStore", episode_id: str, affect: Any) -> None:
    """Update the affect fields of an episode."""
    assert store._db is not None
    await store._db.execute(
        """
        UPDATE episodes SET
            affect_primary = ?,
            affect_valence = ?,
            affect_arousal = ?,
            affect_certainty = ?,
            affect_intensity = ?,
            affect_social_target = ?,
            affect_tags = ?
        WHERE episode_id = ?
        """,
        (*affect_db_params(affect), episode_id),
    )
    await store._db.commit()


async def mark_episode_successful(store: "MemoryStore", episode_id: str) -> None:
    """Increment the successful-use counter for an episode."""
    assert store._db is not None
    await store._db.execute(
        "UPDATE episodes SET used_successfully = used_successfully + 1 WHERE episode_id = ?",
        (episode_id,),
    )
    await store._db.commit()


async def mark_episode_failed(store: "MemoryStore", episode_id: str) -> None:
    """Increment the unsuccessful-use counter for an episode."""
    assert store._db is not None
    await store._db.execute(
        "UPDATE episodes SET used_unsuccessfully = used_unsuccessfully + 1 WHERE episode_id = ?",
        (episode_id,),
    )
    await store._db.commit()


async def list_episodes_by_embedding_ids(
    store: "MemoryStore",
    embedding_ids: List[str],
) -> List[Episode]:
    """Fetch episodes whose embedding_id is in the given list."""
    if not embedding_ids:
        return []
    placeholders = ", ".join("?" for _ in embedding_ids)
    cursor = await store._execute(
        f"SELECT * FROM episodes WHERE embedding_id IN ({placeholders})",
        tuple(embedding_ids),
    )
    rows = await cursor.fetchall()
    return [deserialize_episode(row) for row in rows]


async def list_non_compacted_episodes(
    store: "MemoryStore",
    session_id: Optional[str] = None,
    limit: int = 100,
) -> List[Episode]:
    """Return episodes that have not been compacted, oldest first."""
    episodes = await store.list_episodes(session_id=session_id, compacted=False, limit=limit)
    episodes.reverse()
    return episodes


async def list_identity_core_episodes(
    store: "MemoryStore",
    limit: int = 100,
) -> List[Episode]:
    """Return episodes marked as identity core, most recent first."""
    assert store._db is not None
    cursor = await store._db.execute(
        "SELECT * FROM episodes WHERE identity_core = 1 ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = await cursor.fetchall()
    return [deserialize_episode(row) for row in rows]


async def prune_episodes_by_salience(store: "MemoryStore", threshold: float) -> int:
    """Remove episodes with salience below threshold. Returns deleted count."""
    assert store._db is not None
    cursor = await store._db.execute(
        "DELETE FROM episodes WHERE salience < ?",
        (threshold,),
    )
    await store._db.commit()
    return cursor.rowcount
