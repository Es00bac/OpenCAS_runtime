"""Row and parameter serialization helpers for the SQLite memory store."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping

from .models import CompactionRecord, EdgeKind, Episode, EpisodeEdge, EpisodeKind, Memory


def affect_db_params(affect: Any) -> tuple[Any, ...]:
    return (
        affect.primary_emotion.value if affect else None,
        affect.valence if affect else None,
        affect.arousal if affect else None,
        affect.certainty if affect else None,
        affect.intensity if affect else None,
        affect.social_target.value if affect else None,
        json.dumps(affect.emotion_tags) if affect and getattr(affect, 'emotion_tags', None) else None,
    )


def episode_db_params(episode: Episode) -> tuple[Any, ...]:
    return (
        str(episode.episode_id),
        episode.created_at.isoformat(),
        episode.kind.value,
        episode.session_id,
        episode.content,
        episode.embedding_id,
        episode.somatic_tag,
        *affect_db_params(episode.affect),
        episode.salience,
        int(episode.compacted),
        int(episode.identity_core),
        episode.confidence_score,
        episode.access_count,
        episode.last_accessed.isoformat() if episode.last_accessed else None,
        episode.used_successfully,
        episode.used_unsuccessfully,
        episode.model_dump_json(include={'payload'}),
    )


def edge_db_params(edge: EpisodeEdge) -> tuple[Any, ...]:
    return (
        str(edge.edge_id),
        edge.source_id,
        edge.target_id,
        edge.kind.value,
        edge.semantic_weight,
        edge.emotional_weight,
        edge.recency_weight,
        edge.structural_weight,
        edge.salience_weight,
        edge.causal_weight,
        edge.verification_weight,
        edge.actor_affinity_weight,
        edge.confidence,
        edge.created_at.isoformat(),
    )


def memory_db_params(memory: Memory) -> tuple[Any, ...]:
    return (
        str(memory.memory_id),
        memory.created_at.isoformat(),
        memory.updated_at.isoformat(),
        memory.content,
        memory.embedding_id,
        memory.model_dump_json(include={'source_episode_ids'}),
        memory.model_dump_json(include={'tags'}),
        memory.salience,
        memory.access_count,
        memory.last_accessed.isoformat() if memory.last_accessed else None,
    )


def compaction_db_params(record: CompactionRecord) -> tuple[Any, ...]:
    return (
        str(record.compaction_id),
        record.created_at.isoformat(),
        record.model_dump_json(include={'episode_ids'}),
        record.summary,
        record.removed_count,
    )


def row_to_edge(row: Mapping[str, Any]) -> EpisodeEdge:
    return EpisodeEdge(
        edge_id=row['edge_id'],
        source_id=row['source_id'],
        target_id=row['target_id'],
        kind=EdgeKind(row['kind']) if row['kind'] else EdgeKind.SEMANTIC,
        semantic_weight=row['semantic_weight'],
        emotional_weight=row['emotional_weight'],
        recency_weight=row['recency_weight'],
        structural_weight=row['structural_weight'],
        salience_weight=row['salience_weight'],
        causal_weight=row['causal_weight'],
        verification_weight=row['verification_weight'],
        actor_affinity_weight=row['actor_affinity_weight'],
        confidence=row['confidence'],
        created_at=datetime.fromisoformat(row['created_at']),
    )


def row_to_episode(row: Mapping[str, Any]) -> Episode:
    raw_payload = row['payload']
    payload = json.loads(raw_payload) if raw_payload else {}
    if 'payload' in payload:
        payload = payload['payload']

    affect = None
    if row['affect_primary'] is not None:
        from opencas.somatic.models import AffectState, PrimaryEmotion, SocialTarget

        raw_tags = row['affect_tags']
        tags = json.loads(raw_tags) if raw_tags else []
        affect = AffectState(
            primary_emotion=PrimaryEmotion(row['affect_primary']),
            valence=row['affect_valence'] or 0.0,
            arousal=row['affect_arousal'] or 0.0,
            certainty=row['affect_certainty'] or 0.0,
            intensity=row['affect_intensity'] or 0.0,
            social_target=SocialTarget(row['affect_social_target'] or 'user'),
            emotion_tags=tags,
        )

    return Episode(
        episode_id=row['episode_id'],
        created_at=datetime.fromisoformat(row['created_at']),
        kind=EpisodeKind(row['kind']),
        session_id=row['session_id'],
        content=row['content'],
        embedding_id=row['embedding_id'],
        somatic_tag=row['somatic_tag'],
        affect=affect,
        salience=row['salience'],
        compacted=bool(row['compacted']),
        identity_core=bool(row['identity_core']),
        confidence_score=row['confidence_score'],
        access_count=row['access_count'] if row['access_count'] is not None else 0,
        last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None,
        used_successfully=row['used_successfully'],
        used_unsuccessfully=row['used_unsuccessfully'],
        payload=payload,
    )


def row_to_memory(row: Mapping[str, Any]) -> Memory:
    raw_source_ids = row['source_episode_ids']
    source_ids = json.loads(raw_source_ids) if raw_source_ids else []
    if isinstance(source_ids, dict) and 'source_episode_ids' in source_ids:
        source_ids = source_ids['source_episode_ids']
    raw_tags = row['tags']
    tags = json.loads(raw_tags) if raw_tags else []
    if isinstance(tags, dict) and 'tags' in tags:
        tags = tags['tags']
    return Memory(
        memory_id=row['memory_id'],
        created_at=datetime.fromisoformat(row['created_at']),
        updated_at=datetime.fromisoformat(row['updated_at']),
        content=row['content'],
        embedding_id=row['embedding_id'],
        source_episode_ids=source_ids,
        tags=tags,
        salience=row['salience'],
        access_count=row['access_count'],
        last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None,
    )
