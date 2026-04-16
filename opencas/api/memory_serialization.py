"""Shared memory API serialization helpers.

These helpers keep the memory routes thin and provide one place to adjust the
operator-facing JSON shape for episodes, memories, and graph edges.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from opencas.context.resonance import compute_edge_strength


def truncate_memory_text(text: str, limit: int = 200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def affect_to_dict(affect: Any) -> Optional[Dict[str, Any]]:
    if affect is None:
        return None
    return {
        "primary_emotion": affect.primary_emotion.value,
        "valence": affect.valence,
        "arousal": affect.arousal,
        "certainty": affect.certainty,
        "intensity": affect.intensity,
        "social_target": affect.social_target.value,
        "emotion_tags": affect.emotion_tags,
    }


def episode_to_dict(ep: Any) -> Dict[str, Any]:
    artifact = (ep.payload or {}).get("artifact") if getattr(ep, "payload", None) else None
    return {
        "episode_id": str(ep.episode_id),
        "created_at": ep.created_at.isoformat(),
        "kind": ep.kind.value,
        "session_id": ep.session_id,
        "content": ep.content,
        "content_preview": truncate_memory_text(ep.content, 240),
        "salience": ep.salience,
        "compacted": ep.compacted,
        "identity_core": ep.identity_core,
        "confidence_score": ep.confidence_score,
        "used_successfully": ep.used_successfully,
        "used_unsuccessfully": ep.used_unsuccessfully,
        "somatic_tag": ep.somatic_tag,
        "embedding_id": ep.embedding_id,
        "affect": affect_to_dict(ep.affect),
        "artifact": artifact,
    }


def memory_to_dict(memory: Any) -> Dict[str, Any]:
    return {
        "memory_id": str(memory.memory_id),
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
        "content": memory.content,
        "content_preview": truncate_memory_text(memory.content, 240),
        "embedding_id": memory.embedding_id,
        "source_episode_ids": list(memory.source_episode_ids),
        "tags": list(memory.tags),
        "salience": memory.salience,
        "access_count": memory.access_count,
        "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
    }


def edge_to_dict(edge: Any) -> Dict[str, Any]:
    return {
        "edge_id": str(edge.edge_id),
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "kind": edge.kind.value,
        "confidence": edge.confidence,
        "semantic_weight": edge.semantic_weight,
        "emotional_weight": edge.emotional_weight,
        "recency_weight": edge.recency_weight,
        "structural_weight": edge.structural_weight,
        "salience_weight": edge.salience_weight,
        "causal_weight": edge.causal_weight,
        "verification_weight": edge.verification_weight,
        "actor_affinity_weight": edge.actor_affinity_weight,
        "strength": round(float(compute_edge_strength(edge)), 6),
        "created_at": edge.created_at.isoformat(),
    }


def edge_signal_summary(edge_payload: Dict[str, Any]) -> Dict[str, Any]:
    signal_weights = {
        "semantic": float(edge_payload.get("semantic_weight", 0.0)),
        "emotional": float(edge_payload.get("emotional_weight", 0.0)),
        "recency": float(edge_payload.get("recency_weight", 0.0)),
        "structural": float(edge_payload.get("structural_weight", 0.0)),
        "salience": float(edge_payload.get("salience_weight", 0.0)),
        "causal": float(edge_payload.get("causal_weight", 0.0)),
        "verification": float(edge_payload.get("verification_weight", 0.0)),
        "actor_affinity": float(edge_payload.get("actor_affinity_weight", 0.0)),
    }
    ordered = sorted(signal_weights.items(), key=lambda item: item[1], reverse=True)
    strongest_kind, strongest_value = ordered[0]
    return {
        "strongest_signal": strongest_kind,
        "strongest_signal_weight": round(strongest_value, 6),
        "signal_weights": signal_weights,
    }
