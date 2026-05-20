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


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _payload_value(payload: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def episode_context_metadata(ep: Any) -> Dict[str, Any]:
    """Return lane/authority labels for dashboard and manager display.

    These labels are observational. Historical episodes can explain where a
    record came from, but they do not by themselves authorize new work.
    """

    payload = dict(getattr(ep, "payload", {}) or {})
    kind = _enum_value(getattr(ep, "kind", ""))
    role = str(payload.get("role") or "").lower()
    origin = " ".join(
        str(value or "").lower()
        for value in (
            payload.get("origin"),
            payload.get("source"),
            payload.get("activation_source"),
            payload.get("proposal_kind"),
            payload.get("context_material"),
        )
    )

    lane = _payload_value(
        payload,
        "source_lane",
        "origin_context_lane",
        "context_lane",
    )
    lane_source = "payload"
    inferred = False
    if isinstance(lane, dict):
        lane = None
    if lane:
        lane = str(lane).lower()
    else:
        inferred = True
        lane_source = "kind_role_heuristic"
        if any(token in origin for token in ("daydream", "reflect", "proposal", "association")):
            lane = "reflective"
        elif kind in {"compaction", "consolidation", "procedural"}:
            lane = "reflective"
        elif kind == "artifact" or "artifact" in payload:
            lane = "reflective"
        elif kind in {"turn", "action", "observation"} or role in {"user", "assistant", "tool"}:
            lane = "executive"
        else:
            lane = "unknown"

    material = str(payload.get("context_material") or "").strip()
    if not material:
        if "proposal" in origin:
            material = "proposal_evidence"
        elif any(token in origin for token in ("daydream", "reflect")):
            material = "reflective_record"
        elif kind == "artifact":
            material = "artifact"
        else:
            material = "episode_record"

    authority = str(payload.get("context_authority") or payload.get("authority") or "").strip()
    if not authority:
        authority = "interpretation" if lane == "reflective" else "live_observation"

    proposal_ids = payload.get("accepted_proposal_ids") or payload.get("proposal_ids") or []
    if isinstance(proposal_ids, str):
        proposal_ids = [proposal_ids]
    elif not isinstance(proposal_ids, list):
        proposal_ids = []

    return {
        "source_lane": lane,
        "source_lane_inferred": inferred,
        "lane_source": lane_source,
        "authority": authority,
        "context_material": material,
        "can_authorize_work": False,
        "accepted_proposal_ids": [str(item) for item in proposal_ids if str(item or "").strip()],
        "truth_snapshot_id": str(payload.get("truth_snapshot_id") or payload.get("source_snapshot_id") or ""),
        "truth_epoch": payload.get("truth_epoch") or payload.get("source_epoch"),
    }


def memory_context_metadata(memory: Any) -> Dict[str, Any]:
    return {
        "source_lane": "database",
        "source_lane_inferred": True,
        "lane_source": "memory_material",
        "authority": "retrieved_memory",
        "context_material": "distilled_memory",
        "can_authorize_work": False,
        "accepted_proposal_ids": [],
        "truth_snapshot_id": "",
        "truth_epoch": None,
    }


def episode_to_dict(ep: Any) -> Dict[str, Any]:
    artifact = (ep.payload or {}).get("artifact") if getattr(ep, "payload", None) else None
    context_meta = episode_context_metadata(ep)
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
        **context_meta,
        "dual_context": context_meta,
    }


def memory_to_dict(memory: Any) -> Dict[str, Any]:
    context_meta = memory_context_metadata(memory)
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
        **context_meta,
        "dual_context": context_meta,
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
