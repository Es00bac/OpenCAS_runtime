"""Identity API routes for the OpenCAS dashboard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["identity"])


class SomaticPatchRequest(BaseModel):
    arousal: Optional[float] = None
    fatigue: Optional[float] = None
    tension: Optional[float] = None
    valence: Optional[float] = None
    focus: Optional[float] = None
    energy: Optional[float] = None
    certainty: Optional[float] = None
    somatic_tag: Optional[str] = None


class SelfModelResponse(BaseModel):
    model_id: str
    updated_at: str
    name: str
    version: str
    narrative: Optional[str]
    values: List[str]
    traits: List[str]
    current_goals: List[str]
    current_intention: Optional[str]
    recent_activity: List[Dict[str, Any]]
    self_beliefs: Dict[str, Any]
    relational_state_id: Optional[str]
    source_system: Optional[str] = None
    imported_identity_profile: Dict[str, Any] = Field(default_factory=dict)
    memory_anchors: List[Dict[str, Any]] = Field(default_factory=list)
    recent_themes: List[Dict[str, Any]] = Field(default_factory=list)
    identity_rebuild_audit: Dict[str, Any] = Field(default_factory=dict)


class UserModelResponse(BaseModel):
    model_id: str
    updated_at: str
    explicit_preferences: Dict[str, Any]
    inferred_goals: List[str]
    known_boundaries: List[str]
    trust_level: float
    uncertainty_areas: List[str]
    partner_user_id: Optional[str] = None
    partner_musubi: Optional[float] = None
    partner_trust_raw: Optional[float] = None
    partner_musubi_raw: Optional[float] = None


class ContinuityResponse(BaseModel):
    state_id: str
    updated_at: str
    last_session_id: Optional[str]
    last_shutdown_time: Optional[str]
    boot_count: int
    version: str
    source_system: Optional[str] = None
    temporal_bridges: Dict[str, Any] = Field(default_factory=dict)
    integrity_report: Dict[str, Any] = Field(default_factory=dict)


class MusubiStateResponse(BaseModel):
    state_id: str
    updated_at: str
    musubi: float
    dimensions: Dict[str, float]
    source_tag: Optional[str]
    continuity_breadcrumb: Optional[str] = None


class SomaticStateResponse(BaseModel):
    state_id: str
    updated_at: str
    arousal: float
    fatigue: float
    tension: float
    valence: float
    focus: float
    energy: float
    certainty: float
    somatic_tag: Optional[str]
    primary_emotion: Optional[str] = None


class IdentitySummaryResponse(BaseModel):
    self_model: SelfModelResponse
    user_model: UserModelResponse
    continuity: ContinuityResponse
    musubi: Optional[MusubiStateResponse]
    somatic: Optional[SomaticStateResponse]


class TomSummaryResponse(BaseModel):
    available: bool
    belief_counts: Dict[str, Any] = Field(default_factory=dict)
    intention_counts: Dict[str, Any] = Field(default_factory=dict)
    recent_beliefs: List[Dict[str, Any]] = Field(default_factory=list)
    recent_intentions: List[Dict[str, Any]] = Field(default_factory=list)
    consistency: Dict[str, Any] = Field(default_factory=dict)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _self_model_to_dict(sm: Any) -> Dict[str, Any]:
    return {
        "model_id": str(sm.model_id),
        "updated_at": sm.updated_at.isoformat(),
        "name": sm.name,
        "version": sm.version,
        "narrative": sm.narrative,
        "values": sm.values,
        "traits": sm.traits,
        "current_goals": sm.current_goals,
        "current_intention": sm.current_intention,
        "recent_activity": sm.recent_activity,
        "self_beliefs": dict(sm.self_beliefs),
        "relational_state_id": sm.relational_state_id,
        "source_system": getattr(sm, "source_system", None),
        "imported_identity_profile": dict(getattr(sm, "imported_identity_profile", {}) or {}),
        "memory_anchors": list(getattr(sm, "memory_anchors", []) or []),
        "recent_themes": list(getattr(sm, "recent_themes", []) or []),
        "identity_rebuild_audit": dict(getattr(sm, "identity_rebuild_audit", {}) or {}),
    }


def _user_model_to_dict(um: Any) -> Dict[str, Any]:
    return {
        "model_id": str(um.model_id),
        "updated_at": um.updated_at.isoformat(),
        "explicit_preferences": dict(um.explicit_preferences),
        "inferred_goals": um.inferred_goals,
        "known_boundaries": um.known_boundaries,
        "trust_level": um.trust_level,
        "uncertainty_areas": um.uncertainty_areas,
        "partner_user_id": getattr(um, "partner_user_id", None),
        "partner_musubi": getattr(um, "partner_musubi", None),
        "partner_trust_raw": getattr(um, "partner_trust_raw", None),
        "partner_musubi_raw": getattr(um, "partner_musubi_raw", None),
    }


def _continuity_to_dict(cs: Any) -> Dict[str, Any]:
    return {
        "state_id": str(cs.state_id),
        "updated_at": cs.updated_at.isoformat(),
        "last_session_id": cs.last_session_id,
        "last_shutdown_time": cs.last_shutdown_time.isoformat() if cs.last_shutdown_time else None,
        "boot_count": cs.boot_count,
        "version": cs.version,
        "source_system": getattr(cs, "source_system", None),
        "temporal_bridges": dict(getattr(cs, "temporal_bridges", {}) or {}),
        "integrity_report": dict(getattr(cs, "integrity_report", {}) or {}),
    }


def _belief_payload(belief: Any) -> Dict[str, Any]:
    timestamp = getattr(belief, "timestamp", None)
    return {
        "belief_id": str(getattr(belief, "belief_id", "")),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "subject": _enum_value(getattr(belief, "subject", "")),
        "predicate": getattr(belief, "predicate", ""),
        "confidence": getattr(belief, "confidence", None),
        "reinforcement_count": getattr(belief, "reinforcement_count", 0),
        "belief_revision_score": getattr(belief, "belief_revision_score", 0.0),
        "meta": dict(getattr(belief, "meta", {}) or {}),
    }


def _intention_payload(intention: Any) -> Dict[str, Any]:
    timestamp = getattr(intention, "timestamp", None)
    resolved_at = getattr(intention, "resolved_at", None)
    return {
        "intention_id": str(getattr(intention, "intention_id", "")),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "actor": _enum_value(getattr(intention, "actor", "")),
        "content": getattr(intention, "content", ""),
        "status": _enum_value(getattr(intention, "status", "")),
        "resolved_at": resolved_at.isoformat() if resolved_at is not None else None,
        "meta": dict(getattr(intention, "meta", {}) or {}),
    }


def _count_by(items: List[Any], attr: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        key = _enum_value(getattr(item, attr, "")) or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_identity_router(runtime: Any) -> APIRouter:
    """Build identity routes wired to *runtime*."""
    r = APIRouter(prefix="/api/identity", tags=["identity"])

    @r.get("", response_model=IdentitySummaryResponse)
    async def get_identity_summary() -> IdentitySummaryResponse:
        identity = runtime.ctx.identity
        musubi: Optional[MusubiStateResponse] = None
        somatic: Optional[SomaticStateResponse] = None

        rel_engine = getattr(runtime.ctx, "relational", None)
        if rel_engine is not None:
            try:
                ms = rel_engine.state
                musubi = MusubiStateResponse(
                    state_id=str(ms.state_id),
                    updated_at=ms.updated_at.isoformat(),
                    musubi=ms.musubi,
                    dimensions=dict(ms.dimensions),
                    source_tag=ms.source_tag,
                    continuity_breadcrumb=getattr(ms, "continuity_breadcrumb", None),
                )
            except Exception:
                pass

        somatic_mgr = getattr(runtime.ctx, "somatic", None)
        if somatic_mgr is not None:
            try:
                ss = somatic_mgr.state
                from opencas.somatic.modulators import SomaticModulators
                mod = SomaticModulators(ss)
                somatic = SomaticStateResponse(
                    state_id=str(ss.state_id),
                    updated_at=ss.updated_at.isoformat(),
                    arousal=ss.arousal,
                    fatigue=ss.fatigue,
                    tension=ss.tension,
                    valence=ss.valence,
                    focus=ss.focus,
                    energy=ss.energy,
                    certainty=ss.certainty,
                    somatic_tag=ss.somatic_tag,
                    primary_emotion=mod._infer_primary_emotion().value,
                )
            except Exception:
                pass

        return IdentitySummaryResponse(
            self_model=SelfModelResponse(**_self_model_to_dict(identity.self_model)),
            user_model=UserModelResponse(**_user_model_to_dict(identity.user_model)),
            continuity=ContinuityResponse(**_continuity_to_dict(identity.continuity)),
            musubi=musubi,
            somatic=somatic,
        )

    @r.get("/self", response_model=SelfModelResponse)
    async def get_self_model() -> SelfModelResponse:
        return SelfModelResponse(**_self_model_to_dict(runtime.ctx.identity.self_model))

    @r.get("/user", response_model=UserModelResponse)
    async def get_user_model() -> UserModelResponse:
        return UserModelResponse(**_user_model_to_dict(runtime.ctx.identity.user_model))

    @r.get("/continuity", response_model=ContinuityResponse)
    async def get_continuity() -> ContinuityResponse:
        return ContinuityResponse(**_continuity_to_dict(runtime.ctx.identity.continuity))

    @r.get("/musubi", response_model=Optional[MusubiStateResponse])
    async def get_musubi() -> Optional[MusubiStateResponse]:
        rel_engine = getattr(runtime.ctx, "relational", None)
        if rel_engine is None:
            return None
        ms = rel_engine.state
        return MusubiStateResponse(
            state_id=str(ms.state_id),
            updated_at=ms.updated_at.isoformat(),
            musubi=ms.musubi,
            dimensions=dict(ms.dimensions),
            source_tag=ms.source_tag,
        )

    @r.get("/somatic", response_model=Optional[SomaticStateResponse])
    async def get_somatic() -> Optional[SomaticStateResponse]:
        somatic_mgr = getattr(runtime.ctx, "somatic", None)
        if somatic_mgr is None:
            return None
        ss = somatic_mgr.state
        primary_emotion: Optional[str] = None
        try:
            from opencas.somatic.modulators import SomaticModulators
            primary_emotion = SomaticModulators(ss)._infer_primary_emotion().value
        except Exception:
            pass
        return SomaticStateResponse(
            state_id=str(ss.state_id),
            updated_at=ss.updated_at.isoformat(),
            arousal=ss.arousal,
            fatigue=ss.fatigue,
            tension=ss.tension,
            valence=ss.valence,
            focus=ss.focus,
            energy=ss.energy,
            certainty=ss.certainty,
            somatic_tag=ss.somatic_tag,
            primary_emotion=primary_emotion,
        )

    @r.get("/tom", response_model=TomSummaryResponse)
    async def get_tom_summary(limit: int = 12) -> TomSummaryResponse:
        tom = getattr(runtime, "tom", None) or getattr(runtime.ctx, "tom", None)
        empty_counts = {
            "belief_counts": {"total": 0, "by_subject": {}},
            "intention_counts": {"total": 0, "active": 0, "by_status": {}},
        }
        if tom is None:
            return TomSummaryResponse(available=False, **empty_counts)
        bounded_limit = max(1, min(int(limit), 50))
        try:
            beliefs = list(tom.list_beliefs())
            intentions = list(tom.list_intentions())
        except Exception:
            return TomSummaryResponse(available=False, **empty_counts)

        consistency: Dict[str, Any] = {}
        checker = getattr(tom, "check_consistency", None)
        if callable(checker):
            try:
                result = checker()
                timestamp = getattr(result, "timestamp", None)
                consistency = {
                    "check_id": str(getattr(result, "check_id", "")),
                    "timestamp": timestamp.isoformat() if timestamp is not None else None,
                    "contradictions": list(getattr(result, "contradictions", []) or []),
                    "warnings": list(getattr(result, "warnings", []) or []),
                    "belief_count": getattr(result, "belief_count", len(beliefs)),
                    "intention_count": getattr(result, "intention_count", len(intentions)),
                }
            except Exception:
                consistency = {}
        by_status = _count_by(intentions, "status")
        return TomSummaryResponse(
            available=True,
            belief_counts={
                "total": len(beliefs),
                "by_subject": _count_by(beliefs, "subject"),
            },
            intention_counts={
                "total": len(intentions),
                "active": by_status.get("active", 0),
                "by_status": by_status,
            },
            recent_beliefs=[_belief_payload(item) for item in beliefs[:bounded_limit]],
            recent_intentions=[_intention_payload(item) for item in intentions[:bounded_limit]],
            consistency=consistency,
        )

    @r.patch("/somatic", response_model=Optional[SomaticStateResponse])
    async def patch_somatic(body: SomaticPatchRequest) -> Optional[SomaticStateResponse]:
        somatic_mgr = getattr(runtime.ctx, "somatic", None)
        if somatic_mgr is None:
            return None
        if body.arousal is not None:
            somatic_mgr.set_arousal(body.arousal)
        if body.fatigue is not None:
            somatic_mgr.set_fatigue(body.fatigue)
        if body.tension is not None:
            somatic_mgr.set_tension(body.tension)
        if body.valence is not None:
            somatic_mgr.set_valence(body.valence)
        if body.focus is not None:
            somatic_mgr.set_focus(body.focus)
        if body.energy is not None:
            somatic_mgr.set_energy(body.energy)
        if body.certainty is not None:
            somatic_mgr.set_certainty(body.certainty)
        if body.somatic_tag is not None:
            somatic_mgr.set_tag(body.somatic_tag)
        ss = somatic_mgr.state
        return SomaticStateResponse(
            state_id=str(ss.state_id),
            updated_at=ss.updated_at.isoformat(),
            arousal=ss.arousal,
            fatigue=ss.fatigue,
            tension=ss.tension,
            valence=ss.valence,
            focus=ss.focus,
            energy=ss.energy,
            certainty=ss.certainty,
            somatic_tag=ss.somatic_tag,
        )

    return r
