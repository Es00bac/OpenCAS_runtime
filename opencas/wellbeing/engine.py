"""Assessment engine for operational wellbeing."""

from __future__ import annotations

import inspect
from typing import Any

from opencas.cognition import CognitionGrounding, GroundingKind, GroundingSource
from opencas.cognition.self_inspection import SelfInspectionRecord

from .models import WellbeingAssessment, WellbeingState


class WellbeingEngine:
    """Derive operational wellbeing from existing runtime evidence."""

    async def assess(self, runtime: Any) -> WellbeingAssessment:
        """Build a grounded wellbeing assessment for the current runtime."""

        somatic_state = _somatic_state(runtime)
        active_commitments = await _list_active_commitments(runtime)
        recent_inspections = await _list_recent_self_inspections(runtime)
        gap_records = await _list_commitment_gap_records(runtime)
        recent_daydreams = await _list_recent_daydreams(runtime)
        tom_result = _check_tom(runtime)
        relational_state = _relational_state(runtime)

        grounding: list[CognitionGrounding] = []
        recovery_need, recovery_grounding = _recovery_need(somatic_state)
        grounding.extend(recovery_grounding)
        truth_pressure, truth_grounding = _truth_pressure(somatic_state, tom_result)
        grounding.extend(truth_grounding)
        promise_load, promise_grounding = _promise_load(active_commitments, gap_records)
        grounding.extend(promise_grounding)
        drift_load, drift_grounding = _drift_load(recent_inspections)
        grounding.extend(drift_grounding)
        curiosity, curiosity_grounding = _curiosity(recent_daydreams)
        grounding.extend(curiosity_grounding)
        relationship_pressure, boundary_pressure, relationship_grounding = _relationship_pressures(
            relational_state=relational_state,
            truth_pressure=truth_pressure,
            promise_load=promise_load,
        )
        grounding.extend(relationship_grounding)

        coherence = round(
            max(0.0, min(1.0, 0.82 - (truth_pressure * 0.28) - (drift_load * 0.24))),
            3,
        )
        autonomy = round(
            max(
                0.0,
                min(1.0, 0.78 - (relationship_pressure * 0.26) - (promise_load * 0.12)),
            ),
            3,
        )

        state = WellbeingState(
            coherence=coherence,
            autonomy=autonomy,
            recovery_need=recovery_need,
            promise_load=promise_load,
            boundary_pressure=boundary_pressure,
            relationship_pressure=relationship_pressure,
            curiosity=curiosity,
            drift_load=drift_load,
            truth_pressure=truth_pressure,
            grounding=grounding,
            meta={
                "active_commitment_count": len(active_commitments),
                "self_inspection_record_count": len(recent_inspections),
                "commitment_gap_record_count": len(gap_records),
                "recent_daydream_count": len(recent_daydreams),
            },
        )
        return WellbeingAssessment(state=state, grounding=grounding)


def _somatic_state(runtime: Any) -> Any | None:
    return getattr(getattr(getattr(runtime, "ctx", None), "somatic", None), "state", None)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _list_active_commitments(runtime: Any) -> list[Any]:
    store = getattr(runtime, "commitment_store", None)
    if store is None:
        return []
    list_active = getattr(store, "list_active", None)
    if not callable(list_active):
        return []
    try:
        return list(await _maybe_await(list_active(limit=100)))
    except Exception:
        return []


async def _list_recent_self_inspections(runtime: Any) -> list[SelfInspectionRecord]:
    store = getattr(runtime, "self_inspection_store", None)
    if store is None:
        return []
    list_recent = getattr(store, "list_recent", None)
    if not callable(list_recent):
        return []
    try:
        return list(await _maybe_await(list_recent(limit=20)))
    except Exception:
        return []


async def _list_commitment_gap_records(runtime: Any) -> list[SelfInspectionRecord]:
    store = getattr(runtime, "self_inspection_store", None)
    if store is None:
        return []
    list_gaps = getattr(store, "list_unresolved_commitment_gaps", None)
    if not callable(list_gaps):
        return []
    try:
        return list(await _maybe_await(list_gaps(limit=20)))
    except Exception:
        return []


async def _list_recent_daydreams(runtime: Any) -> list[Any]:
    store = getattr(getattr(runtime, "ctx", None), "daydream_store", None)
    if store is None:
        return []
    list_recent = getattr(store, "list_recent", None)
    if not callable(list_recent):
        return []
    try:
        return list(await _maybe_await(list_recent(limit=10)))
    except Exception:
        return []


def _check_tom(runtime: Any) -> Any | None:
    tom = getattr(runtime, "tom", None)
    check = getattr(tom, "check_consistency", None)
    if not callable(check):
        return None
    try:
        return check()
    except Exception:
        return None


def _relational_state(runtime: Any) -> Any | None:
    relational = getattr(runtime, "relational", None) or getattr(
        getattr(runtime, "ctx", None), "relational", None
    )
    try:
        return getattr(relational, "state", None)
    except AssertionError:
        return None


def _recovery_need(somatic_state: Any | None) -> tuple[float, list[CognitionGrounding]]:
    if somatic_state is None:
        return 0.0, []
    fatigue = _bounded_float(getattr(somatic_state, "fatigue", 0.0))
    tension = _bounded_float(getattr(somatic_state, "tension", 0.0))
    value = round(max(fatigue, (fatigue * 0.6) + (tension * 0.4)), 3)
    if value <= 0:
        return value, []
    return value, [
        CognitionGrounding(
            kind=GroundingKind.SOMATIC_METRIC,
            source=GroundingSource.SOMATIC,
            subject="recovery_need",
            claim=f"fatigue={fatigue:.3f}, tension={tension:.3f}",
            confidence=0.82,
            allowed_surface="internal",
        )
    ]


def _truth_pressure(somatic_state: Any | None, tom_result: Any | None) -> tuple[float, list[CognitionGrounding]]:
    certainty = 1.0
    if somatic_state is not None:
        certainty = _bounded_float(getattr(somatic_state, "certainty", 1.0))
    contradictions = list(getattr(tom_result, "contradictions", []) or []) if tom_result else []
    warnings = list(getattr(tom_result, "warnings", []) or []) if tom_result else []
    value = max(0.0, 1.0 - certainty)
    value = min(1.0, value + min(0.35, len(contradictions) * 0.15) + min(0.2, len(warnings) * 0.08))
    grounding: list[CognitionGrounding] = []
    if value > 0:
        grounding.append(
            CognitionGrounding(
                kind=GroundingKind.DERIVED,
                source=GroundingSource.RUNTIME,
                subject="truth_pressure",
                claim=(
                    f"certainty={certainty:.3f}, "
                    f"contradictions={len(contradictions)}, warnings={len(warnings)}"
                ),
                confidence=0.7,
                allowed_surface="internal",
            )
        )
    return round(value, 3), grounding


def _promise_load(active_commitments: list[Any], gap_records: list[SelfInspectionRecord]) -> tuple[float, list[CognitionGrounding]]:
    gap_count = sum(len(record.commitment_gaps) for record in gap_records)
    value = min(1.0, (len(active_commitments) / 8.0) + (gap_count * 0.12))
    grounding: list[CognitionGrounding] = []
    if value > 0:
        grounding.append(
            CognitionGrounding(
                kind=GroundingKind.DERIVED,
                source=GroundingSource.RUNTIME,
                subject="promise_load",
                claim=f"active_commitments={len(active_commitments)}, commitment_gaps={gap_count}",
                confidence=0.78,
                allowed_surface="internal",
            )
        )
    return round(value, 3), grounding


def _drift_load(records: list[SelfInspectionRecord]) -> tuple[float, list[CognitionGrounding]]:
    observations = [
        observation
        for record in records
        for observation in getattr(record, "drift_observations", []) or []
    ]
    value = min(1.0, len(observations) * 0.42)
    grounding: list[CognitionGrounding] = []
    if value > 0:
        grounding.append(
            CognitionGrounding(
                kind=GroundingKind.DERIVED,
                source=GroundingSource.RUNTIME,
                subject="drift_load",
                claim=f"recent_drift_observations={len(observations)}",
                evidence_ids=[str(record.record_id) for record in records[:5]],
                confidence=0.74,
                allowed_surface="internal",
            )
        )
    return round(value, 3), grounding


def _curiosity(recent_daydreams: list[Any]) -> tuple[float, list[CognitionGrounding]]:
    if not recent_daydreams:
        return 0.25, [
            CognitionGrounding(
                kind=GroundingKind.DERIVED,
                source=GroundingSource.DAYDREAM,
                subject="curiosity",
                claim="No recent daydream reflections were available for curiosity continuity.",
                confidence=0.55,
                allowed_surface="internal",
            )
        ]
    keepers = sum(1 for item in recent_daydreams if bool(getattr(item, "keeper", False)))
    thought_count = sum(len(getattr(item, "thoughts", []) or []) for item in recent_daydreams)
    value = min(1.0, 0.35 + (keepers * 0.06) + (thought_count * 0.03))
    return round(value, 3), [
        CognitionGrounding(
            kind=GroundingKind.DERIVED,
            source=GroundingSource.DAYDREAM,
            subject="curiosity",
            claim=f"recent_daydreams={len(recent_daydreams)}, keepers={keepers}, thoughts={thought_count}",
            confidence=0.62,
            allowed_surface="internal",
        )
    ]


def _relationship_pressures(
    *,
    relational_state: Any | None,
    truth_pressure: float,
    promise_load: float,
) -> tuple[float, float, list[CognitionGrounding]]:
    if relational_state is None:
        return 0.0, 0.0, []
    dimensions = getattr(relational_state, "dimensions", {}) or {}
    trust = _bounded_relational_float(dimensions.get("trust", 0.0))
    resonance = _bounded_relational_float(dimensions.get("resonance", 0.0))
    musubi = _bounded_relational_float(getattr(relational_state, "musubi", 0.0))
    closeness = max(0.0, musubi, trust, resonance)
    relationship_pressure = round(
        min(1.0, max(0.0, (closeness * 0.35) + (truth_pressure * 0.4) + (promise_load * 0.25))),
        3,
    )
    boundary_pressure = round(
        min(1.0, max(0.0, (relationship_pressure * 0.7) + (truth_pressure * 0.25))),
        3,
    )
    return relationship_pressure, boundary_pressure, [
        CognitionGrounding(
            kind=GroundingKind.RELATIONAL_METRIC,
            source=GroundingSource.RELATIONAL,
            subject="relationship_pressure",
            claim=(
                f"musubi={musubi:.3f}, trust={trust:.3f}, resonance={resonance:.3f}, "
                f"truth_pressure={truth_pressure:.3f}, promise_load={promise_load:.3f}"
            ),
            confidence=0.62,
            allowed_surface="internal",
        )
    ]


def _bounded_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, parsed))


def _bounded_relational_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(-1.0, min(1.0, parsed))
