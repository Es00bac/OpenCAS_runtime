"""Drift-observation accounting for wellbeing maintenance effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from opencas.cognition.self_inspection import DriftObservation, SelfInspectionRecord

from .models import MaintenanceOutcome

CONFIRMED_DRAIN_BASES = {"measured", "artifact_created", "observed"}
ESTIMATED_DRAIN_BASIS = "estimated"
DRIFT_DRAIN_SCALE = 50
MAX_DRAIN_PER_OUTCOME = 5


@dataclass(frozen=True)
class DriftObservationRef:
    """Addressed drift observation pointer kept in outcome metadata."""

    record_id: str
    observation_index: int
    reason: str

    def model_dump(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "observation_index": self.observation_index,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DriftDrainApplication:
    """Result of applying one maintenance outcome to drift observations."""

    status: str
    drained_count: int = 0
    refs: list[DriftObservationRef] = field(default_factory=list)


def drift_observation_is_addressed(observation: DriftObservation) -> bool:
    """Return whether a drift observation already has an outcome linkage."""

    meta = observation.meta or {}
    return bool(
        observation.addressed_by_outcome_id
        or meta.get("addressed_by_outcome_id")
        or meta.get("addressed_by_outcome")
    )


def split_drift_observations(
    records: list[SelfInspectionRecord],
) -> tuple[list[tuple[SelfInspectionRecord, int, DriftObservation]], int]:
    """Return unaddressed drift observations plus the addressed count."""

    open_items: list[tuple[SelfInspectionRecord, int, DriftObservation]] = []
    addressed_count = 0
    for record in records:
        observations = list(getattr(record, "drift_observations", []) or [])
        for index, observation in enumerate(observations):
            if drift_observation_is_addressed(observation):
                addressed_count += 1
            else:
                open_items.append((record, index, observation))
    return open_items, addressed_count


def drift_drain_count_for_outcome(outcome: MaintenanceOutcome) -> int:
    """Return the bounded number of observations this outcome can address."""

    if not outcome_confirms_drift_decrease(outcome):
        return 0
    count = int(abs(_risk_delta(outcome)) * DRIFT_DRAIN_SCALE)
    return max(0, min(MAX_DRAIN_PER_OUTCOME, count))


def outcome_confirms_drift_decrease(outcome: MaintenanceOutcome) -> bool:
    """Return whether an outcome is confirmed enough to drain drift load."""

    meta = outcome.meta or {}
    basis = str(meta.get("effect_basis") or "").strip().lower()
    direction = str(meta.get("effect_direction") or "").strip().lower()
    return (
        basis in CONFIRMED_DRAIN_BASES
        and direction.endswith("_lower")
        and _risk_delta(outcome) < 0
    )


def outcome_is_pending_drift_validation(outcome: MaintenanceOutcome) -> bool:
    """Return whether an estimated lower-risk outcome is awaiting validation."""

    meta = outcome.meta or {}
    basis = str(meta.get("effect_basis") or "").strip().lower()
    direction = str(meta.get("effect_direction") or "").strip().lower()
    status = str(meta.get("drift_drain_status") or "").strip().lower()
    return (
        basis == ESTIMATED_DRAIN_BASIS
        and direction == "expected_lower"
        and _risk_delta(outcome) < 0
        and status not in {"drained", "measured", "validation_failed", "validated_no_change"}
    )


def pending_drift_drain_count(outcomes: list[MaintenanceOutcome]) -> int:
    """Count estimated maintenance outcomes that have not been validated."""

    return sum(1 for outcome in outcomes if outcome_is_pending_drift_validation(outcome))


async def annotate_drift_drain_effect(
    self_inspection_store: Any,
    outcome: MaintenanceOutcome,
) -> DriftDrainApplication:
    """Apply or mark the drift-drain status for a maintenance outcome.

    Estimated outcomes are deliberately left pending; only measured,
    artifact-backed, or observed lower-risk effects can mark drift observations
    as addressed.
    """

    if outcome_is_pending_drift_validation(outcome):
        outcome.meta = {
            **(outcome.meta or {}),
            "drift_drain_status": "pending_validation",
        }
        return DriftDrainApplication(status="pending_validation")

    drain_count = drift_drain_count_for_outcome(outcome)
    if drain_count <= 0:
        return DriftDrainApplication(status="not_applicable")

    if self_inspection_store is None or not callable(
        getattr(self_inspection_store, "list_recent", None)
    ):
        outcome.meta = {
            **(outcome.meta or {}),
            "drift_drain_status": "self_inspection_store_unavailable",
        }
        return DriftDrainApplication(status="self_inspection_store_unavailable")

    records = await self_inspection_store.list_recent(limit=50)
    refs: list[DriftObservationRef] = []
    changed_records: dict[str, SelfInspectionRecord] = {}
    addressed_at = datetime.now(timezone.utc)
    outcome_id = str(outcome.outcome_id)

    for record, index, observation in split_drift_observations(records)[0]:
        if len(refs) >= drain_count:
            break
        observation.addressed_by_outcome_id = outcome_id
        observation.addressed_at = addressed_at
        observation.meta = {
            **(observation.meta or {}),
            "addressed_by_outcome_id": outcome_id,
            "addressed_at": addressed_at.isoformat(),
            "drain_effect_basis": str((outcome.meta or {}).get("effect_basis") or ""),
        }
        refs.append(
            DriftObservationRef(
                record_id=str(record.record_id),
                observation_index=index,
                reason=observation.reason,
            )
        )
        changed_records[str(record.record_id)] = record

    for record in changed_records.values():
        save = getattr(self_inspection_store, "save", None)
        if callable(save):
            await save(record)

    status = "drained" if refs else "no_unaddressed_observations"
    outcome.meta = {
        **(outcome.meta or {}),
        "drift_drain_status": status,
        "drift_observations_drained": len(refs),
        "addressed_drift_observations": [ref.model_dump() for ref in refs],
    }
    for ref in refs:
        if ref.record_id not in outcome.evidence_ids:
            outcome.evidence_ids.append(ref.record_id)
    return DriftDrainApplication(status=status, drained_count=len(refs), refs=refs)


def _risk_delta(outcome: MaintenanceOutcome) -> float:
    meta = outcome.meta or {}
    try:
        return float(meta.get("risk_delta"))
    except (TypeError, ValueError):
        return round(float(outcome.after_risk) - float(outcome.before_risk), 3)
