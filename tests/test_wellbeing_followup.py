from datetime import datetime, timedelta, timezone

import pytest

from opencas.cognition import SelfInspectionStore
from opencas.cognition.self_inspection import (
    DriftObservation,
    SelfInspectionPhase,
    SelfInspectionRecord,
)
from opencas.wellbeing import MaintenanceActionType, MaintenanceOutcome, WellbeingStore
from opencas.wellbeing.followup import RecordOnlyMaintenanceFollowupService


@pytest.mark.asyncio
async def test_record_only_followup_service_creates_review_recommendation(tmp_path):
    store = await WellbeingStore(tmp_path / "wellbeing.db").connect()
    try:
        for _ in range(3):
            await store.save_maintenance_outcome(
                MaintenanceOutcome(
                    action_type=MaintenanceActionType.DEEP_THINK.value,
                    before_risk=0.52,
                    after_risk=0.52,
                    outcome="deep_think_seed_recorded",
                    meta={
                        "effect_basis": "record_only",
                        "effect_direction": "unmeasured",
                        "followup_required": True,
                        "risk_delta": 0.0,
                    },
                )
            )

        service = RecordOnlyMaintenanceFollowupService(store, repeat_threshold=3)
        summary = await service.evaluate()

        assert summary.stuck_loop_count == 1
        assert summary.repeat_count == 3
        assert summary.next_followup_action is not None
        assert summary.next_followup_action["type"] == "self_modification_proposal"
        assert summary.next_followup_action["effect_remains_unmeasured"] is True
        assert summary.linked_followup_ids

        recommendations = await store.list_recommendations(status="proposed", limit=5)
        assert len(recommendations) == 1
        assert recommendations[0].meta["followup_kind"] == "repeated_record_only_maintenance"
        assert recommendations[0].meta["repeat_count"] == 3
        assert recommendations[0].meta["effect_remains_unmeasured"] is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_estimated_drift_decrease_verification_marks_measured_and_drains(tmp_path):
    wellbeing = await WellbeingStore(tmp_path / "wellbeing.db").connect()
    inspections = await SelfInspectionStore(tmp_path / "self_inspection.db").connect()
    try:
        base = datetime(2026, 5, 6, 12, 0, tzinfo=timezone.utc)
        before_record = SelfInspectionRecord(
            created_at=base,
            session_id="estimated-drift",
            phase=SelfInspectionPhase.POST_TURN,
            drift_observations=[
                DriftObservation(reason=f"before {index}") for index in range(5)
            ],
        )
        after_record = SelfInspectionRecord(
            created_at=base + timedelta(minutes=30),
            session_id="estimated-drift",
            phase=SelfInspectionPhase.POST_TURN,
            drift_observations=[
                DriftObservation(reason=f"after {index}") for index in range(4)
            ],
        )
        await inspections.save(before_record)
        await inspections.save(after_record)
        for index in range(3):
            await wellbeing.save_maintenance_outcome(
                MaintenanceOutcome(
                    created_at=base + timedelta(minutes=10 + index),
                    action_type=MaintenanceActionType.RECOVER.value,
                    before_risk=0.52,
                    after_risk=0.48,
                    outcome="background_throttle_recorded",
                    meta={
                        "effect_basis": "estimated",
                        "effect_direction": "expected_lower",
                        "risk_delta": -0.04,
                        "drift_drain_status": "pending_validation",
                    },
                )
            )

        service = RecordOnlyMaintenanceFollowupService(
            wellbeing,
            self_inspection_store=inspections,
        )
        summary = await service.evaluate()

        assert summary.next_followup_action is None
        outcomes = await wellbeing.list_maintenance_outcomes(
            action_type=MaintenanceActionType.RECOVER.value,
            limit=5,
        )
        measured = [
            outcome
            for outcome in outcomes
            if outcome.meta.get("effect_basis") == "measured"
        ]
        assert measured
        assert measured[0].meta["effect_direction"] == "measured_lower"
        assert measured[0].meta["drift_drain_status"] == "drained"
        records = await inspections.list_recent(limit=5)
        assert any(
            observation.addressed_by_outcome_id == str(measured[0].outcome_id)
            for record in records
            for observation in record.drift_observations
        )
    finally:
        await wellbeing.close()
        await inspections.close()


@pytest.mark.asyncio
async def test_estimated_drift_verification_failures_trigger_proposal_followup(tmp_path):
    wellbeing = await WellbeingStore(tmp_path / "wellbeing.db").connect()
    inspections = await SelfInspectionStore(tmp_path / "self_inspection.db").connect()
    try:
        base = datetime(2026, 5, 6, 13, 0, tzinfo=timezone.utc)
        before_record = SelfInspectionRecord(
            created_at=base,
            session_id="estimated-drift-failure",
            phase=SelfInspectionPhase.POST_TURN,
            drift_observations=[DriftObservation(reason="before")],
        )
        after_record = SelfInspectionRecord(
            created_at=base + timedelta(minutes=30),
            session_id="estimated-drift-failure",
            phase=SelfInspectionPhase.POST_TURN,
            drift_observations=[
                DriftObservation(reason=f"after {index}") for index in range(3)
            ],
        )
        await inspections.save(before_record)
        await inspections.save(after_record)
        for index in range(5):
            await wellbeing.save_maintenance_outcome(
                MaintenanceOutcome(
                    created_at=base + timedelta(minutes=10 + index),
                    action_type=MaintenanceActionType.RECOVER.value,
                    before_risk=0.52,
                    after_risk=0.48,
                    outcome="background_throttle_recorded",
                    meta={
                        "effect_basis": "estimated",
                        "effect_direction": "expected_lower",
                        "risk_delta": -0.04,
                        "drift_drain_status": "pending_validation",
                    },
                )
            )

        service = RecordOnlyMaintenanceFollowupService(
            wellbeing,
            self_inspection_store=inspections,
        )
        summary = await service.evaluate()

        assert summary.next_followup_action is not None
        assert summary.next_followup_action["type"] == "self_modification_proposal"
        assert summary.next_followup_action["source_action_type"] == "recover"
        recommendations = await wellbeing.list_recommendations(status="proposed", limit=5)
        assert recommendations
        assert recommendations[0].meta["followup_kind"] == "estimated_drift_drain_unverified"
    finally:
        await wellbeing.close()
        await inspections.close()
