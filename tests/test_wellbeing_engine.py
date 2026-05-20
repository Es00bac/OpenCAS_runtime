from types import SimpleNamespace

import pytest

from opencas.cognition import SelfInspectionStore
from opencas.cognition.self_inspection import (
    DriftObservation,
    SelfInspectionPhase,
    SelfInspectionRecord,
)
from opencas.wellbeing import MaintenanceOutcome, WellbeingDimension, WellbeingEngine
from opencas.wellbeing.drift import annotate_drift_drain_effect


@pytest.mark.asyncio
async def test_engine_marks_recovery_need_from_fatigue_and_tension():
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            somatic=SimpleNamespace(
                state=SimpleNamespace(fatigue=0.82, tension=0.72, certainty=0.35)
            ),
            daydream_store=None,
        ),
        commitment_store=None,
        self_inspection_store=None,
        relational=None,
        tom=None,
    )

    assessment = await WellbeingEngine().assess(runtime)

    assert assessment.state.recovery_need >= 0.7
    assert assessment.state.truth_pressure >= 0.3
    assert WellbeingDimension.RECOVERY_NEED in assessment.concern_dimensions
    assert assessment.state.grounding


@pytest.mark.asyncio
async def test_engine_marks_promise_load_from_active_commitments():
    class FakeCommitments:
        async def list_active(self, limit=100):
            return [object(), object(), object(), object()]

    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            somatic=SimpleNamespace(
                state=SimpleNamespace(fatigue=0.0, tension=0.1, certainty=0.8)
            ),
            daydream_store=None,
        ),
        commitment_store=FakeCommitments(),
        self_inspection_store=None,
        relational=None,
        tom=None,
    )

    assessment = await WellbeingEngine().assess(runtime)

    assert assessment.state.promise_load >= 0.4
    assert WellbeingDimension.PROMISE_LOAD in assessment.concern_dimensions


@pytest.mark.asyncio
async def test_engine_marks_drift_and_low_curiosity_from_recent_records_and_daydreams():
    class FakeSelfInspection:
        async def list_recent(self, **kwargs):
            return [
                SelfInspectionRecord(
                    session_id="s1",
                    phase=SelfInspectionPhase.POST_TURN,
                    drift_observations=[
                        DriftObservation(
                            reason="repeated response shape",
                            severity="warning",
                        )
                    ],
                )
            ]

        async def list_unresolved_commitment_gaps(self, **kwargs):
            return []

    class FakeDaydreams:
        async def list_recent(self, limit=10):
            return []

    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            somatic=SimpleNamespace(
                state=SimpleNamespace(fatigue=0.0, tension=0.1, certainty=0.8)
            ),
            daydream_store=FakeDaydreams(),
        ),
        commitment_store=None,
        self_inspection_store=FakeSelfInspection(),
        relational=None,
        tom=None,
    )

    assessment = await WellbeingEngine().assess(runtime)

    assert assessment.state.drift_load >= 0.4
    assert assessment.state.curiosity <= 0.35
    assert WellbeingDimension.DRIFT_LOAD in assessment.concern_dimensions
    assert WellbeingDimension.CURIOSITY in assessment.concern_dimensions


@pytest.mark.asyncio
async def test_measured_maintenance_outcome_drains_addressed_drift_observations(tmp_path):
    store = await SelfInspectionStore(tmp_path / "self_inspection.db").connect()
    try:
        record = SelfInspectionRecord(
            session_id="drift-drain",
            phase=SelfInspectionPhase.POST_TURN,
            drift_observations=[
                DriftObservation(reason="shape repeat 1"),
                DriftObservation(reason="shape repeat 2"),
                DriftObservation(reason="shape repeat 3"),
            ],
        )
        await store.save(record)
        outcome = MaintenanceOutcome(
            action_type="truth_repair",
            before_risk=0.52,
            after_risk=0.48,
            outcome="truth_repair_receipt",
            meta={
                "effect_basis": "measured",
                "effect_direction": "measured_lower",
                "risk_delta": -0.04,
            },
        )

        drain = await annotate_drift_drain_effect(store, outcome)

        assert drain.drained_count == 2
        assert outcome.meta["drift_drain_status"] == "drained"
        assert str(record.record_id) in outcome.evidence_ids
        records = await store.list_recent(limit=5)
        addressed = [
            observation
            for saved in records
            for observation in saved.drift_observations
            if observation.addressed_by_outcome_id == str(outcome.outcome_id)
        ]
        assert len(addressed) == 2

        runtime = SimpleNamespace(
            ctx=SimpleNamespace(
                somatic=SimpleNamespace(
                    state=SimpleNamespace(fatigue=0.0, tension=0.1, certainty=0.8)
                ),
                daydream_store=None,
            ),
            commitment_store=None,
            self_inspection_store=store,
            wellbeing_store=None,
            relational=None,
            tom=None,
        )
        assessment = await WellbeingEngine().assess(runtime)

        assert assessment.state.drift_load == 0.42
        assert assessment.state.meta["recent_drift_observation_count"] == 1
        assert assessment.state.meta["addressed_drift_observation_count"] == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_estimated_maintenance_outcome_does_not_drain_drift_observations(tmp_path):
    self_store = await SelfInspectionStore(tmp_path / "self_inspection.db").connect()
    try:
        record = SelfInspectionRecord(
            session_id="drift-pending",
            phase=SelfInspectionPhase.POST_TURN,
            drift_observations=[
                DriftObservation(reason="shape repeat 1"),
                DriftObservation(reason="shape repeat 2"),
            ],
        )
        await self_store.save(record)
        outcome = MaintenanceOutcome(
            action_type="recover",
            before_risk=0.52,
            after_risk=0.48,
            outcome="background_throttle_recorded",
            meta={
                "effect_basis": "estimated",
                "effect_direction": "expected_lower",
                "risk_delta": -0.04,
            },
        )

        drain = await annotate_drift_drain_effect(self_store, outcome)

        assert drain.status == "pending_validation"
        assert outcome.meta["drift_drain_status"] == "pending_validation"
        records = await self_store.list_recent(limit=5)
        assert all(
            not observation.addressed_by_outcome_id
            for saved in records
            for observation in saved.drift_observations
        )

        class FakeWellbeingStore:
            async def list_maintenance_outcomes(self, **kwargs):
                return [outcome]

        runtime = SimpleNamespace(
            ctx=SimpleNamespace(
                somatic=SimpleNamespace(
                    state=SimpleNamespace(fatigue=0.0, tension=0.1, certainty=0.8)
                ),
                daydream_store=None,
            ),
            commitment_store=None,
            self_inspection_store=self_store,
            wellbeing_store=FakeWellbeingStore(),
            relational=None,
            tom=None,
        )
        assessment = await WellbeingEngine().assess(runtime)

        assert assessment.state.drift_load == 0.84
        assert assessment.state.meta["drift_drain_pending"] == 1
    finally:
        await self_store.close()
