from types import SimpleNamespace

import pytest

from opencas.cognition.self_inspection import (
    DriftObservation,
    SelfInspectionPhase,
    SelfInspectionRecord,
)
from opencas.wellbeing import WellbeingDimension, WellbeingEngine


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
