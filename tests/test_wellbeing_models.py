from opencas.cognition import CognitionGrounding, GroundingKind, GroundingSource
from opencas.wellbeing import (
    MaintenanceAction,
    MaintenanceActionType,
    WellbeingAssessment,
    WellbeingDimension,
    WellbeingEvent,
    WellbeingRecommendation,
    WellbeingState,
)


def test_wellbeing_state_clamps_dimensions_and_preserves_grounding():
    grounding = CognitionGrounding(
        kind=GroundingKind.DERIVED,
        source=GroundingSource.RUNTIME,
        claim="High promise load was derived from active commitments.",
        subject="promise_load",
        confidence=0.8,
    )
    state = WellbeingState(
        coherence=1.4,
        recovery_need=-0.5,
        promise_load=0.7,
        grounding=[grounding],
    )

    assert state.coherence == 1.0
    assert state.recovery_need == 0.0
    assert state.promise_load == 0.7
    assert state.grounding[0].subject == "promise_load"


def test_assessment_identifies_primary_dimensions():
    assessment = WellbeingAssessment(
        state=WellbeingState(
            coherence=0.45,
            autonomy=0.35,
            relationship_pressure=0.72,
            curiosity=0.2,
        )
    )

    assert WellbeingDimension.RELATIONSHIP_PRESSURE in assessment.concern_dimensions
    assert WellbeingDimension.CURIOSITY in assessment.concern_dimensions
    assert WellbeingDimension.COHERENCE in assessment.concern_dimensions
    assert assessment.state.overall_risk > 0


def test_recommendation_normalizes_actions_and_evidence():
    recommendation = WellbeingRecommendation(
        reason="  preserve promises before taking new work ",
        actions=[
            MaintenanceAction(
                action_type=MaintenanceActionType.PRESERVE_PROMISES,
                reason=" active user-facing commitments exist ",
                priority=1.4,
            )
        ],
        status=" proposed ",
    )
    event = WellbeingEvent(event_type=" assessment ", summary="  fresh state derived ")

    assert recommendation.reason == "preserve promises before taking new work"
    assert recommendation.status == "proposed"
    assert recommendation.actions[0].priority == 1.0
    assert event.event_type == "assessment"
    assert event.summary == "fresh state derived"
