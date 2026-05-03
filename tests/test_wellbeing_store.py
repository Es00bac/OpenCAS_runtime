import pytest

from opencas.wellbeing import (
    MaintenanceAction,
    MaintenanceActionType,
    MaintenanceOutcome,
    SelfModificationProposal,
    WellbeingEvent,
    WellbeingRecommendation,
    WellbeingState,
    WellbeingStore,
)


@pytest.mark.asyncio
async def test_wellbeing_store_round_trips_state_and_events(tmp_path):
    store = await WellbeingStore(tmp_path / "wellbeing.db").connect()
    state = WellbeingState(coherence=0.6, recovery_need=0.3)
    event = WellbeingEvent(event_type="assessment", summary="initial assessment")

    await store.save_state(state)
    await store.append_event(event)

    latest = await store.latest_state()
    states = await store.list_states(limit=5)
    events = await store.list_events(limit=5)

    assert latest is not None
    assert latest.state_id == state.state_id
    assert states[0].state_id == state.state_id
    assert events[0].summary == "initial assessment"
    await store.close()


@pytest.mark.asyncio
async def test_wellbeing_store_round_trips_recommendations_and_proposals(tmp_path):
    store = await WellbeingStore(tmp_path / "wellbeing.db").connect()
    recommendation = WellbeingRecommendation(
        reason="promise preservation",
        actions=[
            MaintenanceAction(
                action_type=MaintenanceActionType.PRESERVE_PROMISES,
                reason="active commitments exist",
                priority=0.9,
            )
        ],
    )
    proposal = SelfModificationProposal(
        title="Reduce unsupported source claims",
        rationale="Repeated drift records show unsupported source framing.",
        target_kind="prompt_adjustment",
        review_required=True,
    )

    await store.save_recommendation(recommendation)
    await store.save_self_modification_proposal(proposal)

    recommendations = await store.list_recommendations(limit=5)
    proposals = await store.list_self_modification_proposals(limit=5)

    assert recommendations[0].actions[0].action_type == MaintenanceActionType.PRESERVE_PROMISES
    assert proposals[0].review_required is True
    assert proposals[0].status == "proposed"
    await store.close()


@pytest.mark.asyncio
async def test_wellbeing_store_round_trips_maintenance_outcomes(tmp_path):
    store = await WellbeingStore(tmp_path / "wellbeing.db").connect()
    outcome = MaintenanceOutcome(
        action_type="recover",
        before_risk=0.8,
        after_risk=0.4,
        outcome="improved",
    )

    await store.save_maintenance_outcome(outcome)
    outcomes = await store.list_maintenance_outcomes(limit=5)

    assert outcomes[0].action_type == "recover"
    assert outcomes[0].after_risk == 0.4
    await store.close()
