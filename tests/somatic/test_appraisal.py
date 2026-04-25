"""Tests for somatic appraisal event taxonomy."""

import pytest
import pytest_asyncio

from opencas.somatic import AppraisalEventType, SomaticAppraisalEvent, SomaticManager
from opencas.somatic.models import PrimaryEmotion
from opencas.somatic.store import SomaticStore


@pytest_asyncio.fixture
async def manager(tmp_path):
    store = SomaticStore(tmp_path / "somatic.db")
    await store.connect()
    mgr = SomaticManager(tmp_path / "somatic.json", store=store)
    yield mgr
    await store.close()


@pytest.mark.asyncio
async def test_emit_appraisal_event_with_snapshot(manager):
    event = await manager.emit_appraisal_event(
        AppraisalEventType.USER_INPUT_RECEIVED,
        source_text="I am so happy today",
        trigger_event_id="turn-1",
        snapshot=True,
    )
    assert event.event_type == AppraisalEventType.USER_INPUT_RECEIVED
    assert event.source_text == "I am so happy today"
    assert event.trigger_event_id == "turn-1"
    assert event.affect_state is not None
    assert event.affect_state.primary_emotion == PrimaryEmotion.JOY
    # Snapshot should have been persisted
    recent = await manager.store.list_recent(limit=1)
    assert len(recent) == 1
    assert recent[0].source == "user_input_received"


@pytest.mark.asyncio
async def test_emit_appraisal_event_no_text(manager):
    event = await manager.emit_appraisal_event(
        AppraisalEventType.TOOL_REJECTED,
        source_text="",
        snapshot=False,
    )
    assert event.event_type == AppraisalEventType.TOOL_REJECTED
    assert event.affect_state is None


@pytest.mark.asyncio
async def test_emit_appraisal_event_nudges_state(manager):
    manager.set_valence(0.0)
    event = await manager.emit_appraisal_event(
        AppraisalEventType.CONFLICT_DETECTED,
        source_text="I am so afraid",
        snapshot=False,
    )
    assert event.affect_state is not None
    assert event.affect_state.primary_emotion == PrimaryEmotion.FEAR
    # State should have been nudged toward negative valence
    assert manager.state.valence < 0.0


@pytest.mark.asyncio
async def test_appraise_generated_returns_affect(manager):
    """appraise_generated should use keyword matching on assistant output."""
    affect = await manager.appraise_generated("I'm happy to help you with that!")
    assert affect is not None
    assert affect.primary_emotion == PrimaryEmotion.JOY


@pytest.mark.asyncio
async def test_reconcile_detects_masking(manager):
    """High internal tension + calm expressed text → masking detected."""
    from opencas.somatic.models import SomaticState, AffectState
    from datetime import datetime, timezone

    # Simulate high-tension internal state
    pre = SomaticState(tension=0.8, valence=-0.2, certainty=0.7)

    # Assistant says something warm/calm
    expressed = AffectState(
        primary_emotion=PrimaryEmotion.TRUST,
        valence=0.6,
        arousal=0.3,
        certainty=0.8,
        intensity=0.5,
        emotion_tags=["trust"],
    )

    result = await manager.reconcile(pre, expressed)
    assert result["masking_detected"] is True
    assert "valence_down_masking" in result["adjustments"]
    # Valence should have been nudged down
    assert manager.state.valence < 0.0


@pytest.mark.asyncio
async def test_reconcile_warm_affect_reduces_tension(manager):
    """Warm expressed affect should reduce tension."""
    from opencas.somatic.models import SomaticState, AffectState

    manager.set_tension(0.5)
    pre = SomaticState(tension=0.5)

    expressed = AffectState(
        primary_emotion=PrimaryEmotion.CARING,
        valence=0.6,
        arousal=0.4,
        certainty=0.7,
        intensity=0.5,
        emotion_tags=["caring"],
    )

    result = await manager.reconcile(pre, expressed)
    assert "tension_down_warm" in result["adjustments"]
    assert manager.state.tension < 0.5


@pytest.mark.asyncio
async def test_reconcile_apologetic_drops_certainty(manager):
    """Apologetic expressed text while internally confident → certainty drops."""
    from opencas.somatic.models import SomaticState, AffectState

    manager.set_certainty(0.8)
    pre = SomaticState(certainty=0.8)

    expressed = AffectState(
        primary_emotion=PrimaryEmotion.APOLOGETIC,
        valence=-0.1,
        arousal=0.2,
        certainty=0.3,
        intensity=0.4,
        emotion_tags=["apologetic"],
    )

    result = await manager.reconcile(pre, expressed)
    assert "certainty_down_apologetic" in result["adjustments"]
    assert manager.state.certainty < 0.8


@pytest.mark.asyncio
async def test_self_response_generated_event_type():
    """SELF_RESPONSE_GENERATED event type should exist."""
    assert AppraisalEventType.SELF_RESPONSE_GENERATED.value == "self_response_generated"
