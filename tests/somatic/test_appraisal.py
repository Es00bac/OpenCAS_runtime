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
