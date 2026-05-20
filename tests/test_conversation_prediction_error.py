from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.cognition import CognitiveEventKind, CognitiveStateStore
from opencas.runtime.conversation_turns import _record_prediction_error_from_user_turn


@pytest.mark.asyncio
async def test_user_correction_records_prediction_error_for_next_turn(tmp_path) -> None:
    cognitive = CognitiveStateStore(tmp_path / "cognitive.db")
    await cognitive.connect()
    try:
        runtime = SimpleNamespace(
            cognitive_state_store=cognitive,
            ctx=SimpleNamespace(cognitive_state_store=cognitive),
            tom=None,
        )

        await _record_prediction_error_from_user_turn(
            runtime,
            "s1",
            "Actually, you do have shell access. I told you to check your capabilities.",
        )

        events = await cognitive.list_recent_events(kind=CognitiveEventKind.SURPRISE, limit=5)
        working = await cognitive.list_working_memory(limit=5)

        assert any("prediction-error" in event.summary for event in events)
        assert any(item.slot == "prediction_error_review" for item in working)
    finally:
        await cognitive.close()
