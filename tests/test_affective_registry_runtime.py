from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.affective_registry.models import ExecutionPhase
from opencas.runtime.agent_loop import _capture_affective_registry_turn_end
from opencas.somatic import SomaticState


class FakeWriter:
    def __init__(self) -> None:
        self.calls = []

    def append_from_somatic_state(self, somatic_state, **kwargs):
        self.calls.append({"somatic_state": somatic_state, "kwargs": kwargs})
        return SimpleNamespace(entry_id="entry-1")


@pytest.mark.asyncio
async def test_capture_affective_registry_turn_end_writes_somatic_state() -> None:
    writer = FakeWriter()
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            affective_registry_writer=writer,
            somatic=SimpleNamespace(state=SomaticState(tension=0.2)),
        ),
        _trace=lambda *args, **kwargs: None,
    )

    await _capture_affective_registry_turn_end(runtime, session_id="s1", outcome="assistant_response_persisted")

    assert len(writer.calls) == 1
    assert writer.calls[0]["kwargs"]["phase"] is ExecutionPhase.TURN_END
    assert writer.calls[0]["kwargs"]["session_id"] == "s1"
    assert writer.calls[0]["kwargs"]["payload"]["outcome"] == "assistant_response_persisted"
