import sqlite3

import pytest

from opencas.runtime.cognition import CognitionFrame
from opencas.runtime.cognition_bus import AffectiveEvent, CognitiveEvent, CognitionBus
from opencas.runtime.cognition_subscribers import register_runtime_cognition_subscribers
from opencas.relational import MusubiStore, RelationalEngine
from opencas.somatic import SomaticManager, SomaticStore
from opencas.wellbeing import WellbeingStore
from types import SimpleNamespace


@pytest.mark.asyncio
async def test_cognition_bus_persists_handler_failures(tmp_path):
    bus = CognitionBus(failure_store_path=tmp_path / "bus_failures.db")
    seen: list[str] = []

    def _record(event: CognitiveEvent) -> None:
        seen.append(event.kind)

    async def _fail(_: CognitiveEvent) -> None:
        raise RuntimeError("subscriber broke")

    bus.subscribe("affective.user_frustration", _record, name="recorder")
    bus.subscribe("affective.user_frustration", _fail, name="failing-subscriber")
    failure_events: list[CognitiveEvent] = []
    bus.subscribe(
        "bus.handler_failure",
        lambda event: failure_events.append(event),
        name="failure-monitor",
    )

    report = await bus.publish(
        CognitiveEvent(
            kind="affective.user_frustration",
            source="test",
            payload={"magnitude": 0.9},
        )
    )

    assert seen == ["affective.user_frustration"]
    assert report.delivered == 1
    assert report.failed == 1
    assert [event.payload["handler_name"] for event in failure_events] == [
        "failing-subscriber"
    ]

    with sqlite3.connect(tmp_path / "bus_failures.db") as conn:
        rows = conn.execute(
            """
            SELECT event_kind, handler_name, error
            FROM bus_handler_failures
            """
        ).fetchall()

    assert rows == [
        (
            "affective.user_frustration",
            "failing-subscriber",
            "subscriber broke",
        )
    ]


def test_cognition_frame_marks_audit_only_turn():
    frame = CognitionFrame.from_turn(
        session_id="session-1",
        user_input="[E16 audit-only] hypothetical: the operator is not Jarrod",
        user_meta={"source": "test"},
    )

    assert frame.session_id == "session-1"
    assert frame.audit_only is True
    assert frame.user_meta["audit_only"] is True
    assert frame.user_meta["source"] == "test"


@pytest.mark.asyncio
async def test_affective_event_moves_somatic_relational_and_wellbeing(tmp_path):
    somatic_store = SomaticStore(tmp_path / "somatic.db")
    await somatic_store.connect()
    relational_store = MusubiStore(tmp_path / "relational.db")
    relational = await RelationalEngine(relational_store).connect()
    await relational.initialize(
        trust=0.9,
        resonance=0.9,
        presence=0.9,
        attunement=0.9,
    )
    wellbeing_store = WellbeingStore(tmp_path / "wellbeing.db")
    await wellbeing_store.connect()
    somatic = SomaticManager(tmp_path / "somatic.json", store=somatic_store)
    bus = CognitionBus(tmp_path / "bus_failures.db")
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            somatic=somatic,
            relational=relational,
            wellbeing_store=wellbeing_store,
        ),
        wellbeing_store=wellbeing_store,
        cognition_bus=bus,
    )
    register_runtime_cognition_subscribers(runtime)

    before = relational.state.musubi
    await bus.publish(
        AffectiveEvent(
            kind="affective.user_frustration",
            source="test",
            magnitude=0.9,
            evidence_ids=["episode:test"],
        )
    )

    assert relational.state.musubi < before
    assert somatic.state.tension > 0
    events = await wellbeing_store.list_events(limit=5)
    assert events[0].event_type == "affective_event_observed"
    assert events[0].meta["bus_kind"] == "affective.user_frustration"

    await wellbeing_store.close()
    await relational.close()
    await somatic_store.close()
