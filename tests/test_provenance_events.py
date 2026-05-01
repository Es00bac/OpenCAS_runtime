from __future__ import annotations

import json
from urllib.parse import quote

import pytest

from opencas.api.provenance_events import (
    ProvenanceEvent,
    ProvenanceEventType,
    build_provenance_event,
    emit_provenance_event,
    parse_provenance_event,
    provenance_event_to_dict,
    ProvenanceEventValidationError,
)


def test_build_provenance_event_generates_stable_link() -> None:
    event = build_provenance_event(
        event_type=ProvenanceEventType.CHECK,
        triggering_artifact="file|workspace|notes.md",
        triggering_action="VERIFY",
        recorded_at="2026-04-21T12:00:00+00:00",
    )

    expected_link = f"opencas://provenance/check/{quote('file|workspace|notes.md', safe='')}?action={quote('VERIFY', safe='')}"
    assert event.source_link == expected_link
    assert event.event_type is ProvenanceEventType.CHECK
    assert event.recorded_at == "2026-04-21T12:00:00+00:00"


def test_emit_provenance_event_appends_to_record_and_round_trips() -> None:
    record = {"existing": "value"}
    event = emit_provenance_event(
        record,
        event_type=ProvenanceEventType.MUTATION,
        triggering_artifact="setting|phone|runtime",
        triggering_action="UPDATE",
        parent_link_id="state/phone/config.json",
        linked_link_ids=["state/phone/config.json", "setting|phone|runtime"],
        details={"field": "enabled"},
    )

    assert record["provenance_events"][0]["event_type"] == "MUTATION"
    assert record["provenance_events"][0]["triggering_artifact"] == "setting|phone|runtime"
    assert record["provenance_events"][0]["parent_link_id"] == "state/phone/config.json"
    assert record["provenance_events"][0]["linked_link_ids"] == ["state/phone/config.json", "setting|phone|runtime"]
    assert event.source_link.endswith("?action=UPDATE")
    parsed = parse_provenance_event(json.dumps(provenance_event_to_dict(event)))
    assert parsed.triggering_action == "UPDATE"
    assert parsed.parent_link_id == "state/phone/config.json"
    assert parsed.linked_link_ids == ("state/phone/config.json", "setting|phone|runtime")


def test_append_provenance_event_rejects_unknown_fields() -> None:
    with pytest.raises(ProvenanceEventValidationError):
        ProvenanceEvent.from_mapping(
            {
                "v": "1",
                "event_type": "CHECK",
                "triggering_artifact": "tool|default|demo",
                "triggering_action": "VERIFY",
                "source_link": "opencas://custom",
                "recorded_at": "2026-04-21T12:00:00+00:00",
                "unexpected": True,
            }
        )
