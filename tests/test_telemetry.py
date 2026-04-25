"""Tests for the telemetry module."""

import pytest
from pathlib import Path
from opencas.telemetry import EventKind, TelemetryStore, Tracer


def test_telemetry_store_append_and_query(tmp_path: Path) -> None:
    store = TelemetryStore(tmp_path)
    tracer = Tracer(store)

    tracer.set_session("session-1")
    event = tracer.log(EventKind.MEMORY_WRITE, "Stored episode", {"episode_id": "ep-1"})

    assert event.session_id == "session-1"
    assert event.kind == EventKind.MEMORY_WRITE

    results = store.query(kinds=[EventKind.MEMORY_WRITE])
    assert len(results) == 1
    assert results[0].payload["episode_id"] == "ep-1"


def test_telemetry_span(tmp_path: Path) -> None:
    store = TelemetryStore(tmp_path)
    tracer = Tracer(store)

    with tracer.span("test-span", {"extra": 42}) as span_id:
        tracer.log(EventKind.TOOL_CALL, "tool called")

    results = store.query()
    kinds = [r.kind for r in results]
    assert EventKind.SPAN_START in kinds
    assert EventKind.SPAN_END in kinds
    assert EventKind.TOOL_CALL in kinds

    span_starts = store.query(kinds=[EventKind.SPAN_START])
    assert span_starts[0].payload["extra"] == 42
    assert span_starts[0].span_id == span_id


def test_telemetry_query_filter_session(tmp_path: Path) -> None:
    store = TelemetryStore(tmp_path)
    tracer = Tracer(store)

    tracer.set_session("session-a")
    tracer.log(EventKind.MEMORY_WRITE, "a")

    tracer.set_session("session-b")
    tracer.log(EventKind.MEMORY_WRITE, "b")

    results = store.query(session_id="session-a")
    assert len(results) == 1
    assert results[0].message == "a"
