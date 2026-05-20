from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opencas.api.routes.memory import build_memory_router
from opencas.api.routes.memory import _cognitive_flow_node
from opencas.telemetry import EventKind


def _event(kind: EventKind, message: str, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        event_id="event-1",
        kind=kind,
        message=message,
        payload=payload,
        timestamp=datetime(2026, 5, 12, 14, 0, tzinfo=timezone.utc),
        session_id="session-1",
        span_id=None,
        parent_span_id=None,
        continuity_backlink=None,
        noted_as_such=False,
        activated_at=None,
    )


def test_cognitive_flow_node_exposes_observable_memory_thought_text() -> None:
    node = _cognitive_flow_node(
        _event(
            EventKind.MEMORY_ACTIVATED,
            "Memory node activated",
            {
                "content_preview": "retrieved continuity memory",
                "query": "current objective",
                "rank": 2,
                "score": 0.72,
            },
        ),
        now=datetime(2026, 5, 12, 14, 1, tzinfo=timezone.utc),
        span_depth=0,
    )

    assert node["is_noise"] is False
    assert node["flow_priority"] >= 0.9
    assert "Accessed memory/context evidence" in node["thought_text"]
    assert "retrieved continuity memory" in node["thought_text"]


def test_cognitive_flow_node_marks_low_value_diagnostics_as_noise() -> None:
    node = _cognitive_flow_node(
        _event(EventKind.DIAGNOSTIC_RUN, "Doctor telemetry probe", {}),
        now=datetime(2026, 5, 12, 14, 1, tzinfo=timezone.utc),
        span_depth=0,
    )

    assert node["is_noise"] is True
    assert node["flow_priority"] < 0.1


def test_cognitive_flow_node_marks_tool_registration_as_noise() -> None:
    node = _cognitive_flow_node(
        _event(
            EventKind.TOOL_CALL,
            "ToolRegistry: tool_registered",
            {"name": "fs_read_file", "risk_tier": "readonly"},
        ),
        now=datetime(2026, 5, 12, 14, 1, tzinfo=timezone.utc),
        span_depth=0,
    )

    assert node["is_noise"] is True
    assert node["flow_priority"] < 0.1


def test_cognitive_flow_node_marks_scheduler_lifecycle_as_noise() -> None:
    node = _cognitive_flow_node(
        _event(EventKind.TOOL_CALL, "AgentScheduler: scheduler_start", {}),
        now=datetime(2026, 5, 12, 14, 1, tzinfo=timezone.utc),
        span_depth=0,
    )

    assert node["is_noise"] is True
    assert node["flow_priority"] < 0.1


def test_cognitive_flow_node_marks_skipped_schedule_trigger_as_noise() -> None:
    node = _cognitive_flow_node(
        _event(
            EventKind.TOOL_CALL,
            "ScheduleService: schedule_triggered",
            {"status": "skipped"},
        ),
        now=datetime(2026, 5, 12, 14, 1, tzinfo=timezone.utc),
        span_depth=0,
    )

    assert node["is_noise"] is True
    assert node["flow_priority"] < 0.1


def test_cognitive_flow_node_marks_embedding_batch_as_noise() -> None:
    node = _cognitive_flow_node(
        _event(EventKind.LLM_CALL, "LLM embed_batch: google/embeddinggemma-300m", {}),
        now=datetime(2026, 5, 12, 14, 1, tzinfo=timezone.utc),
        span_depth=0,
    )

    assert node["is_noise"] is True
    assert node["flow_priority"] < 0.1


def test_cognitive_flow_endpoint_links_no_span_events_by_cognitive_sequence() -> None:
    now = datetime.now(timezone.utc)
    events = [
        SimpleNamespace(
            event_id="memory-event",
            kind=EventKind.MEMORY_ACTIVATED,
            message="Memory node activated",
            payload={"content_preview": "prior project context", "query": "current work"},
            timestamp=now - timedelta(seconds=20),
            session_id="session-1",
            span_id=None,
            parent_span_id=None,
            continuity_backlink=None,
            noted_as_such=False,
            activated_at=None,
        ),
        SimpleNamespace(
            event_id="approval-event",
            kind=EventKind.SELF_APPROVAL,
            message="SelfApprovalLadder: ordinary action approved",
            payload={"tool_name": "web_fetch", "level": "ordinary", "reason": "low risk research"},
            timestamp=now - timedelta(seconds=10),
            session_id="session-1",
            span_id=None,
            parent_span_id=None,
            continuity_backlink=None,
            noted_as_such=False,
            activated_at=None,
        ),
        SimpleNamespace(
            event_id="tool-event",
            kind=EventKind.TOOL_CALL,
            message="BrowserTool: fetch_complete",
            payload={"tool": "browser_fetch", "status": "complete", "result": "thread evidence collected"},
            timestamp=now - timedelta(seconds=3),
            session_id="session-1",
            span_id=None,
            parent_span_id=None,
            continuity_backlink=None,
            noted_as_such=False,
            activated_at=None,
        ),
    ]

    class Store:
        def query(self, **_kwargs):
            return events

    app = FastAPI()
    app.include_router(build_memory_router(SimpleNamespace(tracer=SimpleNamespace(store=Store()))))

    response = TestClient(app).get("/api/memory/cognitive-flow?window_seconds=120&limit=20")

    assert response.status_code == 200
    data = response.json()
    assert data["event_count"] == 3
    assert data["stats"]["edge_count"] >= 2
    assert "cognitive_sequence" in {edge["kind"] for edge in data["edges"]}
    assert {node["flow_lane"] for node in data["nodes"]} >= {
        "memory and recall",
        "decision and approval",
        "tools and execution",
    }
    assert [node["sequence_index"] for node in data["nodes"]] == [1, 2, 3]
    assert [node["temporal_order_label"] for node in data["nodes"]] == ["1/3", "2/3", "3/3"]
    assert all(node["temporal_has_metadata"] for node in data["nodes"])
    assert all(edge["temporal_has_metadata"] for edge in data["edges"])
    assert data["stats"]["high_signal_event_count"] == 3


def test_cognitive_flow_endpoint_exposes_recent_readable_thought_ledger() -> None:
    now = datetime.now(timezone.utc)
    events = [
        SimpleNamespace(
            event_id="memory-event",
            kind=EventKind.MEMORY_ACTIVATED,
            message="Memory node activated",
            payload={
                "content_preview": "retrieved preference about autonomous research",
                "query": "ordinary approval calibration",
                "rank": 1,
                "score": 0.86,
            },
            timestamp=now - timedelta(seconds=20),
            session_id="session-1",
            span_id=None,
            parent_span_id=None,
            continuity_backlink=None,
            noted_as_such=False,
            activated_at=None,
        ),
        SimpleNamespace(
            event_id="approval-event",
            kind=EventKind.SELF_APPROVAL,
            message="SelfApprovalLadder: ordinary action approved",
            payload={
                "tool_name": "browser_navigate",
                "level": "can_do_with_caution",
                "reasoning": "ordinary browser navigation for research",
            },
            timestamp=now - timedelta(seconds=10),
            session_id="session-1",
            span_id=None,
            parent_span_id=None,
            continuity_backlink=None,
            noted_as_such=False,
            activated_at=None,
        ),
        SimpleNamespace(
            event_id="generic-tom-event",
            kind=EventKind.TOM_EVAL,
            message="ToMEngine: consistency_check",
            payload={},
            timestamp=now - timedelta(seconds=5),
            session_id="session-1",
            span_id=None,
            parent_span_id=None,
            continuity_backlink=None,
            noted_as_such=False,
            activated_at=None,
        ),
    ]

    class Store:
        def query(self, **_kwargs):
            return events

    app = FastAPI()
    app.include_router(build_memory_router(SimpleNamespace(tracer=SimpleNamespace(store=Store()))))

    response = TestClient(app).get("/api/memory/cognitive-flow?window_seconds=120&limit=20")

    assert response.status_code == 200
    data = response.json()
    ledger = data["stats"]["recent_readable_thoughts"]
    assert data["stats"]["readable_thought_count"] == 2
    assert len(ledger) == 2
    assert ledger[0]["event_id"] == "memory-event"
    assert "retrieved preference" in ledger[0]["thought_text"]
    assert ledger[1]["event_id"] == "approval-event"
    assert "ordinary browser navigation" in ledger[1]["reasoning_text"]


def test_cognitive_flow_readable_ledger_filters_scheduler_bookkeeping_but_keeps_tool_evidence() -> None:
    now = datetime.now(timezone.utc)
    events = [
        SimpleNamespace(
            event_id="schedule-skip",
            kind=EventKind.TOOL_CALL,
            message="ScheduleService: schedule_triggered",
            payload={"status": "skipped"},
            timestamp=now - timedelta(seconds=20),
            session_id="session-1",
            span_id=None,
            parent_span_id=None,
            continuity_backlink=None,
            noted_as_such=False,
            activated_at=None,
        ),
        SimpleNamespace(
            event_id="tool-evidence",
            kind=EventKind.TOOL_CALL,
            message="BrowserTool: fetch_complete",
            payload={"tool": "browser_fetch", "result": "thread evidence collected"},
            timestamp=now - timedelta(seconds=10),
            session_id="session-1",
            span_id=None,
            parent_span_id=None,
            continuity_backlink=None,
            noted_as_such=False,
            activated_at=None,
        ),
    ]

    class Store:
        def query(self, **_kwargs):
            return events

    app = FastAPI()
    app.include_router(build_memory_router(SimpleNamespace(tracer=SimpleNamespace(store=Store()))))

    response = TestClient(app).get("/api/memory/cognitive-flow?window_seconds=120&limit=20")

    assert response.status_code == 200
    data = response.json()
    ledger = data["stats"]["recent_readable_thoughts"]
    assert data["stats"]["readable_thought_count"] == 1
    assert ledger[0]["event_id"] == "tool-evidence"
    assert "thread evidence collected" in ledger[0]["thought_text"]


def test_cognitive_flow_readable_ledger_deduplicates_repeated_pause_events() -> None:
    now = datetime.now(timezone.utc)
    events = []
    for index in range(3):
        events.append(
            SimpleNamespace(
                event_id=f"cycle-backoff-{index}",
                kind=EventKind.TOOL_CALL,
                message="AgentScheduler: cycle_backoff",
                payload={"reason": "executive_recommended_pause"},
                timestamp=now - timedelta(seconds=30 - index),
                session_id="session-1",
                span_id=None,
                parent_span_id=None,
                continuity_backlink=None,
                noted_as_such=False,
                activated_at=None,
            )
        )

    class Store:
        def query(self, **_kwargs):
            return events

    app = FastAPI()
    app.include_router(build_memory_router(SimpleNamespace(tracer=SimpleNamespace(store=Store()))))

    response = TestClient(app).get("/api/memory/cognitive-flow?window_seconds=120&limit=20")

    assert response.status_code == 200
    data = response.json()
    ledger = data["stats"]["recent_readable_thoughts"]
    assert data["stats"]["readable_thought_count"] == 3
    assert len(ledger) == 1
    assert ledger[0]["event_id"] == "cycle-backoff-2"
    assert ledger[0]["thought_text"] == "Backed off an autonomous work cycle because executive_recommended_pause"


def test_cognitive_flow_endpoint_honors_limit_and_keeps_high_signal_events() -> None:
    now = datetime.now(timezone.utc)
    events = []
    for index in range(50):
        events.append(
            SimpleNamespace(
                event_id=f"noise-{index}",
                kind=EventKind.TOOL_CALL,
                message="AgentScheduler: identity_persistence_heartbeat",
                payload={},
                timestamp=now - timedelta(seconds=100 - index),
                session_id="session-1",
                span_id=None,
                parent_span_id=None,
                continuity_backlink=None,
                noted_as_such=False,
                activated_at=None,
            )
        )
    events.extend(
        [
            SimpleNamespace(
                event_id="memory-high-signal",
                kind=EventKind.MEMORY_ACTIVATED,
                message="Memory node activated",
                payload={"content_preview": "important continuity", "query": "active work"},
                timestamp=now - timedelta(seconds=80),
                session_id="session-1",
                span_id=None,
                parent_span_id=None,
                continuity_backlink=None,
                noted_as_such=False,
                activated_at=None,
            ),
            SimpleNamespace(
                event_id="approval-high-signal",
                kind=EventKind.SELF_APPROVAL,
                message="SelfApprovalLadder: ordinary action approved",
                payload={"reason": "safe low-risk follow-up"},
                timestamp=now - timedelta(seconds=70),
                session_id="session-1",
                span_id=None,
                parent_span_id=None,
                continuity_backlink=None,
                noted_as_such=False,
                activated_at=None,
            ),
        ]
    )

    class Store:
        def query(self, **_kwargs):
            return events

    app = FastAPI()
    app.include_router(build_memory_router(SimpleNamespace(tracer=SimpleNamespace(store=Store()))))

    response = TestClient(app).get("/api/memory/cognitive-flow?window_seconds=120&limit=20")

    assert response.status_code == 200
    data = response.json()
    node_ids = {node["event_id"] for node in data["nodes"]}
    assert data["event_count"] == 20
    assert data["stats"]["candidate_event_count"] == 52
    assert data["stats"]["selected_event_count"] == 20
    assert data["stats"]["truncated_by_limit"] is True
    assert "memory-high-signal" in node_ids
    assert "approval-high-signal" in node_ids


def test_cognitive_flow_endpoint_can_hide_low_information_noise() -> None:
    now = datetime.now(timezone.utc)
    events = [
        SimpleNamespace(
            event_id="noise-event",
            kind=EventKind.TOOL_CALL,
            message="AgentScheduler: identity_persistence_heartbeat",
            payload={},
            timestamp=now - timedelta(seconds=20),
            session_id="session-1",
            span_id=None,
            parent_span_id=None,
            continuity_backlink=None,
            noted_as_such=False,
            activated_at=None,
        ),
        SimpleNamespace(
            event_id="memory-event",
            kind=EventKind.MEMORY_ACTIVATED,
            message="Memory node activated",
            payload={"content_preview": "relevant recall", "query": "active task"},
            timestamp=now - timedelta(seconds=10),
            session_id="session-1",
            span_id=None,
            parent_span_id=None,
            continuity_backlink=None,
            noted_as_such=False,
            activated_at=None,
        ),
    ]

    class Store:
        def query(self, **_kwargs):
            return events

    app = FastAPI()
    app.include_router(build_memory_router(SimpleNamespace(tracer=SimpleNamespace(store=Store()))))

    response = TestClient(app).get("/api/memory/cognitive-flow?window_seconds=120&limit=20&include_noise=false")

    assert response.status_code == 200
    data = response.json()
    assert data["event_count"] == 1
    assert data["nodes"][0]["event_id"] == "memory-event"
    assert data["stats"]["include_noise"] is False
    assert data["stats"]["excluded_noise_event_count"] == 1
