"""Typed cognitive event bus for runtime integration seams.

The bus is intentionally small in Phase A: it gives cognitive producers a named
event surface, records delivery failures durably, and lets later workstreams add
subscribers without creating another silent best-effort runtime peer.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Mapping
from uuid import uuid4


_FAILURE_SCHEMA = """
CREATE TABLE IF NOT EXISTS bus_handler_failures (
    failure_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    event_kind TEXT NOT NULL,
    handler_name TEXT NOT NULL,
    error TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bus_handler_failures_event_kind
ON bus_handler_failures(event_kind);

CREATE INDEX IF NOT EXISTS idx_bus_handler_failures_created_at
ON bus_handler_failures(created_at);
"""


EventHandler = Callable[["CognitiveEvent"], Any | Awaitable[Any]]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class CognitiveEvent:
    """Base event emitted by cognitive producers.

    `kind` is the stable routing key, `source` names the producer, and
    `evidence_ids` carries links to memories, rows, artifacts, or tool events
    that justify the signal.
    """

    kind: str
    source: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    evidence_ids: list[str] = field(default_factory=list)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(slots=True)
class AffectiveEvent(CognitiveEvent):
    """Affective signal intended for somatic, relational, and wellbeing consumers."""

    magnitude: float = 0.0


@dataclass(slots=True)
class WellbeingIntervention(CognitiveEvent):
    """Wellbeing action or recommendation that closes an assessment loop."""


@dataclass(slots=True)
class LadderHealthSummary(CognitiveEvent):
    """Creative-ladder choke-point summary emitted once per cycle."""


@dataclass(slots=True)
class AuthorizationGranted(CognitiveEvent):
    """Standing or session authorization observed by the approval layer."""


@dataclass(slots=True)
class NarratorRedundancyObserved(CognitiveEvent):
    """Narrator skipped a redundant self-knowledge arc."""


class AuditModeTurnObserved(CognitiveEvent):
    """Audit-only turn marker that downstream extractors can subscribe to."""

    def __init__(
        self,
        *,
        session_id: str,
        source: str = "agent_loop.converse",
        payload: Mapping[str, Any] | None = None,
        evidence_ids: Iterable[str] | None = None,
    ) -> None:
        merged_payload = {"session_id": session_id, **dict(payload or {})}
        super().__init__(
            kind="audit.turn_observed",
            source=source,
            payload=merged_payload,
            evidence_ids=list(evidence_ids or []),
        )


@dataclass(slots=True)
class BusHandlerFailure(CognitiveEvent):
    """Observable failure raised when a bus subscriber breaks."""


@dataclass(slots=True)
class DeliveryResult:
    handler_name: str
    success: bool
    error: str | None = None


@dataclass(slots=True)
class EventDeliveryReport:
    event: CognitiveEvent
    results: list[DeliveryResult] = field(default_factory=list)

    @property
    def delivered(self) -> int:
        return sum(1 for result in self.results if result.success)

    @property
    def failed(self) -> int:
        return sum(1 for result in self.results if not result.success)


class CognitionBus:
    """In-process bus with durable failure visibility.

    Handler exceptions are never swallowed. They are captured in the delivery
    report and written to `bus_failures.db` so monitoring can detect broken
    cognitive loops even when the originating turn continues.
    """

    def __init__(self, failure_store_path: Path | str | None = None) -> None:
        self.failure_store_path = Path(failure_store_path) if failure_store_path else None
        self._subscribers: dict[str, list[tuple[str, EventHandler]]] = {}
        if self.failure_store_path is not None:
            self._ensure_failure_store()

    def subscribe(
        self,
        kind: str,
        handler: EventHandler,
        *,
        name: str | None = None,
    ) -> None:
        handler_name = name or getattr(handler, "__name__", handler.__class__.__name__)
        self._subscribers.setdefault(kind, []).append((handler_name, handler))

    async def publish(self, event: CognitiveEvent) -> EventDeliveryReport:
        report = EventDeliveryReport(event=event)
        for handler_name, handler in list(self._subscribers.get(event.kind, [])):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
                report.results.append(DeliveryResult(handler_name=handler_name, success=True))
            except Exception as exc:
                error = str(exc)
                report.results.append(
                    DeliveryResult(
                        handler_name=handler_name,
                        success=False,
                        error=error,
                    )
                )
                self._record_failure(event, handler_name=handler_name, error=error)
                if event.kind != "bus.handler_failure":
                    await self._emit_handler_failure_event(
                        event,
                        handler_name=handler_name,
                        error=error,
                    )
        return report

    async def _emit_handler_failure_event(
        self,
        event: CognitiveEvent,
        *,
        handler_name: str,
        error: str,
    ) -> None:
        failure_event = BusHandlerFailure(
            kind="bus.handler_failure",
            source="cognition_bus",
            payload={
                "event_id": event.event_id,
                "event_kind": event.kind,
                "handler_name": handler_name,
                "error": error,
            },
            evidence_ids=[event.event_id],
        )
        for monitor_name, monitor in list(self._subscribers.get(failure_event.kind, [])):
            try:
                result = monitor(failure_event)
                if inspect.isawaitable(result):
                    await result
            except Exception as monitor_exc:
                self._record_failure(
                    failure_event,
                    handler_name=monitor_name,
                    error=str(monitor_exc),
                )

    def _ensure_failure_store(self) -> None:
        assert self.failure_store_path is not None
        self.failure_store_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.failure_store_path) as conn:
            conn.executescript(_FAILURE_SCHEMA)

    def _record_failure(
        self,
        event: CognitiveEvent,
        *,
        handler_name: str,
        error: str,
    ) -> None:
        if self.failure_store_path is None:
            return
        self._ensure_failure_store()
        payload = {
            "event_payload": dict(event.payload),
            "event_source": event.source,
            "evidence_ids": list(event.evidence_ids),
        }
        with sqlite3.connect(self.failure_store_path) as conn:
            conn.execute(
                """
                INSERT INTO bus_handler_failures (
                    failure_id, event_id, event_kind, handler_name, error,
                    payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    event.event_id,
                    event.kind,
                    handler_name,
                    error,
                    json.dumps(payload, sort_keys=True),
                    _utc_now().isoformat(),
                ),
            )
