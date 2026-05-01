"""High-level tracer with span support and living handshake."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional
from uuid import UUID, uuid4

from .models import EventKind, TelemetryEvent
from .store import TelemetryStore


_current_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "span_id", default=None
)
_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id", default=None
)


class Tracer:
    """High-level telemetry tracer for OpenCAS.

    The living handshake mechanism turns the log from a passive record into
    an active bridge between memory and action. Three key operations:

    1. note_as_such: Explicitly acknowledge an event, marking it as integrated
       into the system's working memory.
    2. link_memory_to_action: Create a bidirectional bridge between a memory
       event and the action it enabled.
    3. activate_memory: Record when a memory was last activated into action,
       making continuity visible and queryable.
    """

    def __init__(self, store: TelemetryStore) -> None:
        self.store = store

    def _emit(
        self,
        kind: EventKind,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        span_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        noted_as_such: bool = False,
        continuity_backlink: Optional[str] = None,
        activated_at: Optional[datetime] = None,
    ) -> TelemetryEvent:
        event = TelemetryEvent(
            kind=kind,
            message=message,
            payload=payload or {},
            span_id=span_id or _current_span_id.get(),
            session_id=session_id or _current_session_id.get(),
            parent_span_id=parent_span_id,
            noted_as_such=noted_as_such,
            continuity_backlink=continuity_backlink,
            activated_at=activated_at,
        )
        self.store.append(event)
        return event

    def set_session(self, session_id: str) -> None:
        _current_session_id.set(session_id)

    def log(
        self,
        kind: EventKind,
        message: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> TelemetryEvent:
        return self._emit(kind=kind, message=message, payload=payload)

    # ── Living Handshake Methods ──────────────────────────────────────

    def note_as_such(self, event_id: UUID) -> TelemetryEvent:
        """Mark an existing event as noted-as-such.

        This creates a new MEMORY_NOTED event that references the original,
        making the acknowledgment explicit and traceable in the log.
        """
        return self._emit(
            kind=EventKind.MEMORY_NOTED,
            message=f"Memory noted-as-such: {event_id}",
            payload={"referenced_event_id": str(event_id)},
            noted_as_such=True,
        )

    def link_memory_to_action(
        self,
        memory_event_id: UUID,
        action_event_id: UUID,
    ) -> TelemetryEvent:
        """Create a bidirectional backlink between memory and action.

        This turns continuity into something visible: you can query from
        a memory to the actions it enabled, and from an action back to
        the memories that motivated it.
        """
        backlink = str(action_event_id)
        return self._emit(
            kind=EventKind.ACTION_BACKLINK,
            message=f"Memory {memory_event_id} enabled action {action_event_id}",
            payload={
                "memory_event_id": str(memory_event_id),
                "action_event_id": str(action_event_id),
            },
            continuity_backlink=backlink,
            noted_as_such=True,
        )

    def activate_memory(self, event_id: UUID) -> TelemetryEvent:
        """Record that a memory was activated into action.

        Updates the activated_at timestamp, making the last-use time
        of any memory visible and queryable. This is the heartbeat
        of the living handshake.
        """
        now = datetime.now(timezone.utc)
        return self._emit(
            kind=EventKind.MEMORY_ACTIVATED,
            message=f"Memory activated: {event_id}",
            payload={"referenced_event_id": str(event_id)},
            activated_at=now,
            noted_as_such=True,
        )

    # ── Span Context Manager ──────────────────────────────────────────

    @contextmanager
    def span(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Open a tracing span. Yields the span_id."""
        span_id = str(uuid4())
        parent_span_id = _current_span_id.get()
        _current_span_id.set(span_id)
        self._emit(
            kind=EventKind.SPAN_START,
            message=f"Span start: {name}",
            payload={"name": name, **(payload or {})},
            span_id=span_id,
            parent_span_id=parent_span_id,
        )
        try:
            yield span_id
        except Exception as exc:
            self._emit(
                kind=EventKind.ERROR,
                message=f"Span error: {name}: {exc}",
                payload={"name": name, "error": str(exc)},
                span_id=span_id,
                parent_span_id=parent_span_id,
            )
            raise
        finally:
            self._emit(
                kind=EventKind.SPAN_END,
                message=f"Span end: {name}",
                payload={"name": name},
                span_id=span_id,
                parent_span_id=parent_span_id,
            )
            _current_span_id.set(parent_span_id)