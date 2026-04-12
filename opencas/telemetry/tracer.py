"""High-level tracer with span support."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional
from uuid import uuid4

from .models import EventKind, TelemetryEvent
from .store import TelemetryStore


_current_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "span_id", default=None
)
_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "session_id", default=None
)


class Tracer:
    """High-level telemetry tracer for OpenCAS."""

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
    ) -> TelemetryEvent:
        event = TelemetryEvent(
            kind=kind,
            message=message,
            payload=payload or {},
            span_id=span_id or _current_span_id.get(),
            session_id=session_id or _current_session_id.get(),
            parent_span_id=parent_span_id,
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
