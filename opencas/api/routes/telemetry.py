"""Telemetry / log viewer API routes for the OpenCAS dashboard."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from opencas.telemetry import EventKind, TelemetryEvent


class TelemetryEventsResponse(BaseModel):
    events: List[Dict[str, Any]]
    total: int
    has_more: bool


class TelemetryKindsResponse(BaseModel):
    kinds: List[Dict[str, Any]]


class TelemetrySessionsResponse(BaseModel):
    sessions: List[str]


class TelemetryStatsResponse(BaseModel):
    total_events: int
    events_by_kind: Dict[str, int]
    events_last_hour: int
    events_last_24h: int


def _event_to_dict(e: TelemetryEvent) -> Dict[str, Any]:
    return {
        "event_id": str(e.event_id),
        "timestamp": e.timestamp.isoformat(),
        "kind": e.kind.value,
        "message": e.message,
        "payload": e.payload,
        "session_id": e.session_id,
        "span_id": e.span_id,
        "parent_span_id": e.parent_span_id,
    }


def build_telemetry_router(runtime: Any) -> APIRouter:
    """Build telemetry routes wired to *runtime*."""
    r = APIRouter(prefix="/api/telemetry", tags=["telemetry"])

    @r.get("/events", response_model=TelemetryEventsResponse)
    async def get_telemetry_events(
        kinds: Optional[str] = None,
        session_id: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        since_minutes: Optional[int] = None,
    ) -> TelemetryEventsResponse:
        """Query telemetry events with filters."""
        store = runtime.tracer.store
        kind_list: Optional[List[EventKind]] = None
        if kinds:
            kind_list = [EventKind(k.strip()) for k in kinds.split(",") if k.strip()]

        def predicate(e: TelemetryEvent) -> bool:
            if search:
                text = f"{e.message or ''} {str(e.payload)}".lower()
                if search.lower() not in text:
                    return False
            if since_minutes:
                cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
                if e.timestamp < cutoff:
                    return False
            return True

        all_events = store.query(
            kinds=kind_list,
            session_id=session_id,
            limit=limit + offset + 1,
            predicate=predicate,
        )

        total = len(all_events)
        sliced = all_events[offset : offset + limit]
        has_more = total > offset + len(sliced)

        return TelemetryEventsResponse(
            events=[_event_to_dict(e) for e in sliced],
            total=total,
            has_more=has_more,
        )

    @r.get("/kinds", response_model=TelemetryKindsResponse)
    async def get_telemetry_kinds() -> TelemetryKindsResponse:
        """Return all event kinds with approximate counts from recent events."""
        store = runtime.tracer.store
        recent = store.query(limit=2000)
        counts: Dict[str, int] = {}
        for e in recent:
            counts[e.kind.value] = counts.get(e.kind.value, 0) + 1

        kinds = []
        for ek in EventKind:
            kinds.append(
                {
                    "kind": ek.value,
                    "count": counts.get(ek.value, 0),
                    "label": ek.value.replace("_", " ").title(),
                }
            )
        kinds.sort(key=lambda k: -k["count"])
        return TelemetryKindsResponse(kinds=kinds)

    @r.get("/sessions", response_model=TelemetrySessionsResponse)
    async def get_telemetry_sessions(limit: int = 50) -> TelemetrySessionsResponse:
        """Return recent session IDs from telemetry events."""
        store = runtime.tracer.store
        events = store.query(limit=2000)
        sessions: List[str] = []
        seen = set()
        for e in reversed(events):
            sid = e.session_id
            if sid and sid not in seen:
                seen.add(sid)
                sessions.append(sid)
                if len(sessions) >= limit:
                    break
        return TelemetrySessionsResponse(sessions=sessions)

    @r.get("/stats", response_model=TelemetryStatsResponse)
    async def get_telemetry_stats() -> TelemetryStatsResponse:
        """Return aggregate telemetry statistics."""
        store = runtime.tracer.store
        all_events = store.query(limit=5000)
        events_by_kind: Dict[str, int] = {}
        now = datetime.now(timezone.utc)
        last_hour = 0
        last_24h = 0
        for e in all_events:
            events_by_kind[e.kind.value] = events_by_kind.get(e.kind.value, 0) + 1
            age = now - e.timestamp
            if age <= timedelta(hours=1):
                last_hour += 1
            if age <= timedelta(hours=24):
                last_24h += 1

        return TelemetryStatsResponse(
            total_events=len(all_events),
            events_by_kind=events_by_kind,
            events_last_hour=last_hour,
            events_last_24h=last_24h,
        )

    @r.get("/stream")
    async def telemetry_stream(
        request: Request,
        kinds: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> StreamingResponse:
        """Server-Sent Events stream of live telemetry events."""
        store = runtime.tracer.store
        kind_set: Optional[set[str]] = None
        if kinds:
            kind_set = {k.strip() for k in kinds.split(",") if k.strip()}

        queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue()

        def _on_event(event: TelemetryEvent) -> None:
            try:
                if kind_set and event.kind.value not in kind_set:
                    return
                if session_id and event.session_id != session_id:
                    return
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

        store.subscribe(_on_event)

        async def _event_generator() -> AsyncGenerator[str, None]:
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        # Send a keep-alive comment to keep the connection alive
                        yield ":heartbeat\n\n"
                        continue
                    data = json.dumps(_event_to_dict(event))
                    yield f"data: {data}\n\n"
            finally:
                store.unsubscribe(_on_event)

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return r
