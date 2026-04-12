"""EventBus-to-WebSocket bridge for real-time telemetry."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket

from opencas.infra import BaaCompletedEvent, BaaProgressEvent


class WebSocketBridge:
    """Subscribes to EventBus events and forwards them to connected WebSocket clients."""

    def __init__(self, event_bus: Any) -> None:
        self._clients: Set[WebSocket] = set()
        self._event_bus = event_bus
        self._subscribed = False

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        if not self._subscribed:
            self._event_bus.subscribe(BaaProgressEvent, self._on_baa_progress)
            self._event_bus.subscribe(BaaCompletedEvent, self._on_baa_completed)
            self._subscribed = True

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        if not self._clients:
            return
        disconnected: Set[WebSocket] = set()
        text = json.dumps(message)
        for client in self._clients:
            try:
                await client.send_text(text)
            except Exception:
                disconnected.add(client)
        for client in disconnected:
            self._clients.discard(client)

    async def _on_baa_progress(self, event: BaaProgressEvent) -> None:
        await self.broadcast(
            {
                "type": "baa_progress",
                "task_id": event.task_id,
                "stage": event.stage,
                "objective": event.objective,
                "attempt": event.attempt,
                "timestamp": event.timestamp,
            }
        )

    async def _on_baa_completed(self, event: BaaCompletedEvent) -> None:
        await self.broadcast(
            {
                "type": "baa_completed",
                "task_id": event.task_id,
                "success": event.success,
                "stage": event.stage,
                "objective": event.objective,
                "output": event.output,
                "timestamp": event.timestamp,
            }
        )
