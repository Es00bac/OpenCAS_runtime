"""Reliability coordinator for detecting anomaly spikes in BAA execution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from opencas.infra import BaaCompletedEvent, BaaPauseEvent, EventBus


@dataclass
class _ResultRecord:
    success: bool
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ReliabilityCoordinator:
    """Monitors BAA completion events and emits pause signals on anomaly spikes."""

    def __init__(
        self,
        event_bus: EventBus,
        window_size: int = 10,
        failure_threshold: float = 0.7,
        cooldown_seconds: int = 300,
    ) -> None:
        self.event_bus = event_bus
        self.window_size = max(1, window_size)
        self.failure_threshold = max(0.0, min(1.0, failure_threshold))
        self.cooldown_seconds = max(0, cooldown_seconds)
        self._history: deque[_ResultRecord] = deque(maxlen=self.window_size)
        self._last_pause: Optional[datetime] = None
        self._handler_id = self._subscribe()

    def _subscribe(self) -> int:
        self.event_bus.subscribe(BaaCompletedEvent, self._on_baa_completed)
        return id(self._on_baa_completed)

    def stop(self) -> None:
        self.event_bus.unsubscribe(BaaCompletedEvent, self._on_baa_completed)

    async def _on_baa_completed(self, event: BaaCompletedEvent) -> None:
        self._history.append(_ResultRecord(success=event.success))
        if len(self._history) < self.window_size:
            return
        failure_rate = 1.0 - (sum(1 for r in self._history if r.success) / len(self._history))
        if failure_rate >= self.failure_threshold:
            if self._can_emit_pause():
                self._last_pause = datetime.now(timezone.utc)
                await self.event_bus.emit(
                    BaaPauseEvent(
                        reason="BAA failure rate spike detected",
                        failure_rate=round(failure_rate, 3),
                        window_size=len(self._history),
                    )
                )

    def _can_emit_pause(self) -> bool:
        if self._last_pause is None or self.cooldown_seconds == 0:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_pause).total_seconds()
        return elapsed >= self.cooldown_seconds

    def get_stats(self) -> dict:
        if not self._history:
            return {"failure_rate": 0.0, "window_size": 0, "paused": False}
        failure_rate = 1.0 - (sum(1 for r in self._history if r.success) / len(self._history))
        return {
            "failure_rate": round(failure_rate, 3),
            "window_size": len(self._history),
            "paused": self._last_pause is not None,
        }
