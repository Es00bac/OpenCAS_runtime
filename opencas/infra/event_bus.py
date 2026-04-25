"""Typed async event bus for cross-module reactive communication."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Type, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

EventHandler = Callable[[T], Coroutine[Any, Any, None]]


@dataclass
class BaaProgressEvent:
    """Emitted when a BAA task advances through a lifecycle stage."""

    task_id: str
    stage: str
    objective: str
    attempt: int
    timestamp: str | None = None


@dataclass
class BaaCompletedEvent:
    """Emitted when a BAA task reaches a terminal state."""

    task_id: str
    success: bool
    stage: str
    objective: str
    output: str = ""
    timestamp: str | None = None


@dataclass
class BaaPauseEvent:
    """Emitted when the reliability coordinator requests a BAA pause."""

    reason: str
    failure_rate: float
    window_size: int
    timestamp: str | None = None


@dataclass
class HealthCheckEvent:
    """Emitted by the health monitor after a periodic diagnostic sweep."""

    overall: str
    failures: int
    warnings: int
    checks: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str | None = None


class EventBus:
    """Lightweight in-memory async event bus with typed subscriptions."""

    def __init__(self) -> None:
        self._handlers: Dict[Type[Any], List[EventHandler[Any]]] = {}
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: Type[T], handler: EventHandler[T]) -> None:
        """Register *handler* to receive events of *event_type*."""
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: Type[T], handler: EventHandler[T]) -> None:
        """Remove *handler* from *event_type* listeners."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def emit(self, event: Any) -> None:
        """Broadcast *event* to all subscribed handlers concurrently."""
        event_type = type(event)
        handlers = list(self._handlers.get(event_type, []))
        if not handlers:
            return
        await asyncio.gather(
            *(self._invoke(handler, event) for handler in handlers),
            return_exceptions=True,
        )

    @staticmethod
    async def _invoke(handler: EventHandler[Any], event: Any) -> None:
        try:
            await handler(event)
        except Exception:
            # Handlers must not crash the bus; swallow errors to preserve
            # delivery to other subscribers.
            logger.exception("EventBus handler failed for event %s", type(event).__name__)
