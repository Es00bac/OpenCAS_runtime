"""Async-safe circuit breaker for upstream LLM providers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CircuitState:
    failures: int = 0
    last_failure: float = 0.0
    open_since: Optional[float] = None
    recovery_timeout: float = 60.0

    def is_open(self) -> bool:
        if self.open_since is None:
            return False
        return (time.time() - self.open_since) < self.recovery_timeout


class ProviderCircuitOpen(RuntimeError):
    """Raised when a provider circuit is open and no fallback is wired."""

    def __init__(self, provider_name: str, opened_at: float) -> None:
        super().__init__(f"Provider {provider_name} circuit open since {opened_at}")
        self.provider_name = provider_name
        self.opened_at = opened_at


class ProviderCircuitBreaker:
    """Open a provider circuit after repeated failures."""

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 60.0) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_timeout = max(0.1, float(recovery_timeout))
        self._circuits: Dict[str, CircuitState] = {}
        self._lock = asyncio.Lock()

    async def record_success(self, provider_name: str) -> None:
        async with self._lock:
            self._circuits.pop(provider_name, None)

    async def record_failure(self, provider_name: str) -> None:
        async with self._lock:
            state = self._circuits.setdefault(
                provider_name,
                CircuitState(recovery_timeout=self.recovery_timeout),
            )
            state.failures += 1
            state.last_failure = time.time()
            if state.failures >= self.failure_threshold:
                state.open_since = time.time()

    async def is_open(self, provider_name: str) -> bool:
        async with self._lock:
            state = self._circuits.get(provider_name)
            if state is None:
                return False
            if state.open_since is None:
                return False
            if state.is_open():
                return True
            self._circuits.pop(provider_name, None)
            return False

    async def opened_at(self, provider_name: str) -> float:
        async with self._lock:
            state = self._circuits.get(provider_name)
            return float(state.open_since or 0.0) if state is not None else 0.0

    async def status(self) -> Dict[str, Dict[str, Any]]:
        async with self._lock:
            return {
                name: {
                    "failures": state.failures,
                    "last_failure": state.last_failure,
                    "open": state.is_open(),
                    "open_since": state.open_since,
                }
                for name, state in self._circuits.items()
            }
