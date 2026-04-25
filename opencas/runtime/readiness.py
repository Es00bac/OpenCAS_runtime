"""Agent readiness state machine for OpenCAS."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ReadinessState(str, Enum):
    """Lifecycle states for runtime readiness."""

    BOOTING = "booting"
    READY = "ready"
    DEGRADED = "degraded"
    PAUSED = "paused"
    FAILED = "failed"
    SHUTTING_DOWN = "shutting_down"


class AgentReadiness:
    """State machine tracking the runtime's readiness lifecycle."""

    def __init__(self) -> None:
        self._state = ReadinessState.BOOTING
        self._history: List[Dict[str, Any]] = []
        self._since = datetime.now(timezone.utc)
        self._reason: Optional[str] = "initialized"
        self._record(ReadinessState.BOOTING, self._reason)

    @property
    def state(self) -> ReadinessState:
        return self._state

    @property
    def since(self) -> datetime:
        return self._since

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def transition(self, to: ReadinessState, reason: Optional[str] = None) -> None:
        """Move to a new readiness state and record the transition."""
        if to == self._state:
            return
        self._state = to
        self._since = datetime.now(timezone.utc)
        self._reason = reason
        self._record(to, reason)

    def ready(self, reason: Optional[str] = None) -> None:
        self.transition(ReadinessState.READY, reason)

    def degraded(self, reason: str) -> None:
        self.transition(ReadinessState.DEGRADED, reason)

    def pause(self, reason: Optional[str] = None) -> None:
        self.transition(ReadinessState.PAUSED, reason)

    def fail(self, reason: str) -> None:
        self.transition(ReadinessState.FAILED, reason)

    def shutdown(self, reason: Optional[str] = None) -> None:
        self.transition(ReadinessState.SHUTTING_DOWN, reason)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self._state.value,
            "since": self._since.isoformat(),
            "reason": self._reason,
            "history": self._history[-10:],
        }

    def _record(self, state: ReadinessState, reason: Optional[str]) -> None:
        self._history.append(
            {
                "state": state.value,
                "at": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
            }
        )
