"""Circuit breaker for iterative tool-use loops."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple


class ToolLoopGuard:
    """Tracks consecutive tool calls per session and breaks runaway loops."""

    # Complex but legitimate operator work can span planning, browser inspection,
    # PTY interaction, and filesystem verification before yielding a user-facing
    # answer. Keep the runaway-loop breaker, but allow a wider bounded budget.
    MAX_ROUNDS: int = 24
    _RECENT_WINDOW: int = 3
    # When round depth exceeds this threshold the agent is clearly in deep
    # task-execution mode; background loops should yield via focus mode.
    FOCUS_MODE_DEPTH: int = 8

    def __init__(self, *, max_rounds: Optional[int] = None) -> None:
        self.max_rounds = max(1, int(max_rounds or self.MAX_ROUNDS))
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def record_call(
        self,
        session_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
    ) -> Optional[str]:
        """Record a tool call and return an error message if the circuit breaks."""
        state = self._sessions.setdefault(session_id, {"count": 0, "recent": []})
        state["count"] += 1

        if state["count"] > self.max_rounds:
            return (
                f"Tool loop circuit breaker: exceeded {self.max_rounds} "
                "consecutive tool calls in this session."
            )

        # Track recent calls for identical-repeat detection
        sig: Tuple[str, str] = (tool_name, json.dumps(tool_args, sort_keys=True))
        state["recent"].append(sig)
        if len(state["recent"]) > self._RECENT_WINDOW:
            state["recent"].pop(0)

        if (
            len(state["recent"]) >= self._RECENT_WINDOW
            and len(set(state["recent"])) == 1
        ):
            return (
                f"Tool loop circuit breaker: tool '{tool_name}' was called "
                f"{self._RECENT_WINDOW} times with identical arguments."
            )

        return None

    def is_deep(self, session_id: str) -> bool:
        """Return True when the session has exceeded FOCUS_MODE_DEPTH rounds."""
        state = self._sessions.get(session_id)
        if state is None:
            return False
        return state["count"] >= self.FOCUS_MODE_DEPTH

    def reset(self, session_id: str) -> None:
        """Clear tracking state for a session."""
        self._sessions.pop(session_id, None)
