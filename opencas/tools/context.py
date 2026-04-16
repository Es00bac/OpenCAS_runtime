"""Context and result types for the ToolUseLoop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from opencas.runtime import AgentRuntime


class UserInputRequired(Exception):
    """Raised when a tool requires interactive user input to proceed."""

    def __init__(self, question: str, options: Optional[List[str]] = None) -> None:
        self.question = question
        self.options = options or []
        super().__init__(question)


@dataclass
class ToolUseContext:
    """Runtime context for a single ToolUseLoop invocation."""

    runtime: "AgentRuntime"
    session_id: str
    task_id: Optional[str] = None
    max_iterations: int = 32
    plan_mode: bool = False
    active_plan_id: Optional[str] = None


@dataclass
class ToolUseResult:
    """Outcome of a ToolUseLoop run."""

    final_output: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    guard_fired: bool = False
    guard_reason: Optional[str] = None
