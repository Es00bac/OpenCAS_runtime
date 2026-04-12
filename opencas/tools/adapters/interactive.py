"""Interactive tool adapter for pausing the loop to ask the user."""

from __future__ import annotations

from typing import Any, Dict

from ..models import ToolResult


class InteractiveToolAdapter:
    """Adapter that surfaces questions to the operator."""

    def __init__(self) -> None:
        pass

    def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "ask_user_question":
                return self._ask_user_question(args)
            return ToolResult(success=False, output=f"Unknown interactive tool: {name}", metadata={})
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    def _ask_user_question(self, args: Dict[str, Any]) -> ToolResult:
        question = str(args.get("question", ""))
        if not question:
            return ToolResult(success=False, output="question is required", metadata={})
        # Returning success=False signals ToolUseLoop to raise UserInputRequired
        return ToolResult(success=False, output=question, metadata={})
