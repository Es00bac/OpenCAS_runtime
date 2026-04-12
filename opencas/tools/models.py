"""Data models for the tools subsystem."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from opencas.autonomy.models import ActionRiskTier


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    output: str
    metadata: Dict[str, Any]


@dataclass
class ToolEntry:
    """Registered tool with its adapter and risk tier."""

    name: str
    description: str
    adapter: Callable[[str, Dict[str, Any]], ToolResult]
    risk_tier: ActionRiskTier
    parameters: Optional[Dict[str, Any]] = None
