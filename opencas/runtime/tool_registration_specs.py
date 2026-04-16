"""Shared tool-registration spec helpers for AgentRuntime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from opencas.autonomy.models import ActionRiskTier


@dataclass(frozen=True)
class ToolRegistrationSpec:
    """Declarative description of one runtime tool registration."""

    name: str
    description: str
    risk_tier: ActionRiskTier
    schema: dict[str, Any]


def register_tool_specs(
    runtime: Any,
    adapter: Any,
    specs: Iterable[ToolRegistrationSpec],
) -> None:
    """Register a homogeneous set of tools that share one adapter instance."""
    for spec in specs:
        runtime.tools.register(
            spec.name,
            spec.description,
            adapter,
            spec.risk_tier,
            spec.schema,
            plugin_id="core",
        )
