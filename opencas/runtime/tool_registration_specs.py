"""Shared tool-registration spec helpers for AgentRuntime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from opencas.autonomy.models import ActionRiskTier
from opencas.platform import CapabilityDescriptor, CapabilitySource, CapabilityStatus


@dataclass(frozen=True)
class ToolRegistrationSpec:
    """Declarative description of one runtime tool registration."""

    name: str
    description: str
    risk_tier: ActionRiskTier
    schema: dict[str, Any]
    capability_id: str | None = None
    display_name: str | None = None


def register_tool_specs(
    runtime: Any,
    adapter: Any,
    specs: Iterable[ToolRegistrationSpec],
) -> None:
    """Register a homogeneous set of tools that share one adapter instance."""
    capability_registry = getattr(runtime, "capability_registry", None)
    for spec in specs:
        runtime.tools.register(
            spec.name,
            spec.description,
            adapter,
            spec.risk_tier,
            spec.schema,
            plugin_id="core",
        )
        if capability_registry is not None:
            capability_registry.register(
                CapabilityDescriptor(
                    capability_id=spec.capability_id or f"core:{spec.name}",
                    display_name=spec.display_name or spec.name,
                    kind="tool",
                    source=CapabilitySource.CORE,
                    owner_id="core",
                    status=CapabilityStatus.ENABLED,
                    description=spec.description,
                    tool_names=[spec.name],
                    config_schema=spec.schema,
                    metadata={"risk_tier": spec.risk_tier.value},
                )
            )
