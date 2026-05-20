from __future__ import annotations

from dataclasses import dataclass

import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.tools import ToolRegistry, ToolResult


@dataclass
class FakeViolation:
    value_name: str = "truthfulness"
    weight: float = 0.8
    description: str = "Do not fabricate evidence."

    def to_policy_evidence(self):
        return {
            "value_name": self.value_name,
            "weight": self.weight,
            "description": self.description,
        }


class FakeValuesEngine:
    def __init__(self, violations):
        self.violations = violations
        self.calls = []

    async def check_alignment_semantic(self, action_description, **kwargs):
        self.calls.append({"action_description": action_description, "kwargs": kwargs})
        return self.violations


def _register_tool(registry: ToolRegistry, *, tier: ActionRiskTier) -> None:
    async def _adapter(name, args):
        return ToolResult(success=True, output="ran", metadata={})

    registry.register("test_tool", "test tool", _adapter, tier)


@pytest.mark.asyncio
async def test_values_gate_only_reviews_high_risk_tools() -> None:
    registry = ToolRegistry()
    engine = FakeValuesEngine([])
    registry.values_engine = engine
    _register_tool(registry, tier=ActionRiskTier.WORKSPACE_WRITE)

    result = await registry.execute_async("test_tool", {"path": "x"})

    assert result.success is True
    assert engine.calls == []


@pytest.mark.asyncio
async def test_values_gate_blocks_high_risk_violation_and_caches() -> None:
    registry = ToolRegistry()
    engine = FakeValuesEngine([FakeViolation()])
    registry.values_engine = engine
    _register_tool(registry, tier=ActionRiskTier.EXTERNAL_WRITE)

    first = await registry.execute_async("test_tool", {"path": "x"})
    second = await registry.execute_async("test_tool", {"path": "x"})

    assert first.success is False
    assert first.metadata["values_violation"] is True
    assert first.metadata["violation"]["value_name"] == "truthfulness"
    assert second.success is False
    assert len(engine.calls) == 1
