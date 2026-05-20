from __future__ import annotations

import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.autonomy.trust_engine import TrustEngine


class FakeIdentity:
    def __init__(self) -> None:
        class _UserModel:
            trust_level = 0.5

        class _SelfModel:
            self_beliefs = {}

        self.user_model = _UserModel()
        self.self_model = _SelfModel()
        self.saved = 0

    async def save(self) -> None:
        self.saved += 1


@pytest.fixture
def engine() -> TrustEngine:
    return TrustEngine(FakeIdentity())


def test_initial_trust(engine: TrustEngine) -> None:
    assert engine.current_trust() == 0.5


@pytest.mark.asyncio
async def test_success_increases_trust(engine: TrustEngine) -> None:
    first = await engine.record_success(tool_name="fs_read_file", tier=ActionRiskTier.READONLY)
    second = await engine.record_success(tool_name="fs_read_file", tier=ActionRiskTier.READONLY)

    assert first > 0.5
    assert second > first
    assert engine.identity.saved == 2


@pytest.mark.asyncio
async def test_failure_decreases_trust(engine: TrustEngine) -> None:
    engine.identity.user_model.trust_level = 0.8

    trust = await engine.record_failure(
        tool_name="fs_write_file",
        tier=ActionRiskTier.WORKSPACE_WRITE,
    )

    assert trust < 0.8
    assert engine.identity.saved == 1


def test_threshold_rises_with_trust(engine: TrustEngine) -> None:
    engine.identity.user_model.trust_level = 0.9
    high = engine.threshold_for(ActionRiskTier.SHELL_LOCAL)
    engine.identity.user_model.trust_level = 0.2
    low = engine.threshold_for(ActionRiskTier.SHELL_LOCAL)

    assert high > low


@pytest.mark.asyncio
async def test_max_trust_cap(engine: TrustEngine) -> None:
    engine.identity.user_model.trust_level = 0.98
    for _ in range(100):
        await engine.record_success()

    assert engine.current_trust() <= 0.98


@pytest.mark.asyncio
async def test_tool_success_rate_tracking(engine: TrustEngine) -> None:
    await engine.record_success(tool_name="fs_read_file", tier=ActionRiskTier.READONLY)

    beliefs = engine.identity.self_model.self_beliefs
    assert beliefs["success_rate_tool_fs_read_file"] > 0.5
    assert beliefs["success_rate_tier_readonly"] > 0.5
