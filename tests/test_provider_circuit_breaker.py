from __future__ import annotations

import pytest

from opencas.api.provider_circuit_breaker import ProviderCircuitBreaker, ProviderCircuitOpen


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold_and_resets_on_success() -> None:
    breaker = ProviderCircuitBreaker(failure_threshold=2, recovery_timeout=60)

    assert await breaker.is_open("provider-a") is False
    await breaker.record_failure("provider-a")
    assert await breaker.is_open("provider-a") is False
    await breaker.record_failure("provider-a")
    assert await breaker.is_open("provider-a") is True
    await breaker.record_success("provider-a")
    assert await breaker.is_open("provider-a") is False


def test_provider_circuit_open_exposes_provider_name() -> None:
    exc = ProviderCircuitOpen("provider-a", 123.0)

    assert exc.provider_name == "provider-a"
    assert exc.opened_at == 123.0
    assert "provider-a" in str(exc)
