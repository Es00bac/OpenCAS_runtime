"""Tests for persisted web-domain trust and learned evidence."""

from __future__ import annotations

import pytest

from opencas.governance import WebActionClass, WebTrustLevel, WebTrustService, WebTrustStore


@pytest.mark.asyncio
async def test_web_trust_learns_gray_then_trusted(tmp_path) -> None:
    service = await WebTrustService(WebTrustStore(tmp_path / "web_trust.db")).connect()

    initial = service.assess(
        url="https://docs.python.org/3/library/pathlib.html",
        domain=None,
        action_class=WebActionClass.FETCH,
    )
    assert initial is not None
    assert initial.level == WebTrustLevel.UNKNOWN

    for _ in range(2):
        await service.record_outcome(
            url="https://docs.python.org/3/library/pathlib.html",
            domain=None,
            action_class=WebActionClass.FETCH,
            success=True,
        )
    gray = service.assess(
        url="https://docs.python.org/3/library/pathlib.html",
        domain=None,
        action_class=WebActionClass.FETCH,
    )
    assert gray is not None
    assert gray.level == WebTrustLevel.GRAY

    for _ in range(10):
        await service.record_outcome(
            url="https://docs.python.org/3/library/pathlib.html",
            domain=None,
            action_class=WebActionClass.NAVIGATE,
            success=True,
        )
    trusted = service.assess(
        url="https://docs.python.org/3/library/pathlib.html",
        domain=None,
        action_class=WebActionClass.NAVIGATE,
    )
    assert trusted is not None
    assert trusted.level == WebTrustLevel.TRUSTED
    assert trusted.certainty >= 0.96

    await service.close()


@pytest.mark.asyncio
async def test_web_trust_block_policy_overrides_learned_state(tmp_path) -> None:
    service = await WebTrustService(WebTrustStore(tmp_path / "web_trust.db")).connect()
    await service.set_policy("example.com", WebTrustLevel.BLOCKED, note="manual block")

    assessment = service.assess(
        url="https://sub.example.com/forms/login",
        domain=None,
        action_class=WebActionClass.INTERACT,
    )

    assert assessment is not None
    assert assessment.blocked is True
    assert assessment.level == WebTrustLevel.BLOCKED
    assert assessment.matched_policy_domain == "example.com"

    await service.close()
