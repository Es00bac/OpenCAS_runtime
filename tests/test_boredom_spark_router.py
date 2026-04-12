"""Tests for BoredomPhysics and SparkRouter."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest

from opencas.autonomy.boredom import BoredomPhysics
from opencas.autonomy.models import WorkObject
from opencas.autonomy.spark_router import SparkRouter, SparkRung


def test_boredom_zero_at_start() -> None:
    bp = BoredomPhysics()
    assert bp.compute_boredom() == pytest.approx(0.0, abs=0.01)


def test_boredom_grows_with_idle_time() -> None:
    bp = BoredomPhysics()
    now = datetime.now(timezone.utc)
    # Simulate 4 hours of idle time by backdating both activity and reset
    bp._last_activity_at = now - timedelta(hours=4)
    bp._last_reset_at = now - timedelta(hours=4)
    boredom = bp.compute_boredom(now)
    assert boredom > 0.9


def test_motivation_blend() -> None:
    bp = BoredomPhysics()
    now = datetime.now(timezone.utc)
    bp._last_activity_at = now - timedelta(hours=2)
    bp._last_reset_at = now - timedelta(hours=2)
    motivation = bp.compute_motivation(somatic_readiness=0.5, now=now)
    # boredom at 2h ~0.76, motivation = 0.76*0.76 + 0.24*0.5 ~0.70
    assert motivation > 0.55


def test_should_daydream_threshold() -> None:
    bp = BoredomPhysics()
    now = datetime.now(timezone.utc)
    bp._last_activity_at = now - timedelta(hours=0.1)
    bp._last_reset_at = now - timedelta(hours=0.1)
    assert bp.should_daydream(somatic_readiness=0.5, now=now) is False

    bp._last_activity_at = now - timedelta(hours=3)
    bp._last_reset_at = now - timedelta(hours=3)
    assert bp.should_daydream(somatic_readiness=0.5, now=now) is True


def test_record_activity_resets_boredom() -> None:
    bp = BoredomPhysics()
    now = datetime.now(timezone.utc)
    bp._last_reset_at = now - timedelta(hours=3)
    bp.record_activity()
    assert bp.compute_boredom() < 0.1


def test_spark_router_rejects_low_boredom_or_score() -> None:
    router = SparkRouter()
    wo = WorkObject(content="spark", promotion_score=0.2)
    assert router.route(wo, None, boredom=0.1) == SparkRung.REJECT
    assert router.route(wo, None, boredom=0.5) == SparkRung.REJECT


def test_spark_router_note_micro_task_full_task() -> None:
    router = SparkRouter()
    wo = WorkObject(content="spark", promotion_score=0.35)
    assert router.route(wo, None, boredom=0.5) == SparkRung.NOTE

    wo.promotion_score = 0.55
    assert router.route(wo, None, boredom=0.5) == SparkRung.MICRO_TASK

    wo.promotion_score = 0.7
    assert router.route(wo, None, boredom=0.5) == SparkRung.FULL_TASK


def test_persistent_intent_bypass() -> None:
    router = SparkRouter()
    wo = WorkObject(content="recurring spark", promotion_score=0.65, meta={"intensity": 0.7})
    now = datetime.now(timezone.utc)

    # Reject three times in the last 24h
    for _ in range(3):
        router.route(wo, None, boredom=0.1, now=now)

    # Next route with sufficient score should bypass to FULL_TASK
    rung = router.route(wo, None, boredom=0.5, now=now)
    assert rung == SparkRung.FULL_TASK


def test_no_bypass_without_intensity() -> None:
    router = SparkRouter()
    wo = WorkObject(content="recurring spark", promotion_score=0.64, meta={"intensity": 0.3})
    now = datetime.now(timezone.utc)

    for i in range(3):
        router.route(wo, None, boredom=0.1, now=now + timedelta(minutes=i))

    rung = router.route(wo, None, boredom=0.5, now=now + timedelta(hours=1))
    assert rung == SparkRung.MICRO_TASK
