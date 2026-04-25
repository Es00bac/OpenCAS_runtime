"""Tests for the AgentReadiness state machine."""

import pytest

from opencas.runtime import AgentReadiness, ReadinessState


def test_readiness_starts_in_booting() -> None:
    r = AgentReadiness()
    assert r.state == ReadinessState.BOOTING
    assert r.reason == "initialized"
    assert len(r.history) == 1


def test_readiness_transition_changes_state() -> None:
    r = AgentReadiness()
    r.ready("boot_complete")
    assert r.state == ReadinessState.READY
    assert r.reason == "boot_complete"
    assert len(r.history) == 2


def test_readiness_noop_when_same_state() -> None:
    r = AgentReadiness()
    r.transition(ReadinessState.BOOTING, "still booting")
    assert r.state == ReadinessState.BOOTING
    assert len(r.history) == 1


def test_readiness_pause_and_degraded() -> None:
    r = AgentReadiness()
    r.ready()
    r.pause("user_requested")
    assert r.state == ReadinessState.PAUSED
    r.degraded("memory_high")
    assert r.state == ReadinessState.DEGRADED


def test_readiness_fail_and_shutdown() -> None:
    r = AgentReadiness()
    r.fail("unrecoverable")
    assert r.state == ReadinessState.FAILED
    r.shutdown("sigterm")
    assert r.state == ReadinessState.SHUTTING_DOWN


def test_readiness_snapshot() -> None:
    r = AgentReadiness()
    r.ready("done")
    snap = r.snapshot()
    assert snap["state"] == "ready"
    assert snap["reason"] == "done"
    assert len(snap["history"]) == 2
