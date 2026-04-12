"""Tests for ToolLoopGuard circuit breaker."""

import pytest

from opencas.tools.loop_guard import ToolLoopGuard


class TestToolLoopGuard:
    def test_initial_calls_allowed(self):
        guard = ToolLoopGuard()
        for i in range(ToolLoopGuard.MAX_ROUNDS):
            assert guard.record_call("s1", "fs_read_file", {"path": f"/tmp/{i}"}) is None

    def test_max_rounds_circuit_breaker(self):
        guard = ToolLoopGuard()
        for i in range(ToolLoopGuard.MAX_ROUNDS):
            guard.record_call("s1", "fs_read_file", {"path": f"/tmp/{i}"})

        reason = guard.record_call("s1", "fs_read_file", {"path": "/tmp/extra"})
        assert reason is not None
        assert f"exceeded {ToolLoopGuard.MAX_ROUNDS}" in reason

    def test_identical_call_circuit_breaker(self):
        guard = ToolLoopGuard()
        args = {"path": "/tmp"}
        assert guard.record_call("s1", "fs_read_file", args) is None
        assert guard.record_call("s1", "fs_read_file", args) is None
        reason = guard.record_call("s1", "fs_read_file", args)
        assert reason is not None
        assert "fs_read_file" in reason
        assert "3 times" in reason

    def test_different_tools_do_not_trigger_identical_guard(self):
        guard = ToolLoopGuard()
        for _ in range(5):
            assert guard.record_call("s1", "tool_a", {"x": 1}) is None
            assert guard.record_call("s1", "tool_b", {"x": 1}) is None

    def test_reset_clears_state(self):
        guard = ToolLoopGuard()
        for i in range(ToolLoopGuard.MAX_ROUNDS):
            guard.record_call("s1", "fs_read_file", {"path": f"/tmp/{i}"})

        guard.reset("s1")
        assert guard.record_call("s1", "fs_read_file", {"path": "/tmp"}) is None

    def test_isolation_per_session(self):
        guard = ToolLoopGuard()
        for i in range(ToolLoopGuard.MAX_ROUNDS):
            guard.record_call("s1", "tool", {"i": i})
        assert guard.record_call("s1", "tool", {"i": 99}) is not None
        assert guard.record_call("s2", "tool", {"i": 99}) is None
