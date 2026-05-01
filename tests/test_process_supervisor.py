"""Tests for ProcessSupervisor and ProcessToolAdapter."""

import time

import pytest

from opencas.execution.process_supervisor import (
    _MAX_PROCESS_STREAM_LINE_CHARS,
    _MAX_PROCESS_STREAM_LINES,
    ProcessSupervisor,
)
from opencas.tools.adapters.process import ProcessToolAdapter


class TestProcessSupervisor:
    def test_start_poll_kill_lifecycle(self):
        supervisor = ProcessSupervisor()
        pid = supervisor.start("default", "echo hello")
        assert pid
        time.sleep(0.3)
        result = supervisor.poll("default", pid)
        assert result["found"] is True
        assert result["running"] is False
        assert result["returncode"] == 0
        assert result["stdout"] == "hello\n"
        assert supervisor.kill("default", pid) is True
        supervisor.remove("default", pid)
        supervisor.shutdown()

    def test_write_to_stdin(self):
        supervisor = ProcessSupervisor()
        pid = supervisor.start("default", "cat")
        assert supervisor.write("default", pid, "hello world\n") is True
        time.sleep(0.3)
        result = supervisor.poll("default", pid)
        assert result["stdout"] == "hello world\n"
        assert supervisor.kill("default", pid) is True
        supervisor.remove("default", pid)
        supervisor.shutdown()

    def test_scope_isolation(self):
        supervisor = ProcessSupervisor()
        pid = supervisor.start("scope_a", "echo scope_a")
        result = supervisor.poll("scope_b", pid)
        assert result["found"] is False
        assert supervisor.kill("scope_b", pid) is False
        supervisor.remove("scope_a", pid)
        supervisor.shutdown()

    def test_clear_removes_all_in_scope(self):
        supervisor = ProcessSupervisor()
        pid1 = supervisor.start("scope_x", "sleep 10")
        pid2 = supervisor.start("scope_x", "sleep 10")
        pid3 = supervisor.start("scope_y", "sleep 10")
        removed = supervisor.clear("scope_x")
        assert removed == 2
        assert supervisor.poll("scope_x", pid1)["found"] is False
        assert supervisor.poll("scope_x", pid2)["found"] is False
        assert supervisor.poll("scope_y", pid3)["found"] is True
        supervisor.kill("scope_y", pid3)
        supervisor.remove("scope_y", pid3)
        supervisor.shutdown()

    def test_remove_kills_running_process(self):
        supervisor = ProcessSupervisor()
        pid = supervisor.start("default", "sleep 10")
        assert supervisor.poll("default", pid)["running"] is True
        assert supervisor.remove("default", pid) is True
        time.sleep(0.2)
        assert supervisor.poll("default", pid)["found"] is False
        supervisor.shutdown()

    def test_send_signal(self):
        supervisor = ProcessSupervisor()
        pid = supervisor.start("default", "sleep 10")
        assert supervisor.send_signal("default", pid, 15) is True
        time.sleep(0.3)
        result = supervisor.poll("default", pid)
        assert result["running"] is False
        supervisor.remove("default", pid)
        supervisor.shutdown()

    def test_shutdown_clears_all(self):
        supervisor = ProcessSupervisor()
        pid1 = supervisor.start("s1", "sleep 10")
        pid2 = supervisor.start("s2", "sleep 10")
        supervisor.shutdown()
        assert supervisor.poll("s1", pid1)["found"] is False
        assert supervisor.poll("s2", pid2)["found"] is False

    def test_snapshot_surfaces_counts_and_entries(self):
        supervisor = ProcessSupervisor()
        pid = supervisor.start("scope_a", "sleep 1")
        try:
            snapshot = supervisor.snapshot()
            assert snapshot["total_count"] == 1
            assert snapshot["running_count"] == 1
            assert snapshot["scope_count"] == 1
            assert snapshot["entries"][0]["process_id"] == pid
            assert snapshot["entries"][0]["scope_key"] == "scope_a"
            assert snapshot["entries"][0]["cwd"]
        finally:
            supervisor.remove("scope_a", pid)
            supervisor.shutdown()

    def test_process_output_is_bounded(self):
        # Long-running command output should be capped to avoid unbounded growth.
        supervisor = ProcessSupervisor()
        pid = supervisor.start(
            "default",
            "python -u -c \"import sys; "
            "print('x' * 64); "
            "[sys.stdout.write('y' * 4096 + '\\\\n') or sys.stdout.flush() for _ in range(500)]\"",
        )
        result = {"stdout": "", "running": True}
        for _ in range(20):
            time.sleep(0.05)
            result = supervisor.poll("default", pid)
            if not result["running"]:
                break
        if result["running"]:
            supervisor.kill("default", pid)
            result = supervisor.poll("default", pid)
        assert result["found"] is True
        lines = (result["stdout"] or "").splitlines()
        assert len(lines) <= _MAX_PROCESS_STREAM_LINES
        if lines:
            assert max(len(line) for line in lines) <= _MAX_PROCESS_STREAM_LINE_CHARS
        assert supervisor.remove("default", pid) is True
        supervisor.shutdown()


class TestProcessToolAdapter:
    def test_adapter_start_and_poll(self):
        supervisor = ProcessSupervisor()
        adapter = ProcessToolAdapter(supervisor, default_cwd=".")
        result = adapter("process_start", {"command": "echo adapter_test"})
        assert result.success is True
        import json

        data = json.loads(result.output)
        pid = data["process_id"]
        time.sleep(0.3)
        poll_result = adapter("process_poll", {"process_id": pid})
        assert poll_result.success is True
        poll_data = json.loads(poll_result.output)
        assert poll_data["stdout"] == "adapter_test\n"
        supervisor.shutdown()

    def test_adapter_missing_command(self):
        supervisor = ProcessSupervisor()
        adapter = ProcessToolAdapter(supervisor, default_cwd=".")
        result = adapter("process_start", {})
        assert result.success is False
        assert "Missing required argument: command" in result.output
        supervisor.shutdown()

    def test_adapter_missing_process_id(self):
        supervisor = ProcessSupervisor()
        adapter = ProcessToolAdapter(supervisor, default_cwd=".")
        for name in ["process_poll", "process_write", "process_send_signal", "process_kill", "process_remove"]:
            result = adapter(name, {})
            assert result.success is False
            assert "Missing required argument: process_id" in result.output
        supervisor.shutdown()

    def test_adapter_write_and_kill(self):
        supervisor = ProcessSupervisor()
        adapter = ProcessToolAdapter(supervisor, default_cwd=".")
        start = adapter("process_start", {"command": "cat"})
        import json

        pid = json.loads(start.output)["process_id"]
        write = adapter("process_write", {"process_id": pid, "input": "hi\n"})
        assert write.success is True
        time.sleep(0.3)
        poll = adapter("process_poll", {"process_id": pid})
        assert json.loads(poll.output)["stdout"] == "hi\n"
        kill = adapter("process_kill", {"process_id": pid})
        assert kill.success is True
        supervisor.shutdown()

    def test_adapter_uses_default_cwd(self, tmp_path):
        supervisor = ProcessSupervisor()
        adapter = ProcessToolAdapter(supervisor, default_cwd=str(tmp_path))
        start = adapter("process_start", {"command": "pwd"})
        import json

        pid = json.loads(start.output)["process_id"]
        time.sleep(0.3)
        poll = adapter("process_poll", {"process_id": pid})
        poll_data = json.loads(poll.output)
        assert poll_data["stdout"].strip() == str(tmp_path)
        snapshot = supervisor.snapshot()
        assert snapshot["entries"][0]["cwd"] == str(tmp_path)
        supervisor.shutdown()

    def test_adapter_clear(self):
        supervisor = ProcessSupervisor()
        adapter = ProcessToolAdapter(supervisor, default_cwd=".")
        adapter("process_start", {"command": "sleep 10", "scope_key": "sc"})
        adapter("process_start", {"command": "sleep 10", "scope_key": "sc"})
        result = adapter("process_clear", {"scope_key": "sc"})
        assert result.success is True
        import json

        assert json.loads(result.output)["removed"] == 2
        supervisor.shutdown()

    def test_adapter_unknown_tool(self):
        supervisor = ProcessSupervisor()
        adapter = ProcessToolAdapter(supervisor, default_cwd=".")
        result = adapter("process_unknown", {})
        assert result.success is False
        assert "Unknown process tool" in result.output
        supervisor.shutdown()
