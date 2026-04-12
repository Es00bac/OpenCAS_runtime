"""Tests for Docker sandbox option."""

from pathlib import Path
from subprocess import TimeoutExpired
from unittest.mock import MagicMock, patch

import pytest

from opencas.sandbox import SandboxConfig, SandboxMode
from opencas.sandbox.docker import DockerSandbox
from opencas.tools.adapters.shell import ShellToolAdapter


class TestDockerSandbox:
    def test_check_available_true(self):
        sandbox = DockerSandbox(allowed_roots=[Path("/tmp")])
        with patch(
            "opencas.sandbox.docker.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert sandbox.check_available() is True

    def test_check_available_false(self):
        sandbox = DockerSandbox(allowed_roots=[Path("/tmp")])
        with patch(
            "opencas.sandbox.docker.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert sandbox.check_available() is False

    def test_ensure_running_creates_container(self):
        sandbox = DockerSandbox(allowed_roots=[Path("/tmp")])
        with patch(
            "opencas.sandbox.docker.subprocess.run"
        ) as mock_run:
            # 1st call (docker --version in check_available)
            # 2nd call (inspect) fails → container not running
            # 3rd call (rm -f) succeeds
            # 4th call (run) succeeds
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=1, stdout=""),
                MagicMock(returncode=0),
                MagicMock(returncode=0, stdout="container_id"),
            ]
            assert sandbox._ensure_running() is True

    def test_run_command_without_container(self):
        sandbox = DockerSandbox(allowed_roots=[Path("/tmp")])
        with patch.object(
            sandbox, "_ensure_running", return_value=False
        ):
            result = sandbox.run_command("echo hello")
            assert result["ok"] is False
            assert "unavailable" in result["error"]

    def test_run_command_success(self):
        sandbox = DockerSandbox(allowed_roots=[Path("/tmp")])
        with patch.object(sandbox, "_ensure_running", return_value=True):
            with patch(
                "opencas.sandbox.docker.subprocess.run"
            ) as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="hello\n",
                    stderr="",
                )
                result = sandbox.run_command("echo hello")
                assert result["ok"] is True
                assert result["code"] == 0
                assert result["stdout"] == "hello\n"

    def test_run_command_timeout(self):
        sandbox = DockerSandbox(allowed_roots=[Path("/tmp")], timeout=1.0)
        with patch.object(sandbox, "_ensure_running", return_value=True):
            with patch(
                "opencas.sandbox.docker.subprocess.run",
                side_effect=TimeoutExpired(cmd="docker", timeout=1.0),
            ):
                result = sandbox.run_command("sleep 10")
                assert result["ok"] is False
                assert "timed out" in result["error"]


class TestShellToolAdapterDockerRouting:
    def test_routes_through_docker_when_sandbox_present(self):
        docker = MagicMock()
        docker.run_command.return_value = {
            "ok": True,
            "code": 0,
            "stdout": "hi",
            "stderr": "",
        }
        shell = ShellToolAdapter(cwd="/tmp", docker_sandbox=docker)
        result = shell("bash_run_command", {"command": "echo hi"})
        docker.run_command.assert_called_once_with("echo hi", cwd="/tmp")
        assert result.success is True
        assert result.metadata.get("sandboxed") is True

    def test_falls_back_to_subprocess_without_docker(self):
        shell = ShellToolAdapter(cwd="/tmp")
        with patch(
            "opencas.tools.adapters.shell.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="hi\n",
                stderr="",
            )
            result = shell("bash_run_command", {"command": "echo hi"})
            mock_run.assert_called_once()
            assert mock_run.call_args.kwargs["shell"] is False
            assert result.success is True
            assert "sandboxed" not in result.output

    def test_uses_shell_for_shell_syntax(self):
        shell = ShellToolAdapter(cwd="/tmp")
        with patch(
            "opencas.tools.adapters.shell.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="hi\n",
                stderr="",
            )
            result = shell("bash_run_command", {"command": "echo hi | cat"})
            mock_run.assert_called_once()
            assert mock_run.call_args.kwargs["shell"] is True
            assert result.success is True


class TestSandboxConfig:
    def test_docker_mode_added(self):
        cfg = SandboxConfig(mode=SandboxMode.DOCKER)
        assert cfg.mode == SandboxMode.DOCKER
