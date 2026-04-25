"""Tests for sandbox configuration and container detection."""

from pathlib import Path

import pytest

from opencas.sandbox import SandboxConfig, SandboxMode


def test_sandbox_default_mode() -> None:
    cfg = SandboxConfig()
    assert cfg.mode == SandboxMode.WORKSPACE_ONLY
    assert cfg.allowed_roots == []


def test_sandbox_report_isolation_off() -> None:
    cfg = SandboxConfig(mode=SandboxMode.OFF, allowed_roots=[Path("/tmp")])
    report = cfg.report_isolation()
    assert report["mode"] == "off"
    assert report["fallback"] is False
    assert "/tmp" in report["allowed_roots"]


def test_sandbox_report_isolation_workspace_fallback() -> None:
    # When not in a container and mode is not OFF, fallback should be True
    cfg = SandboxConfig(mode=SandboxMode.WORKSPACE_ONLY)
    report = cfg.report_isolation()
    assert report["mode"] == "workspace-only"
    # fallback is True when not in a container
    assert isinstance(report["container_detected"], bool)
    if report["container_detected"]:
        assert report["fallback"] is False
    else:
        assert report["fallback"] is True


def test_sandbox_detect_container_returns_bool() -> None:
    result = SandboxConfig.detect_container()
    assert isinstance(result, bool)
