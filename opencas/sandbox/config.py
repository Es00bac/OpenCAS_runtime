"""Sandbox configuration with explicit modes and container detection."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SandboxMode(str, Enum):
    """Isolation modes for OpenCAS execution."""

    OFF = "off"
    WORKSPACE_ONLY = "workspace-only"
    ALLOW_LIST = "allow-list"
    DOCKER = "docker"


class SandboxConfig(BaseModel):
    """Configuration for filesystem and execution sandboxing."""

    mode: SandboxMode = SandboxMode.WORKSPACE_ONLY
    allowed_roots: List[Path] = Field(default_factory=list)

    @staticmethod
    def detect_container() -> bool:
        """Return True if running inside a container or Linux namespace."""
        # Docker/Containerd marker file
        if Path("/.dockerenv").exists():
            return True

        # Check cgroup controllers for container signatures
        cgroup_path = Path("/proc/self/cgroup")
        if cgroup_path.exists():
            try:
                contents = cgroup_path.read_text()
                markers = ("docker", "containerd", "kubepods", "lxc", "libpod")
                if any(m in contents for m in markers):
                    return True
            except Exception:
                pass

        # Environment hints
        env_hints = ("KUBERNETES_SERVICE_HOST", "container", "CONTAINER_ID")
        if any(os.environ.get(k) for k in env_hints):
            return True

        return False

    def report_isolation(self) -> Dict[str, Any]:
        """Return a diagnostic dict describing current isolation level."""
        container = self.detect_container()
        return {
            "mode": self.mode.value,
            "container_detected": container,
            "allowed_roots": [str(r) for r in self.allowed_roots],
            "fallback": container is False and self.mode != SandboxMode.OFF,
        }
