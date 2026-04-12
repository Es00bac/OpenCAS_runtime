"""Docker sandbox for shell command isolation."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Isolate shell commands inside a Docker container.

    Maps host *allowed_roots* to ``/workspace`` inside the container
    and routes ``bash_run_command`` through ``docker exec``.
    """

    def __init__(
        self,
        allowed_roots: List[Path],
        image: str = "python:3.11-slim",
        container_name: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.allowed_roots = [Path(r).resolve() for r in allowed_roots]
        self.image = image
        self.container_name = container_name or "opencas-sandbox"
        self.timeout = timeout
        self._available: Optional[bool] = None

    def check_available(self) -> bool:
        """Return True if the Docker CLI is reachable."""
        if self._available is not None:
            return self._available
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self._available = result.returncode == 0
        except Exception as exc:
            logger.warning("Docker availability check failed: %s", exc)
            self._available = False
        return bool(self._available)

    def _ensure_running(self) -> bool:
        """Ensure the sandbox container exists and is running."""
        if not self.check_available():
            return False
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip() == "true":
                return True
        except Exception:
            pass

        # Stop and remove any stale container
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
            timeout=10,
            check=False,
        )

        if not self.allowed_roots:
            logger.error("No allowed_roots configured for DockerSandbox")
            return False

        mounts = []
        for root in self.allowed_roots:
            mounts.extend(["-v", f"{root}:/workspace:rw"])

        cmd = [
            "docker", "run", "-d", "--name", self.container_name,
            "--workdir", "/workspace",
            *mounts,
            self.image,
            "sleep", "infinity",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            if result.returncode == 0:
                return True
            logger.error("Failed to start sandbox container: %s", result.stderr)
        except Exception as exc:
            logger.error("Exception starting sandbox container: %s", exc)
        return False

    def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run *command* inside the sandbox container via ``docker exec``."""
        if not self._ensure_running():
            return {
                "ok": False,
                "error": "Docker sandbox unavailable or container failed to start",
            }

        exec_cmd = ["docker", "exec"]
        if cwd:
            exec_cmd.extend(["-w", cwd])
        if env:
            for key, value in env.items():
                exec_cmd.extend(["-e", f"{key}={value}"])
        exec_cmd.extend([self.container_name, "sh", "-c", command])

        try:
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            return {
                "ok": result.returncode == 0,
                "code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"Command timed out after {self.timeout}s",
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
            }
