"""Local Qdrant startup helpers for OpenCAS."""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional, Sequence
from urllib.parse import urlparse

import httpx


LOCAL_QDRANT_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class QdrantStartupResult:
    status: str
    message: str
    url: str
    command: Optional[Sequence[str]] = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


RunCommand = Callable[[Sequence[str]], Awaitable[subprocess.CompletedProcess[str]]]
Probe = Callable[[str], Awaitable[bool]]
Sleep = Callable[[float], Awaitable[None]]


def is_local_qdrant_url(url: str) -> bool:
    return (urlparse(url).hostname or "").lower() in LOCAL_QDRANT_HOSTS


async def probe_qdrant_collections(url: str, timeout: float = 2.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url.rstrip('/')}/collections")
            response.raise_for_status()
        return True
    except Exception:
        return False


async def run_command(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return subprocess.CompletedProcess(
        args=list(cmd),
        returncode=process.returncode,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


async def ensure_local_qdrant(
    url: str,
    *,
    state_dir: Path | str,
    container_name: str = "opencas_qdrant",
    fallback_container_names: Sequence[str] = ("openbulma_v4_qdrant",),
    image: str = "qdrant/qdrant:v1.17.0",
    timeout_seconds: float = 30.0,
    probe: Probe = probe_qdrant_collections,
    run: RunCommand = run_command,
    sleep: Sleep = asyncio.sleep,
) -> QdrantStartupResult:
    """Ensure a localhost Qdrant endpoint is reachable, starting Docker if needed."""
    if await probe(url):
        return QdrantStartupResult(
            status="ready",
            message="Qdrant already reachable",
            url=url,
        )
    if not is_local_qdrant_url(url):
        return QdrantStartupResult(
            status="skipped",
            message="Qdrant URL is not local; startup management skipped",
            url=url,
        )

    for name in (container_name, *fallback_container_names):
        inspect = await run(
            ["docker", "inspect", "--format", "{{.State.Running}}", name]
        )
        if inspect.returncode != 0:
            continue
        command: Sequence[str]
        if inspect.stdout.strip().lower() == "true":
            command = ["docker", "inspect", name]
        else:
            command = ["docker", "start", name]
            started = await run(command)
            if started.returncode != 0:
                return QdrantStartupResult(
                    status="failed",
                    message=f"Failed to start Qdrant container {name}: {started.stderr.strip()}",
                    url=url,
                    command=command,
                )
        if await _wait_until_ready(url, timeout_seconds, probe, sleep):
            return QdrantStartupResult(
                status="started",
                message=f"Qdrant container {name} is reachable",
                url=url,
                command=command,
            )

    command = _docker_run_command(
        url=url,
        state_dir=Path(state_dir),
        container_name=container_name,
        image=image,
    )
    created = await run(command)
    if created.returncode != 0:
        return QdrantStartupResult(
            status="failed",
            message=f"Failed to create Qdrant container {container_name}: {created.stderr.strip()}",
            url=url,
            command=command,
        )
    if await _wait_until_ready(url, timeout_seconds, probe, sleep):
        return QdrantStartupResult(
            status="started",
            message=f"Created Qdrant container {container_name}",
            url=url,
            command=command,
        )
    return QdrantStartupResult(
        status="failed",
        message="Qdrant did not become reachable before timeout",
        url=url,
        command=command,
    )


def _docker_run_command(
    *,
    url: str,
    state_dir: Path,
    container_name: str,
    image: str,
) -> list[str]:
    parsed = urlparse(url)
    port = parsed.port or 6333
    grpc_port = port + 1
    storage = (state_dir / "qdrant_storage").expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    return [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--restart",
        "unless-stopped",
        "-p",
        f"127.0.0.1:{port}:6333",
        "-p",
        f"127.0.0.1:{grpc_port}:6334",
        "-v",
        f"{storage}:/qdrant/storage",
        image,
    ]


async def _wait_until_ready(
    url: str,
    timeout_seconds: float,
    probe: Probe,
    sleep: Sleep,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() <= deadline:
        if await probe(url):
            return True
        await sleep(0.5)
    return False
