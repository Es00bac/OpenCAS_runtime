from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.embeddings.qdrant_startup import ensure_local_qdrant


def _completed(stdout: str = "", returncode: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


@pytest.mark.asyncio
async def test_ensure_local_qdrant_skips_non_local_urls(tmp_path) -> None:
    async def probe(_url: str) -> bool:
        return False

    async def run(_cmd):
        raise AssertionError("non-local Qdrant URLs must not start local Docker")

    result = await ensure_local_qdrant(
        "http://qdrant.example.test:6333",
        state_dir=tmp_path,
        probe=probe,
        run=run,
    )

    assert result.status == "skipped"
    assert "not local" in result.message


@pytest.mark.asyncio
async def test_ensure_local_qdrant_starts_existing_container(tmp_path) -> None:
    probes = iter([False, True])
    commands: list[list[str]] = []

    async def probe(_url: str) -> bool:
        return next(probes)

    async def run(cmd):
        commands.append(list(cmd))
        if cmd[:2] == ["docker", "inspect"]:
            return _completed(stdout="false\n")
        if cmd[:2] == ["docker", "start"]:
            return _completed(stdout="opencas_qdrant\n")
        raise AssertionError(f"unexpected command: {cmd}")

    async def sleep(_seconds: float) -> None:
        return None

    result = await ensure_local_qdrant(
        "http://127.0.0.1:6333",
        state_dir=tmp_path,
        container_name="opencas_qdrant",
        fallback_container_names=(),
        probe=probe,
        run=run,
        sleep=sleep,
        timeout_seconds=1,
    )

    assert result.status == "started"
    assert ["docker", "start", "opencas_qdrant"] in commands


@pytest.mark.asyncio
async def test_ensure_local_qdrant_creates_container_when_absent(tmp_path) -> None:
    probes = iter([False, True])
    commands: list[list[str]] = []

    async def probe(_url: str) -> bool:
        return next(probes)

    async def run(cmd):
        commands.append(list(cmd))
        if cmd[:2] == ["docker", "inspect"]:
            return _completed(returncode=1)
        if cmd[:3] == ["docker", "run", "-d"]:
            return _completed(stdout="new-container\n")
        raise AssertionError(f"unexpected command: {cmd}")

    async def sleep(_seconds: float) -> None:
        return None

    result = await ensure_local_qdrant(
        "http://127.0.0.1:6333",
        state_dir=tmp_path,
        container_name="opencas_qdrant",
        fallback_container_names=(),
        probe=probe,
        run=run,
        sleep=sleep,
        timeout_seconds=1,
        image="qdrant/qdrant:v1.17.0",
    )

    assert result.status == "started"
    run_command = next(cmd for cmd in commands if cmd[:3] == ["docker", "run", "-d"])
    assert "--name" in run_command
    assert "opencas_qdrant" in run_command
    assert "127.0.0.1:6333:6333" in run_command
    assert "qdrant/qdrant:v1.17.0" in run_command
