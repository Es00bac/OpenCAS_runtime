"""Tests for OpenCAS FastAPI server surface."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
import pytest_asyncio

from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from opencas.api.server import create_app
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime


def _build_app(tmp_path: Path) -> TestClient:
    """Synchronous helper to bootstrap runtime and wrap in TestClient."""
    config = BootstrapConfig(state_dir=tmp_path, session_id="server-test")
    ctx = asyncio.run(BootstrapPipeline(config).run())
    runtime = AgentRuntime(ctx)
    app = create_app(runtime)
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client(tmp_path: Path):
    config = BootstrapConfig(state_dir=tmp_path, session_id="server-test")
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    app = create_app(runtime)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_health(async_client: AsyncClient) -> None:
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readiness(async_client: AsyncClient) -> None:
    response = await async_client.get("/readiness")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == "ready"


@pytest.mark.asyncio
async def test_chat_endpoint(async_client: AsyncClient) -> None:
    response = await async_client.post(
        "/chat",
        json={"session_id": "test-session", "message": "hello"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data


@pytest.mark.asyncio
async def test_telemetry_recent(async_client: AsyncClient) -> None:
    response = await async_client.get("/telemetry/recent?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_websocket_chat(tmp_path: Path) -> None:
    client = _build_app(tmp_path)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "chat", "session_id": "ws-test", "payload": "hi"})
        raw = websocket.receive_json()
        assert raw["type"] == "chat_response"
        assert raw["session_id"] == "ws-test"
        assert "payload" in raw


def test_websocket_receives_baa_events(tmp_path: Path) -> None:
    config = BootstrapConfig(state_dir=tmp_path, session_id="server-test")
    ctx = asyncio.run(BootstrapPipeline(config).run())
    runtime = AgentRuntime(ctx)
    app = create_app(runtime)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        from opencas.execution.models import RepairTask, ExecutionStage, RepairResult
        from opencas.infra import BaaCompletedEvent

        # Emit a completed event on the bus
        event = BaaCompletedEvent(
            task_id="task-123",
            success=True,
            stage=ExecutionStage.DONE.value,
            objective="test objective",
            output="done",
        )
        asyncio.run(runtime.ctx.event_bus.emit(event))

        # The bridge forwards BaaCompletedEvent to all connected websockets.
        # Drain messages until we see either the baa_completed or chat_response.
        websocket.send_json({"type": "chat", "payload": "ping", "session_id": "x"})
        seen_baa = False
        for _ in range(2):
            raw = websocket.receive_json()
            if raw["type"] == "baa_completed":
                seen_baa = True
                continue
            assert raw["type"] == "chat_response"
            break
        assert seen_baa
