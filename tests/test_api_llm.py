"""Tests for the LLM adapter using open_llm_auth."""

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock

from opencas.api.llm import LLMClient
from opencas.telemetry import EventKind, TelemetryStore, Tracer


@pytest.fixture
def mock_provider_manager():
    mgr = MagicMock()
    resolved = MagicMock()
    resolved.provider_id = "test-provider"
    resolved.model_id = "test-model"
    resolved.provider = MagicMock()
    resolved.provider.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "hi"}}]})
    async def _fake_stream():
        for chunk in ["data: chunk1", "data: chunk2"]:
            yield chunk
    resolved.provider.chat_completion_stream = AsyncMock(return_value=_fake_stream())
    resolved.provider.embeddings = AsyncMock(return_value={
        "data": [{"embedding": [0.1, 0.2, 0.3]}]
    })
    mgr.resolve.return_value = resolved
    return mgr


@pytest.fixture
def tracer(tmp_path):
    store = TelemetryStore(tmp_path / "telemetry")
    return Tracer(store)


def test_llm_client_list_models() -> None:
    mgr = MagicMock()
    client = LLMClient(mgr)
    models = client.list_available_models()
    assert len(models) > 0
    assert all("/" in m for m in models)


@pytest.mark.asyncio
async def test_llm_chat_completion(mock_provider_manager: MagicMock, tracer: Tracer) -> None:
    client = LLMClient(mock_provider_manager, default_model="test/model", tracer=tracer)
    response = await client.chat_completion(messages=[{"role": "user", "content": "hello"}])
    assert response["choices"][0]["message"]["content"] == "hi"
    mock_provider_manager.resolve.assert_called_once_with("test/model")

    events = tracer.store.query(kinds=[EventKind.TOOL_CALL])
    assert len(events) == 1
    assert "LLM chat_completion" in events[0].message


@pytest.mark.asyncio
async def test_llm_chat_completion_stream(mock_provider_manager: MagicMock, tracer: Tracer) -> None:
    client = LLMClient(mock_provider_manager, default_model="test/model", tracer=tracer)
    chunks = []
    async for chunk in client.chat_completion_stream(messages=[{"role": "user", "content": "hello"}]):
        chunks.append(chunk)
    assert chunks == ["data: chunk1", "data: chunk2"]


@pytest.mark.asyncio
async def test_llm_embed(mock_provider_manager: MagicMock, tracer: Tracer) -> None:
    client = LLMClient(mock_provider_manager, default_model="test/model", tracer=tracer)
    vector = await client.embed("hello world")
    assert vector == [0.1, 0.2, 0.3]
    mock_provider_manager.resolve.assert_called_with("openai/text-embedding-3-small")


@pytest.mark.asyncio
async def test_llm_embed_unsupported_provider(mock_provider_manager: MagicMock) -> None:
    del mock_provider_manager.resolve.return_value.provider.embeddings
    client = LLMClient(mock_provider_manager)
    with pytest.raises(RuntimeError):
        await client.embed("hello world")


@pytest_asyncio.fixture
async def token_telemetry(tmp_path):
    from opencas.telemetry import TokenTelemetry
    return TokenTelemetry(tmp_path / "token_telemetry", buffer_flush_size=1)


@pytest.mark.asyncio
async def test_llm_resolve_failure() -> None:
    mgr = MagicMock()
    mgr.resolve.return_value = None
    client = LLMClient(mgr)
    with pytest.raises(ValueError):
        await client.chat_completion(messages=[])


@pytest.mark.asyncio
async def test_llm_chat_completion_records_token_telemetry(
    mock_provider_manager: MagicMock, token_telemetry, tracer: Tracer
) -> None:
    mock_provider_manager.resolve.return_value.provider.chat_completion = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
    )
    client = LLMClient(
        mock_provider_manager,
        default_model="test/model",
        tracer=tracer,
        token_telemetry=token_telemetry,
    )
    response = await client.chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        session_id="session-1",
        task_id="task-1",
    )
    assert response["choices"][0]["message"]["content"] == "hi"
    summary = token_telemetry.get_session_summary("session-1")
    assert summary.total_calls == 1
    assert summary.total_tokens == 8


@pytest.mark.asyncio
async def test_llm_embed_records_token_telemetry(
    mock_provider_manager: MagicMock, token_telemetry, tracer: Tracer
) -> None:
    client = LLMClient(
        mock_provider_manager,
        default_model="test/model",
        tracer=tracer,
        token_telemetry=token_telemetry,
    )
    vector = await client.embed("hello world", session_id="session-2")
    assert vector == [0.1, 0.2, 0.3]
    summary = token_telemetry.get_session_summary("session-2")
    assert summary.total_calls == 1


@pytest.mark.asyncio
async def test_llm_stream_records_token_telemetry(
    mock_provider_manager: MagicMock, token_telemetry, tracer: Tracer
) -> None:
    client = LLMClient(
        mock_provider_manager,
        default_model="test/model",
        tracer=tracer,
        token_telemetry=token_telemetry,
    )
    chunks = []
    async for chunk in client.chat_completion_stream(
        messages=[{"role": "user", "content": "hello"}],
        session_id="session-3",
    ):
        chunks.append(chunk)
    assert chunks == ["data: chunk1", "data: chunk2"]
    summary = token_telemetry.get_session_summary("session-3")
    assert summary.total_calls == 1
    assert summary.avg_latency_ms >= 0
