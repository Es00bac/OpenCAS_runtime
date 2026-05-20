"""Tests for the LLM adapter using open_llm_auth."""

from unittest.mock import AsyncMock, MagicMock

import httpx

import pytest
import pytest_asyncio

from opencas.api.llm import LLMClient
from opencas.api.provider_circuit_breaker import ProviderCircuitBreaker, ProviderCircuitOpen
from opencas.generation.policy import (
    GenerationDomain,
    GenerationPhase,
    GenerationPolicyRequest,
)
from opencas.model_routing import ModelRoutingConfig, ModelRoutingMode, ReasoningEffort
from opencas.telemetry import EventKind, TelemetryStore, Tracer


@pytest.fixture
def mock_provider_manager():
    mgr = MagicMock()
    resolved = MagicMock()
    resolved.provider_id = "test-provider"
    resolved.model_id = "test-model"
    resolved.provider = MagicMock()
    resolved.profile_id = "test-profile"
    resolved.auth_source = "test-auth"
    resolved.provider.supports_reasoning_effort = MagicMock(return_value=True)
    resolved.provider.chat_completion = AsyncMock(return_value={"choices": [{"message": {"content": "hi"}}]})
    async def _fake_stream():
        for chunk in ["data: chunk1", "data: chunk2"]:
            yield chunk
    resolved.provider.chat_completion_stream = AsyncMock(return_value=_fake_stream())
    resolved.provider.embeddings = AsyncMock(return_value={
        "data": [{"embedding": [0.1, 0.2, 0.3]}]
    })
    mgr.resolve.return_value = resolved
    mgr.default_embedding_model_ref.return_value = "gateway/default-embedding"
    return mgr


@pytest.fixture
def tracer(tmp_path):
    store = TelemetryStore(tmp_path / "telemetry")
    return Tracer(store)


def test_llm_client_list_models() -> None:
    mgr = MagicMock()
    mgr.available_model_refs.return_value = ["openai/gpt-5.5", "codex-cli/gpt-5.5", "kimi-coding/k2p6"]
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

    events = tracer.store.query(kinds=[EventKind.LLM_CALL])
    assert any("LLM chat_completion" in event.message for event in events)


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
    mock_provider_manager.resolve.assert_called_with("gateway/default-embedding")


@pytest.mark.asyncio
async def test_llm_embed_passes_requested_dimensions(
    mock_provider_manager: MagicMock,
) -> None:
    client = LLMClient(mock_provider_manager)

    await client.embed("hello world", model="google/embeddinggemma-300m", dimensions=3072)

    mock_provider_manager.resolve.return_value.provider.embeddings.assert_awaited_once()
    _, kwargs = mock_provider_manager.resolve.return_value.provider.embeddings.await_args
    assert kwargs["payload"]["dimensions"] == 3072


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
    llm_events = tracer.store.query(kinds=[EventKind.LLM_CALL])
    assert any(
        event.payload.get("session_id") == "session-1"
        and event.payload.get("task_id") == "task-1"
        for event in llm_events
    )


@pytest.mark.asyncio
async def test_llm_chat_completion_adds_openai_prompt_cache_hints(
    mock_provider_manager: MagicMock, token_telemetry, tracer: Tracer
) -> None:
    resolved = mock_provider_manager.resolve.return_value
    resolved.provider_id = "openai"
    resolved.provider.chat_completion = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "cached"}}],
            "usage": {
                "prompt_tokens": 1200,
                "completion_tokens": 10,
                "total_tokens": 1210,
                "prompt_tokens_details": {"cached_tokens": 512},
            },
        }
    )
    client = LLMClient(
        mock_provider_manager,
        default_model="openai/gpt-5.5",
        tracer=tracer,
        token_telemetry=token_telemetry,
    )

    await client.chat_completion(
        messages=[{"role": "user", "content": "What do you recall?"}],
        complexity="light",
        source="conversation_prefetched_recall",
        session_id="session-cache",
    )

    payload = resolved.provider.chat_completion.await_args.kwargs["payload"]
    assert payload["prompt_cache_key"].startswith(
        "opencas:conversation_prefetched_recall:light:"
    )
    assert payload["prompt_cache_retention"] == "in_memory"
    events = token_telemetry.get_session_events("session-cache")
    assert events[0].cached_prompt_tokens == 512
    complete_events = [
        event
        for event in tracer.store.query(kinds=[EventKind.LLM_CALL])
        if event.message.startswith("LLM chat_completion complete")
    ]
    assert any(event.payload.get("cached_prompt_tokens") == 512 for event in complete_events)


@pytest.mark.asyncio
async def test_llm_chat_completion_records_codex_automatic_cache_without_unsupported_hints(
    mock_provider_manager: MagicMock, token_telemetry, tracer: Tracer
) -> None:
    resolved = mock_provider_manager.resolve.return_value
    resolved.provider_id = "openai-codex"
    resolved.provider.chat_completion = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "cached"}}],
            "usage": {
                "prompt_tokens": 1600,
                "completion_tokens": 20,
                "total_tokens": 1620,
                "prompt_tokens_details": {"cached_tokens": 384},
            },
        }
    )
    client = LLMClient(
        mock_provider_manager,
        default_model="openai/gpt-5.5",
        tracer=tracer,
        token_telemetry=token_telemetry,
    )

    await client.chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        complexity="light",
        source="conversation_prefetched_recall",
        session_id="session-cache",
    )

    payload = resolved.provider.chat_completion.await_args.kwargs["payload"]
    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload
    events = token_telemetry.get_session_events("session-cache")
    assert events[0].cached_prompt_tokens == 384
    llm_events = tracer.store.query(kinds=[EventKind.LLM_CALL])
    assert any(
        event.message.startswith("LLM chat_completion:")
        and event.payload.get("prompt_cache_mode") == "provider_automatic"
        and event.payload.get("prompt_cache_hints_supported") is False
        for event in llm_events
    )
    assert any(
        event.message.startswith("LLM chat_completion complete")
        and event.payload.get("prompt_cache_mode") == "provider_automatic"
        and event.payload.get("cached_prompt_tokens") == 384
        for event in llm_events
    )


@pytest.mark.asyncio
async def test_llm_chat_completion_applies_generation_policy(
    mock_provider_manager: MagicMock,
    tracer: Tracer,
) -> None:
    resolved = mock_provider_manager.resolve.return_value
    resolved.provider_id = "openai-codex"
    client = LLMClient(mock_provider_manager, default_model="openai/gpt-5.5", tracer=tracer)

    await client.chat_completion(
        messages=[{"role": "user", "content": "Brainstorm three opening scenes."}],
        source="creative_brainstorm_test",
        generation_request=GenerationPolicyRequest(
            phase=GenerationPhase.BRAINSTORM,
            domain=GenerationDomain.CREATIVE_WRITING,
            novelty_pressure=0.7,
        ),
    )

    payload = resolved.provider.chat_completion.await_args.kwargs["payload"]
    assert payload["temperature"] >= 0.85
    assert "top_p" not in payload

    llm_events = tracer.store.query(kinds=[EventKind.LLM_CALL])
    assert any(
        event.message.startswith("LLM chat_completion:")
        and event.payload.get("generation_phase") == "brainstorm"
        and event.payload.get("generation_profile") == "brainstorm"
        for event in llm_events
    )


@pytest.mark.asyncio
async def test_llm_chat_completion_preserves_explicit_sampling_override(
    mock_provider_manager: MagicMock,
    tracer: Tracer,
) -> None:
    resolved = mock_provider_manager.resolve.return_value
    resolved.provider_id = "openai-codex"
    client = LLMClient(mock_provider_manager, default_model="openai/gpt-5.5", tracer=tracer)

    await client.chat_completion(
        messages=[{"role": "user", "content": "Check this JSON."}],
        payload={"temperature": 0},
        source="verify_override_test",
        generation_request=GenerationPolicyRequest(
            phase=GenerationPhase.BRAINSTORM,
            domain=GenerationDomain.CREATIVE_WRITING,
            novelty_pressure=1.0,
        ),
    )

    payload = resolved.provider.chat_completion.await_args.kwargs["payload"]
    assert payload["temperature"] == 0
    assert "top_p" not in payload

    llm_events = tracer.store.query(kinds=[EventKind.LLM_CALL])
    assert any(
        event.payload.get("generation_explicit_overrides") == ["temperature"]
        for event in llm_events
    )


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


@pytest.mark.asyncio
async def test_llm_chat_completion_uses_tiered_routing(
    mock_provider_manager: MagicMock,
) -> None:
    client = LLMClient(
        mock_provider_manager,
        default_model="anthropic/claude-sonnet-4-6",
        model_routing=ModelRoutingConfig(
            mode="tiered",
            light_model="google/gemini-2.5-flash",
            standard_model="anthropic/claude-sonnet-4-6",
            high_model="openai/gpt-5.3-codex",
            extra_high_model="codex-cli/gpt-5.3-codex",
            high_reasoning_effort=ReasoningEffort.HIGH,
        ),
    )
    await client.chat_completion(
        messages=[{"role": "user", "content": "hello"}],
        complexity="high",
    )
    mock_provider_manager.resolve.assert_called_once_with("openai/gpt-5.3-codex")
    payload = mock_provider_manager.resolve.return_value.provider.chat_completion.await_args.kwargs["payload"]
    assert payload["reasoning_effort"] == "high"
    lane = client.current_lane_meta()
    assert lane["resolved_model"] == "test-provider/test-model"
    assert lane["reasoning_effort"] == "high"
    assert lane["reasoning_supported"] is True
    assert lane["complexity"] == "high"


def test_llm_resolve_model_for_single_mode_uses_same_model_for_every_tier() -> None:
    mgr = MagicMock()
    client = LLMClient(
        mgr,
        default_model="anthropic/claude-sonnet-4-6",
        model_routing=ModelRoutingConfig(
            mode="single",
            single_model="codex-cli/gpt-5.3-codex",
        ),
    )

    assert client.resolve_model_for_complexity(complexity="light") == "codex-cli/gpt-5.3-codex"
    assert client.resolve_model_for_complexity(complexity="standard") == "codex-cli/gpt-5.3-codex"
    assert client.resolve_model_for_complexity(complexity="high") == "codex-cli/gpt-5.3-codex"
    assert client.resolve_model_for_complexity(complexity="extra_high") == "codex-cli/gpt-5.3-codex"


def test_llm_resolve_reasoning_effort_for_tiers() -> None:
    mgr = MagicMock()
    client = LLMClient(
        mgr,
        default_model="codex-cli/gpt-5.4",
        model_routing=ModelRoutingConfig(
            mode="tiered",
            light_model="codex-cli/gpt-5.4",
            standard_model="codex-cli/gpt-5.4",
            high_model="codex-cli/gpt-5.4",
            extra_high_model="codex-cli/gpt-5.4",
            light_reasoning_effort=ReasoningEffort.LOW,
            standard_reasoning_effort=ReasoningEffort.MEDIUM,
            high_reasoning_effort=ReasoningEffort.HIGH,
            extra_high_reasoning_effort=ReasoningEffort.EXTRA_HIGH,
        ),
    )

    assert client.resolve_reasoning_effort_for_complexity(complexity="light") == "low"
    assert client.resolve_reasoning_effort_for_complexity(complexity="standard") == "medium"
    assert client.resolve_reasoning_effort_for_complexity(complexity="high") == "high"
    assert client.resolve_reasoning_effort_for_complexity(complexity="extra_high") == "xhigh"


def test_llm_single_route_keeps_light_reasoning_low_by_default() -> None:
    mgr = MagicMock()
    client = LLMClient(
        mgr,
        default_model="codex-cli/gpt-5.5",
        model_routing=ModelRoutingConfig(
            mode=ModelRoutingMode.SINGLE,
            single_model="codex-cli/gpt-5.5",
            single_reasoning_effort=ReasoningEffort.HIGH,
        ),
    )

    assert client.resolve_reasoning_effort_for_complexity(complexity="light") == "low"
    assert client.resolve_reasoning_effort_for_complexity(complexity="standard") == "high"


def test_llm_provider_supports_reasoning_effort_uses_resolved_provider() -> None:
    mgr = MagicMock()
    resolved = MagicMock()
    resolved.model_id = "gpt-5.4"
    resolved.provider = MagicMock()
    resolved.provider.supports_reasoning_effort.return_value = True
    mgr.resolve.return_value = resolved
    client = LLMClient(mgr, default_model="codex-cli/gpt-5.4")

    assert client.provider_supports_reasoning_effort() is True
    resolved.provider.supports_reasoning_effort.assert_called_once_with(model="gpt-5.4")


def test_llm_reports_builtin_context_window_and_prompt_budget() -> None:
    mgr = MagicMock()
    resolved = MagicMock()
    resolved.provider_id = "kimi-coding"
    resolved.model_id = "k2p5"
    mgr.resolve.return_value = resolved
    mgr.model_definition.return_value = {
        "id": "k2p5",
        "contextWindow": 262144,
        "maxTokens": 32768,
    }
    client = LLMClient(mgr, default_model="kimi-coding/k2p5")

    meta = client.model_context_metadata()

    assert meta["context_window"] == 262144
    assert meta["max_output_tokens"] == 32768
    assert meta["prompt_context_budget"] > 200_000
    assert meta["resolved_model"] == "kimi-coding/k2p5"


@pytest.mark.asyncio
async def test_llm_chat_completion_opens_provider_circuit(mock_provider_manager: MagicMock) -> None:
    mock_provider_manager.resolve.return_value.provider.chat_completion = AsyncMock(
        side_effect=RuntimeError("HTTP 500 upstream")
    )
    client = LLMClient(mock_provider_manager, default_model="test/model")
    client._circuit_breaker = ProviderCircuitBreaker(failure_threshold=1, recovery_timeout=60)

    with pytest.raises(RuntimeError):
        await client.chat_completion(messages=[{"role": "user", "content": "hello"}])

    with pytest.raises(ProviderCircuitOpen) as exc:
        await client.chat_completion(messages=[{"role": "user", "content": "hello"}])

    assert exc.value.provider_name == "test-provider"
    assert mock_provider_manager.resolve.return_value.provider.chat_completion.await_count == 1


@pytest.mark.asyncio
async def test_llm_chat_completion_retries_transient_rate_limit(
    mock_provider_manager: MagicMock,
) -> None:
    request = httpx.Request("POST", "https://api.kimi.com/coding/v1/messages")
    response = httpx.Response(429, request=request, headers={"retry-after": "0"})
    transient = httpx.HTTPStatusError(
        "429 Too Many Requests",
        request=request,
        response=response,
    )
    provider = mock_provider_manager.resolve.return_value.provider
    provider.chat_completion = AsyncMock(
        side_effect=[
            transient,
            {"choices": [{"message": {"content": "Recovered."}}]},
        ]
    )
    client = LLMClient(mock_provider_manager, default_model="test/model")
    client._chat_retry_base_delay = 0.0
    client._chat_retry_max_delay = 0.0

    result = await client.chat_completion(messages=[{"role": "user", "content": "hello"}])

    assert result["choices"][0]["message"]["content"] == "Recovered."
    assert provider.chat_completion.await_count == 2
    assert await client._circuit_breaker.is_open("test-provider") is False


@pytest.mark.asyncio
async def test_llm_chat_completion_opens_circuit_after_retry_exhaustion(
    mock_provider_manager: MagicMock,
) -> None:
    provider = mock_provider_manager.resolve.return_value.provider
    provider.chat_completion = AsyncMock(side_effect=httpx.ReadError("connection closed"))
    client = LLMClient(mock_provider_manager, default_model="test/model")
    client._chat_retry_base_delay = 0.0
    client._chat_retry_max_delay = 0.0
    client._circuit_breaker = ProviderCircuitBreaker(failure_threshold=1, recovery_timeout=60)

    with pytest.raises(httpx.ReadError):
        await client.chat_completion(messages=[{"role": "user", "content": "hello"}])

    assert provider.chat_completion.await_count == client._chat_retry_attempts
    assert await client._circuit_breaker.is_open("test-provider") is True


@pytest.mark.asyncio
async def test_llm_embed_batch_respects_open_provider_circuit(mock_provider_manager: MagicMock) -> None:
    client = LLMClient(mock_provider_manager)
    client._circuit_breaker = ProviderCircuitBreaker(failure_threshold=1, recovery_timeout=60)
    await client._circuit_breaker.record_failure("test-provider")

    with pytest.raises(ProviderCircuitOpen):
        await client.embed_batch(["hello"], model="openai/text-embedding-3-small")
