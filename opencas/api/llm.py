"""LLM adapter for OpenCAS using open_llm_auth's ProviderManager.

This module wraps the multi-provider gateway so OpenCAS can route all LLM calls
through a single auth-aware layer.
"""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from open_llm_auth.auth.manager import ProviderManager, ResolvedProvider
from open_llm_auth.provider_catalog import BUILTIN_MODELS

from opencas.telemetry import EventKind, TokenTelemetry, Tracer


class LLMClient:
    """Async client for LLM operations backed by open_llm_auth."""

    def __init__(
        self,
        provider_manager: ProviderManager,
        default_model: Optional[str] = None,
        tracer: Optional[Tracer] = None,
        token_telemetry: Optional[TokenTelemetry] = None,
    ) -> None:
        self.manager = provider_manager
        self.default_model = default_model or "anthropic/claude-sonnet-4-6"
        self.tracer = tracer
        self.token_telemetry = token_telemetry

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        source: str = "chat",
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        execution_mode: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request and return the response object."""
        model_ref = model or self.default_model
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"LLM chat_completion: {model_ref}",
                {"model": model_ref, "message_count": len(messages)},
            )
        resolved = self._resolve(model_ref)
        started = time.perf_counter()
        merged_payload = dict(payload or {})
        if tools is not None:
            merged_payload["tools"] = tools
        if tool_choice is not None:
            merged_payload["tool_choice"] = tool_choice
        response = await resolved.provider.chat_completion(
            model=resolved.model_id,
            messages=messages,
            payload=merged_payload,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        if self.token_telemetry:
            usage = response.get("usage") or {}
            await self.token_telemetry.record(
                model=model_ref,
                provider=resolved.provider_id,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                latency_ms=latency_ms,
                source=source,
                session_id=session_id,
                task_id=task_id,
                execution_mode=execution_mode,
            )
        return response

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        source: str = "chat_stream",
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        execution_mode: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion as server-sent event strings."""
        model_ref = model or self.default_model
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"LLM chat_completion_stream: {model_ref}",
                {"model": model_ref, "message_count": len(messages)},
            )
        resolved = self._resolve(model_ref)
        stream = await resolved.provider.chat_completion_stream(
            model=resolved.model_id,
            messages=messages,
            payload=payload or {},
        )
        if not self.token_telemetry:
            async for chunk in stream:
                yield chunk
            return

        started = time.perf_counter()

        async def _wrapped() -> AsyncGenerator[str, None]:
            async for chunk in stream:
                yield chunk
            latency_ms = int((time.perf_counter() - started) * 1000)
            await self.token_telemetry.record(  # type: ignore[union-attr]
                model=model_ref,
                provider=resolved.provider_id,
                latency_ms=latency_ms,
                source=source,
                session_id=session_id,
                task_id=task_id,
                execution_mode=execution_mode,
            )

        async for chunk in _wrapped():
            yield chunk

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
        source: str = "embed",
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        execution_mode: Optional[str] = None,
    ) -> List[float]:
        """Compute an embedding vector for *text* via the LLM gateway.

        Falls back to the local hashing embedder if no gateway embedding
        provider is resolved.
        """
        model_ref = model or "openai/text-embedding-3-small"
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"LLM embed: {model_ref}",
                {"model": model_ref, "text_length": len(text)},
            )
        resolved = self._resolve(model_ref)
        started = time.perf_counter()
        if hasattr(resolved.provider, "embeddings"):
            result = await resolved.provider.embeddings(
                model=resolved.model_id,
                input_texts=[text],
                payload={},
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            if self.token_telemetry:
                await self.token_telemetry.record(
                    model=model_ref,
                    provider=resolved.provider_id,
                    latency_ms=latency_ms,
                    source=source,
                    session_id=session_id,
                    task_id=task_id,
                    execution_mode=execution_mode,
                )
            return result["data"][0]["embedding"]
        raise RuntimeError(
            f"Provider {resolved.provider_id} does not support embeddings"
        )

    def list_available_models(self) -> List[str]:
        """Return a flat list of available model references."""
        # Start from builtin catalog model IDs
        models: List[str] = []
        for provider_id, model_list in BUILTIN_MODELS.items():
            for model in model_list:
                models.append(f"{provider_id}/{model['id']}")
        return sorted(set(models))

    def _resolve(self, model_ref: str) -> ResolvedProvider:
        """Resolve a model reference to a concrete provider instance."""
        resolved = self.manager.resolve(model_ref)
        if resolved is None:
            raise ValueError(f"Could not resolve model: {model_ref}")
        return resolved
