"""LLM adapter for OpenCAS using open_llm_auth's ProviderManager.

This module wraps the multi-provider gateway so OpenCAS can route all LLM calls
through a single auth-aware layer.
"""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from open_llm_auth.auth.manager import ProviderManager, ResolvedProvider
from open_llm_auth.provider_catalog import BUILTIN_MODELS

from opencas.model_routing import (
    ComplexityTier,
    ModelRoutingConfig,
    ReasoningEffort,
    next_complexity_tier,
    normalize_complexity_tier,
)
from opencas.telemetry import EventKind, TokenTelemetry, Tracer


class LLMClient:
    """Async client for LLM operations backed by open_llm_auth."""

    def __init__(
        self,
        provider_manager: ProviderManager,
        default_model: Optional[str] = None,
        model_routing: Optional[ModelRoutingConfig] = None,
        tracer: Optional[Tracer] = None,
        token_telemetry: Optional[TokenTelemetry] = None,
    ) -> None:
        self.manager = provider_manager
        self.default_model = default_model or "anthropic/claude-sonnet-4-6"
        self.model_routing = (model_routing or ModelRoutingConfig()).normalized(
            self.default_model
        )
        self.tracer = tracer
        self.token_telemetry = token_telemetry
        self._last_lane_meta: Dict[str, Any] = {}

    def set_model_routing(
        self,
        *,
        default_model: Optional[str] = None,
        model_routing: Optional[ModelRoutingConfig] = None,
    ) -> None:
        """Update the active routing policy without rebuilding the client."""
        if default_model:
            self.default_model = default_model
        self.model_routing = (model_routing or self.model_routing).normalized(
            self.default_model
        )

    def resolve_model_for_complexity(
        self,
        *,
        model: Optional[str] = None,
        complexity: ComplexityTier | str | None = None,
    ) -> str:
        """Resolve the model reference used for a specific reasoning tier."""
        if model:
            return model
        tier = normalize_complexity_tier(complexity)
        resolved = self.model_routing.resolve_model(
            default_model=self.default_model,
            complexity=tier,
        )
        return resolved or self.default_model

    def resolve_reasoning_effort_for_complexity(
        self,
        *,
        complexity: ComplexityTier | str | None = None,
    ) -> Optional[str]:
        """Resolve the reasoning-effort override used for a specific tier."""
        tier = normalize_complexity_tier(complexity)
        resolved = self.model_routing.resolve_reasoning_effort(complexity=tier)
        if isinstance(resolved, ReasoningEffort):
            return resolved.value
        return str(resolved).strip() or None if resolved else None

    def provider_supports_reasoning_effort(
        self,
        *,
        model: Optional[str] = None,
    ) -> bool:
        """Return whether the resolved provider honors reasoning-effort hints."""
        model_ref = model or self.default_model
        if not model_ref:
            return False
        try:
            resolved = self._resolve(model_ref)
        except Exception:
            return False
        provider = getattr(resolved, "provider", None)
        checker = getattr(provider, "supports_reasoning_effort", None)
        if callable(checker):
            try:
                return bool(checker(model=resolved.model_id))
            except TypeError:
                return bool(checker())
            except Exception:
                return False
        return False

    def current_lane_meta(self) -> Dict[str, Any]:
        """Return the most recent resolved lane metadata for runtime surfaces."""
        return dict(self._last_lane_meta)

    def _record_lane_meta(
        self,
        *,
        requested_model: str,
        resolved: ResolvedProvider,
        complexity: ComplexityTier,
        reasoning_effort: Optional[str],
    ) -> None:
        provider = getattr(resolved, "provider", None)
        checker = getattr(provider, "supports_reasoning_effort", None)
        supports_reasoning = False
        if callable(checker):
            try:
                supports_reasoning = bool(checker(model=resolved.model_id))
            except TypeError:
                supports_reasoning = bool(checker())
            except Exception:
                supports_reasoning = False
        lane_meta: Dict[str, Any] = {
            "model": requested_model,
            "provider": resolved.provider_id,
            "resolved_model": f"{resolved.provider_id}/{resolved.model_id}",
            "profile_id": resolved.profile_id,
            "auth_source": resolved.auth_source,
            "complexity": complexity.value,
        }
        if supports_reasoning:
            lane_meta["reasoning_supported"] = True
            lane_meta["reasoning_effort"] = (
                str(reasoning_effort).strip() if reasoning_effort else None
            )
        self._last_lane_meta = {
            key: value for key, value in lane_meta.items() if value is not None
        }

    @staticmethod
    def escalate_complexity(
        complexity: ComplexityTier | str | None,
    ) -> ComplexityTier:
        """Move one step up the reasoning ladder."""
        return next_complexity_tier(complexity)

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        complexity: ComplexityTier | str | None = None,
        payload: Optional[Dict[str, Any]] = None,
        source: str = "chat",
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        execution_mode: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request and return the response object."""
        tier = normalize_complexity_tier(complexity)
        model_ref = self.resolve_model_for_complexity(
            model=model,
            complexity=tier,
        )
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"LLM chat_completion: {model_ref}",
                {
                    "model": model_ref,
                    "message_count": len(messages),
                    "complexity": tier.value,
                    "routing_mode": self.model_routing.mode.value,
                },
            )
        resolved = self._resolve(model_ref)
        started = time.perf_counter()
        merged_payload = dict(payload or {})
        if not merged_payload.get("reasoning_effort"):
            reasoning_effort = self.resolve_reasoning_effort_for_complexity(
                complexity=tier,
            )
            if reasoning_effort:
                merged_payload["reasoning_effort"] = reasoning_effort
        reasoning_effort = (
            str(merged_payload.get("reasoning_effort")).strip()
            if merged_payload.get("reasoning_effort")
            else None
        )
        self._record_lane_meta(
            requested_model=model_ref,
            resolved=resolved,
            complexity=tier,
            reasoning_effort=reasoning_effort,
        )
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
        complexity: ComplexityTier | str | None = None,
        payload: Optional[Dict[str, Any]] = None,
        source: str = "chat_stream",
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        execution_mode: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion as server-sent event strings."""
        tier = normalize_complexity_tier(complexity)
        model_ref = self.resolve_model_for_complexity(
            model=model,
            complexity=tier,
        )
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"LLM chat_completion_stream: {model_ref}",
                {
                    "model": model_ref,
                    "message_count": len(messages),
                    "complexity": tier.value,
                    "routing_mode": self.model_routing.mode.value,
                },
            )
        resolved = self._resolve(model_ref)
        merged_payload = dict(payload or {})
        if not merged_payload.get("reasoning_effort"):
            reasoning_effort = self.resolve_reasoning_effort_for_complexity(
                complexity=tier,
            )
            if reasoning_effort:
                merged_payload["reasoning_effort"] = reasoning_effort
        reasoning_effort = (
            str(merged_payload.get("reasoning_effort")).strip()
            if merged_payload.get("reasoning_effort")
            else None
        )
        self._record_lane_meta(
            requested_model=model_ref,
            resolved=resolved,
            complexity=tier,
            reasoning_effort=reasoning_effort,
        )
        stream = await resolved.provider.chat_completion_stream(
            model=resolved.model_id,
            messages=messages,
            payload=merged_payload,
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
