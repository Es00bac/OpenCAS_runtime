"""LLM adapter for OpenCAS using open_llm_auth's ProviderManager.

This module wraps the multi-provider gateway so OpenCAS can route all LLM calls
through a single auth-aware layer.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from open_llm_auth.auth.manager import ProviderManager, ResolvedProvider

from opencas.api.provider_circuit_breaker import ProviderCircuitBreaker, ProviderCircuitOpen
from opencas.generation.policy import (
    GenerationPolicyConfig,
    GenerationPolicyRequest,
    GenerationPolicyResolver,
)
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
        generation_policy: Optional[GenerationPolicyConfig] = None,
        tracer: Optional[Tracer] = None,
        token_telemetry: Optional[TokenTelemetry] = None,
    ) -> None:
        self.manager = provider_manager
        self.default_model = default_model or getattr(
            getattr(provider_manager, "_config", None), "default_model", None
        )
        self.model_routing = (model_routing or ModelRoutingConfig()).normalized(
            self.default_model
        )
        self.generation_policy = (generation_policy or GenerationPolicyConfig()).normalized()
        self.generation_policy_resolver = GenerationPolicyResolver(self.generation_policy)
        self.tracer = tracer
        self.token_telemetry = token_telemetry
        self._last_lane_meta: Dict[str, Any] = {}
        self._circuit_breaker = ProviderCircuitBreaker()
        self._chat_retry_attempts = 3
        self._chat_retry_base_delay = 0.75
        self._chat_retry_max_delay = 8.0

    @staticmethod
    def _supports_prompt_cache_hints(provider_id: str) -> bool:
        provider = str(provider_id or "").strip().lower()
        return provider == "openai"

    @classmethod
    def _prompt_cache_trace_fields(cls, provider_id: str) -> Dict[str, Any]:
        """Describe the cache contract for telemetry without mutating payloads."""

        provider = str(provider_id or "").strip().lower()
        hints_supported = cls._supports_prompt_cache_hints(provider)
        if hints_supported:
            mode = "explicit_hints"
        elif provider == "openai-codex":
            # The OpenAI Codex Responses backend does not accept explicit cache
            # keys, but it may report provider-managed cache hits in usage.
            mode = "provider_automatic"
        else:
            mode = "none"
        return {
            "prompt_cache_hints_supported": hints_supported,
            "prompt_cache_mode": mode,
        }

    @staticmethod
    def _safe_cache_group(source: str) -> str:
        cleaned = re.sub(r"[^a-z0-9_.-]+", "-", str(source or "chat").casefold()).strip("-")
        return (cleaned or "chat")[:48]

    def _apply_prompt_cache_hints(
        self,
        payload: Dict[str, Any],
        *,
        provider_id: str,
        model_ref: str,
        complexity: ComplexityTier,
        source: str,
        session_id: Optional[str],
    ) -> None:
        """Add provider cache routing hints without changing prompt semantics."""

        if not self._supports_prompt_cache_hints(provider_id):
            return
        if not payload.get("prompt_cache_key"):
            scope = "|".join(
                [
                    "opencas",
                    str(provider_id or ""),
                    str(model_ref or ""),
                    complexity.value,
                    str(session_id or "global"),
                ]
            )
            scope_hash = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16]
            payload["prompt_cache_key"] = (
                f"opencas:{self._safe_cache_group(source)}:{complexity.value}:{scope_hash}"
            )
        payload.setdefault("prompt_cache_retention", "in_memory")

    @staticmethod
    def _cached_prompt_tokens(usage: Dict[str, Any]) -> Optional[int]:
        for key in ("prompt_tokens_details", "input_tokens_details"):
            details = usage.get(key)
            if not isinstance(details, dict):
                continue
            value = details.get("cached_tokens")
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            return parsed if parsed >= 0 else None
        return None

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

    def set_generation_policy(self, generation_policy: GenerationPolicyConfig) -> None:
        """Update the active generation policy without rebuilding the client."""
        self.generation_policy = generation_policy.normalized()
        self.generation_policy_resolver.set_config(self.generation_policy)

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

    def model_context_metadata(
        self,
        *,
        model: Optional[str] = None,
        complexity: ComplexityTier | str | None = None,
    ) -> Dict[str, Any]:
        """Return context-window metadata for the model selected for a lane.

        OpenLLMAuth already carries model catalog data. OpenCAS uses this to size
        prompt construction instead of assuming every model has the same small
        context budget.
        """
        tier = normalize_complexity_tier(complexity)
        requested_model = self.resolve_model_for_complexity(model=model, complexity=tier)
        provider_id, model_id = self._resolved_provider_and_model(requested_model)
        return self._model_context_metadata_for_parts(
            requested_model=requested_model,
            provider_id=provider_id,
            model_id=model_id,
            complexity=tier,
        )

    def _model_context_metadata_for_parts(
        self,
        *,
        requested_model: str,
        provider_id: str,
        model_id: str,
        complexity: ComplexityTier,
    ) -> Dict[str, Any]:
        model_def = self._model_definition(provider_id, model_id) or {}
        context_window = self._positive_int(
            model_def.get("context_window", model_def.get("contextWindow"))
        )
        max_output = self._positive_int(
            model_def.get("max_tokens", model_def.get("maxTokens"))
        )
        prompt_budget = self.prompt_context_budget_for_window(
            context_window=context_window,
            max_output_tokens=max_output,
        )
        return {
            "requested_model": requested_model,
            "resolved_model": f"{provider_id}/{model_id}",
            "provider": provider_id,
            "context_window": context_window,
            "max_output_tokens": max_output,
            "prompt_context_budget": prompt_budget,
            "budget_source": "model_metadata" if context_window else "legacy_fallback",
            "complexity": complexity.value,
        }

    def context_window_for_model(
        self,
        *,
        model: Optional[str] = None,
        complexity: ComplexityTier | str | None = None,
    ) -> Optional[int]:
        """Return the selected model's total context window, when known."""
        return self.model_context_metadata(model=model, complexity=complexity).get(
            "context_window"
        )

    def prompt_context_budget(
        self,
        *,
        model: Optional[str] = None,
        complexity: ComplexityTier | str | None = None,
    ) -> int:
        """Return the prompt-side token budget for the selected model."""
        return int(
            self.model_context_metadata(model=model, complexity=complexity).get(
                "prompt_context_budget"
            )
            or 6000
        )

    @staticmethod
    def prompt_context_budget_for_window(
        *,
        context_window: Optional[int],
        max_output_tokens: Optional[int] = None,
    ) -> int:
        """Compute usable prompt budget from total context window metadata."""
        if not context_window or context_window <= 0:
            return 6000
        output_reserve = min(
            max_output_tokens or 4096,
            max(1024, context_window // 4),
        )
        protocol_overhead = max(512, min(8192, context_window // 50))
        return max(1024, context_window - output_reserve - protocol_overhead)

    def _resolved_provider_and_model(self, requested_model: str) -> tuple[str, str]:
        try:
            resolved = self._resolve(requested_model)
            return str(resolved.provider_id), str(resolved.model_id)
        except Exception:
            if "/" in requested_model:
                provider_id, model_id = requested_model.split("/", 1)
                return provider_id.strip(), model_id.strip()
            return "", requested_model.strip()

    def _model_definition(self, provider_id: str, model_id: str) -> Dict[str, Any]:
        configured = self._configured_model_definition(provider_id, model_id)
        if configured:
            return configured
        model_definition = getattr(self.manager, "model_definition", None)
        if callable(model_definition):
            try:
                definition = model_definition(f"{provider_id}/{model_id}")
                if definition:
                    return dict(definition)
            except Exception:
                return {}
        return {}

    def _configured_model_definition(self, provider_id: str, model_id: str) -> Dict[str, Any]:
        provider_cfg = None
        resolver = getattr(self.manager, "_resolve_provider_config", None)
        if callable(resolver):
            try:
                provider_cfg = resolver(provider_id)
            except Exception:
                provider_cfg = None
        if provider_cfg is None:
            config = getattr(self.manager, "_config", None)
            configs_fn = getattr(config, "all_provider_configs", None)
            if callable(configs_fn):
                try:
                    provider_cfg = (configs_fn() or {}).get(provider_id)
                except Exception:
                    provider_cfg = None
        for model in getattr(provider_cfg, "models", []) or []:
            if str(getattr(model, "id", "") or "").strip() != model_id:
                continue
            dump = getattr(model, "model_dump", None)
            if callable(dump):
                return dict(dump(mode="json", by_alias=True))
            return {
                "id": getattr(model, "id", None),
                "contextWindow": getattr(model, "context_window", None),
                "maxTokens": getattr(model, "max_tokens", None),
            }
        return {}

    @staticmethod
    def _positive_int(value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

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
        context_meta = self._model_context_metadata_for_parts(
            requested_model=requested_model,
            provider_id=str(resolved.provider_id),
            model_id=str(resolved.model_id),
            complexity=complexity,
        )
        lane_meta["context_window"] = context_meta.get("context_window")
        lane_meta["prompt_context_budget"] = context_meta.get("prompt_context_budget")
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
        generation_request: Optional[GenerationPolicyRequest | Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request and return the response object."""
        tier = normalize_complexity_tier(complexity)
        model_ref = self.resolve_model_for_complexity(
            model=model,
            complexity=tier,
        )
        resolved = self._resolve(model_ref)
        started = time.perf_counter()
        merged_payload = dict(payload or {})
        reasoning_effort = None
        if not merged_payload.get("reasoning_effort"):
            reasoning_effort = self.resolve_reasoning_effort_for_complexity(
                complexity=tier,
            )
            if reasoning_effort:
                merged_payload["reasoning_effort"] = reasoning_effort
        effective_generation_policy = None
        if generation_request is not None:
            effective_generation_policy = self.generation_policy_resolver.resolve(
                generation_request,
                provider_id=str(resolved.provider_id),
                model_ref=model_ref,
                explicit_payload=merged_payload,
            )
            for key, value in effective_generation_policy.effective.items():
                merged_payload[key] = value
        reasoning_effort = (
            str(merged_payload.get("reasoning_effort")).strip()
            if merged_payload.get("reasoning_effort")
            else None
        )
        self._apply_prompt_cache_hints(
            merged_payload,
            provider_id=str(resolved.provider_id),
            model_ref=model_ref,
            complexity=tier,
            source=source,
            session_id=session_id,
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
        cache_trace_fields = self._prompt_cache_trace_fields(str(resolved.provider_id))
        generation_trace_fields = (
            effective_generation_policy.trace_fields()
            if effective_generation_policy is not None
            else {}
        )
        trace_context = {
            "model": model_ref,
            "provider": resolved.provider_id,
            "message_count": len(messages),
            "complexity": tier.value,
            "routing_mode": self.model_routing.mode.value,
            "reasoning_effort": reasoning_effort,
            "source": source,
            "session_id": session_id,
            "task_id": task_id,
            "execution_mode": execution_mode,
            "prompt_cache_key": merged_payload.get("prompt_cache_key"),
            "prompt_cache_retention": merged_payload.get("prompt_cache_retention"),
            **cache_trace_fields,
            **generation_trace_fields,
        }
        trace_context = {
            key: value for key, value in trace_context.items() if value is not None
        }
        if self.tracer:
            self.tracer.log(
                EventKind.LLM_CALL,
                f"LLM chat_completion: {model_ref}",
                trace_context,
            )
        await self._raise_if_circuit_open(resolved.provider_id)
        try:
            response = await self._chat_completion_with_retry(
                resolved=resolved,
                model_ref=model_ref,
                messages=messages,
                payload=merged_payload,
                source=source,
            )
            if effective_generation_policy is not None and isinstance(response, dict):
                response.setdefault(
                    "_opencas_generation_policy",
                    {
                        **generation_trace_fields,
                        "generation_provider": str(resolved.provider_id),
                        "generation_model": model_ref,
                    },
                )
            await self._circuit_breaker.record_success(resolved.provider_id)
        except Exception as exc:
            await self._circuit_breaker.record_failure(resolved.provider_id)
            if self.tracer:
                self.tracer.log(
                    EventKind.ERROR,
                    f"LLM chat_completion failed: {model_ref}: {exc}",
                    {
                        "model": model_ref,
                        "provider": resolved.provider_id,
                        "error": str(exc),
                        "complexity": tier.value,
                        "reasoning_effort": reasoning_effort,
                        "source": source,
                        "session_id": session_id,
                        "task_id": task_id,
                        "execution_mode": execution_mode,
                    },
                )
            raise
        latency_ms = int((time.perf_counter() - started) * 1000)
        if self.tracer:
            usage = response.get("usage") or {}
            cached_prompt_tokens = self._cached_prompt_tokens(usage)
            complete_context = {
                    "model": model_ref,
                    "provider": resolved.provider_id,
                    "latency_ms": latency_ms,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "cached_prompt_tokens": cached_prompt_tokens,
                    "complexity": tier.value,
                    "reasoning_effort": reasoning_effort,
                    "source": source,
                    "session_id": session_id,
                    "task_id": task_id,
                    "execution_mode": execution_mode,
                    "prompt_cache_key": merged_payload.get("prompt_cache_key"),
                    "prompt_cache_retention": merged_payload.get("prompt_cache_retention"),
                    **cache_trace_fields,
                    **generation_trace_fields,
                }
            self.tracer.log(
                EventKind.LLM_CALL,
                f"LLM chat_completion complete: {model_ref}",
                {
                    key: value
                    for key, value in complete_context.items()
                    if value is not None
                },
            )
        if self.token_telemetry:
            usage = response.get("usage") or {}
            await self.token_telemetry.record(
                model=model_ref,
                provider=resolved.provider_id,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                cached_prompt_tokens=self._cached_prompt_tokens(usage),
                latency_ms=latency_ms,
                source=source,
                session_id=session_id,
                task_id=task_id,
                execution_mode=execution_mode,
            )
        return response

    async def _chat_completion_with_retry(
        self,
        *,
        resolved: ResolvedProvider,
        model_ref: str,
        messages: List[Dict[str, Any]],
        payload: Dict[str, Any],
        source: str,
    ) -> Dict[str, Any]:
        attempts = max(1, int(self._chat_retry_attempts))
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await resolved.provider.chat_completion(
                    model=resolved.model_id,
                    messages=messages,
                    payload=payload,
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts or not self._is_transient_chat_error(exc):
                    raise
                delay = self._retry_delay_for_exception(exc, attempt=attempt)
                if self.tracer:
                    self.tracer.log(
                        EventKind.ERROR,
                        f"LLM chat_completion transient retry: {model_ref}: {self._error_label(exc)}",
                        {
                            "model": model_ref,
                            "provider": resolved.provider_id,
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "delay_seconds": delay,
                            "error_type": type(exc).__name__,
                            "source": source,
                        },
                    )
                if delay > 0:
                    await asyncio.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM chat_completion retry loop exited without a response")

    @classmethod
    def _is_transient_chat_error(cls, exc: Exception) -> bool:
        status = cls._exception_http_status(exc)
        if status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        name = type(exc).__name__.lower()
        return any(
            marker in name
            for marker in (
                "readerror",
                "connecterror",
                "timeout",
                "remoteprotocolerror",
                "networkerror",
            )
        )

    def _retry_delay_for_exception(self, exc: Exception, *, attempt: int) -> float:
        retry_after = self._exception_retry_after_seconds(exc)
        if retry_after is not None:
            return min(self._chat_retry_max_delay, max(0.0, retry_after))
        base = max(0.0, float(self._chat_retry_base_delay))
        return min(self._chat_retry_max_delay, base * (2 ** max(0, attempt - 1)))

    @staticmethod
    def _exception_http_status(exc: Exception) -> Optional[int]:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        try:
            return int(status)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _exception_retry_after_seconds(exc: Exception) -> Optional[float]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        try:
            value = headers.get("retry-after")
        except Exception:
            return None
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _error_label(exc: Exception) -> str:
        status = LLMClient._exception_http_status(exc)
        if status is not None:
            return f"{type(exc).__name__} HTTP {status}"
        text = str(exc).strip()
        return f"{type(exc).__name__}: {text}" if text else type(exc).__name__

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
        resolved = self._resolve(model_ref)
        if self.tracer:
            self.tracer.log(
                EventKind.LLM_CALL,
                f"LLM chat_completion_stream: {model_ref}",
                {
                    "model": model_ref,
                    "provider": resolved.provider_id,
                    "message_count": len(messages),
                    "complexity": tier.value,
                    "routing_mode": self.model_routing.mode.value,
                    "source": source,
                },
            )
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
        self._apply_prompt_cache_hints(
            merged_payload,
            provider_id=str(resolved.provider_id),
            model_ref=model_ref,
            complexity=tier,
            source=source,
            session_id=session_id,
        )
        self._record_lane_meta(
            requested_model=model_ref,
            resolved=resolved,
            complexity=tier,
            reasoning_effort=reasoning_effort,
        )
        await self._raise_if_circuit_open(resolved.provider_id)
        try:
            stream = await resolved.provider.chat_completion_stream(
                model=resolved.model_id,
                messages=messages,
                payload=merged_payload,
            )
        except Exception:
            await self._circuit_breaker.record_failure(resolved.provider_id)
            raise
        if not self.token_telemetry:
            async for chunk in self._circuit_wrapped_stream(stream, resolved.provider_id):
                yield chunk
            return

        started = time.perf_counter()

        async def _wrapped() -> AsyncGenerator[str, None]:
            async for chunk in self._circuit_wrapped_stream(stream, resolved.provider_id):
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
        task_type: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> List[float]:
        """Compute an embedding vector for *text* via the LLM gateway.

        Falls back to the local hashing embedder if no gateway embedding
        provider is resolved.
        """
        results = await self.embed_batch(
            [text],
            model=model,
            source=source,
            session_id=session_id,
            task_id=task_id,
            execution_mode=execution_mode,
            task_type=task_type,
            dimensions=dimensions,
        )
        return results[0]

    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        source: str = "embed_batch",
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        execution_mode: Optional[str] = None,
        task_type: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> List[List[float]]:
        """Compute embedding vectors for a batch of texts via the LLM gateway."""
        if not texts:
            return []

        model_ref = model or self._default_embedding_model_ref()
        resolved = self._resolve(model_ref)
        if self.tracer:
            self.tracer.log(
                EventKind.LLM_CALL,
                f"LLM embed_batch: {model_ref}",
                {
                    "model": model_ref,
                    "provider": resolved.provider_id,
                    "batch_size": len(texts),
                    "total_text_length": sum(len(text) for text in texts),
                    "source": source,
                },
            )
        started = time.perf_counter()
        payload: Dict[str, Any] = {}
        if task_type is not None:
            payload["task_type"] = task_type
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if hasattr(resolved.provider, "embeddings"):
            await self._raise_if_circuit_open(resolved.provider_id)
            try:
                result = await resolved.provider.embeddings(
                    model=resolved.model_id,
                    input_texts=texts,
                    payload=payload,
                )
                await self._circuit_breaker.record_success(resolved.provider_id)
            except Exception:
                await self._circuit_breaker.record_failure(resolved.provider_id)
                raise
            latency_ms = int((time.perf_counter() - started) * 1000)
            if self.tracer:
                self.tracer.log(
                    EventKind.LLM_CALL,
                    f"LLM embed_batch complete: {model_ref}",
                    {
                        "model": model_ref,
                        "provider": resolved.provider_id,
                        "latency_ms": latency_ms,
                        "source": source,
                    },
                )
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
            # The result is expected to be in OpenAI format: {"data": [{"embedding": [...]}, ...]}
            return [item["embedding"] for item in result["data"]]
        await self._circuit_breaker.record_failure(resolved.provider_id)
        raise RuntimeError(f"Provider {resolved.provider_id} does not support embeddings")

    def list_available_models(self) -> List[str]:
        """Return a flat list of available model references."""
        inventory = getattr(self.manager, "available_model_refs", None)
        if callable(inventory):
            try:
                return list(inventory())
            except Exception:
                return []
        return []

    def _default_embedding_model_ref(self) -> str:
        """Return the OpenLLMAuth-owned default embedding model ref."""

        resolver = getattr(self.manager, "default_embedding_model_ref", None)
        if callable(resolver):
            return str(resolver())
        return ProviderManager.default_embedding_model_ref()

    def _resolve(self, model_ref: str) -> ResolvedProvider:
        """Resolve a model reference to a concrete provider instance."""
        resolved = self.manager.resolve(model_ref)
        if resolved is None:
            raise ValueError(f"Could not resolve model: {model_ref}")
        return resolved

    async def _raise_if_circuit_open(self, provider_id: str) -> None:
        if await self._circuit_breaker.is_open(provider_id):
            opened_at = await self._circuit_breaker.opened_at(provider_id)
            raise ProviderCircuitOpen(provider_id, opened_at)

    async def _circuit_wrapped_stream(
        self,
        stream: AsyncGenerator[str, None],
        provider_id: str,
    ) -> AsyncGenerator[str, None]:
        try:
            async for chunk in stream:
                yield chunk
        except Exception:
            await self._circuit_breaker.record_failure(provider_id)
            raise
        await self._circuit_breaker.record_success(provider_id)
