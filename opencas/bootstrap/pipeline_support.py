"""Shared support helpers for the bootstrap pipeline."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from typing import Any, Optional

from open_llm_auth.auth.manager import ProviderManager

from opencas.embeddings.backfill import EmbeddingBackfill
from opencas.telemetry import EventKind, Tracer
from opencas.bootstrap.responsibility import BOOTSTRAP_RESPONSIBILITY_WARNING


def emit_moral_warning(stage_callback: Any) -> None:
    stage_callback("moral_warning", {"warning": BOOTSTRAP_RESPONSIBILITY_WARNING})


async def run_embedding_backfill(
    backfill: EmbeddingBackfill,
    memory: Any,
    stage_callback: Any,
) -> None:
    try:
        # Align distilled memories first (high priority, small volume)
        memories = await memory.list_memories(limit=2000)
        memories_count = await backfill.align_memory_embeddings(memories)
        if memories_count > 0:
            stage_callback("memory_backfill_complete", {"backfilled": memories_count})

        # Align history episodes (larger volume, incremental)
        sample = await memory.list_episodes(compacted=False, limit=5000)
        backfilled = await backfill.backfill_missing_embeddings(sample)
        if backfilled > 0:
            stage_callback("embedding_backfill_complete", {"backfilled": backfilled})
    except asyncio.CancelledError:
        return
    except Exception as exc:
        stage_callback("embedding_backfill_failed", {"error": str(exc)})


def resolve_embedding_model(config: Any, llm: Any) -> str:
    """Resolve the configured embedding model via OpenLLMAuth authority."""
    if config.embedding_model_id:
        return config.embedding_model_id

    manager = getattr(llm, "manager", None) or getattr(llm, "provider_manager", None)
    resolver = getattr(manager, "default_embedding_model_ref", None)
    if callable(resolver):
        return str(resolver())
    return ProviderManager.default_embedding_model_ref()


def resolve_embedding_dimensions(model_id: Optional[str], llm: Any = None) -> Optional[int]:
    """Return provider request dimensions using OpenLLMAuth model metadata."""
    if not model_id:
        return None
    manager = getattr(llm, "manager", None) or getattr(llm, "provider_manager", None)
    definition = None
    resolver = getattr(manager, "embedding_model_definition", None)
    if callable(resolver):
        try:
            definition = resolver(model_id)
        except Exception:
            definition = None
    if definition is None:
        definition = ProviderManager.local_embedding_model_definition(model_id)
    if isinstance(definition, dict):
        dimensions = definition.get("dimensions")
        if dimensions is not None:
            try:
                return int(dimensions)
            except (TypeError, ValueError):
                return None
    return None


def embedding_model_uses_local_runtime(model_id: Optional[str], llm: Any = None) -> bool:
    """Return whether OpenCAS should compute the embedding model locally."""
    if not model_id:
        return False
    manager = getattr(llm, "manager", None) or getattr(llm, "provider_manager", None)
    definition = None
    resolver = getattr(manager, "embedding_model_definition", None)
    if callable(resolver):
        try:
            definition = resolver(model_id)
        except Exception:
            definition = None
    if definition is None:
        definition = ProviderManager.local_embedding_model_definition(model_id)
    return bool(isinstance(definition, dict) and definition.get("local_runtime"))


def runtime_guard(config: Any) -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError(f"OpenCAS requires Python >= 3.11, found {sys.version}")

    critical_deps = ["pydantic", "open_llm_auth"]
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError as exc:
            raise RuntimeError(f"Missing critical dependency: {dep}") from exc

    if config.qdrant_url:
        try:
            import qdrant_client  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                f"Qdrant is configured but qdrant_client is unavailable: {exc}"
            ) from exc


def stage(tracer: Optional[Tracer], name: str, payload: Optional[dict] = None) -> None:
    if tracer:
        tracer.log(
            EventKind.BOOTSTRAP_STAGE,
            f"Bootstrap stage: {name}",
            payload or {},
        )


def hnsw_runtime_supported() -> bool:
    """Return whether the local interpreter/runtime is safe for HNSW use."""
    if importlib.util.find_spec("hnswlib") is None:
        return False
    if sys.version_info >= (3, 14):
        return False
    return True
