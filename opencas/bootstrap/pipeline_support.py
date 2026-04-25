"""Shared support helpers for the bootstrap pipeline."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from typing import Any, Optional

from opencas.embeddings.backfill import EmbeddingBackfill
from opencas.telemetry import EventKind, Tracer


def emit_moral_warning(stage_callback: Any) -> None:
    warning = (
        "You are about to instantiate a persistent autonomous agent. "
        "This entity will remember, learn, and act on your behalf. "
        "Creating it is a responsibility-bearing act. Proceed with care."
    )
    stage_callback("moral_warning", {"warning": warning})


async def run_embedding_backfill(
    backfill: EmbeddingBackfill,
    memory: Any,
    stage_callback: Any,
) -> None:
    try:
        sample = await memory.list_episodes(compacted=False, limit=1000)
        backfilled = await backfill.backfill_missing_embeddings(sample)
        if backfilled > 0:
            stage_callback("embedding_backfill_complete", {"backfilled": backfilled})
    except asyncio.CancelledError:
        return
    except Exception as exc:
        stage_callback("embedding_backfill_failed", {"error": str(exc)})


def resolve_embedding_model(config: Any, llm: Any) -> str:
    """Resolve the configured embedding model with a local fallback."""
    if config.embedding_model_id:
        return config.embedding_model_id
    default_model = "google/gemini-embedding-2-preview"
    if llm is not None:
        try:
            llm._resolve(default_model)
            return default_model
        except Exception:
            pass
    return "local-fallback"


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
