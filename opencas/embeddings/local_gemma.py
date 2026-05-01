import asyncio
import logging
import os
import threading
from typing import List, Optional, Sequence
import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger("embeddings.local_gemma")


def _configure_torch_threads() -> int:
    """Apply an explicit torch thread count when OPENCAS_TORCH_THREADS is set."""
    raw_value = os.environ.get("OPENCAS_TORCH_THREADS")
    if not raw_value:
        return torch.get_num_threads()
    try:
        threads = int(raw_value)
    except ValueError:
        logger.warning("Ignoring invalid OPENCAS_TORCH_THREADS=%r", raw_value)
        return torch.get_num_threads()
    if threads < 1:
        logger.warning("Ignoring non-positive OPENCAS_TORCH_THREADS=%r", raw_value)
        return torch.get_num_threads()
    torch.set_num_threads(threads)
    return torch.get_num_threads()


class GemmaEmbedder:
    """High-performance local embedder with dedicated threading."""

    def __init__(self, model_id: Optional[str] = None, device: str = "cpu"):
        self.model_id = model_id or "google/embeddinggemma-300m"
        self.device = device
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None
        self._lock = threading.Lock()

        torch_threads = _configure_torch_threads()
        logger.info(f"Initialized local Gemma embedder (torch threads: {torch_threads}).")

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence-transformers not installed.")

            token = os.environ.get("HF_TOKEN")
            logger.info(f"Loading local model {self.model_id}...")
            try:
                self._model = SentenceTransformer(self.model_id, device=self.device, token=token, trust_remote_code=True)
                test_vec = self._model.encode("test")
                self._dimension = len(test_vec)
                logger.info(f"Model ready. Threads: {torch.get_num_threads()}, Dim: {self._dimension}")
            except Exception as e:
                logger.error(f"Load failed: {e}")
                raise

    async def embed(self, text: str) -> List[float]:
        await asyncio.to_thread(self._ensure_model)
        vector = await asyncio.to_thread(self._model.encode, text, show_progress_bar=False)
        return vector.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        await asyncio.to_thread(self._ensure_model)
        vectors = await asyncio.to_thread(self._model.encode, texts, batch_size=64, show_progress_bar=False)
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_model()
        return self._dimension
