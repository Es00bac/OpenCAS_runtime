"""Embeddings module for OpenCAS: compute-once, cache-many vector service."""

from .models import EmbeddingHealth, EmbeddingRecord
from .qdrant_backend import QdrantVectorBackend
from .hnsw_backend import HnswVectorBackend
from .service import EmbeddingCache, EmbeddingService

__all__ = [
    "EmbeddingCache",
    "EmbeddingHealth",
    "EmbeddingRecord",
    "EmbeddingService",
    "HnswVectorBackend",
    "QdrantVectorBackend",
]
