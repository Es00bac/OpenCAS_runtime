"""Embedding data models for OpenCAS."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EmbeddingRecord(BaseModel):
    """A cached embedding with provenance."""

    embedding_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Source identifier: hash of the content that generated this embedding
    source_hash: str

    # Model identifier used to compute the embedding
    model_id: str

    # Dimension of the vector
    dimension: int

    # Serialized vector as list of floats
    vector: List[float]

    # Optional metadata (e.g., text length, language)
    meta: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingHealth(BaseModel):
    """Health snapshot of the embedding service."""

    total_records: int = 0
    total_models: int = 0
    cache_hit_rate_1h: Optional[float] = None
    backfill_needed: int = 0
    last_backfill_at: Optional[datetime] = None
    average_vector_dimension: Optional[int] = None
    semantic_success_count_1h: int = 0
    lexical_fallback_count_1h: int = 0
    avg_latency_ms_1h: Optional[float] = None
    avg_embed_latency_ms_1h: Optional[float] = None
    ready_ratio: float = 1.0
