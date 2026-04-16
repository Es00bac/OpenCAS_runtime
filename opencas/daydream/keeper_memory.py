"""KeeperMemory: high-salience, pruning-immune daydream outputs.

Promoted from sparks with high existential weight. Keeper memories
survive consolidation pruning and remain accessible across sessions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class KeeperMemory(BaseModel):
    """A durable, high-salience memory promoted from existential daydreams."""

    keeper_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_spark_id: Optional[str] = None
    content: str
    fascination_thread: Optional[str] = None
    existential_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)

    def is_prunable(self) -> bool:
        """Keeper memories are never pruned."""
        return False


def promote_to_keeper(
    spark_content: str,
    source_spark_id: Optional[str] = None,
    fascination_thread: Optional[str] = None,
    existential_weight: float = 0.0,
    tags: Optional[List[str]] = None,
) -> KeeperMemory:
    """Promote a high-existential-weight spark to a KeeperMemory."""
    return KeeperMemory(
        source_spark_id=source_spark_id,
        content=spark_content,
        fascination_thread=fascination_thread,
        existential_weight=existential_weight,
        tags=tags or ["keeper", "existential"],
    )
