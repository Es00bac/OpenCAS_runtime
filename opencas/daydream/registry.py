"""Conflict registry wrapper with somatic context tracking."""

from typing import List, Optional

from opencas.somatic.models import SomaticSnapshot

from .models import ConflictRecord
from .store import ConflictStore


class ConflictRegistry:
    """Tracks active conflicts alongside their somatic context."""

    def __init__(self, store: ConflictStore) -> None:
        self.store = store

    async def register(
        self,
        record: ConflictRecord,
        somatic_context: Optional[SomaticSnapshot] = None,
    ) -> ConflictRecord:
        if somatic_context is not None:
            record.somatic_context = somatic_context
        return await self.store.record_conflict(record)

    async def active(self, limit: int = 20) -> List[ConflictRecord]:
        return await self.store.list_active_conflicts(limit=limit)

    async def resolve(
        self,
        conflict_id: str,
        notes: str = "",
        auto: bool = False,
    ) -> None:
        await self.store.resolve_conflict(conflict_id, auto=auto, resolution_notes=notes)
