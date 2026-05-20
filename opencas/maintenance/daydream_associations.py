"""Backfill retrievable semantic memories from stored daydream reflections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opencas.daydream.association_memory import (
    attach_daydream_association_context,
    build_daydream_association_memory,
    daydream_association_memory_id,
    persist_daydream_association_memory,
    should_persist_daydream_association,
)


@dataclass(slots=True)
class DaydreamAssociationBackfillReport:
    scanned: int = 0
    eligible: int = 0
    created: int = 0
    updated: int = 0
    skipped: int = 0
    dry_run: bool = True

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "scanned": self.scanned,
            "eligible": self.eligible,
            "created": self.created,
            "updated": self.updated,
            "skipped": self.skipped,
            "dry_run": self.dry_run,
        }


async def backfill_daydream_association_memories(
    *,
    daydream_store: Any,
    memory_store: Any,
    embeddings: Any,
    limit: int = 1000,
    dry_run: bool = True,
) -> DaydreamAssociationBackfillReport:
    """Create deterministic semantic association memories for stored reflections."""
    report = DaydreamAssociationBackfillReport(dry_run=dry_run)
    if daydream_store is None or memory_store is None:
        return report
    list_recent = getattr(daydream_store, "list_recent", None)
    if not callable(list_recent):
        return report

    reflections = await list_recent(limit=max(1, int(limit)))
    for reflection in reflections:
        report.scanned += 1
        if not should_persist_daydream_association(reflection):
            report.skipped += 1
            continue
        report.eligible += 1
        memory_id = str(daydream_association_memory_id(reflection))
        created = True
        get_memory = getattr(memory_store, "get_memory", None)
        if callable(get_memory):
            created = await get_memory(memory_id) is None

        if dry_run:
            build_daydream_association_memory(reflection)
        else:
            result = await persist_daydream_association_memory(
                memory_store=memory_store,
                embeddings=embeddings,
                reflection=reflection,
            )
            if result is None:
                report.skipped += 1
                continue
            created = result.created
            attach_daydream_association_context(reflection, result.memory)
            save_reflection = getattr(daydream_store, "save_reflection", None)
            if callable(save_reflection):
                await save_reflection(reflection)

        if created:
            report.created += 1
        else:
            report.updated += 1

    return report
