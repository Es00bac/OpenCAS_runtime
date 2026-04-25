"""Support helpers for ContextBuilder prompt assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Tuple

import numpy as np

from .models import MessageEntry, MessageRole, RetrievalResult


def is_soul_foundation_episode(episode: Any) -> bool:
    """Return whether an episode is an authoritative identity foundation."""
    source = str(episode.payload.get("source") or episode.payload.get("legacy_source", "")).lower()
    if source.startswith("soul:") or source.startswith("foundation:"):
        return True
    metadata = episode.payload.get("metadata") or episode.payload.get("legacy_metadata") or {}
    ep_type = str(metadata.get("type", "")).lower()
    return ep_type in ("foundation_soul", "foundation_document")


def is_workspace_derived_source(source: str) -> bool:
    """Return whether an identity candidate came from workspace bookkeeping."""
    normalized = source.lower()
    return (
        normalized.startswith("workspace:")
        or normalized.startswith("workspace-meta:")
        or normalized.startswith("workspace-manifest:")
        or normalized.startswith("workspace-usage:")
    )


async def build_identity_anchors(builder: Any) -> Tuple[List[str], List[str]]:
    """Fetch identity-core episodes and format SOUL + identity anchors."""
    soul_anchors: List[str] = []
    identity_anchors: List[str] = []
    if builder.retriever.memory is None:
        return soul_anchors, identity_anchors

    episodes = await builder.retriever.memory.list_identity_core_episodes(limit=20)
    if not episodes:
        return soul_anchors, identity_anchors

    soul_eps = [ep for ep in episodes if is_soul_foundation_episode(ep)]
    soul_eps.sort(key=lambda ep: ep.salience, reverse=True)
    for ep in soul_eps[:6]:
        ts = ep.created_at.isoformat()[:19]
        excerpt = str(ep.content)[:340]
        soul_anchors.append(f"- {ts}: {excerpt}")

    identity_eps = [
        ep
        for ep in episodes
        if not is_soul_foundation_episode(ep)
        and not is_workspace_derived_source(str(ep.payload.get("source") or ep.payload.get("legacy_source", "")))
    ]
    identity_eps.sort(key=lambda ep: ep.salience, reverse=True)
    for ep in identity_eps[:8]:
        ts = ep.created_at.isoformat()[:19]
        source = ep.payload.get("source") or ep.payload.get("legacy_source", "unknown")
        excerpt = str(ep.content)[:400]
        identity_anchors.append(f"- {ts} [{source}]: {excerpt}")

    return soul_anchors, identity_anchors


def estimate_tokens(texts: List[str]) -> int:
    """Sum token estimates for a list of texts."""
    return int(sum(len(t) for t in texts) * 0.25)


async def prune_by_redundancy(builder: Any, results: List[RetrievalResult], target_budget: int) -> List[RetrievalResult]:
    """Greedy redundancy removal: drop the result with highest average similarity."""
    system_entry = await builder._build_system_entry(
        style_note=builder.modulators.to_prompt_style_note() if builder.modulators is not None else ""
    )
    history = await builder.store.list_recent(session_id="default", limit=builder.recent_limit)
    base_tokens = estimate_tokens([system_entry.content] if system_entry else []) + estimate_tokens(
        [h.content for h in history]
    )

    embeddings: List[np.ndarray] = []
    for result in results:
        record = await builder.retriever.embeddings.embed(
            result.content,
            task_type="retrieval_context",
        )
        vec = np.array(record.vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        embeddings.append(vec)

    working = list(results)
    while working:
        current_tokens = base_tokens + estimate_tokens([r.content for r in working])
        if current_tokens <= target_budget:
            break
        n = len(working)
        if n == 1:
            working.pop()
            continue
        avg_sims: List[float] = []
        for i in range(n):
            sims = []
            for j in range(n):
                if i == j:
                    continue
                if embeddings[i].shape != embeddings[j].shape:
                    continue
                sims.append(float(np.dot(embeddings[i], embeddings[j])))
            avg_sims.append(float(np.mean(sims)) if sims else 0.0)
        highest = int(np.argmax(avg_sims))
        working.pop(highest)
        embeddings.pop(highest)
    return working


async def record_retrieval_usage(builder: Any, results: List[RetrievalResult]) -> None:
    """Record only the memories that actually made it into prompt context."""
    for result in results:
        if result.source_type == "episode":
            await builder.retriever.memory.touch_episode(result.source_id)
            continue
        if result.source_type == "memory":
            await builder.retriever.memory.touch_memory(result.source_id)
            memory = getattr(result, "memory", None)
            for episode_id in getattr(memory, "source_episode_ids", []) or []:
                await builder.retriever.memory.touch_episode(str(episode_id))


def to_memory_entries(results: List[RetrievalResult]) -> List[MessageEntry]:
    """Convert retrieval results into memory-role message entries."""
    entries: List[MessageEntry] = []
    for result in results:
        label = result.source_type.capitalize()
        content = f"[{label}] {result.content}"
        entries.append(
            MessageEntry(
                role=MessageRole.MEMORY,
                content=content,
                meta={"source_type": result.source_type, "source_id": result.source_id},
            )
        )
    return entries
