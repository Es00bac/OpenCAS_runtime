"""Retrieval behavioral evals.

Measures whether MemoryRetriever surfaces relevant episodes via keyword
(FTS) matching, salience ranking, and recency bias. Uses local-fallback
embeddings so no API calls are needed — semantic similarity is not
meaningful under hash embeddings, but FTS + salience + recency are.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

from opencas.context.retriever import MemoryRetriever
from opencas.embeddings.service import EmbeddingCache, EmbeddingService
from opencas.memory import Episode, EpisodeKind, Memory, MemoryStore


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float  # 0.0–1.0
    notes: str
    details: dict = field(default_factory=dict)


async def _make_store(tmp: Path) -> MemoryStore:
    store = MemoryStore(tmp / "memory.db")
    await store.connect()
    return store


async def _make_embeddings(tmp: Path) -> EmbeddingService:
    cache = EmbeddingCache(tmp / "embeddings.db")
    await cache.connect()
    return EmbeddingService(cache, model_id="local-fallback")


# ---------------------------------------------------------------------------
# Eval 1: keyword recall @ 3
# Store 10 episodes with distinct rare words. Query each rare word. Measure
# how often the correct episode appears in the top-3 results.
# ---------------------------------------------------------------------------
async def eval_keyword_recall(tmp: Path) -> EvalResult:
    store = await _make_store(tmp / "keyword")
    embeddings = await _make_embeddings(tmp / "keyword")

    targets = [
        ("aardvark", "The aardvark digs burrows at night."),
        ("quokka", "The quokka is a small marsupial from Australia."),
        ("axolotl", "The axolotl can regenerate its limbs entirely."),
        ("pangolin", "The pangolin is the only mammal covered in scales."),
        ("tardigrade", "The tardigrade survives extreme radiation and vacuum."),
    ]
    filler = [
        "The cat sat on the mat.",
        "Sunlight filtered through the curtains.",
        "A brief history of time includes many chapters.",
        "Running is good for cardiovascular health.",
        "The meeting was postponed until further notice.",
    ]

    for content in filler:
        await store.save_episode(Episode(
            kind=EpisodeKind.TURN, session_id="eval", content=content
        ))
    for _, content in targets:
        await store.save_episode(Episode(
            kind=EpisodeKind.TURN, session_id="eval", content=content
        ))

    retriever = MemoryRetriever(store, embeddings)
    hits = 0
    details = {}
    for keyword, content in targets:
        results = await retriever.retrieve(keyword, limit=3)
        found_contents = [r.content for r in results]
        hit = any(keyword in (c or "") for c in found_contents)
        details[keyword] = {"hit": hit, "top3": found_contents[:3]}
        if hit:
            hits += 1

    await store.close()
    score = hits / len(targets)
    return EvalResult(
        name="retrieval.keyword_recall_at_3",
        passed=score >= 0.6,
        score=score,
        notes=f"{hits}/{len(targets)} keyword queries found target in top-3",
        details=details,
    )


# ---------------------------------------------------------------------------
# Eval 2: salience ranking
# Store two episodes with the same keyword. One has high salience, one low.
# Verify the high-salience episode ranks first.
# ---------------------------------------------------------------------------
async def eval_salience_ranking(tmp: Path) -> EvalResult:
    store = await _make_store(tmp / "salience")
    embeddings = await _make_embeddings(tmp / "salience")

    low_ep = Episode(
        kind=EpisodeKind.TURN, session_id="eval",
        content="The zygomorphic flower has bilateral symmetry.",
        salience=1.0,
    )
    high_ep = Episode(
        kind=EpisodeKind.TURN, session_id="eval",
        content="The zygomorphic bloom is prized by botanists.",
        salience=9.5,
    )
    # Insert low first so recency would favour it absent salience scoring
    await store.save_episode(low_ep)
    await store.save_episode(high_ep)

    retriever = MemoryRetriever(store, embeddings)
    results = await retriever.retrieve("zygomorphic", limit=5)
    top_ids = [r.source_id for r in results if r.source_id]
    high_id = str(high_ep.episode_id)
    low_id = str(low_ep.episode_id)

    high_rank = top_ids.index(high_id) if high_id in top_ids else 99
    low_rank = top_ids.index(low_id) if low_id in top_ids else 99
    passed = high_rank < low_rank

    await store.close()
    return EvalResult(
        name="retrieval.salience_ranking",
        passed=passed,
        score=1.0 if passed else 0.0,
        notes=f"High-salience rank={high_rank}, low-salience rank={low_rank}",
        details={"high_rank": high_rank, "low_rank": low_rank, "top_ids": top_ids[:5]},
    )


# ---------------------------------------------------------------------------
# Eval 3: recency bias
# Store 6 episodes with the same keyword. Verify that an episode saved more
# recently (but otherwise equal) scores higher than an old one.
# ---------------------------------------------------------------------------
async def eval_recency_bias(tmp: Path) -> EvalResult:
    store = await _make_store(tmp / "recency")
    embeddings = await _make_embeddings(tmp / "recency")

    now = datetime.now(timezone.utc)
    old_ep = Episode(
        kind=EpisodeKind.TURN, session_id="eval",
        content="The vellichor word refers to old bookshops.",
        salience=5.0,
        created_at=now - timedelta(days=30),
    )
    new_ep = Episode(
        kind=EpisodeKind.TURN, session_id="eval",
        content="The vellichor concept evokes nostalgia in browsers.",
        salience=5.0,
        created_at=now - timedelta(minutes=5),
    )
    for i in range(4):
        await store.save_episode(Episode(
            kind=EpisodeKind.TURN, session_id="eval",
            content=f"Filler episode number {i} about nothing relevant.",
            salience=5.0,
            created_at=now - timedelta(hours=i),
        ))
    await store.save_episode(old_ep)
    await store.save_episode(new_ep)

    retriever = MemoryRetriever(store, embeddings)
    results = await retriever.retrieve("vellichor", limit=5)
    top_ids = [r.source_id for r in results if r.source_id]
    new_id = str(new_ep.episode_id)
    old_id = str(old_ep.episode_id)

    new_rank = top_ids.index(new_id) if new_id in top_ids else 99
    old_rank = top_ids.index(old_id) if old_id in top_ids else 99
    passed = new_rank < old_rank

    await store.close()
    return EvalResult(
        name="retrieval.recency_bias",
        passed=passed,
        score=1.0 if passed else 0.0,
        notes=f"Recent rank={new_rank}, old rank={old_rank}",
        details={"new_rank": new_rank, "old_rank": old_rank, "top_ids": top_ids[:5]},
    )


# ---------------------------------------------------------------------------
# Eval 4: memory vs episode mixing
# Store a Memory (distilled fact) and an Episode (turn) with the same keyword.
# Both should appear in results — verify retrieval surfaces both types.
# ---------------------------------------------------------------------------
async def eval_memory_episode_mixing(tmp: Path) -> EvalResult:
    store = await _make_store(tmp / "mixing")
    embeddings = await _make_embeddings(tmp / "mixing")

    ep = Episode(
        kind=EpisodeKind.TURN, session_id="eval",
        content="Ephemeral streams only flow after rain events.",
        salience=5.0,
    )
    mem = Memory(
        content="Ephemeral channels are dry except during storm runoff.",
        tags=["hydrology"],
        salience=7.0,
    )
    await store.save_episode(ep)
    await store.save_memory(mem)

    retriever = MemoryRetriever(store, embeddings)
    results = await retriever.retrieve("ephemeral", limit=5)
    contents = [r.content or "" for r in results]
    found_ep = any("streams" in c for c in contents)
    found_mem = any("channels" in c for c in contents)
    passed = found_ep and found_mem

    await store.close()
    return EvalResult(
        name="retrieval.memory_episode_mixing",
        passed=passed,
        score=(int(found_ep) + int(found_mem)) / 2.0,
        notes=f"Episode found={found_ep}, Memory found={found_mem}",
        details={"found_episode": found_ep, "found_memory": found_mem, "contents": contents[:5]},
    )


async def run_all(tmp_root: Path) -> List[EvalResult]:
    tmp_root.mkdir(parents=True, exist_ok=True)
    return await asyncio.gather(
        eval_keyword_recall(tmp_root),
        eval_salience_ranking(tmp_root),
        eval_recency_bias(tmp_root),
        eval_memory_episode_mixing(tmp_root),
    )
