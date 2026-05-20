from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from opencas.context.models import RetrievalResult
from opencas.memory.models import Episode, EpisodeKind
from opencas.tools.memory_tools import MemoryToolAdapter


class _FakeRetriever:
    def __init__(self, results):
        self.results = results

    async def inspect(self, query: str, *, limit: int = 10, offset: int = 0):
        self.query = query
        self.limit = limit
        self.offset = offset
        sliced = self.results[offset : offset + limit]
        return {
            "results": sliced,
            "meta": {
                "total_count": len(self.results),
                "limit": limit,
                "offset": offset,
            },
        }


@pytest.mark.asyncio
async def test_search_memories_renders_episode_kind_timestamp_and_tool_name() -> None:
    created = datetime(2026, 5, 1, 18, 3, 33, tzinfo=timezone.utc)
    action = Episode(
        created_at=created,
        kind=EpisodeKind.ACTION,
        session_id="s1",
        content="tool fs_write_file path=/tmp/probe.md bytes=12 checksum=abc",
        payload={"tool_name": "fs_write_file"},
    )
    turn = Episode(
        created_at=datetime(2026, 5, 1, 18, 4, 0, tzinfo=timezone.utc),
        kind=EpisodeKind.TURN,
        session_id="s1",
        content="asking about /tmp/probe.md",
    )
    runtime = SimpleNamespace(
        retriever=_FakeRetriever(
            [
                RetrievalResult(
                    source_type="episode",
                    source_id=str(action.episode_id),
                    content=action.content,
                    score=0.91,
                    episode=action,
                ),
                RetrievalResult(
                    source_type="episode",
                    source_id=str(turn.episode_id),
                    content=turn.content,
                    score=0.73,
                    episode=turn,
                ),
            ]
        )
    )

    output, meta = await MemoryToolAdapter(runtime)._search_memories(
        {"query": "probe.md", "limit": 2}
    )

    lines = output.splitlines()
    assert "2026-05-01T18:03:33+00:00 [ACTION][tool=fs_write_file]" in lines[0]
    assert "tool fs_write_file path=/tmp/probe.md" in lines[0]
    assert "2026-05-01T18:04:00+00:00 [TURN]" in lines[1]
    assert "[tool=" not in lines[1]


@pytest.mark.asyncio
async def test_search_memories_empty_result_names_artifact_lookup() -> None:
    runtime = SimpleNamespace(retriever=_FakeRetriever([]))

    output, meta = await MemoryToolAdapter(runtime)._search_memories(
        {"query": "missing.md"}
    )

    assert "No matching memories found" in output
    assert "artifact_lookup" in output
