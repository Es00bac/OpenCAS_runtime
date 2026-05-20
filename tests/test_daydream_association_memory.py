"""Regression tests for retrievable daydream association memories."""

import pytest

from opencas.daydream.association_memory import (
    DAYDREAM_ASSOCIATION_TAG,
    build_daydream_association_memory,
    daydream_association_memory_id,
)
from opencas.daydream.models import (
    DaydreamDialogueTurn,
    DaydreamReflection,
    DaydreamThought,
    DaydreamThoughtKind,
    DaydreamThoughtRoute,
)
from opencas.maintenance.daydream_associations import (
    backfill_daydream_association_memories,
)


def test_non_keeper_daydream_builds_retrievable_association_memory() -> None:
    reflection = DaydreamReflection(
        spark_content="A bad Chapter 3 idea: make the Void Node explain everything directly.",
        synthesis="This is probably too explicit, but it marks a path to avoid.",
        open_question="Can the Void Node stay technical without overexplaining?",
        alignment_score=0.12,
        novelty_score=0.82,
        keeper=False,
        thoughts=[
            DaydreamThought(
                kind=DaydreamThoughtKind.STORY_SEED,
                route=DaydreamThoughtRoute.DISCARD,
                summary="Overexplaining the Void Node would flatten Chapter 3's tension.",
                inner_dialogue=[
                    DaydreamDialogueTurn(
                        voice="curious",
                        text="What if the Void Node explained itself directly?",
                    ),
                    DaydreamDialogueTurn(
                        voice="skeptical",
                        text="That would remove the reader's productive uncertainty.",
                    ),
                ],
                usefulness=0.35,
                novelty=0.74,
                confidence=0.62,
                risk=0.7,
            )
        ],
    )

    memory = build_daydream_association_memory(reflection, embedding_id="assoc-hash")

    assert memory.memory_id == daydream_association_memory_id(reflection)
    assert DAYDREAM_ASSOCIATION_TAG in memory.tags
    assert "daydream_non_keeper" in memory.tags
    assert "daydream_caution" in memory.tags
    assert "not a factual claim" in memory.content
    assert "Void Node" in memory.content
    assert "Chapter 3" in memory.content
    assert "path to avoid" in memory.content
    assert "Self-dialogue:" in memory.content
    assert "skeptical" in memory.content
    assert "productive uncertainty" in memory.content
    assert "daydream_self_dialogue" in memory.tags
    assert 0.0 < memory.salience < 5.0


@pytest.mark.asyncio
async def test_backfill_updates_reflection_context_with_association_memory() -> None:
    reflection = DaydreamReflection(
        spark_content="A stray thought about Chapter 3's dry technical logbook.",
        synthesis="This may help the Void Node stay tactile later.",
        novelty_score=0.7,
    )
    daydream_store = _FakeDaydreamStore([reflection])
    memory_store = _FakeMemoryStore()

    report = await backfill_daydream_association_memories(
        daydream_store=daydream_store,
        memory_store=memory_store,
        embeddings=None,
        dry_run=False,
    )

    memory_id = str(daydream_association_memory_id(reflection))
    assert report.as_dict()["created"] == 1
    assert memory_id in memory_store.memories
    assert daydream_store.saved[0].experience_context["association_memory_id"] == memory_id


class _FakeDaydreamStore:
    def __init__(self, reflections: list[DaydreamReflection]) -> None:
        self.reflections = reflections
        self.saved: list[DaydreamReflection] = []

    async def list_recent(self, limit: int = 10):
        return self.reflections[:limit]

    async def save_reflection(self, reflection: DaydreamReflection) -> None:
        self.saved.append(reflection)


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.memories = {}

    async def get_memory(self, memory_id: str):
        return self.memories.get(memory_id)

    async def save_memory(self, memory) -> None:
        self.memories[str(memory.memory_id)] = memory
