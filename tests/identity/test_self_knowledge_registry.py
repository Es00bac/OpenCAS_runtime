"""Tests for the self-knowledge registry."""

from pathlib import Path
import pytest

from opencas.identity import KnowledgeEntry, SelfKnowledgeRegistry


@pytest.fixture
def registry(tmp_path: Path):
    return SelfKnowledgeRegistry(tmp_path / "self_knowledge.jsonl")


def test_record_and_get(registry: SelfKnowledgeRegistry) -> None:
    entry = registry.record("tom", "belief_focused", {"predicate": "focused"}, confidence=0.8)
    assert entry.domain == "tom"
    assert entry.key == "belief_focused"
    fetched = registry.get("tom", "belief_focused")
    assert fetched is not None
    assert fetched.value == {"predicate": "focused"}


def test_list_by_domain(registry: SelfKnowledgeRegistry) -> None:
    registry.record("tom", "a", 1)
    registry.record("tom", "b", 2)
    registry.record("memory", "c", 3)
    results = registry.list_by_domain("tom")
    assert len(results) == 2


def test_search(registry: SelfKnowledgeRegistry) -> None:
    registry.record("tom", "belief_focused", 1)
    registry.record("tom", "belief_tired", 2)
    results = registry.search("focus")
    assert len(results) == 1
    assert results[0].key == "belief_focused"


def test_to_self_beliefs(registry: SelfKnowledgeRegistry) -> None:
    registry.record("tom", "k1", "v1")
    registry.record("tom", "k2", "v2")
    registry.record("memory", "k1", "v3")
    beliefs = registry.to_self_beliefs()
    assert beliefs["tom"]["k1"] == "v1"
    assert beliefs["tom"]["k2"] == "v2"
    assert beliefs["memory"]["k1"] == "v3"


def test_latest_value_wins(registry: SelfKnowledgeRegistry) -> None:
    registry.record("tom", "k1", "first")
    registry.record("tom", "k1", "second")
    beliefs = registry.to_self_beliefs()
    assert beliefs["tom"]["k1"] == "second"
    fetched = registry.get("tom", "k1")
    assert fetched is not None
    assert fetched.value == "second"


def test_persistence(tmp_path: Path) -> None:
    path = tmp_path / "self_knowledge.jsonl"
    r1 = SelfKnowledgeRegistry(path)
    r1.record("tom", "persisted", {"x": 1})
    del r1
    r2 = SelfKnowledgeRegistry(path)
    entry = r2.get("tom", "persisted")
    assert entry is not None
    assert entry.value == {"x": 1}
