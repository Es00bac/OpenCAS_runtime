"""Tests for legacy loaders."""

from pathlib import Path

from opencas.legacy.loader import load_json, stream_jsonl


FIXTURES = Path(__file__).parent / "fixtures"


def test_stream_jsonl_parses_episodes():
    path = FIXTURES / "episodes.jsonl"
    rows = list(stream_jsonl(path))
    assert len(rows) == 2
    assert rows[0]["id"] == "2026-02-19_0"
    assert rows[1]["id"] == "2026-02-19_2"


def test_stream_jsonl_handles_missing_file() -> None:
    rows = list(stream_jsonl(FIXTURES / "nonexistent.jsonl"))
    assert rows == []


def test_load_json_reads_profile() -> None:
    data = load_json(FIXTURES / "profile.json")
    assert data is not None
    assert data["coreNarrative"] == "Bulma prioritizes continuity, care."
    assert data["partner"]["userId"] == "test_user"


def test_load_json_missing_returns_none() -> None:
    assert load_json(FIXTURES / "nonexistent.json") is None
