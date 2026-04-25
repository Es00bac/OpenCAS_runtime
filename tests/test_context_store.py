"""Tests for SessionContextStore."""

import pytest
import pytest_asyncio

from opencas.context import MessageRole, SessionContextStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SessionContextStore(tmp_path / "context.db")
    await s.connect()
    yield s
    await s.close()


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_append_and_list_recent(store):
    await store.append("s1", MessageRole.USER, "hello")
    await store.append("s1", MessageRole.ASSISTANT, "hi there")
    recent = await store.list_recent("s1", limit=10)
    assert len(recent) == 2
    assert recent[0].role == MessageRole.USER
    assert recent[0].content == "hello"
    assert recent[1].role == MessageRole.ASSISTANT
    assert recent[1].content == "hi there"


@pytest.mark.asyncio
async def test_list_recent_session_isolation(store):
    await store.append("s1", MessageRole.USER, "a")
    await store.append("s2", MessageRole.USER, "b")
    recent = await store.list_recent("s1", limit=10)
    assert len(recent) == 1
    assert recent[0].content == "a"


@pytest.mark.asyncio
async def test_count(store):
    await store.append("s1", MessageRole.USER, "x")
    await store.append("s1", MessageRole.USER, "y")
    assert await store.count("s1") == 2
    assert await store.count("s2") == 0


@pytest.mark.asyncio
async def test_search_basic(store):
    await store.append("s1", MessageRole.USER, "hello world")
    await store.append("s1", MessageRole.ASSISTANT, "goodbye moon")
    results = await store.search("hello")
    assert len(results) >= 1
    assert any("hello" in r.content for r in results)


@pytest.mark.asyncio
async def test_search_session_isolation(store):
    await store.append("s1", MessageRole.USER, "alpha beta")
    await store.append("s2", MessageRole.USER, "gamma delta")
    results = await store.search("alpha", session_id="s1")
    assert len(results) == 1
    assert results[0].content == "alpha beta"


@pytest.mark.asyncio
async def test_ensure_session_lists_empty_session_without_hidden_message(store):
    await store.ensure_session("empty-session")

    sessions = await store.list_session_ids(limit=10, status="active")
    meta = await store.get_session_meta("empty-session")
    recent = await store.list_recent("empty-session", limit=10)

    assert any(item["session_id"] == "empty-session" for item in sessions)
    assert meta is not None
    assert meta["status"] == "active"
    assert meta["message_count"] == 0
    assert recent == []


@pytest.mark.asyncio
async def test_session_metadata_supports_rename_archive_and_search(store):
    await store.ensure_session("portfolio-review")
    await store.update_session_name("portfolio-review", "Portfolio Review")
    await store.set_session_status("portfolio-review", "archived")

    archived = await store.list_session_ids(limit=10, status="archived")
    active = await store.list_session_ids(limit=10, status="active")
    searched = await store.search_sessions("portfolio", status="archived", limit=10)
    meta = await store.get_session_meta("portfolio-review")

    assert any(item["session_id"] == "portfolio-review" for item in archived)
    assert all(item["session_id"] != "portfolio-review" for item in active)
    assert len(searched) == 1
    assert searched[0]["name"] == "Portfolio Review"
    assert meta is not None
    assert meta["name"] == "Portfolio Review"
    assert meta["status"] == "archived"


@pytest.mark.asyncio
async def test_merge_message_meta_updates_existing_entry(store):
    entry = await store.append("voice-session", MessageRole.ASSISTANT, "hello there", meta={"lane": {"resolved_model": "codex-cli/gpt-5.4"}})

    updated = await store.merge_message_meta(
        "voice-session",
        entry.message_id,
        {"voice_output": {"provider": "elevenlabs", "model": "eleven_flash_v2_5"}},
    )

    assert updated is not None
    assert updated.meta["lane"]["resolved_model"] == "codex-cli/gpt-5.4"
    assert updated.meta["voice_output"]["provider"] == "elevenlabs"
