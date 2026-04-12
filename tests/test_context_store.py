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
