"""Tests for procedural memory extraction (PR-016)."""

import pytest
import pytest_asyncio

from opencas.context import MemoryRetriever
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.execution import BoundedAssistantAgent, RepairTask
from opencas.execution.models import RepairResult
from opencas.memory import Episode, EpisodeKind, Memory, MemoryStore
from opencas.tools import ToolRegistry


@pytest_asyncio.fixture
async def stores(tmp_path):
    mem_store = MemoryStore(tmp_path / "memory.db")
    await mem_store.connect()
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    embed_service = EmbeddingService(
        cache=cache,
        model_id="local-fallback",
    )
    yield mem_store, embed_service
    await mem_store.close()
    await cache.close()


@pytest_asyncio.fixture
async def baa(stores):
    mem_store, embed_service = stores
    tools = ToolRegistry()
    agent = BoundedAssistantAgent(
        tools=tools,
        memory=mem_store,
        embeddings=embed_service,
        max_concurrent=1,
    )
    await agent.start()
    yield agent
    await agent.stop()


@pytest.mark.asyncio
async def test_extract_procedural_memory_creates_episode(baa, stores):
    mem_store, _embed_service = stores
    task = RepairTask(objective="run tests and fix failures")
    task_id = str(task.task_id)

    # Seed action episodes for this task session
    await mem_store.save_episode(
        Episode(
            kind=EpisodeKind.ACTION,
            session_id=task_id,
            content='tool bash_run_command: {"command": "pytest"}',
        )
    )
    await mem_store.save_episode(
        Episode(
            kind=EpisodeKind.ACTION,
            session_id=task_id,
            content='tool fs_read_file: {"file_path": "tests/test_app.py"}',
        )
    )
    await mem_store.save_episode(
        Episode(
            kind=EpisodeKind.OBSERVATION,
            session_id=task_id,
            content="tool bash_run_command failed: test output",
        )
    )

    result = RepairResult(
        task_id=task.task_id,
        success=True,
        stage=task.stage,
        output="tests passed after fix",
    )

    await baa._extract_procedural_memory(task, result)

    # Verify procedural episode was created
    episodes = await mem_store.list_episodes(session_id=task_id)
    procedural = [ep for ep in episodes if ep.kind == EpisodeKind.PROCEDURAL]
    assert len(procedural) == 1
    proc = procedural[0]
    assert "run tests and fix failures" in proc.content
    assert "bash_run_command" in proc.content
    assert "fs_read_file" in proc.content
    assert proc.embedding_id is not None


@pytest.mark.asyncio
async def test_extract_procedural_memory_no_tools_does_nothing(baa, stores):
    mem_store, _embed_service = stores
    task = RepairTask(objective="do nothing")
    task_id = str(task.task_id)

    result = RepairResult(
        task_id=task.task_id,
        success=True,
        stage=task.stage,
        output="done",
    )

    await baa._extract_procedural_memory(task, result)

    episodes = await mem_store.list_episodes(session_id=task_id)
    procedural = [ep for ep in episodes if ep.kind == EpisodeKind.PROCEDURAL]
    assert len(procedural) == 0


@pytest.mark.asyncio
async def test_semantic_search_includes_procedural_episodes(stores):
    mem_store, embed_service = stores
    retriever = MemoryRetriever(memory=mem_store, embeddings=embed_service)

    # Use the same content for episode and query so local-fallback embeddings match
    content = "run tests and fix failures"
    embed_record = await embed_service.embed(content)

    proc_ep = Episode(
        kind=EpisodeKind.PROCEDURAL,
        session_id="session-1",
        content=content,
        embedding_id=embed_record.source_hash,
    )
    await mem_store.save_episode(proc_ep)

    results = await retriever.retrieve(content, limit=5)
    assert any(
        r.source_type == "episode" and r.content == content
        for r in results
    )


@pytest.mark.asyncio
async def test_keyword_search_includes_procedural_episodes(stores):
    mem_store, embed_service = stores
    retriever = MemoryRetriever(memory=mem_store, embeddings=embed_service)

    proc_ep = Episode(
        kind=EpisodeKind.PROCEDURAL,
        session_id="session-2",
        content="Objective: deploy to staging\nTool sequence:\n- tool bash_run_command\nOutcome: success",
    )
    await mem_store.save_episode(proc_ep)

    results = await retriever.retrieve("deploy to staging", limit=5)
    assert any(
        r.source_type == "episode" and "deploy to staging" in r.content
        for r in results
    )


@pytest.mark.asyncio
async def test_list_episodes_by_embedding_ids(stores):
    mem_store, embed_service = stores
    embed_record = await embed_service.embed("procedural episode content")

    proc_ep = Episode(
        kind=EpisodeKind.PROCEDURAL,
        session_id="session-3",
        content="procedural episode content",
        embedding_id=embed_record.source_hash,
    )
    await mem_store.save_episode(proc_ep)

    fetched = await mem_store.list_episodes_by_embedding_ids([embed_record.source_hash])
    assert len(fetched) == 1
    assert fetched[0].episode_id == proc_ep.episode_id
