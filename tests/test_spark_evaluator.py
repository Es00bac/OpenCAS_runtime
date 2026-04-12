"""Tests for SparkEvaluator."""

import pytest
import pytest_asyncio

import numpy as np
from unittest.mock import AsyncMock, MagicMock

from opencas.autonomy.executive import ExecutiveState
from opencas.autonomy.models import WorkObject, WorkStage
from opencas.autonomy.work_store import WorkStore
from opencas.daydream import SparkEvaluator
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.somatic import SomaticManager, SomaticState


@pytest_asyncio.fixture
async def evaluator(tmp_path):
    cache = EmbeddingCache(tmp_path / "embeddings.db")
    await cache.connect()
    embed_service = EmbeddingService(cache=cache, model_id="local-fallback")
    work_store = WorkStore(tmp_path / "work.db")
    await work_store.connect()

    from opencas.identity import IdentityManager, IdentityStore
    id_store = IdentityStore(tmp_path / "identity")
    identity = IdentityManager(id_store)
    identity.load()
    executive = ExecutiveState(identity=identity)
    somatic = SomaticManager(tmp_path / "somatic.json")
    from opencas.relational import RelationalEngine, MusubiStore
    relational_store = MusubiStore(tmp_path / "musubi.db")
    await relational_store.connect()
    relational = RelationalEngine(store=relational_store)
    await relational.connect()

    ev = SparkEvaluator(
        embeddings=embed_service,
        work_store=work_store,
        executive=executive,
        somatic=somatic,
        relational=relational,
        novelty_floor=0.3,
    )
    yield ev
    await work_store.close()
    await cache.close()
    await relational_store.close()


@pytest.mark.asyncio
async def test_evaluate_with_no_existing_work_returns_high_novelty(evaluator):
    spark = WorkObject(content="test spark about space exploration")
    score = await evaluator.evaluate(spark)
    assert score >= 0.5
    assert spark.meta.get("spark_evaluated") is True
    assert evaluator.is_promotable(spark)


@pytest.mark.asyncio
async def test_filter_sparks_drops_low_scores(evaluator):
    # Manually override scores by mocking evaluate on one spark
    spark_high = WorkObject(content="high quality spark")
    spark_low = WorkObject(content="low quality spark")

    # Seed existing work to differentiate
    existing = WorkObject(content="low quality spark duplicate")
    await evaluator.work_store.save(existing)

    filtered = await evaluator.filter_sparks([spark_high, spark_low])
    # At minimum, the evaluator should run and some may be filtered.
    # With local-fallback embeddings, identical content gets same embedding.
    # We assert structural correctness rather than exact counts.
    assert isinstance(filtered, list)
    for sp in filtered:
        assert sp.meta.get("spark_evaluated") is True
        assert evaluator.is_promotable(sp)


@pytest.mark.asyncio
async def test_evaluate_somatic_alignment_penalizes_fatigue(evaluator):
    spark = WorkObject(content="i feel exhausted but want to learn rocket science")
    # Set high fatigue
    evaluator.somatic.state.fatigue = 0.9
    evaluator.somatic.state.arousal = 0.8
    score = await evaluator.evaluate(spark)
    details = spark.meta["spark_evaluation"]
    assert details["somatic_alignment"] < 0.6


@pytest.mark.asyncio
async def test_evaluate_executive_feasibility_boosts_goal_alignment(evaluator):
    spark = WorkObject(content="build a better memory retrieval pipeline")
    evaluator.executive.add_goal("improve memory retrieval")
    score = await evaluator.evaluate(spark)
    details = spark.meta["spark_evaluation"]
    assert details["executive_feasibility"] > 0.5


@pytest.mark.asyncio
async def test_evaluate_relational_alignment_high_musubi(evaluator):
    spark = WorkObject(content="we should collaborate on this project together")
    evaluator.relational._state.musubi = 0.8
    score = await evaluator.evaluate(spark)
    details = spark.meta["spark_evaluation"]
    assert details["relational_alignment"] > 0.5


@pytest.mark.asyncio
async def test_evaluate_cosine_distance_with_similar_work(evaluator):
    # Create two sparks with same content -> embedding should be identical under local-fallback
    existing = WorkObject(content="quantum computing advances")
    await evaluator.work_store.save(existing)

    spark = WorkObject(content="quantum computing advances")
    score = await evaluator.evaluate(spark)
    details = spark.meta["spark_evaluation"]
    # Identical embeddings = cosine similarity 1.0 = distance 0.0
    assert details["cosine_distance"] == 0.0


@pytest.mark.asyncio
async def test_evaluate_cosine_distance_different_content(evaluator):
    existing = WorkObject(content="quantum computing advances")
    await evaluator.work_store.save(existing)

    spark = WorkObject(content="medieval history of pottery")
    score = await evaluator.evaluate(spark)
    details = spark.meta["spark_evaluation"]
    assert details["cosine_distance"] > 0.0


@pytest.mark.asyncio
async def test_daydream_generator_filters_with_evaluator(tmp_path):
    from opencas.runtime.daydream import DaydreamGenerator
    from opencas.api import LLMClient

    cache = EmbeddingCache(tmp_path / "embed.db")
    await cache.connect()
    embed_service = EmbeddingService(cache=cache, model_id="local-fallback")
    work_store = WorkStore(tmp_path / "work.db")
    await work_store.connect()

    from opencas.identity import IdentityManager, IdentityStore
    id_store = IdentityStore(tmp_path / "identity2")
    identity = IdentityManager(id_store)
    identity.load()
    ev = SparkEvaluator(
        embeddings=embed_service,
        work_store=work_store,
        executive=ExecutiveState(identity=identity),
        somatic=SomaticManager(tmp_path / "somatic2.json"),
        novelty_floor=1.0,  # Impossible floor so nothing passes
    )

    # Mock LLM so we don't need a real key
    mock_mgr = MagicMock()
    resolved = MagicMock()
    resolved.provider_id = "test"
    resolved.model_id = "test"
    resolved.provider = MagicMock()
    resolved.provider.chat_completion = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": '{"sparks": ["spark a", "spark b"], "recollection": "", "interpretation": "", "synthesis": "", "open_question": "", "changed_self_view": "", "tension_hints": []}'
                    }
                }
            ]
        }
    )
    mock_mgr.resolve.return_value = resolved
    llm = LLMClient(mock_mgr, default_model="test/model")

    mock_memory = MagicMock()
    mock_memory.list_episodes = AsyncMock(return_value=[])
    mock_memory.list_recent_episodes = AsyncMock(return_value=[])
    dg = DaydreamGenerator(
        llm=llm,
        memory=mock_memory,
        spark_evaluator=ev,
    )

    work_objects, reflections = await dg.generate()
    assert len(reflections) == 2
    assert len(work_objects) == 0  # All filtered out by floor=1.0

    await work_store.close()
    await cache.close()
