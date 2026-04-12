"""Tests for memory fabric scorers."""

from datetime import datetime, timezone, timedelta

import pytest

from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.memory import EdgeKind, Episode, EpisodeKind
from opencas.memory.fabric.scorers import (
    CausalScorer,
    ConceptualScorer,
    EmotionalScorer,
    RelationalScorer,
    TemporalScorer,
)
from opencas.somatic.models import AffectState, PrimaryEmotion, SocialTarget


@pytest.fixture
def conceptual_scorer():
    # We can't async-connect in a sync fixture easily, so we'll create inline
    return None


@pytest.mark.asyncio
async def test_conceptual_scorer_missing_embeddings():
    scorer = ConceptualScorer(
        EmbeddingService(EmbeddingCache(":memory:"), model_id="local-fallback")
    )
    ep_a = Episode(kind=EpisodeKind.TURN, content="a")
    ep_b = Episode(kind=EpisodeKind.TURN, content="b")
    score = await scorer.score(ep_a, ep_b)
    assert score == 0.0


@pytest.mark.asyncio
async def test_conceptual_scorer_similar_content():
    cache = EmbeddingCache(":memory:")
    await cache.connect()
    embeddings = EmbeddingService(cache=cache, model_id="local-fallback")
    scorer = ConceptualScorer(embeddings)

    ep_a = Episode(kind=EpisodeKind.TURN, content="rust programming basics")
    ep_b = Episode(kind=EpisodeKind.TURN, content="rust ownership concepts")
    rec_a = await embeddings.embed(ep_a.content)
    rec_b = await embeddings.embed(ep_b.content)
    ep_a.embedding_id = rec_a.source_hash
    ep_b.embedding_id = rec_b.source_hash

    score = await scorer.score(ep_a, ep_b)
    assert 0.0 < score <= 1.0

    await cache.close()


@pytest.mark.asyncio
async def test_emotional_scorer_missing_affect():
    scorer = EmotionalScorer()
    ep_a = Episode(kind=EpisodeKind.TURN, content="a")
    ep_b = Episode(kind=EpisodeKind.TURN, content="b")
    assert await scorer.score(ep_a, ep_b) == 0.0


@pytest.mark.asyncio
async def test_emotional_scorer_matching_emotion():
    scorer = EmotionalScorer()
    ep_a = Episode(
        kind=EpisodeKind.TURN,
        content="a",
        affect=AffectState(primary_emotion=PrimaryEmotion.JOY, valence=0.8, arousal=0.6),
    )
    ep_b = Episode(
        kind=EpisodeKind.TURN,
        content="b",
        affect=AffectState(primary_emotion=PrimaryEmotion.JOY, valence=0.7, arousal=0.5),
    )
    score = await scorer.score(ep_a, ep_b)
    assert score > 0.5


@pytest.mark.asyncio
async def test_relational_scorer_same_project_and_session():
    scorer = RelationalScorer()
    ep_a = Episode(
        kind=EpisodeKind.TURN,
        content="a",
        session_id="s1",
        payload={"project_id": "p1"},
        affect=AffectState(social_target=SocialTarget.USER),
    )
    ep_b = Episode(
        kind=EpisodeKind.TURN,
        content="b",
        session_id="s1",
        payload={"project_id": "p1"},
        affect=AffectState(social_target=SocialTarget.USER),
    )
    score = await scorer.score(ep_a, ep_b)
    assert score == 1.0


@pytest.mark.asyncio
async def test_temporal_scorer_recent():
    scorer = TemporalScorer(decay_days=120.0)
    now = datetime.now(timezone.utc)
    ep_a = Episode(kind=EpisodeKind.TURN, content="a", created_at=now)
    ep_b = Episode(kind=EpisodeKind.TURN, content="b", created_at=now)
    assert await scorer.score(ep_a, ep_b) == 1.0


@pytest.mark.asyncio
async def test_temporal_scorer_distant():
    scorer = TemporalScorer(decay_days=120.0)
    now = datetime.now(timezone.utc)
    ep_a = Episode(kind=EpisodeKind.TURN, content="a", created_at=now)
    ep_b = Episode(
        kind=EpisodeKind.TURN,
        content="b",
        created_at=now - timedelta(days=200),
    )
    score = await scorer.score(ep_a, ep_b)
    assert score == 0.0


@pytest.mark.asyncio
async def test_causal_scorer_action_to_observation():
    scorer = CausalScorer()
    now = datetime.now(timezone.utc)
    ep_a = Episode(
        kind=EpisodeKind.ACTION,
        content="triggered the build",
        created_at=now - timedelta(minutes=1),
        payload={"project_id": "p1"},
    )
    ep_b = Episode(
        kind=EpisodeKind.OBSERVATION,
        content="build succeeded",
        created_at=now,
        payload={"project_id": "p1"},
    )
    score = await scorer.score(ep_a, ep_b)
    assert score >= 0.3


@pytest.mark.asyncio
async def test_causal_scorer_no_cues():
    scorer = CausalScorer()
    now = datetime.now(timezone.utc)
    ep_a = Episode(
        kind=EpisodeKind.TURN,
        content="hello world",
        created_at=now,
    )
    ep_b = Episode(
        kind=EpisodeKind.TURN,
        content="goodbye world",
        created_at=now,
    )
    score = await scorer.score(ep_a, ep_b)
    assert score == 0.0
