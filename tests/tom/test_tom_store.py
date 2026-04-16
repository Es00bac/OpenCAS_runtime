"""Tests for the persistent ToM store."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.tom import Belief, BeliefSubject, Intention, IntentionStatus, TomStore


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    db = tmp_path / "tom.db"
    s = TomStore(db)
    await s.connect()
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_save_and_list_beliefs(store: TomStore) -> None:
    b = Belief(subject=BeliefSubject.USER, predicate="likes tea", confidence=0.8)
    await store.save_belief(b)
    results = await store.list_beliefs()
    assert len(results) == 1
    assert results[0].predicate == "likes tea"


@pytest.mark.asyncio
async def test_list_beliefs_filtered(store: TomStore) -> None:
    b1 = Belief(subject=BeliefSubject.USER, predicate="likes tea")
    b2 = Belief(subject=BeliefSubject.SELF, predicate="is focused")
    await store.save_belief(b1)
    await store.save_belief(b2)
    user_beliefs = await store.list_beliefs(subject=BeliefSubject.USER)
    assert len(user_beliefs) == 1
    assert user_beliefs[0].subject == BeliefSubject.USER


@pytest.mark.asyncio
async def test_save_and_list_intentions(store: TomStore) -> None:
    i = Intention(actor=BeliefSubject.SELF, content="finish report")
    await store.save_intention(i)
    results = await store.list_intentions()
    assert len(results) == 1
    assert results[0].content == "finish report"


@pytest.mark.asyncio
async def test_resolve_intention(store: TomStore) -> None:
    i = Intention(actor=BeliefSubject.SELF, content="finish report")
    await store.save_intention(i)
    await store.resolve_intention(str(i.intention_id), IntentionStatus.COMPLETED)
    results = await store.list_intentions(status=IntentionStatus.COMPLETED)
    assert len(results) == 1
    assert results[0].resolved_at is not None


@pytest.mark.asyncio
async def test_hydration_capped(store: TomStore) -> None:
    from opencas.tom.engine import ToMEngine
    from opencas.identity import IdentityManager, IdentityStore
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        identity = IdentityManager(IdentityStore(Path(td) / "identity"))
        identity.load()
        engine = ToMEngine(identity=identity, store=store)
        for idx in range(5):
            await engine.record_belief(BeliefSubject.SELF, f"belief {idx}")
            await engine.record_intention(BeliefSubject.SELF, f"intention {idx}")

        # Create a fresh engine and hydrate from store
        engine2 = ToMEngine(identity=identity, store=store)
        await engine2.load()
        assert len(engine2._beliefs) == 5
        assert len(engine2._intentions) == 5


@pytest.mark.asyncio
async def test_get_belief_by_predicate(store: TomStore) -> None:
    await store.save_belief(Belief(subject=BeliefSubject.USER, predicate="kind", confidence=0.7))
    await store.save_belief(Belief(subject=BeliefSubject.SELF, predicate="focused", confidence=0.9))

    found = await store.get_belief_by_predicate(BeliefSubject.USER, "kind")
    assert found is not None
    assert found.predicate == "kind"
    assert found.confidence == 0.7

    missing = await store.get_belief_by_predicate(BeliefSubject.USER, "cruel")
    assert missing is None


@pytest.mark.asyncio
async def test_increment_belief_reinforcement(store: TomStore) -> None:
    from datetime import datetime, timezone

    b = Belief(subject=BeliefSubject.USER, predicate="kind", confidence=0.5)
    await store.save_belief(b)

    now = datetime.now(timezone.utc)
    await store.increment_belief_reinforcement(
        str(b.belief_id), confidence=0.6, reinforcement_count=3, last_reinforced=now
    )

    results = await store.list_beliefs(subject=BeliefSubject.USER)
    assert len(results) == 1
    assert results[0].confidence == 0.6
    assert results[0].reinforcement_count == 3
    assert results[0].last_reinforced is not None


@pytest.mark.asyncio
async def test_reinforcement_deduplication(store: TomStore) -> None:
    """Recording the same belief 10 times should produce 1 reinforced row, not 10 duplicates."""
    from opencas.tom.engine import ToMEngine
    from opencas.identity import IdentityManager, IdentityStore
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        identity = IdentityManager(IdentityStore(Path(td) / "identity"))
        identity.load()
        engine = ToMEngine(identity=identity, store=store)

        for _ in range(10):
            await engine.record_belief(BeliefSubject.USER, "kind")

        beliefs = engine.list_beliefs(subject=BeliefSubject.USER, predicate="kind")
        assert len(beliefs) == 1
        assert beliefs[0].reinforcement_count == 10
        assert beliefs[0].confidence > 0.9  # merged upward from initial 1.0


@pytest.mark.asyncio
async def test_different_predicates_not_deduplicated(store: TomStore) -> None:
    """Different predicates should remain separate beliefs."""
    from opencas.tom.engine import ToMEngine
    from opencas.identity import IdentityManager, IdentityStore
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        identity = IdentityManager(IdentityStore(Path(td) / "identity"))
        identity.load()
        engine = ToMEngine(identity=identity, store=store)

        await engine.record_belief(BeliefSubject.USER, "kind")
        await engine.record_belief(BeliefSubject.USER, "curious")

        beliefs = engine.list_beliefs(subject=BeliefSubject.USER)
        assert len(beliefs) == 2
        assert {b.predicate for b in beliefs} == {"kind", "curious"}


@pytest.mark.asyncio
async def test_reinforcement_persists_across_load(store: TomStore) -> None:
    """Reinforced beliefs should survive store hydration."""
    from opencas.tom.engine import ToMEngine
    from opencas.identity import IdentityManager, IdentityStore
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        identity = IdentityManager(IdentityStore(Path(td) / "identity"))
        identity.load()

        engine = ToMEngine(identity=identity, store=store)
        for _ in range(5):
            await engine.record_belief(BeliefSubject.USER, "kind")

        # Hydrate a fresh engine from the same store
        engine2 = ToMEngine(identity=identity, store=store)
        await engine2.load()

        beliefs = engine2.list_beliefs(subject=BeliefSubject.USER, predicate="kind")
        assert len(beliefs) == 1
        assert beliefs[0].reinforcement_count == 5
