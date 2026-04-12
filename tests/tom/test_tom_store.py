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
