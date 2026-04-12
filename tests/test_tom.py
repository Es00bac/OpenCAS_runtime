"""Tests for the Theory of Mind (ToM) engine."""

from pathlib import Path
import pytest

from opencas.identity import IdentityManager, IdentityStore
from opencas.tom import BeliefSubject, IntentionStatus, ToMEngine


@pytest.fixture
def identity(tmp_path: Path):
    store = IdentityStore(tmp_path / "identity")
    mgr = IdentityManager(store)
    mgr.load()
    return mgr


@pytest.fixture
def tom(identity: IdentityManager):
    return ToMEngine(identity=identity)


@pytest.mark.asyncio
async def test_record_belief(tom: ToMEngine) -> None:
    b = await tom.record_belief(BeliefSubject.SELF, "ready to work", confidence=0.9)
    assert b.subject == BeliefSubject.SELF
    assert b.predicate == "ready to work"
    assert b.confidence == 0.9
    assert len(tom.list_beliefs(subject=BeliefSubject.SELF)) == 1


@pytest.mark.asyncio
async def test_record_intention(tom: ToMEngine) -> None:
    i = await tom.record_intention(BeliefSubject.SELF, "plan the day")
    assert i.actor == BeliefSubject.SELF
    assert i.content == "plan the day"
    assert i.status == IntentionStatus.ACTIVE
    assert len(tom.list_intentions(actor=BeliefSubject.SELF, status=IntentionStatus.ACTIVE)) == 1


@pytest.mark.asyncio
async def test_resolve_intention(tom: ToMEngine) -> None:
    await tom.record_intention(BeliefSubject.SELF, "plan the day")
    assert await tom.resolve_intention("plan the day", IntentionStatus.COMPLETED) is True
    assert len(tom.list_intentions(status=IntentionStatus.ACTIVE)) == 0
    assert len(tom.list_intentions(status=IntentionStatus.COMPLETED)) == 1


@pytest.mark.asyncio
async def test_resolve_intention_missing(tom: ToMEngine) -> None:
    assert await tom.resolve_intention("nonexistent") is False


@pytest.mark.asyncio
async def test_boundary_contradiction(tom: ToMEngine, identity: IdentityManager) -> None:
    identity.user_model.known_boundaries = ["no email"]
    identity.save()
    await tom.record_intention(BeliefSubject.SELF, "send email to team")
    result = tom.check_consistency()
    assert any("no email" in c for c in result.contradictions)


@pytest.mark.asyncio
async def test_self_belief_opposite_contradiction(tom: ToMEngine) -> None:
    await tom.record_belief(BeliefSubject.SELF, "tired", confidence=0.8)
    await tom.record_belief(BeliefSubject.SELF, "rested", confidence=0.8)
    result = tom.check_consistency()
    assert any("tired" in c and "rested" in c for c in result.contradictions)


@pytest.mark.asyncio
async def test_low_confidence_user_belief_warning(tom: ToMEngine) -> None:
    await tom.record_belief(BeliefSubject.USER, "likes jazz", confidence=0.1)
    result = tom.check_consistency()
    assert any("likes jazz" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_belief_syncs_to_identity(tom: ToMEngine, identity: IdentityManager) -> None:
    await tom.record_belief(BeliefSubject.SELF, "focused", confidence=0.8)
    assert any("focused" in str(v) for v in identity.self_model.self_beliefs.values())


@pytest.mark.asyncio
async def test_intention_syncs_to_identity(tom: ToMEngine, identity: IdentityManager) -> None:
    await tom.record_intention(BeliefSubject.SELF, "debug failing test")
    assert identity.self_model.current_intention == "debug failing test"
