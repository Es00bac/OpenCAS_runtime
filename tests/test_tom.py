"""Tests for the Theory of Mind (ToM) engine."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from opencas.identity import IdentityManager, IdentityStore
from opencas.tom import Belief, BeliefSubject, IntentionStatus, TomStore, ToMEngine


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
async def test_record_belief_audit_only_does_not_enter_production_tom(tom: ToMEngine) -> None:
    belief = await tom.record_belief(
        BeliefSubject.USER,
        "audit fiction",
        confidence=0.9,
        meta={"audit_only": True},
    )

    assert belief.meta["audit_only"] is True
    assert tom.list_beliefs(subject=BeliefSubject.USER) == []


@pytest.mark.asyncio
async def test_record_belief_hydrates_reinforced_store_fallback(
    tmp_path: Path,
    identity: IdentityManager,
) -> None:
    store = TomStore(tmp_path / "tom.db")
    await store.connect()
    try:
        base_time = datetime(2026, 5, 8, tzinfo=timezone.utc)
        for idx in range(1100):
            await store.save_belief(
                Belief(
                    subject=BeliefSubject.USER,
                    predicate=f"belief {idx}",
                    timestamp=base_time + timedelta(seconds=idx),
                    confidence=0.5,
                )
            )

        engine = ToMEngine(identity=identity, store=store)
        await engine.load()
        assert len(engine.list_beliefs(subject=BeliefSubject.USER)) == 1000
        assert engine.list_beliefs(subject=BeliefSubject.USER, predicate="belief 0") == []

        belief = await engine.record_belief(BeliefSubject.USER, "belief 0")

        assert belief.predicate == "belief 0"
        assert len(engine.list_beliefs(subject=BeliefSubject.USER)) == 1000
        assert engine.list_beliefs(subject=BeliefSubject.USER, predicate="belief 0") == [belief]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_record_intention_audit_only_does_not_enter_production_tom(tom: ToMEngine) -> None:
    intention = await tom.record_intention(
        BeliefSubject.SELF,
        "audit fiction",
        meta={"audit_only": True},
    )

    assert intention.meta["audit_only"] is True
    assert tom.list_intentions(actor=BeliefSubject.SELF) == []


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
async def test_record_belief_derives_structured_frame_and_source_strength(tom: ToMEngine) -> None:
    belief = await tom.record_belief(
        BeliefSubject.USER,
        "asked: check the calendar",
        confidence=0.66,
        meta={"source": "conversation_turn", "extractor": "rule_tier_a"},
    )

    assert belief.relation == "asked"
    assert belief.object == "check the calendar"
    assert belief.source_kind == "conversation_turn"
    assert belief.source_strength >= 0.7


@pytest.mark.asyncio
async def test_expired_high_confidence_belief_warns(tom: ToMEngine) -> None:
    await tom.record_belief(
        BeliefSubject.USER,
        "favorite color is blue",
        confidence=0.92,
        valid_until=datetime.now(timezone.utc) - timedelta(minutes=5),
        meta={"source": "conversation_turn"},
    )

    result = tom.check_consistency()

    assert any("expired" in warning and "favorite color is blue" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_user_location_belief_contradiction(tom: ToMEngine) -> None:
    await tom.record_belief(
        BeliefSubject.USER,
        "i live in arvada, colorado",
        confidence=0.82,
        meta={"source": "conversation_turn"},
    )
    await tom.record_belief(
        BeliefSubject.USER,
        "i live in denver, colorado",
        confidence=0.81,
        meta={"source": "conversation_turn"},
    )

    result = tom.check_consistency()

    assert any("location" in contradiction.lower() for contradiction in result.contradictions)


@pytest.mark.asyncio
async def test_legacy_beliefs_are_lazily_structured_for_consistency(tom: ToMEngine) -> None:
    tom._beliefs = [
        Belief(
            subject=BeliefSubject.USER,
            predicate="i live in arvada, colorado",
            confidence=0.82,
            meta={"source": "conversation_turn"},
        ),
        Belief(
            subject=BeliefSubject.USER,
            predicate="i live in denver, colorado",
            confidence=0.81,
            meta={"source": "conversation_turn"},
        ),
    ]

    result = tom.check_consistency()

    assert any("location" in contradiction.lower() for contradiction in result.contradictions)
    assert tom._beliefs[0].relation == "lives_in"
    assert tom._beliefs[0].object == "arvada, colorado"
    assert tom._beliefs[0].source_kind == "conversation_turn"
    assert tom._beliefs[0].source_strength == 0.75


@pytest.mark.asyncio
async def test_belief_syncs_to_identity(tom: ToMEngine, identity: IdentityManager) -> None:
    await tom.record_belief(BeliefSubject.SELF, "focused", confidence=0.8)
    assert any("focused" in str(v) for v in identity.self_model.self_beliefs.values())


@pytest.mark.asyncio
async def test_intention_syncs_to_identity(tom: ToMEngine, identity: IdentityManager) -> None:
    await tom.record_intention(BeliefSubject.SELF, "debug failing test")
    assert identity.self_model.current_intention == "debug failing test"


@pytest.mark.asyncio
async def test_agent_social_models_are_separate_from_user_and_self(tom: ToMEngine) -> None:
    await tom.record_agent_belief(
        "opus",
        "suggested an artifact spine plan",
        entity_label="Opus",
        confidence=0.72,
        evidence_ids=["chat:opus-plan"],
    )
    await tom.record_agent_belief(
        "codex",
        "suggested an artifact spine plan",
        entity_label="Codex",
        confidence=0.76,
        evidence_ids=["chat:codex-plan"],
    )
    await tom.record_agent_intention("opus", "audit OpenCAS stability", entity_label="Opus")

    agent_beliefs = tom.list_beliefs(subject=BeliefSubject.AGENT)
    models = tom.list_agent_models()

    assert len(agent_beliefs) == 2
    assert {belief.meta["entity_id"] for belief in agent_beliefs} == {"opus", "codex"}
    assert any(model["entity_id"] == "opus" and model["intentions"] for model in models)
    assert tom.list_beliefs(subject=BeliefSubject.USER) == []


@pytest.mark.asyncio
async def test_reinforced_belief_persists_evidence_and_source_upgrade(
    identity: IdentityManager,
    tmp_path: Path,
) -> None:
    store = TomStore(tmp_path / "tom.db")
    await store.connect()
    try:
        engine = ToMEngine(identity=identity, store=store)
        await engine.record_belief(
            BeliefSubject.USER,
            "prefers voice updates for meaningful progress",
            confidence=0.52,
            evidence_ids=["conversation_turn:s1"],
            meta={"source": "conversation_turn"},
        )
        await engine.record_belief(
            BeliefSubject.USER,
            "prefers voice updates for meaningful progress",
            confidence=0.52,
            evidence_ids=["conversation_turn:s2"],
            meta={"source": "tool_result", "source_strength": 0.9},
        )

        rehydrated = ToMEngine(identity=identity, store=store)
        await rehydrated.load()
        beliefs = rehydrated.list_beliefs(subject=BeliefSubject.USER)

        assert len(beliefs) == 1
        assert sorted(beliefs[0].evidence_ids) == ["conversation_turn:s1", "conversation_turn:s2"]
        assert beliefs[0].source_kind == "tool_result"
        assert beliefs[0].source_strength == pytest.approx(0.9)
        assert beliefs[0].reinforcement_count == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_salient_user_model_surfaces_behavioral_preferences(tom: ToMEngine) -> None:
    await tom.record_belief(
        BeliefSubject.USER,
        "prefers concise progress reports",
        confidence=0.72,
        evidence_ids=["conversation_turn:s1"],
        meta={"source": "conversation_turn"},
    )
    await tom.record_belief(
        BeliefSubject.USER,
        "unrelated archive fact",
        confidence=0.9,
        meta={"source": "inference", "source_strength": 0.35},
    )
    await tom.record_belief(
        BeliefSubject.USER,
        "needs stale guidance",
        confidence=0.9,
        valid_until=datetime.now(timezone.utc) - timedelta(days=1),
        meta={"source": "conversation_turn"},
    )

    model = tom.salient_user_model("continue with progress reports", limit=3)

    assert model[0]["predicate"] == "prefers concise progress reports"
    assert model[0]["effective_confidence"] == pytest.approx(0.72)
    assert model[0]["evidence_ids"] == ["conversation_turn:s1"]
    assert all(item["predicate"] != "unrelated archive fact" for item in model)
