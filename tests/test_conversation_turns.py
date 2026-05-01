from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from opencas.runtime.episodic_runtime import extract_runtime_goal_directives
from opencas.runtime.conversation_recovery import (
    complete_conversation_turn_marker,
    load_pending_conversation_turn_markers,
    recover_interrupted_conversation_turns,
    start_conversation_turn_marker,
)
from opencas.runtime.conversation_turns import (
    _apply_goal_directives,
    _maybe_compact_manifest,
    _record_tom_belief,
)


@pytest.mark.asyncio
async def test_record_tom_belief_extracts_preferences_and_world_facts() -> None:
    recorded = []

    class FakeTom:
        async def record_belief(self, subject, predicate, confidence=1.0, **kwargs):
            recorded.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "confidence": confidence,
                    "meta": kwargs.get("meta") or {},
                }
            )

        def check_consistency(self):
            return SimpleNamespace(contradictions=[])

    runtime = SimpleNamespace(
        tom=FakeTom(),
        ctx=SimpleNamespace(config=SimpleNamespace(tom_legacy_said_recorder=False)),
        _trace=lambda *args, **kwargs: None,
    )

    await _record_tom_belief(
        runtime,
        "I prefer EmbeddingGemma at 3072 dimensions. The dashboard uses Gemma embeddings.",
    )

    assert recorded
    assert all(not item["predicate"].startswith("said:") for item in recorded)
    assert {
        "subject": "user",
        "predicate": "prefers embeddinggemma at 3072 dimensions",
    } in [
        {"subject": item["subject"].value, "predicate": item["predicate"]}
        for item in recorded
    ]
    assert {
        "subject": "world",
        "predicate": "dashboard uses gemma embeddings",
    } in [
        {"subject": item["subject"].value, "predicate": item["predicate"]}
        for item in recorded
    ]


@pytest.mark.asyncio
async def test_record_tom_belief_extracts_learned_self_location() -> None:
    recorded = []

    class FakeTom:
        async def record_belief(self, subject, predicate, confidence=1.0, **kwargs):
            recorded.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "confidence": confidence,
                    "meta": kwargs.get("meta") or {},
                }
            )

        def check_consistency(self):
            return SimpleNamespace(contradictions=[])

    runtime = SimpleNamespace(
        tom=FakeTom(),
        ctx=SimpleNamespace(config=SimpleNamespace(tom_legacy_said_recorder=False)),
        _trace=lambda *args, **kwargs: None,
    )

    await _record_tom_belief(
        runtime,
        "She lives with me, in my computer, in my city. You live with me in my computer.",
    )

    recorded_pairs = [
        {"subject": item["subject"].value, "predicate": item["predicate"]}
        for item in recorded
    ]
    assert {
        "subject": "self",
        "predicate": "lives with user, in user's computer, in arvada",
    } in recorded_pairs
    assert {
        "subject": "self",
        "predicate": "lives with user in user's computer",
    } in recorded_pairs
    assert all(item["meta"]["extractor"] == "rule_tier_a" for item in recorded)


@pytest.mark.asyncio
async def test_apply_goal_directives_mirrors_explicit_intention_into_tom() -> None:
    recorded = []

    class FakeExecutive:
        active_goals = []

        def __init__(self) -> None:
            self.intention = None

        def add_goal(self, goal):
            self.active_goals.append(goal)

        def set_intention(self, intention):
            self.intention = intention

        def remove_goal(self, goal):
            self.active_goals.remove(goal)

    class FakeTom:
        def list_intentions(self, actor=None, status=None):
            return []

        async def record_intention(self, actor, content, meta=None):
            recorded.append({"actor": actor, "content": content, "meta": meta or {}})

    runtime = SimpleNamespace(
        executive=FakeExecutive(),
        tom=FakeTom(),
        ctx=SimpleNamespace(identity=None),
        _extract_goal_directives=extract_runtime_goal_directives,
        _sync_executive_snapshot=lambda: None,
        _trace=lambda *args, **kwargs: None,
    )

    await _apply_goal_directives(
        runtime,
        "Please intention is keep ToM aligned with live executive focus.",
        session_id="session-1",
    )

    assert runtime.executive.intention == "keep tom aligned with live executive focus"
    assert recorded
    assert recorded[0]["actor"].value == "self"
    assert recorded[0]["content"] == "keep tom aligned with live executive focus"
    assert recorded[0]["meta"]["source"] == "user_goal_directive"
    assert recorded[0]["meta"]["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_maybe_compact_manifest_skips_below_new_threshold() -> None:
    runtime = SimpleNamespace(
        maybe_compact_session=AsyncMock(),
        _trace=lambda *args, **kwargs: None,
    )
    manifest = SimpleNamespace(token_estimate=5000)

    await _maybe_compact_manifest(runtime, "session-1", manifest)

    runtime.maybe_compact_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_compact_manifest_applies_session_cooldown() -> None:
    traces = []
    runtime = SimpleNamespace(
        maybe_compact_session=AsyncMock(
            return_value=SimpleNamespace(removed_count=4, compaction_id="cmp-1")
        ),
        _trace=lambda event, payload=None: traces.append((event, payload or {})),
    )
    manifest = SimpleNamespace(token_estimate=9000)

    await _maybe_compact_manifest(runtime, "session-1", manifest)
    await _maybe_compact_manifest(runtime, "session-1", manifest)

    assert runtime.maybe_compact_session.await_count == 1
    assert any(event == "compaction_skipped" for event, _ in traces)


def test_conversation_turn_marker_tracks_pending_and_completion(tmp_path) -> None:
    marker = start_conversation_turn_marker(
        tmp_path,
        session_id="session-1",
        user_input="Look into the Colorado Enterprise Fund CDFI route",
        user_meta={"source": "telegram"},
    )

    pending = load_pending_conversation_turn_markers(tmp_path)
    assert len(pending) == 1
    assert pending[0]["marker_id"] == marker["marker_id"]
    assert pending[0]["session_id"] == "session-1"
    assert pending[0]["phase"] == "started"

    complete_conversation_turn_marker(
        tmp_path,
        marker["marker_id"],
        outcome="assistant_response_persisted",
    )

    assert load_pending_conversation_turn_markers(tmp_path) == []


@pytest.mark.asyncio
async def test_recover_interrupted_conversation_turn_adds_visible_marker(tmp_path) -> None:
    start_conversation_turn_marker(
        tmp_path,
        session_id="session-1",
        user_input="Research Colorado Enterprise Fund and CDFI options",
        user_meta={"source": "dashboard"},
    )
    appended = []
    traces = []

    class FakeContextStore:
        async def append(self, session_id, role, content, meta=None):
            appended.append(
                {
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "meta": meta or {},
                }
            )

    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            config=SimpleNamespace(state_dir=tmp_path),
            context_store=FakeContextStore(),
            llm=None,
        ),
        _trace=lambda event, payload=None: traces.append((event, payload or {})),
    )

    recovered = await recover_interrupted_conversation_turns(runtime)

    assert recovered == 1
    assert len(appended) == 1
    assert appended[0]["session_id"] == "session-1"
    assert appended[0]["role"].value == "assistant"
    assert "interrupted" in appended[0]["content"].lower()
    assert "Colorado Enterprise Fund" in appended[0]["content"]
    assert appended[0]["meta"]["recovered_interrupted_turn"] is True
    assert load_pending_conversation_turn_markers(tmp_path) == []
    assert traces[0][0] == "conversation_turn_recovered"
