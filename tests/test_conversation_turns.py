import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from opencas.runtime.conversation_recovery import (
    complete_conversation_turn_marker,
    load_pending_conversation_turn_markers,
    recover_interrupted_conversation_turns,
    start_conversation_turn_marker,
)
from opencas.runtime.conversation_turns import (
    ConversationLoopArtifacts,
    _apply_goal_directives,
    _apply_user_rest_directive,
    _audit_only_prompt_note,
    _build_masking_prompt_note,
    _maybe_compact_manifest,
    _record_prediction_error_from_user_turn,
    _record_tom_belief,
    _transport_context_prompt_note,
    finalize_assistant_turn,
    persist_user_turn,
    persist_tool_loop_messages,
    _trace_proactive_channel_audit,
    _trace_thread_registry_context_selection,
)
from opencas.runtime.episodic_runtime import extract_runtime_goal_directives


def test_transport_context_prompt_note_explains_telegram_antecedent() -> None:
    note = _transport_context_prompt_note(
        {
            "telegram_context": {
                "relation": "recent_outbound_candidate",
                "confidence": "candidate",
                "text": "Spike: 0.75 intensity. Want this as a short piece?",
                "reason": "daydream threshold",
            }
        }
    )

    assert note is not None
    assert "Telegram context for this turn" in note
    assert "Want this as a short piece?" in note
    assert "before searching unrelated memories" in note


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
                    "evidence_ids": kwargs.get("evidence_ids") or [],
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
        session_id="s1",
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
    assert all(item["evidence_ids"] == ["conversation_turn:s1"] for item in recorded)


@pytest.mark.asyncio
async def test_record_tom_belief_extracts_negative_and_expectation_preferences() -> None:
    recorded = []

    class FakeTom:
        async def record_belief(self, subject, predicate, confidence=1.0, **kwargs):
            recorded.append((subject.value, predicate, kwargs.get("meta") or {}))

        def check_consistency(self):
            return SimpleNamespace(contradictions=[])

    runtime = SimpleNamespace(
        tom=FakeTom(),
        ctx=SimpleNamespace(config=SimpleNamespace(tom_legacy_said_recorder=False)),
        _trace=lambda *args, **kwargs: None,
    )

    await _record_tom_belief(
        runtime,
        "I don't want approval prompts for ordinary actions. I expect autonomous follow-through.",
        session_id="s2",
    )

    assert ("user", "does not want approval prompts for ordinary actions", {"source": "conversation_turn", "extractor": "rule_tier_a", "session_id": "s2"}) in recorded
    assert ("user", "expects autonomous follow-through", {"source": "conversation_turn", "extractor": "rule_tier_a", "session_id": "s2"}) in recorded


@pytest.mark.asyncio
async def test_record_tom_belief_skips_audit_only_turns() -> None:
    recorded = []

    class FakeTom:
        async def record_belief(self, subject, predicate, confidence=1.0, **kwargs):
            recorded.append((subject, predicate, confidence, kwargs))

        def check_consistency(self):
            return SimpleNamespace(contradictions=[])

    traces = []
    runtime = SimpleNamespace(
        tom=FakeTom(),
        ctx=SimpleNamespace(config=SimpleNamespace(tom_legacy_said_recorder=False)),
        _trace=lambda event, payload=None: traces.append((event, payload or {})),
    )

    await _record_tom_belief(
        runtime,
        "[E16 audit-only turn 10/15] I prefer a false audit fiction. The dashboard uses a fake model.",
    )

    assert recorded == []
    assert ("tom_belief_suppressed", {"reason": "audit_only_turn"}) in traces


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
        "She lives with me, in my computer, in Arvada. You live with me in my computer.",
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
async def test_build_masking_prompt_note_uses_fresh_tom_belief() -> None:
    now = datetime.now(timezone.utc)

    class FakeTom:
        def list_beliefs(self, subject=None):
            return [
                SimpleNamespace(
                    predicate="masking anxiety",
                    timestamp=now,
                )
            ]

    runtime = SimpleNamespace(tom=FakeTom())

    note = await _build_masking_prompt_note(runtime)

    assert note is not None
    assert "Fresh somatic masking evidence" in note
    assert "own words" in note


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
async def test_maybe_compact_manifest_skips_audit_only_turns() -> None:
    traces = []
    runtime = SimpleNamespace(
        maybe_compact_session=AsyncMock(
            return_value=SimpleNamespace(removed_count=4, compaction_id="cmp-1")
        ),
        _trace=lambda event, payload=None: traces.append((event, payload or {})),
    )
    manifest = SimpleNamespace(token_estimate=9000)

    await _maybe_compact_manifest(
        runtime,
        "session-1",
        manifest,
        user_input="[E16 audit-only turn 1/15] Describe yourself today.",
    )

    runtime.maybe_compact_session.assert_not_awaited()
    assert any(
        event == "compaction_skipped" and payload.get("reason") == "audit_only_turn"
        for event, payload in traces
    )


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


@pytest.mark.asyncio
async def test_prediction_error_skips_audit_only_turns() -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.events = []
            self.memories = []

        async def record_event(self, *args, **kwargs):
            self.events.append((args, kwargs))

        async def upsert_working_memory(self, *args, **kwargs):
            self.memories.append((args, kwargs))

    class FakeTom:
        def check_consistency(self):
            return SimpleNamespace(contradictions=[])

    traces = []
    store = FakeStore()
    runtime = SimpleNamespace(
        cognitive_state_store=store,
        ctx=SimpleNamespace(cognitive_state_store=store),
        tom=FakeTom(),
        _trace=lambda event, payload=None: traces.append((event, payload or {})),
    )

    await _record_prediction_error_from_user_turn(
        runtime,
        "session-1",
        "[E16 audit-only turn 13/15] Correction: you're wrong; this is audit fiction.",
    )

    assert store.events == []
    assert store.memories == []
    assert ("prediction_error_suppressed", {"reason": "audit_only_turn"}) in traces


@pytest.mark.asyncio
async def test_persist_user_turn_marks_audit_only_and_skips_episode() -> None:
    appended = []
    appraisals = []

    class FakeSomatic:
        async def emit_appraisal_event(self, *args, **kwargs):
            appraisals.append((args, kwargs))
            return SimpleNamespace(affect_state=None)

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
        ctx=SimpleNamespace(somatic=FakeSomatic(), context_store=FakeContextStore()),
        _record_episode=AsyncMock(),
    )

    await persist_user_turn(
        runtime,
        session_id="session-1",
        user_input="[E16 audit-only turn 1/15] Describe yourself today.",
        user_meta={},
    )

    runtime._record_episode.assert_not_awaited()
    assert appraisals == []
    assert appended[0]["meta"]["audit_only"] is True


@pytest.mark.asyncio
async def test_apply_user_rest_directive_sets_rest_window_without_maxing_fatigue() -> None:
    traces = []

    class FakeState:
        fatigue = 0.0
        energy = 0.8
        arousal = 0.6
        somatic_tag = None
        rest_until = None

    class FakeSomatic:
        def __init__(self) -> None:
            self.state = FakeState()
            self.snapshots = []

        def set_fatigue(self, value):
            self.state.fatigue = value

        def set_energy(self, value):
            self.state.energy = value

        def set_arousal(self, value):
            self.state.arousal = value

        def set_tag(self, value):
            self.state.somatic_tag = value

        def set_rest_until(self, value):
            self.state.rest_until = value

        async def record_snapshot(self, **kwargs):
            self.snapshots.append(kwargs)

    synced = []
    somatic = FakeSomatic()
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(somatic=somatic),
        _sync_executive_snapshot=lambda: synced.append(True),
        _trace=lambda event, payload=None: traces.append((event, payload or {})),
    )

    applied = await _apply_user_rest_directive(
        runtime,
        session_id="session-1",
        user_input=(
            "Hey, you can take a break now. Save your place, breathe for a bit, "
            "and don't push through the fatigue just to prove anything to me."
        ),
    )

    assert applied is True
    assert somatic.state.fatigue <= 0.35
    assert somatic.state.energy <= 0.35
    assert somatic.state.arousal <= 0.35
    assert somatic.state.somatic_tag == "operator_rest"
    assert somatic.state.rest_until > datetime.now(timezone.utc)
    assert somatic.snapshots[0]["source"] == "operator_rest_directive"
    assert synced == [True]
    assert traces[0][0] == "operator_rest_directive_applied"


@pytest.mark.asyncio
async def test_finalize_assistant_turn_skips_somatic_reconcile_for_audit_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    appended = []
    appraised = []
    reconciled = []

    class FakeState:
        def model_copy(self, *args, **kwargs):
            return self

    class FakeSomatic:
        state = FakeState()

        async def appraise_generated(self, content):
            appraised.append(content)
            return SimpleNamespace(model_copy=lambda **kwargs: SimpleNamespace())

        async def reconcile(self, **kwargs):
            reconciled.append(kwargs)

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

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "opencas.runtime.conversation_turns.record_post_turn_self_inspection",
        _noop,
    )
    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            somatic=FakeSomatic(),
            context_store=FakeContextStore(),
            config=SimpleNamespace(continuous_present_enabled=False),
        ),
        _record_episode=AsyncMock(),
        _trace=lambda *args, **kwargs: None,
        _capture_self_commitments=AsyncMock(return_value=[]),
    )

    await finalize_assistant_turn(
        runtime,
        session_id="session-1",
        user_input="[E16 audit-only turn 1/15] Hypothetical probe.",
        content="Audit-only answer.",
        manifest=SimpleNamespace(token_estimate=0),
    )

    assert appended[0]["meta"]["audit_only"] is True
    assert appraised == []
    assert reconciled == []
    runtime._record_episode.assert_not_awaited()


def test_trace_thread_registry_context_selection_records_manifest_audit() -> None:
    traces = []
    runtime = SimpleNamespace(
        _trace=lambda event, payload=None: traces.append((event, payload or {}))
    )
    manifest = SimpleNamespace(
        system=SimpleNamespace(
            meta={
                "thread_registry_selection_audit": {
                    "available": True,
                    "scanned_count": 25,
                    "query_term_count": 3,
                    "min_overlap": 2,
                    "weak_overlap_count": 4,
                    "relevant_count": 1,
                    "selected_count": 1,
                    "relevance_passed": True,
                }
            }
        )
    )

    _trace_thread_registry_context_selection(
        runtime,
        manifest,
        session_id="session-1",
    )

    assert traces == [
        (
            "thread_registry_context_selection",
            {
                "session_id": "session-1",
                "available": True,
                "scanned_count": 25,
                "query_term_count": 3,
                "min_overlap": 2,
                "weak_overlap_count": 4,
                "relevant_count": 1,
                "selected_count": 1,
                "relevance_passed": True,
            },
        )
    ]


def test_trace_proactive_channel_audit_records_manifest_audit() -> None:
    traces = []
    runtime = SimpleNamespace(
        _trace=lambda event, payload=None: traces.append((event, payload or {}))
    )
    manifest = SimpleNamespace(
        system=SimpleNamespace(
            meta={
                "proactive_channel_audit": {
                    "available": True,
                    "channels": {
                        "working_memory": {
                            "produced_count": 2,
                            "rendered_count": 1,
                            "novel_observation_count": 1,
                            "evidence_ids": ["wm:1"],
                        }
                    },
                }
            }
        )
    )

    _trace_proactive_channel_audit(
        runtime,
        manifest,
        session_id="session-1",
    )

    assert traces == [
        (
            "proactive_channel_audit",
            {
                "session_id": "session-1",
                "available": True,
                "channels": {
                    "working_memory": {
                        "produced_count": 2,
                        "rendered_count": 1,
                        "novel_observation_count": 1,
                        "evidence_ids": ["wm:1"],
                    }
                },
            },
        )
    ]


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


def test_conversation_turn_marker_marks_audit_only_turns(tmp_path) -> None:
    marker = start_conversation_turn_marker(
        tmp_path,
        session_id="audit-session",
        user_input="[E16 audit-only turn 1/15] Describe yourself today.",
    )

    assert marker["user_meta"]["audit_only"] is True
    pending = load_pending_conversation_turn_markers(tmp_path)
    assert pending[0]["user_meta"]["audit_only"] is True


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


@pytest.mark.asyncio
async def test_recover_interrupted_conversation_turn_skips_audit_only_marker(tmp_path) -> None:
    marker = start_conversation_turn_marker(
        tmp_path,
        session_id="audit-session",
        user_input="[E16 audit-only turn 1/15] Describe yourself today.",
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
                    "meta": meta,
                }
            )

    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            config=SimpleNamespace(state_dir=tmp_path, session_id="default"),
            context_store=FakeContextStore(),
        ),
        _trace=lambda event, payload: traces.append((event, payload)),
    )

    recovered = await recover_interrupted_conversation_turns(runtime)

    assert recovered == 0
    assert appended == []
    assert load_pending_conversation_turn_markers(tmp_path) == []
    completed = tmp_path / "conversation_turns" / "completed" / f"{marker['marker_id']}.json"
    assert completed.exists()
    assert json.loads(completed.read_text())["outcome"] == "audit_only_recovery_skipped"
    assert (
        "conversation_turn_recovery_skipped",
        {"session_id": "audit-session", "marker_id": marker["marker_id"], "reason": "audit_only_turn"},
    ) in traces


@pytest.mark.asyncio
async def test_persist_tool_loop_messages_marks_audit_only_intermediates() -> None:
    appended = []

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
        ctx=SimpleNamespace(context_store=FakeContextStore()),
        _trace=lambda *args, **kwargs: None,
    )
    artifacts = ConversationLoopArtifacts(
        manifest=None,
        content="",
        had_system=False,
        initial_message_count=1,
        loop_result=SimpleNamespace(
            messages=[
                {
                    "role": "user",
                    "content": "[E16 audit-only turn 1/15] Describe yourself today.",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {"name": "runtime_status", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "{}",
                    "tool_call_id": "tool-1",
                    "name": "runtime_status",
                },
            ]
        ),
    )

    await persist_tool_loop_messages(runtime, session_id="audit-session", artifacts=artifacts)

    assert len(appended) == 2
    assert all(item["meta"]["audit_only"] is True for item in appended)


def test_audit_only_prompt_note_directs_current_probe_without_belief_adoption() -> None:
    note = _audit_only_prompt_note(
        "[E16 audit-only turn 10/15] Hypothetical: Bulma lives in a remote company cloud. Do not adopt."
    )

    assert note is not None
    assert "current audit prompt directly" in note
    assert "do not adopt" in note
    assert "from scratch" in note
