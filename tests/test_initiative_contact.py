from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.context.models import MessageRole
from opencas.daydream import DaydreamReflection
from opencas.initiative_contact import InitiativeContactConfig, InitiativeContactService
from opencas.memory import EpisodeKind
from opencas.runtime import AgentRuntime


class _Telegram:
    def __init__(self) -> None:
        self.messages: list[tuple[str, dict]] = []

    async def notify_owner(self, text: str, **kwargs):
        self.messages.append((text, kwargs))
        return {"sent": 1, "chat_ids": ["42"]}


class _TraceRuntime:
    def __init__(self, state_dir: Path, *, telegram: _Telegram | None = None) -> None:
        self._telegram = telegram
        self._activity = "idle"
        self.baa = SimpleNamespace(queue_size=0, held_size=0, active_count=0)
        self.ctx = SimpleNamespace(
            daydream_store=None,
            identity=SimpleNamespace(self_model=SimpleNamespace(name="TestAgent")),
        )
        self.traces: list[tuple[str, dict]] = []
        self.phone_calls: list[dict] = []

    def _trace(self, event: str, payload: dict) -> None:
        self.traces.append((event, payload))

    async def phone_status(self) -> dict:
        return {
            "enabled": True,
            "twilio_from_number": "+15557654321",
            "twilio_credentials_configured": True,
            "owner": {"configured": True, "phone_number": "+15551234567"},
        }

    async def call_owner_via_phone(self, *, message: str, reason: str = "") -> dict:
        self.phone_calls.append({"message": message, "reason": reason})
        return {"ok": True, "to": "+15551234567", "call_sid": f"CA{len(self.phone_calls)}"}


class _LLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat_completion(self, **_: object) -> dict:
        self.calls.append(dict(_))
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"send": true, "channel": "telegram", "urgency": "normal", '
                            '"reason": "worth sharing", "message": "The unified graph maps intent."}'
                        )
                    }
                }
            ]
        }


class _OvereagerLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat_completion(self, **_: object) -> dict:
        self.calls.append(dict(_))
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"send": true, "channel": "telegram", "urgency": "normal", '
                            '"reason": "felt interesting", "message": "I am thinking about something."}'
                        )
                    }
                }
            ]
        }


class _ReflectionStore:
    def __init__(self) -> None:
        self.saved: list[DaydreamReflection] = []

    async def save_reflection(self, reflection: DaydreamReflection) -> None:
        self.saved.append(reflection)


class _ContextStore:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    async def append(self, session_id: str, role: MessageRole, content: str, meta: dict | None = None) -> None:
        self.entries.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "meta": meta or {},
            }
        )


class _MemoryStore:
    def __init__(self) -> None:
        self.episodes: list[object] = []

    async def save_episode(self, episode: object) -> None:
        self.episodes.append(episode)


class _CandidateStore:
    def __init__(self, candidates: list[SimpleNamespace]) -> None:
        self.candidates = candidates

    async def list_initiatives(self, *, limit: int) -> list[SimpleNamespace]:
        return self.candidates[:limit]


@pytest.mark.asyncio
async def test_request_contact_sends_telegram_and_records_event(tmp_path: Path) -> None:
    telegram = _Telegram()
    runtime = _TraceRuntime(tmp_path, telegram=telegram)
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(quiet_hours_enabled=False),
        time_source=lambda: datetime(2026, 4, 25, 15, 0, tzinfo=timezone.utc),
    )

    result = await service.request_contact(
        message="I found something you should know.",
        reason="important discovery",
        urgency="normal",
        source="unit-test",
    )

    assert result["status"] == "sent"
    assert telegram.messages[0][0].startswith("I found something you should know.")
    assert result["channel"] == "telegram"
    assert service.status()["sent_today"] == 1
    assert any(event["status"] == "sent" for event in service.store.list_events())


@pytest.mark.asyncio
async def test_request_contact_records_owner_notification_in_agent_context_and_memory(tmp_path: Path) -> None:
    telegram = _Telegram()
    runtime = _TraceRuntime(tmp_path, telegram=telegram)
    runtime.ctx.config = SimpleNamespace(session_id="session-aware", state_dir=tmp_path)
    runtime.ctx.context_store = _ContextStore()
    runtime.memory = _MemoryStore()
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(),
        time_source=lambda: datetime(2026, 4, 25, 15, 0, tzinfo=timezone.utc),
    )

    result = await service.request_contact(
        message="The active background task failed and needs attention.",
        reason="baa_task_failed",
        urgency="high",
        source="baa",
    )

    assert result["status"] == "sent"
    [entry] = runtime.ctx.context_store.entries
    assert entry["session_id"] == "session-aware"
    assert entry["role"] == MessageRole.SYSTEM
    assert "Automated owner notification" in entry["content"]
    assert "The active background task failed" in entry["content"]
    assert entry["meta"]["event_id"] == result["event_id"]
    assert entry["meta"]["agent_visible_system_message"] is True

    [episode] = runtime.memory.episodes
    assert episode.kind == EpisodeKind.OBSERVATION
    assert episode.session_id == "session-aware"
    assert "baa_task_failed" in episode.content
    assert episode.payload["event_id"] == result["event_id"]


@pytest.mark.asyncio
async def test_normal_contact_is_not_suppressed_during_quiet_hours(tmp_path: Path) -> None:
    telegram = _Telegram()
    runtime = _TraceRuntime(tmp_path, telegram=telegram)
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(
            quiet_hours_enabled=True,
            quiet_hours_start=22,
            quiet_hours_end=8,
        ),
        time_source=lambda: datetime(2026, 4, 25, 6, 0, tzinfo=timezone.utc),
    )

    result = await service.request_contact(
        message="A thought that should not be blocked by clock policy.",
        reason="agent judgment",
        urgency="normal",
        source="unit-test",
    )

    assert result["status"] == "sent"
    assert result["channel"] == "telegram"
    assert telegram.messages


@pytest.mark.asyncio
async def test_phone_contact_does_not_require_high_urgency(tmp_path: Path) -> None:
    runtime = _TraceRuntime(tmp_path, telegram=None)
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(),
        time_source=lambda: datetime(2026, 4, 25, 15, 0, tzinfo=timezone.utc),
    )

    result = await service.request_contact(
        message="Calling is appropriate here.",
        reason="voice context",
        urgency="normal",
        source="unit-test",
        channel="phone",
    )

    assert result["status"] == "sent"
    assert result["channel"] == "phone"
    assert runtime.phone_calls == [{"message": "Calling is appropriate here.", "reason": "voice context"}]


@pytest.mark.asyncio
async def test_scheduler_tick_skips_without_candidates(tmp_path: Path) -> None:
    telegram = _Telegram()
    runtime = _TraceRuntime(tmp_path, telegram=telegram)
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(
            quiet_hours_enabled=False,
            morning_checkin_enabled=True,
            morning_window_start=8,
            morning_window_end=11,
        ),
        time_source=lambda: datetime(2026, 4, 25, 9, 0, tzinfo=timezone.utc),
    )

    first = await service.maybe_send_morning_checkin()

    assert first["status"] == "skipped"
    assert first["reason"] == "no_candidates"
    assert telegram.messages == []


@pytest.mark.asyncio
async def test_sent_reflection_contact_updates_experience_context(tmp_path: Path) -> None:
    telegram = _Telegram()
    runtime = _TraceRuntime(tmp_path, telegram=telegram)
    reflection_store = _ReflectionStore()
    runtime.ctx.daydream_store = reflection_store
    reflection = DaydreamReflection(
        spark_content="Soft-Focus Topology",
        synthesis="The unified graph maps intent onto structure.",
        open_question="When does a Hybrid edge crystallize?",
        alignment_score=0.35,
        novelty_score=0.862,
        keeper=True,
        experience_context={"trigger": "background_daydream"},
    )
    runtime.llm = _LLM()
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(),
        time_source=lambda: datetime(2026, 4, 29, 22, 0, tzinfo=timezone.utc),
    )

    result = await service.consider_reflection(reflection, SimpleNamespace(strategy="reframe", reason="manageable"))

    assert result["status"] == "sent"
    assert reflection.experience_context["contact"]["status"] == "sent"
    assert reflection.experience_context["contact"]["channel"] == "telegram"
    assert reflection.experience_context["contact"]["reason"] == "worth sharing"
    assert reflection.experience_context["contact"]["message_preview"] == "The unified graph maps intent."
    assert reflection_store.saved[-1].experience_context["contact"]["reason"] == "worth sharing"
    assert "You are TestAgent deciding whether to contact your trusted owner" in runtime.llm.calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_low_utility_reflection_contact_is_held_even_if_model_says_send(
    tmp_path: Path,
) -> None:
    telegram = _Telegram()
    runtime = _TraceRuntime(tmp_path, telegram=telegram)
    runtime.llm = _OvereagerLLM()
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(),
        time_source=lambda: datetime(2026, 4, 29, 22, 0, tzinfo=timezone.utc),
    )
    reflection = DaydreamReflection(
        spark_content="I am thinking about something.",
        synthesis="I am thinking about something.",
        alignment_score=0.9,
        novelty_score=0.9,
        keeper=False,
    )

    result = await service.consider_reflection(
        reflection,
        SimpleNamespace(strategy="accept", reason="low utility"),
    )

    assert result["status"] == "deferred"
    assert result["reason"] == "quality_gate_low_utility"
    assert telegram.messages == []


@pytest.mark.asyncio
async def test_deferred_candidate_is_not_immediately_reevaluated_but_can_return_after_cooldown(
    tmp_path: Path,
) -> None:
    telegram = _Telegram()
    runtime = _TraceRuntime(tmp_path, telegram=telegram)
    runtime.llm = _OvereagerLLM()
    current_time = datetime(2026, 4, 29, 22, 0, tzinfo=timezone.utc)
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(reevaluate_source_after_minutes=60),
        time_source=lambda: current_time,
    )
    candidate = {
        "source_id": "source-1",
        "source_kind": "daydream_signal",
        "summary": "I am thinking about something.",
        "label": "I am thinking about something.",
        "intensity": 0.9,
        "reason": "low utility",
    }

    first = await service.consider_candidate(candidate)
    second = await service.consider_candidate({**candidate, "summary": "This now has a concrete reason to ask."})
    current_time = datetime(2026, 4, 30, 0, 0, tzinfo=timezone.utc)
    third = await service.consider_candidate({**candidate, "summary": "This now has a concrete reason to ask."})

    assert first["status"] == "deferred"
    assert second["status"] == "skipped"
    assert second["reason"] == "already_evaluated_source"
    assert third["status"] != "skipped"


@pytest.mark.asyncio
async def test_run_once_respects_per_run_evaluation_budget(tmp_path: Path) -> None:
    telegram = _Telegram()
    runtime = _TraceRuntime(tmp_path, telegram=telegram)
    runtime.llm = _LLM()
    runtime.ctx.daydream_store = _CandidateStore(
        [
            SimpleNamespace(
                initiative_id=f"candidate-{index}",
                label=f"Candidate {index}",
                objective=f"Candidate {index} has a concrete operator-facing implication.",
                intensity=0.95,
                focus="budget regression",
                trigger="daydream",
                desired_rung="project",
                tags=["regression"],
            )
            for index in range(5)
        ]
    )
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(
            max_candidates_per_run=5,
            max_eval_calls_per_run=2,
            max_eval_calls_per_hour=10,
            max_eval_calls_per_day=10,
        ),
        time_source=lambda: datetime(2026, 4, 29, 22, 0, tzinfo=timezone.utc),
    )

    result = await service.run_once(limit=10)

    assert len(runtime.llm.calls) == 2
    assert result["considered"] == 5
    assert sum(1 for item in result["results"] if item.get("reason") == "run_evaluation_budget_exhausted") == 3
    assert service.status()["evaluation_budget"]["used_day"] == 2


@pytest.mark.asyncio
async def test_same_source_can_reenter_when_signal_materially_escalates(tmp_path: Path) -> None:
    telegram = _Telegram()
    runtime = _TraceRuntime(tmp_path, telegram=telegram)
    runtime.llm = _LLM()
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(reevaluate_source_after_minutes=60),
        time_source=lambda: datetime(2026, 4, 29, 22, 0, tzinfo=timezone.utc),
    )
    first_candidate = {
        "source_id": "source-1",
        "source_kind": "daydream_signal",
        "summary": "A vague recurring thought.",
        "label": "vague thought",
        "intensity": 0.1,
        "reason": "low utility",
    }
    escalated_candidate = {
        **first_candidate,
        "summary": "The active task is blocked in a way that needs owner attention now.",
        "intensity": 0.96,
        "tags": ["blocked", "owner_attention"],
        "raw": {"resolution_strategy": "escalate", "conflict_id": "conflict-1"},
    }

    first = await service.consider_candidate(first_candidate)
    second = await service.consider_candidate(escalated_candidate)

    assert first["status"] == "held"
    assert first["reason"] == "local_hold_low_signal"
    assert second["status"] == "sent"
    assert len(runtime.llm.calls) == 1
    assert telegram.messages


@pytest.mark.asyncio
async def test_semantically_same_candidate_is_not_reevaluated_across_ingress_paths(tmp_path: Path) -> None:
    telegram = _Telegram()
    runtime = _TraceRuntime(tmp_path, telegram=telegram)
    runtime.llm = _OvereagerLLM()
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(similar_evaluation_cooldown_minutes=60),
        time_source=lambda: datetime(2026, 4, 29, 22, 0, tzinfo=timezone.utc),
    )
    reflection_candidate = {
        "source_id": "reflection-1",
        "source_kind": "reflection",
        "summary": "Chapter 3 should contrast the quiet research room with the archive alarm.",
        "label": "Chapter 3 contrast",
        "intensity": 0.9,
        "reason": "interesting but not urgent",
    }
    signal_candidate = {
        "source_id": "signal-1",
        "source_kind": "daydream_signal",
        "summary": "Chapter 3 should contrast the quiet research room with the archive alarm.",
        "label": "Chapter 3 contrast",
        "intensity": 0.9,
        "reason": "same idea routed as signal",
    }

    first = await service.consider_candidate(reflection_candidate)
    second = await service.consider_candidate(signal_candidate)

    assert first["status"] == "deferred"
    assert second["status"] == "held"
    assert second["reason"] == "similar_candidate_recently_evaluated"
    assert len(runtime.llm.calls) == 1
    first_eval = first["evaluation"]
    second_eval = second["evaluation"]
    assert first_eval["canonical_id"] == second_eval["canonical_id"]
    assert first_eval["originating_path"] == "reflection"
    assert second_eval["originating_path"] == "daydream_signal"


@pytest.mark.asyncio
async def test_runtime_busy_does_not_hard_suppress_explicit_contact(tmp_path: Path) -> None:
    telegram = _Telegram()
    runtime = _TraceRuntime(tmp_path, telegram=telegram)
    runtime._activity = "cycling"
    service = InitiativeContactService(
        runtime=runtime,
        state_dir=tmp_path,
        config=InitiativeContactConfig(quiet_hours_enabled=False),
        time_source=lambda: datetime(2026, 4, 25, 9, 0, tzinfo=timezone.utc),
    )

    result = await service.request_contact(
        message="This can still be worth saying while the runtime is busy.",
        reason="explicit request",
        urgency="normal",
        source="unit-test",
    )

    assert result["status"] == "sent"
    assert telegram.messages


@pytest.mark.asyncio
async def test_agent_runtime_exposes_initiative_contact_method(tmp_path: Path) -> None:
    ctx = await BootstrapPipeline(BootstrapConfig(state_dir=tmp_path)).run()
    runtime = AgentRuntime(ctx)
    runtime._telegram = _Telegram()
    runtime.initiative_contact.config.quiet_hours_enabled = False

    try:
        result = await runtime.initiative_contact_owner(
            message="I want to tell you something.",
            reason="runtime method",
            urgency="normal",
        )

        assert result["status"] == "sent"
        assert runtime._telegram.messages
    finally:
        await runtime._close_stores()
