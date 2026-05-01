from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.daydream import DaydreamReflection
from opencas.initiative_contact import InitiativeContactConfig, InitiativeContactService
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
        self.ctx = SimpleNamespace(daydream_store=None)
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
    async def chat_completion(self, **_: object) -> dict:
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


class _ReflectionStore:
    def __init__(self) -> None:
        self.saved: list[DaydreamReflection] = []

    async def save_reflection(self, reflection: DaydreamReflection) -> None:
        self.saved.append(reflection)


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
