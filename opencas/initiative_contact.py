"""Policy-limited owner contact initiated by the runtime itself."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel

INITIATIVE_CONTACT_CONVERSATION_ID = "99030867-fca4-4a7d-a7a5-752f3ed9345f"
ContactUrgency = Literal["low", "normal", "high", "critical"]
ContactChannel = Literal["auto", "telegram", "phone"]


class InitiativeContactConfig(BaseModel):
    """Runtime policy for proactive owner contact."""

    enabled: bool = True
    preferred_channel: ContactChannel = "auto"
    max_message_chars: int = 1800
    # Legacy compatibility fields. They are no longer hard suppression gates.
    max_daily_contacts: int = 4
    cooldown_minutes: int = 60
    quiet_hours_enabled: bool = False
    quiet_hours_start: int = 22
    quiet_hours_end: int = 8
    morning_checkin_enabled: bool = False
    morning_window_start: int = 8
    morning_window_end: int = 11
    morning_source_conversation_id: str = INITIATIVE_CONTACT_CONVERSATION_ID


class InitiativeContactStore:
    """Append-only JSONL contact log under the private state directory."""

    def __init__(self, state_dir: Path | str) -> None:
        self.path = Path(state_dir).expanduser() / "initiative_contact" / "events.jsonl"

    def append_event(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True))
            handle.write("\n")

    def list_events(self, *, limit: int = 200) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events[-max(1, int(limit)) :]


class InitiativeContactService:
    """Dispatch owner contact requests through configured trusted channels."""

    def __init__(
        self,
        *,
        runtime: Any,
        state_dir: Path | str,
        config: Optional[InitiativeContactConfig] = None,
        time_source: Optional[Callable[[], datetime]] = None,
        store: Optional[InitiativeContactStore] = None,
    ) -> None:
        self.runtime = runtime
        self.config = config or InitiativeContactConfig()
        self._time_source = time_source or (lambda: datetime.now().astimezone())
        self.store = store or InitiativeContactStore(state_dir)

    def status(self) -> dict[str, Any]:
        events = self.store.list_events(limit=200)
        now = self._now()
        return {
            "enabled": self.config.enabled,
            "preferred_channel": self.config.preferred_channel,
            "mode": "agent_judgment",
            "quiet_hours_enforced": False,
            "daily_cap_enforced": False,
            "cooldown_enforced": False,
            "sent_today": self._sent_count_for_day(events, now),
            "last_event": events[-1] if events else None,
            "event_count": len(events),
        }

    async def request_contact(
        self,
        *,
        message: str,
        reason: str = "",
        urgency: ContactUrgency = "normal",
        source: str = "runtime",
        channel: ContactChannel = "auto",
    ) -> dict[str, Any]:
        now = self._now()
        message = self._normalize_message(message)
        urgency = self._normalize_urgency(urgency)
        channel = self._normalize_channel(channel)
        if not self.config.enabled:
            return self._record(now, status="suppressed", reason="disabled", urgency=urgency, source=source)
        if not message:
            return self._record(now, status="suppressed", reason="empty_message", urgency=urgency, source=source)

        result = await self._dispatch(message=message, reason=reason, urgency=urgency, source=source, channel=channel)
        status = "sent" if result.get("sent") else "failed"
        event_reason = str(result.get("reason") or reason or "owner_contact")
        return self._record(
            now,
            status=status,
            reason=event_reason,
            urgency=urgency,
            source=source,
            channel=str(result.get("channel") or channel),
            message_preview=message[:160],
            dispatch=result,
        )

    async def maybe_send_morning_checkin(self) -> dict[str, Any]:
        """Compatibility alias for the scheduler's proactive outreach tick."""
        return await self.run_once()

    async def run_once(self, *, limit: int = 20) -> dict[str, Any]:
        """Evaluate recent daydream material and let the agent choose whether to reach out."""
        if not self.config.enabled:
            return {"status": "skipped", "reason": "disabled", "considered": 0, "sent": 0}
        candidates = await self._collect_candidates(limit=limit)
        if not candidates:
            return {"status": "skipped", "reason": "no_candidates", "considered": 0, "sent": 0}
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            if self._source_already_evaluated(str(candidate.get("source_id") or "")):
                results.append(
                    {
                        "status": "skipped",
                        "reason": "already_evaluated_source",
                        "source_id": candidate.get("source_id"),
                    }
                )
                continue
            results.append(await self.consider_candidate(candidate))
        sent = sum(1 for result in results if result.get("status") == "sent")
        return {
            "status": "sent" if sent else "skipped",
            "reason": "agent_judgment",
            "considered": len(results),
            "sent": sent,
            "results": results,
        }

    async def consider_reflection(self, reflection: Any, resolution: Any) -> dict[str, Any]:
        """Evaluate one live daydream reflection for possible owner contact."""
        candidate = {
            "source_id": str(getattr(reflection, "reflection_id", "")),
            "source_kind": "reflection",
            "label": str(getattr(reflection, "open_question", "") or getattr(reflection, "spark_content", ""))[:160],
            "summary": str(getattr(reflection, "synthesis", "") or getattr(reflection, "spark_content", "")),
            "intensity": max(
                float(getattr(reflection, "alignment_score", 0.0) or 0.0),
                float(getattr(reflection, "novelty_score", 0.0) or 0.0),
            ),
            "reason": str(getattr(resolution, "reason", "") or ""),
            "tags": list(getattr(reflection, "tension_hints", []) or []),
            "raw": {
                "resolution_strategy": str(getattr(resolution, "strategy", "") or ""),
                "conflict_id": getattr(resolution, "conflict_id", None),
                "keeper": bool(getattr(reflection, "keeper", False)),
            },
        }
        result = await self.consider_candidate(candidate)
        self._attach_contact_experience_context(reflection, result)
        await self._persist_reflection_context(reflection)
        return result

    async def consider_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Ask the model if this candidate deserves owner contact, then dispatch if chosen."""
        now = self._now()
        source_id = str(candidate.get("source_id") or "").strip()
        if not source_id:
            return {"status": "skipped", "reason": "missing_source_id"}
        if self._source_already_evaluated(source_id):
            return {"status": "skipped", "reason": "already_evaluated_source", "source_id": source_id}
        channels = await self._available_channels()
        if not channels:
            return {"status": "skipped", "reason": "no_available_channel", "source_id": source_id}

        decision = await self._decide(candidate, channels)
        if not decision.get("send"):
            return self._record(
                now,
                status="held",
                reason=str(decision.get("reason") or "agent_declined"),
                urgency=self._normalize_urgency(str(decision.get("urgency") or "normal")),
                source=str(candidate.get("source_kind") or "candidate"),
                source_id=source_id,
                channel=str(decision.get("channel") or ""),
                message_preview=str(decision.get("message") or "")[:160],
                dispatch={"decision": decision, "candidate": candidate},
            )

        message = self._normalize_message(str(decision.get("message") or self._default_candidate_message(candidate)))
        channel = self._normalize_channel(str(decision.get("channel") or "auto"))
        urgency = self._normalize_urgency(str(decision.get("urgency") or "normal"))
        result = await self._dispatch(
            message=message,
            reason=str(decision.get("reason") or candidate.get("reason") or "agent_judgment"),
            urgency=urgency,
            source=str(candidate.get("source_kind") or "candidate"),
            channel=channel,
        )
        return self._record(
            now,
            status="sent" if result.get("sent") else "failed",
            reason=str(result.get("reason") or decision.get("reason") or "agent_judgment"),
            urgency=urgency,
            source=str(candidate.get("source_kind") or "candidate"),
            source_id=source_id,
            channel=str(result.get("channel") or channel),
            message_preview=message[:160],
            dispatch={"decision": decision, "candidate": candidate, "delivery": result},
        )

    async def _dispatch(
        self,
        *,
        message: str,
        reason: str,
        urgency: ContactUrgency,
        source: str,
        channel: ContactChannel,
    ) -> dict[str, Any]:
        available = await self._available_channels()
        selected = self._select_channel(channel=channel, urgency=urgency, available=available)
        if selected not in available:
            return {"sent": False, "channel": selected, "reason": f"{selected}_unavailable"}

        if selected == "telegram":
            telegram = getattr(self.runtime, "_telegram", None)
            if telegram is not None and hasattr(telegram, "notify_owner"):
                result = await telegram.notify_owner(
                    message,
                    reason=reason,
                    urgency=urgency,
                    source=source,
                )
                sent_count = int(result.get("sent", 0) or 0) if isinstance(result, dict) else 0
                return {
                    "sent": sent_count > 0,
                    "channel": "telegram",
                    "telegram": result,
                    "reason": None if sent_count > 0 else "telegram_no_owner_chat",
                }
            return {"sent": False, "channel": "telegram", "reason": "telegram_unavailable"}

        if selected == "phone":
            caller = getattr(self.runtime, "call_owner_via_phone", None)
            if not callable(caller):
                return {"sent": False, "channel": "phone", "reason": "phone_unavailable"}
            result = await caller(message=message, reason=reason)
            return {"sent": True, "channel": "phone", "phone": result}

        return {"sent": False, "channel": selected, "reason": "unsupported_channel"}

    async def _collect_candidates(self, *, limit: int) -> list[dict[str, Any]]:
        store = getattr(getattr(self.runtime, "ctx", None), "daydream_store", None)
        if store is None:
            return []
        candidates: list[dict[str, Any]] = []
        if hasattr(store, "list_initiatives"):
            try:
                for item in await store.list_initiatives(limit=limit):
                    source_id = str(getattr(item, "initiative_id", "") or getattr(item, "spark_id", "") or "").strip()
                    if not source_id:
                        continue
                    candidates.append(
                        {
                            "source_id": source_id,
                            "source_kind": "initiative",
                            "label": str(getattr(item, "label", "") or getattr(item, "objective", "") or "")[:160],
                            "summary": str(getattr(item, "objective", "") or getattr(item, "summary", "") or getattr(item, "focus", "")),
                            "intensity": float(getattr(item, "intensity", 0.0) or 0.0),
                            "reason": str(getattr(item, "focus", "") or getattr(item, "trigger", "") or getattr(item, "desired_rung", "")),
                            "tags": list(getattr(item, "tags", []) or []),
                            "raw": {
                                "spark_id": getattr(item, "spark_id", None),
                                "rung": getattr(item, "rung", ""),
                                "desired_rung": getattr(item, "desired_rung", ""),
                                "task_id": getattr(item, "task_id", None),
                            },
                        }
                    )
            except Exception:
                pass
        if hasattr(store, "list_recent"):
            try:
                for item in await store.list_recent(limit=limit, keeper_only=None):
                    source_id = str(getattr(item, "reflection_id", "") or "").strip()
                    if not source_id:
                        continue
                    candidates.append(
                        {
                            "source_id": source_id,
                            "source_kind": "reflection",
                            "label": str(getattr(item, "open_question", "") or getattr(item, "spark_content", ""))[:160],
                            "summary": str(getattr(item, "synthesis", "") or getattr(item, "spark_content", "")),
                            "intensity": max(
                                float(getattr(item, "alignment_score", 0.0) or 0.0),
                                float(getattr(item, "novelty_score", 0.0) or 0.0),
                            ),
                            "reason": "recent daydream reflection",
                            "tags": list(getattr(item, "tension_hints", []) or []),
                            "raw": {"keeper": bool(getattr(item, "keeper", False))},
                        }
                    )
            except Exception:
                pass
        return candidates[: max(1, int(limit))]

    async def _available_channels(self) -> list[ContactChannel]:
        channels: list[ContactChannel] = []
        telegram = getattr(self.runtime, "_telegram", None)
        if telegram is not None and hasattr(telegram, "notify_owner"):
            owner_chat_ids = getattr(telegram, "_owner_chat_ids", None)
            if callable(owner_chat_ids):
                try:
                    if await owner_chat_ids():
                        channels.append("telegram")
                except Exception:
                    pass
            else:
                channels.append("telegram")

        phone_status = getattr(self.runtime, "phone_status", None)
        if callable(phone_status):
            try:
                status = await phone_status()
                owner = status.get("owner", {}) if isinstance(status, dict) else {}
                if (
                    status.get("enabled")
                    and owner.get("configured")
                    and status.get("twilio_from_number")
                    and status.get("twilio_credentials_configured")
                ):
                    channels.append("phone")
            except Exception:
                pass
        elif callable(getattr(self.runtime, "call_owner_via_phone", None)):
            channels.append("phone")
        return channels

    async def _decide(self, candidate: dict[str, Any], channels: list[ContactChannel]) -> dict[str, Any]:
        llm = getattr(self.runtime, "llm", None)
        if llm is None:
            return self._fallback_decision(candidate, channels)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the OpenCAS agent deciding whether to contact your trusted owner. "
                    "The owner explicitly allows proactive messages and phone calls at any time. "
                    "Do not apply quiet hours, daily caps, cooldowns, or a rule that silence means rejection. "
                    "If the owner is busy, later response context can teach that. "
                    "You may still choose not to contact them if this is not useful, timely, or clear. "
                    "Choose telegram for lightweight updates and phone for high-context or attention-worthy outreach. "
                    "Return strict JSON: send boolean, channel auto|telegram|phone, urgency low|normal|high|critical, reason, message."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "available_channels": channels,
                        "candidate": candidate,
                        "recent_contact_events": self.store.list_events(limit=8),
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        try:
            response = await llm.chat_completion(
                messages=messages,
                complexity="light",
                payload={"temperature": 0.25, "max_tokens": 220},
                source="initiative_contact_decision",
            )
            parsed = self._parse_json(self._extract_response_text(response))
            if not isinstance(parsed, dict):
                return self._fallback_decision(candidate, channels)
            return {
                "send": bool(parsed.get("send")),
                "channel": str(parsed.get("channel") or "auto"),
                "urgency": str(parsed.get("urgency") or "normal"),
                "reason": str(parsed.get("reason") or ""),
                "message": str(parsed.get("message") or ""),
                "raw": parsed,
            }
        except Exception:
            return self._fallback_decision(candidate, channels)

    def _fallback_decision(self, candidate: dict[str, Any], channels: list[ContactChannel]) -> dict[str, Any]:
        raw = candidate.get("raw") if isinstance(candidate.get("raw"), dict) else {}
        strategy = str(raw.get("resolution_strategy") or "")
        intensity = float(candidate.get("intensity", 0.0) or 0.0)
        should_send = strategy == "escalate" or intensity >= 0.85
        return {
            "send": should_send,
            "channel": self._select_channel(channel="auto", urgency="high" if strategy == "escalate" else "normal", available=channels),
            "urgency": "high" if strategy == "escalate" else "normal",
            "reason": "fallback_escalation" if should_send else "fallback_hold_low_intensity",
            "message": self._default_candidate_message(candidate),
            "fallback": True,
        }

    def _select_channel(
        self,
        *,
        channel: ContactChannel,
        urgency: ContactUrgency,
        available: list[ContactChannel],
    ) -> ContactChannel:
        if channel != "auto":
            return channel
        if self.config.preferred_channel != "auto" and self.config.preferred_channel in available:
            return self.config.preferred_channel
        if urgency in {"high", "critical"} and "phone" in available:
            return "phone"
        if "telegram" in available:
            return "telegram"
        if available:
            return available[0]
        return "telegram"

    def _source_already_evaluated(self, source_id: str) -> bool:
        if not source_id:
            return False
        return any(
            event.get("source_id") == source_id and event.get("status") in {"sent", "held"}
            for event in self.store.list_events(limit=1000)
        )

    def _default_candidate_message(self, candidate: dict[str, Any]) -> str:
        summary = str(candidate.get("summary") or candidate.get("label") or "").strip()
        reason = str(candidate.get("reason") or "").strip()
        if reason:
            return f"I think you should know this: {summary}\n\nWhy now: {reason}"
        return f"I think you should know this: {summary}"

    def _record(
        self,
        now: datetime,
        *,
        status: str,
        reason: str,
        urgency: ContactUrgency,
        source: str,
        source_id: str = "",
        channel: str = "",
        message_preview: str = "",
        dispatch: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": str(uuid4()),
            "created_at": now.isoformat(),
            "date": now.date().isoformat(),
            "status": status,
            "reason": reason,
            "urgency": urgency,
            "source": source,
            "source_id": source_id,
            "channel": channel,
            "message_preview": message_preview,
            "dispatch": dispatch or {},
        }
        self.store.append_event(event)
        self._trace("initiative_contact_event", event)
        return event

    def _attach_contact_experience_context(self, reflection: Any, event: dict[str, Any]) -> None:
        context = getattr(reflection, "experience_context", None)
        if not isinstance(context, dict):
            context = {}
        dispatch = event.get("dispatch") if isinstance(event, dict) else {}
        decision = dispatch.get("decision") if isinstance(dispatch, dict) else {}
        context["contact"] = {
            "status": event.get("status"),
            "created_at": event.get("created_at"),
            "channel": event.get("channel") or decision.get("channel"),
            "urgency": event.get("urgency") or decision.get("urgency"),
            "reason": event.get("reason") or decision.get("reason"),
            "message_preview": event.get("message_preview") or str(decision.get("message") or "")[:160],
        }
        try:
            reflection.experience_context = context
        except Exception:
            pass

    async def _persist_reflection_context(self, reflection: Any) -> None:
        store = getattr(getattr(self.runtime, "ctx", None), "daydream_store", None)
        save_reflection = getattr(store, "save_reflection", None)
        if not callable(save_reflection):
            return
        try:
            await save_reflection(reflection)
        except Exception as exc:
            self._trace(
                "initiative_contact_reflection_context_persist_error",
                {
                    "reflection_id": str(getattr(reflection, "reflection_id", "") or ""),
                    "error": str(exc),
                },
            )

    def _trace(self, event: str, payload: dict[str, Any]) -> None:
        tracer = getattr(self.runtime, "_trace", None)
        if callable(tracer):
            try:
                tracer(event, payload)
            except Exception:
                pass

    def _now(self) -> datetime:
        now = self._time_source()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now

    def _sent_count_for_day(self, events: Iterable[dict[str, Any]], now: datetime) -> int:
        today = now.date().isoformat()
        return sum(1 for event in events if event.get("date") == today and event.get("status") == "sent")

    def _normalize_message(self, message: str) -> str:
        text = str(message or "").strip()
        if len(text) <= self.config.max_message_chars:
            return text
        return text[: max(0, self.config.max_message_chars - 20)].rstrip() + "\n[truncated]"

    @staticmethod
    def _normalize_urgency(value: str) -> ContactUrgency:
        normalized = str(value or "normal").strip().lower()
        if normalized in {"low", "normal", "high", "critical"}:
            return normalized  # type: ignore[return-value]
        return "normal"

    @staticmethod
    def _normalize_channel(value: str) -> ContactChannel:
        normalized = str(value or "auto").strip().lower()
        if normalized in {"auto", "telegram", "phone"}:
            return normalized  # type: ignore[return-value]
        return "auto"

    def _extract_response_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return str(message.get("content") or "")
        return ""

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        text = str(content or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
