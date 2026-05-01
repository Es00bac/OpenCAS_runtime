"""Derived phone call state helpers for dashboard diagnostics."""

from __future__ import annotations

from datetime import datetime
import time
from typing import Any, Mapping


_MENU_STATES = {"screening", "owner_menu"}
_WORKSPACE_STATES = {"workspace_live", "workspace_assistant", "employer"}


class PhoneSessionMachine:
    """Source-of-truth state transitions for one live phone session."""

    def __init__(self, *, mode: str, active_menu_key: str = "") -> None:
        self.mode = str(mode or "").strip().lower() or "owner"
        self.active_menu_key = str(active_menu_key or "").strip()
        self.current_state = _initial_state({"mode": self.mode})
        self.visited_states: list[str] = [self.current_state] if self.current_state else []
        self.state_timeline: list[dict[str, Any]] = []
        self.terminal_action: str | None = None
        self.hangup_reason: str | None = None
        self._started_at = time.monotonic()
        self._marks: dict[str, float] = {"call_started": self._started_at}
        self._record_state(self.current_state, reason="call_started")

    def _mark(self, key: str) -> None:
        self._marks.setdefault(key, time.monotonic())

    def _duration(self, start_key: str, end_key: str) -> float | None:
        start = self._marks.get(start_key)
        end = self._marks.get(end_key)
        if start is None or end is None:
            return None
        return round(end - start, 3)

    def _record_state(self, next_state: str | None, *, reason: str) -> dict[str, Any] | None:
        cleaned = str(next_state or "").strip()
        if not cleaned:
            return None
        previous = self.current_state
        if cleaned == previous and self.state_timeline:
            return None
        self.current_state = cleaned
        if cleaned not in self.visited_states:
            self.visited_states.append(cleaned)
        self.state_timeline.append(
            {
                "offset_seconds": round(time.monotonic() - self._started_at, 3),
                "state": cleaned,
                "reason": reason,
            }
        )
        return {
            "from_state": previous if self.state_timeline[:-1] else None,
            "to_state": cleaned,
            "reason": reason,
            "mode": self.mode,
            "active_menu_key": self.active_menu_key or None,
        }

    def on_dtmf(self) -> dict[str, Any] | None:
        self._mark("first_dtmf")
        if self.current_state == "owner_pin":
            return self._record_state("owner_pin_entry", reason="dtmf_entered")
        if self.current_state in _MENU_STATES or self.current_state in {"menu_navigation", "menu_retry"}:
            return self._record_state("menu_input", reason="dtmf_entered")
        return None

    def on_menu_unmatched(self) -> dict[str, Any] | None:
        return self._record_state("menu_retry", reason="menu_unmatched")

    def on_menu_choice(self, *, action: str, next_mode: str | None = None, next_menu_key: str = "") -> dict[str, Any] | None:
        self._mark("first_menu_choice")
        cleaned_action = str(action or "").strip()
        if cleaned_action in {"say_then_hangup", "time_announcement"}:
            self.terminal_action = cleaned_action
        if next_mode:
            self.mode = str(next_mode).strip().lower() or self.mode
        self.active_menu_key = str(next_menu_key or "").strip()
        if cleaned_action == "workspace_assistant":
            return self._record_state("workspace_live", reason="menu_workspace_selected")
        if cleaned_action == "owner_conversation":
            return self._record_state("owner_live", reason="owner_conversation_selected")
        if cleaned_action == "submenu":
            return self._record_state("menu_navigation", reason="submenu_selected")
        if cleaned_action in {"say_then_hangup", "time_announcement"}:
            return self._record_state("terminal_prompt", reason=cleaned_action)
        return None

    def on_owner_pin_verified(self, *, next_mode: str, next_menu_key: str = "") -> dict[str, Any] | None:
        self.mode = str(next_mode).strip().lower() or self.mode
        self.active_menu_key = str(next_menu_key or "").strip()
        return self._record_state("owner_verified", reason="owner_pin_verified")

    def on_owner_pin_rejected(self, *, attempts: int) -> dict[str, Any] | None:
        if int(attempts) >= 3:
            return self._record_state("owner_pin_failed", reason="owner_pin_rejected")
        return self._record_state("owner_pin_retry", reason="owner_pin_rejected")

    def on_transcribed(self) -> dict[str, Any] | None:
        self._mark("first_transcription")
        if self.current_state in _MENU_STATES or self.current_state in {"menu_input", "menu_retry", "menu_navigation"}:
            return self._record_state("menu_heard", reason="speech_captured")
        if self.current_state in _WORKSPACE_STATES or self.current_state == "workspace_live":
            return self._record_state("workspace_heard", reason="speech_captured")
        return self._record_state("owner_heard", reason="speech_captured")

    def on_reply_started(self, *, workspace: bool) -> dict[str, Any] | None:
        self._mark("first_reply_start")
        return self._record_state(
            "generating_reply",
            reason="workspace_reply_started" if workspace else "owner_reply_started",
        )

    def on_reply_completed(self, *, workspace: bool) -> dict[str, Any] | None:
        return self._record_state(
            "reply_ready",
            reason="workspace_reply_completed" if workspace else "owner_reply_completed",
        )

    def on_tts_sent(self) -> dict[str, Any] | None:
        self._mark("first_tts")
        return self._record_state("speaking", reason="tts_sent")

    def on_error(self, *, reason: str) -> dict[str, Any] | None:
        return self._record_state("error", reason=reason)

    def on_close(self, *, reason: str) -> dict[str, Any] | None:
        self.hangup_reason = str(reason or "").strip() or None
        self._mark("closed")
        return self._record_state("closed", reason=self.hangup_reason or "stream_closed")

    def phase_durations(self) -> dict[str, Any]:
        return {
            "time_to_first_dtmf_seconds": self._duration("call_started", "first_dtmf"),
            "time_to_transcription_seconds": self._duration("call_started", "first_transcription"),
            "time_to_first_menu_choice_seconds": self._duration("call_started", "first_menu_choice"),
            "time_to_first_reply_start_seconds": self._duration("call_started", "first_reply_start"),
            "time_to_first_tts_seconds": self._duration("call_started", "first_tts"),
            "reply_start_to_tts_seconds": self._duration("first_reply_start", "first_tts"),
            "total_call_seconds": self._duration("call_started", "closed"),
        }

    def diagnostics(self) -> dict[str, Any]:
        return {
            "current_state": self.current_state,
            "visited_states": list(self.visited_states),
            "state_timeline": list(self.state_timeline),
            "terminal_action": self.terminal_action,
            "hangup_reason": self.hangup_reason,
            "hangup_class": _hangup_class(self.hangup_reason),
            "phase_durations": self.phase_durations(),
        }


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _seconds_between(start_iso: str | None, end_iso: str | None) -> float | None:
    start = _parse_iso(start_iso)
    end = _parse_iso(end_iso)
    if start is None or end is None:
        return None
    return round((end - start).total_seconds(), 3)


def _initial_state(payload: Mapping[str, Any]) -> str:
    mode = str(payload.get("mode") or "").strip().lower()
    if mode == "owner_pin":
        return "owner_pin"
    if mode == "owner_menu":
        return "owner_menu"
    if mode == "screening":
        return "screening"
    if mode in {"workspace_assistant", "employer"}:
        return "workspace_live"
    if mode == "owner":
        return "owner_live"
    return mode or "unknown"


def _hangup_class(reason: str | None) -> str | None:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"twilio_stop", "websocket_disconnect"}:
        return "remote_disconnect"
    if normalized in {"owner_pin_failed", "caller_resolution_failed"}:
        return "policy_end"
    if "error" in normalized or "failed" in normalized:
        return "error"
    return "system_end"


def analyze_phone_call_timeline(timeline: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Project raw phone trace events into structured call diagnostics."""

    ordered = sorted(timeline, key=lambda item: str(item.get("timestamp") or ""))
    if not ordered:
        return {
            "current_state": None,
            "visited_states": [],
            "state_timeline": [],
            "terminal_action": None,
            "hangup_reason": None,
            "hangup_class": None,
            "phase_durations": {},
        }

    first_payload = dict(ordered[0].get("payload") or {})
    base_time = str(ordered[0].get("timestamp") or "")
    current_state = _initial_state(first_payload)
    visited_states: list[str] = [current_state] if current_state else []
    state_timeline: list[dict[str, Any]] = []
    if current_state:
        state_timeline.append(
            {"timestamp": base_time, "state": current_state, "reason": "call_started"}
        )

    hangup_reason: str | None = None
    terminal_action: str | None = None
    transcribed_at: str | None = None
    first_dtmf_at: str | None = None
    first_menu_choice_at: str | None = None
    first_reply_started_at: str | None = None
    first_tts_at: str | None = None
    closed_at: str | None = None
    phase_durations: dict[str, Any] | None = None

    def set_state(next_state: str | None, *, timestamp: str, reason: str) -> None:
        nonlocal current_state
        cleaned = str(next_state or "").strip()
        if not cleaned:
            return
        if cleaned == current_state:
            return
        current_state = cleaned
        if cleaned not in visited_states:
            visited_states.append(cleaned)
        state_timeline.append({"timestamp": timestamp, "state": cleaned, "reason": reason})

    for item in ordered:
        event = str(item.get("event") or "").strip()
        timestamp = str(item.get("timestamp") or "")
        payload = dict(item.get("payload") or {})
        if event == "phone_session_state_changed":
            next_state = str(payload.get("to_state") or "").strip()
            reason = str(payload.get("reason") or "state_changed").strip()
            if next_state:
                set_state(next_state, timestamp=timestamp, reason=reason)
        elif event == "phone_stream_dtmf":
            first_dtmf_at = first_dtmf_at or timestamp
            if current_state == "owner_pin":
                set_state("owner_pin_entry", timestamp=timestamp, reason="dtmf_entered")
            elif current_state in _MENU_STATES:
                set_state("menu_input", timestamp=timestamp, reason="dtmf_entered")
        elif event == "phone_stream_menu_unmatched":
            set_state("menu_retry", timestamp=timestamp, reason="menu_unmatched")
        elif event == "phone_stream_menu_choice":
            first_menu_choice_at = first_menu_choice_at or timestamp
            action = str(payload.get("action") or "").strip() or None
            if action:
                terminal_action = action if action in {"say_then_hangup", "time_announcement"} else terminal_action
            if action == "workspace_assistant":
                set_state("workspace_live", timestamp=timestamp, reason="menu_workspace_selected")
            elif action == "owner_conversation":
                set_state("owner_live", timestamp=timestamp, reason="owner_conversation_selected")
            elif action == "submenu":
                set_state("menu_navigation", timestamp=timestamp, reason="submenu_selected")
            elif action in {"say_then_hangup", "time_announcement"}:
                set_state("terminal_prompt", timestamp=timestamp, reason=action)
        elif event == "phone_stream_owner_pin_verified":
            set_state("owner_verified", timestamp=timestamp, reason="owner_pin_verified")
        elif event == "phone_stream_owner_pin_rejected":
            attempts = int(payload.get("attempts") or 0)
            set_state(
                "owner_pin_failed" if attempts >= 3 else "owner_pin_retry",
                timestamp=timestamp,
                reason="owner_pin_rejected",
            )
        elif event == "phone_stream_transcribed":
            transcribed_at = transcribed_at or timestamp
            if current_state in _MENU_STATES or current_state in {"menu_input", "menu_retry", "menu_navigation"}:
                set_state("menu_heard", timestamp=timestamp, reason="speech_captured")
            elif current_state in _WORKSPACE_STATES:
                set_state("workspace_heard", timestamp=timestamp, reason="speech_captured")
            else:
                set_state("owner_heard", timestamp=timestamp, reason="speech_captured")
        elif event in {"phone_owner_reply_started", "phone_workspace_reply_started"}:
            first_reply_started_at = first_reply_started_at or timestamp
            set_state(
                "generating_reply",
                timestamp=timestamp,
                reason="workspace_reply_started" if "workspace" in event else "owner_reply_started",
            )
        elif event in {"phone_owner_reply_completed", "phone_workspace_reply_completed"}:
            set_state(
                "reply_ready",
                timestamp=timestamp,
                reason="workspace_reply_completed" if "workspace" in event else "owner_reply_completed",
            )
        elif event == "phone_stream_tts_sent":
            first_tts_at = first_tts_at or timestamp
            set_state("speaking", timestamp=timestamp, reason="tts_sent")
        elif event in {"phone_stream_tts_failed", "phone_stream_event_error"}:
            set_state("error", timestamp=timestamp, reason=event)
        elif event == "phone_stream_closed":
            closed_at = timestamp
            hangup_reason = str(payload.get("reason") or "").strip() or None
            terminal_action = terminal_action or str(payload.get("terminal_action") or "").strip() or None
            close_durations = payload.get("phase_durations")
            if isinstance(close_durations, Mapping):
                phase_durations = {
                    "time_to_first_dtmf_seconds": close_durations.get("time_to_first_dtmf_seconds"),
                    "time_to_transcription_seconds": close_durations.get("time_to_transcription_seconds"),
                    "time_to_first_menu_choice_seconds": close_durations.get("time_to_first_menu_choice_seconds"),
                    "time_to_first_reply_start_seconds": close_durations.get("time_to_first_reply_start_seconds"),
                    "time_to_first_tts_seconds": close_durations.get("time_to_first_tts_seconds"),
                    "reply_start_to_tts_seconds": close_durations.get("reply_start_to_tts_seconds"),
                    "total_call_seconds": close_durations.get("total_call_seconds"),
                }
            if isinstance(payload.get("visited_states"), list) and payload.get("visited_states"):
                visited_states = [str(item) for item in payload.get("visited_states") if str(item or "").strip()]
            if isinstance(payload.get("current_state"), str) and str(payload.get("current_state") or "").strip():
                current_state = str(payload.get("current_state") or "").strip()
            set_state("closed", timestamp=timestamp, reason=hangup_reason or "stream_closed")

    phase_durations = phase_durations or {
        "time_to_first_dtmf_seconds": _seconds_between(base_time, first_dtmf_at),
        "time_to_transcription_seconds": _seconds_between(base_time, transcribed_at),
        "time_to_first_menu_choice_seconds": _seconds_between(base_time, first_menu_choice_at),
        "time_to_first_reply_start_seconds": _seconds_between(base_time, first_reply_started_at),
        "time_to_first_tts_seconds": _seconds_between(base_time, first_tts_at),
        "reply_start_to_tts_seconds": _seconds_between(first_reply_started_at, first_tts_at),
        "total_call_seconds": _seconds_between(base_time, closed_at),
    }
    return {
        "current_state": current_state,
        "visited_states": visited_states,
        "state_timeline": state_timeline,
        "terminal_action": terminal_action,
        "hangup_reason": hangup_reason,
        "hangup_class": _hangup_class(hangup_reason),
        "phase_durations": phase_durations,
    }
