"""Live Twilio Media Streams support for the phone bridge."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import re
import subprocess
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
from xml.sax.saxutils import escape

from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

from opencas.api.voice_service import synthesize_speech, transcribe_audio
from opencas.api.chat_service import chat_upload_dir
from opencas.phone_config import normalize_phone_number
from opencas.phone_lane_policy import resolve_menu_route, resolve_stream_start
from opencas.phone_session_state import PhoneSessionMachine

_SAMPLE_RATE = 8000
_SAMPLE_WIDTH = 2
_CHANNELS = 1
_WAIT_TONE_REPEAT_SECONDS = 1.8

logger = logging.getLogger(__name__)


def build_connect_stream_twiml(websocket_url: str, parameters: Mapping[str, str]) -> str:
    """Build TwiML that upgrades an owner call into a bidirectional media stream."""

    params = "".join(
        f'<Parameter name="{escape(str(name))}" value="{escape(str(value))}"/>'
        for name, value in parameters.items()
        if value is not None and str(value) != ""
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{escape(websocket_url)}">{params}</Stream>'
        "</Connect>"
        "</Response>"
    )


def build_wait_tone_mulaw() -> bytes:
    """Generate a short in-memory wait tone for live phone replies."""

    pcm = bytearray()
    pcm.extend(_sine_pcm(frequency_hz=880, duration_seconds=0.08, amplitude=0.22))
    pcm.extend(_silence_pcm(0.09))
    pcm.extend(_sine_pcm(frequency_hz=660, duration_seconds=0.08, amplitude=0.22))
    pcm.extend(_silence_pcm(0.52))
    return pcm_to_mulaw(bytes(pcm))


def _sine_pcm(*, frequency_hz: float, duration_seconds: float, amplitude: float) -> bytes:
    total_samples = max(1, int(_SAMPLE_RATE * duration_seconds))
    scale = int(32767 * max(0.0, min(amplitude, 1.0)))
    frames = bytearray()
    for index in range(total_samples):
        value = int(scale * math.sin((2.0 * math.pi * frequency_hz * index) / _SAMPLE_RATE))
        frames.extend(int(value).to_bytes(2, byteorder="little", signed=True))
    return bytes(frames)


def _silence_pcm(duration_seconds: float) -> bytes:
    total_samples = max(1, int(_SAMPLE_RATE * duration_seconds))
    return b"\x00\x00" * total_samples


def mulaw_to_wav_bytes(payload: bytes) -> bytes:
    pcm = mulaw_to_pcm(payload)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(_CHANNELS)
        handle.setsampwidth(_SAMPLE_WIDTH)
        handle.setframerate(_SAMPLE_RATE)
        handle.writeframes(pcm)
    return buffer.getvalue()


def transcode_audio_file_to_mulaw(audio_path: Path) -> bytes:
    """Convert a synthesized audio file into raw mulaw/8000 audio for Twilio."""

    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ar",
            str(_SAMPLE_RATE),
            "-ac",
            str(_CHANNELS),
            "-f",
            "mulaw",
            "-",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(stderr or "ffmpeg mulaw transcode failed")
    return bytes(completed.stdout)


@dataclass
class PhoneMediaStreamSession:
    """One live phone session over Twilio bidirectional Media Streams."""

    websocket: WebSocket
    service: Any
    caller: Any | None = None
    call_sid: str = ""
    intro_message: str = ""
    call_token: Optional[str] = None
    stream_sid: str = ""
    mode: str = "owner"
    input_buffer: bytearray = field(default_factory=bytearray)
    heard_speech: bool = False
    last_speech_at: float = 0.0
    assistant_playing: bool = False
    pending_marks: set[str] = field(default_factory=set)
    closed: bool = False
    employer_mode_active: bool = False
    monitor_task: Optional[asyncio.Task[None]] = None
    processing_task: Optional[asyncio.Task[None]] = None
    waiting_task: Optional[asyncio.Task[None]] = None
    custom_parameters: dict[str, str] = field(default_factory=dict)
    owner_pin_digits: str = ""
    owner_pin_attempts: int = 0
    active_menu_key: str = ""
    state_machine: PhoneSessionMachine | None = None
    pre_speech_buffer: bytearray = field(default_factory=bytearray)
    recorded_utterances: list[bytes] = field(default_factory=list)
    close_reason: str = ""

    async def run(self) -> None:
        await self.websocket.accept()
        self.monitor_task = asyncio.create_task(self._monitor_turn_completion())
        try:
            while True:
                message = await self.websocket.receive_text()
                event = json.loads(message)
                event_type = str(event.get("event") or "").strip().lower()
                try:
                    if event_type == "start":
                        await self._handle_start(event)
                    elif event_type == "media":
                        await self._handle_media(event)
                    elif event_type == "dtmf":
                        await self._handle_dtmf(event)
                    elif event_type == "mark":
                        self._handle_mark(event)
                    elif event_type == "stop":
                        self.close_reason = "twilio_stop"
                        break
                except Exception:
                    self.service.trace_phone_event(
                        "phone_stream_event_error",
                        caller=self.caller,
                        call_sid=self.call_sid,
                        event_type=event_type,
                        stream_sid=self.stream_sid,
                        mode=self.mode,
                    )
                    logger.exception(
                        "Phone media stream event failed",
                        extra={
                            "event_type": event_type,
                            "call_sid": self.call_sid,
                            "stream_sid": self.stream_sid,
                            "mode": self.mode,
                        },
                    )
                    if self.state_machine is not None:
                        self._apply_machine_transition(
                            self.state_machine.on_error(reason="phone_stream_event_error")
                        )
                    try:
                        await self._clear_audio()
                    except Exception:
                        logger.exception("Phone media stream clear-audio recovery failed")
        except WebSocketDisconnect:
            if not self.close_reason:
                self.close_reason = "websocket_disconnect"
        finally:
            self.closed = True
            close_diagnostics = {}
            if self.state_machine is not None:
                self._apply_machine_transition(self.state_machine.on_close(reason=self.close_reason or "session_finished"))
                close_diagnostics = self.state_machine.diagnostics()
            self.service.trace_phone_event(
                "phone_stream_closed",
                caller=self.caller,
                call_sid=self.call_sid,
                stream_sid=self.stream_sid,
                mode=self.mode,
                reason=self.close_reason or "session_finished",
                employer_mode_active=self.employer_mode_active,
                current_state=close_diagnostics.get("current_state"),
                visited_states=close_diagnostics.get("visited_states"),
                terminal_action=close_diagnostics.get("terminal_action"),
                hangup_class=close_diagnostics.get("hangup_class"),
                phase_durations=close_diagnostics.get("phase_durations"),
            )
            await self._cancel_task(self.waiting_task)
            await self._cancel_task(self.processing_task)
            await self._cancel_task(self.monitor_task)
            if self.caller is not None and self.employer_mode_active:
                try:
                    await self.service.finalize_employer_call(
                        caller=self.caller,
                        call_sid=self.call_sid,
                        caller_audio_mulaw=self._combined_recorded_audio(),
                    )
                except Exception:
                    pass
            await self._close_websocket()

    async def _handle_start(self, event: Mapping[str, Any]) -> None:
        start = event.get("start") if isinstance(event, Mapping) else None
        if isinstance(start, Mapping):
            self.stream_sid = str(start.get("streamSid") or self.stream_sid or "").strip()
            call_sid = str(start.get("callSid") or "").strip()
            if call_sid:
                self.call_sid = call_sid
            custom_parameters = start.get("customParameters")
            if isinstance(custom_parameters, Mapping):
                self.custom_parameters = {
                    str(key): str(value)
                    for key, value in custom_parameters.items()
                    if value is not None
                }
        self.mode = str(self.custom_parameters.get("streamMode") or self.mode or "owner").strip() or "owner"
        self.call_token = str(self.custom_parameters.get("callToken") or self.call_token or "").strip() or None
        caller_number = normalize_phone_number(self.custom_parameters.get("callerNumber"))
        caller_display_name = str(self.custom_parameters.get("displayName") or "").strip() or None
        self.intro_message = str(self.custom_parameters.get("introMessage") or self.intro_message or "").strip()
        start_decision = resolve_stream_start(
            self.service,
            stream_mode=self.mode,
            caller_number=caller_number,
            display_name=caller_display_name,
            call_token=self.call_token,
            intro_message=self.intro_message,
        )
        if start_decision is None:
            self.close_reason = "caller_resolution_failed"
            self.closed = True
            await self._close_websocket()
            return
        self.caller = start_decision.caller
        self.mode = start_decision.stream_mode
        self.active_menu_key = start_decision.active_menu_key
        self.employer_mode_active = start_decision.employer_mode_active
        self.state_machine = PhoneSessionMachine(
            mode=self.mode,
            active_menu_key=self.active_menu_key,
        )
        self._trace_state_transition(self.state_machine.diagnostics().get("current_state"), reason="call_started")
        self.service.trace_phone_event(
            "phone_stream_started",
            caller=self.caller,
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            mode=self.mode,
            active_menu_key=self.active_menu_key or None,
        )
        greeting = start_decision.greeting
        await self._speak_text(greeting)

    async def _handle_media(self, event: Mapping[str, Any]) -> None:
        media = event.get("media") if isinstance(event, Mapping) else None
        if not isinstance(media, Mapping):
            return
        payload = str(media.get("payload") or "").strip()
        if not payload:
            return
        chunk = base64.b64decode(payload)
        if not chunk:
            return

        is_speech = self._chunk_contains_speech(chunk)
        if is_speech and (self.assistant_playing or self.processing_task is not None):
            await self._interrupt_for_barge_in()

        if is_speech:
            if not self.heard_speech and self.pre_speech_buffer:
                self.input_buffer.extend(self.pre_speech_buffer)
            self.pre_speech_buffer.clear()
            self.input_buffer.extend(chunk)
            self.heard_speech = True
            self.last_speech_at = asyncio.get_running_loop().time()
            return
        if self.heard_speech:
            self.input_buffer.extend(chunk)
            return
        self._append_preroll(chunk)

    async def _handle_dtmf(self, event: Mapping[str, Any]) -> None:
        dtmf = event.get("dtmf") if isinstance(event, Mapping) else None
        if not isinstance(dtmf, Mapping):
            return
        digit = str(dtmf.get("digit") or "").strip()
        if not digit:
            return
        self.service.trace_phone_event(
            "phone_stream_dtmf",
            caller=self.caller,
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            mode=self.mode,
            digit=digit,
            active_menu_key=self.active_menu_key or None,
        )
        if self.state_machine is not None:
            self._apply_machine_transition(self.state_machine.on_dtmf())
        if self.assistant_playing or self.processing_task is not None:
            await self._interrupt_for_barge_in()
        if self.mode == "owner_pin":
            await self._handle_owner_pin_digit(digit)
            return
        if self._is_menu_mode():
            await self._apply_menu_choice(self.service.classify_menu_digit(self.active_menu_key, digit))

    def _handle_mark(self, event: Mapping[str, Any]) -> None:
        mark = event.get("mark") if isinstance(event, Mapping) else None
        if not isinstance(mark, Mapping):
            return
        name = str(mark.get("name") or "").strip()
        if name:
            self.pending_marks.discard(name)
        if not self.pending_marks:
            self.assistant_playing = False

    def _trace_state_transition(self, state: str | None, *, reason: str, previous_state: str | None = None) -> None:
        self.service.trace_phone_event(
            "phone_session_state_changed",
            caller=self.caller,
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            mode=self.mode,
            active_menu_key=self.active_menu_key or None,
            from_state=previous_state,
            to_state=state,
            reason=reason,
        )

    def _apply_machine_transition(self, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        self.mode = str(payload.get("mode") or self.mode).strip() or self.mode
        self.active_menu_key = str(payload.get("active_menu_key") or self.active_menu_key).strip()
        self._trace_state_transition(
            str(payload.get("to_state") or "").strip() or None,
            reason=str(payload.get("reason") or "").strip() or "state_changed",
            previous_state=str(payload.get("from_state") or "").strip() or None,
        )

    async def _monitor_turn_completion(self) -> None:
        loop = asyncio.get_running_loop()
        while not self.closed:
            await asyncio.sleep(0.1)
            if self.processing_task is not None and self.processing_task.done():
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                finally:
                    self.processing_task = None
            if self.assistant_playing or self.processing_task is not None or not self.heard_speech:
                continue
            if len(self.input_buffer) < self._min_utterance_bytes():
                continue
            if (loop.time() - self.last_speech_at) < self._silence_gap_seconds():
                continue
            utterance = bytes(self.input_buffer)
            self.input_buffer.clear()
            self.heard_speech = False
            self.pre_speech_buffer.clear()
            self.processing_task = asyncio.create_task(self._process_utterance(utterance))

    async def _process_utterance(self, payload: bytes) -> None:
        if payload:
            self.recorded_utterances.append(payload)
        transcript = await self._transcribe(payload)
        if not transcript:
            self.service.trace_phone_event(
                "phone_stream_transcription_empty",
                caller=self.caller,
                call_sid=self.call_sid,
                stream_sid=self.stream_sid,
                mode=self.mode,
            )
            return
        self.service.trace_phone_event(
            "phone_stream_transcribed",
            caller=self.caller,
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            mode=self.mode,
            transcript_preview=transcript[:120],
            transcript_length=len(transcript),
        )
        if self.state_machine is not None:
            self._apply_machine_transition(self.state_machine.on_transcribed())

        if self._is_menu_mode():
            await self._apply_menu_choice(
                self.service.classify_menu_transcript(self.active_menu_key, transcript)
            )
            return
        if self.mode == "owner_pin":
            await self._speak_text("Please use the keypad to enter the six digit owner PIN.")
            return

        self.waiting_task = asyncio.create_task(self._play_wait_tone_loop())
        try:
            workspace_mode = self.mode != "owner"
            if self.state_machine is not None:
                self._apply_machine_transition(self.state_machine.on_reply_started(workspace=workspace_mode))
            if self.mode == "owner":
                response_text = await self.service.generate_owner_live_stream_reply(
                    caller=self.caller,
                    transcript=transcript,
                    call_sid=self.call_sid,
                )
            else:
                response_text = await self.service.generate_workspace_live_stream_reply(
                    caller=self.caller,
                    transcript=transcript,
                    call_sid=self.call_sid,
                )
            if self.state_machine is not None:
                self._apply_machine_transition(self.state_machine.on_reply_completed(workspace=workspace_mode))
        finally:
            await self._cancel_task(self.waiting_task)
            self.waiting_task = None
            await self._clear_audio()

        if response_text:
            await self._speak_text(response_text)

    async def _apply_menu_choice(self, choice: str | None) -> None:
        option = self.service.resolve_menu_option(self.active_menu_key, choice)
        if option is None:
            self.service.trace_phone_event(
                "phone_stream_menu_unmatched",
                caller=self.caller,
                call_sid=self.call_sid,
                stream_sid=self.stream_sid,
                mode=self.mode,
                active_menu_key=self.active_menu_key or None,
                choice=choice or None,
            )
            if self.state_machine is not None:
                self._apply_machine_transition(self.state_machine.on_menu_unmatched())
            await self._speak_text(self.service.menu_reprompt(self.active_menu_key))
            return
        self.service.trace_phone_event(
            "phone_stream_menu_choice",
            caller=self.caller,
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            mode=self.mode,
            active_menu_key=self.active_menu_key or None,
            choice=choice or option.key,
            option_key=option.key,
            action=option.action,
            digit=option.digit,
        )
        decision = await resolve_menu_route(
            self.service,
            option=option,
            caller=self.caller,
            active_menu_key=self.active_menu_key,
        )
        if decision.next_caller is not None:
            self.caller = decision.next_caller
        self.mode = decision.next_mode
        self.active_menu_key = decision.next_menu_key
        if decision.employer_mode_active is not None:
            self.employer_mode_active = decision.employer_mode_active
        if self.state_machine is not None:
            self._apply_machine_transition(
                self.state_machine.on_menu_choice(
                    action=str(option.action or ""),
                    next_mode=self.mode,
                    next_menu_key=self.active_menu_key,
                )
            )
        await self._speak_text(decision.announcement)
        if decision.hangup_after_speech:
            await self._wait_for_playback_completion()
            self.close_reason = decision.terminal_action or "menu_terminal"
            self.closed = True
            await self._close_websocket()

    async def _handle_owner_pin_digit(self, digit: str) -> None:
        if not digit.isdigit():
            return
        self.owner_pin_digits = (self.owner_pin_digits + digit)[:6]
        if len(self.owner_pin_digits) < 6:
            return
        candidate = self.owner_pin_digits
        self.owner_pin_digits = ""
        if self.service.validate_owner_pin(candidate):
            self.service.trace_phone_event(
                "phone_stream_owner_pin_verified",
                caller=self.caller,
                call_sid=self.call_sid,
                stream_sid=self.stream_sid,
                mode=self.mode,
                attempts=self.owner_pin_attempts + 1,
            )
            success = self.service.owner_pin_success_message()
            if self.service.owner_menu_enabled():
                self.mode = "owner_menu"
                self.active_menu_key = self.service.owner_menu_key() or ""
                greeting = self.service.menu_prompt(self.active_menu_key)
            else:
                self.mode = "owner"
                self.active_menu_key = ""
                greeting = self.service.default_stream_greeting(self.caller, stream_mode="owner")
            if self.state_machine is not None:
                self._apply_machine_transition(
                    self.state_machine.on_owner_pin_verified(
                        next_mode=self.mode,
                        next_menu_key=self.active_menu_key,
                    )
                )
            await self._speak_text(f"{success} {greeting}".strip())
            return
        self.owner_pin_attempts += 1
        self.service.trace_phone_event(
            "phone_stream_owner_pin_rejected",
            caller=self.caller,
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            mode=self.mode,
            attempts=self.owner_pin_attempts,
        )
        if self.state_machine is not None:
            self._apply_machine_transition(
                self.state_machine.on_owner_pin_rejected(attempts=self.owner_pin_attempts)
            )
        if self.owner_pin_attempts >= 3:
            await self._speak_text(self.service.owner_pin_failure_message())
            await self._wait_for_playback_completion()
            self.close_reason = "owner_pin_failed"
            self.closed = True
            await self._close_websocket()
            return
        await self._speak_text(self.service.owner_pin_retry_prompt())

    async def _transcribe(self, payload: bytes) -> str:
        wav_bytes = mulaw_to_wav_bytes(payload)
        result = await transcribe_audio(
            chat_upload_dir(self.service.runtime),
            audio_bytes=wav_bytes,
            filename=f"phone_input_{uuid.uuid4().hex}.wav",
            media_type="audio/wav",
            prefer_local=False,
            language_code="en",
            state_dir=self.service.runtime.ctx.config.state_dir,
            config=self.service.config,
        )
        return str(result.text or "").strip()

    async def _play_wait_tone_loop(self) -> None:
        tone = build_wait_tone_mulaw()
        while True:
            await self._send_mulaw_audio(tone, mark_name=f"wait-{uuid.uuid4().hex}")
            await asyncio.sleep(_WAIT_TONE_REPEAT_SECONDS)

    async def _speak_text(self, text: str) -> None:
        normalized = str(text or "").strip()
        if not normalized:
            return
        normalized = self._normalize_spoken_text(normalized)
        attempts = (False, True)
        last_error: Exception | None = None
        for prefer_local in attempts:
            try:
                await self._speak_text_once(normalized, prefer_local=prefer_local)
                return
            except Exception as exc:
                self.service.trace_phone_event(
                    "phone_stream_tts_attempt_failed",
                    caller=self.caller,
                    call_sid=self.call_sid,
                    stream_sid=self.stream_sid,
                    mode=self.mode,
                    provider="local" if prefer_local else "hosted",
                    error=str(exc),
                )
                last_error = exc
                logger.warning(
                    "Phone speech attempt failed",
                    extra={
                        "call_sid": self.call_sid,
                        "stream_sid": self.stream_sid,
                        "mode": self.mode,
                        "prefer_local": prefer_local,
                        "error": str(exc),
                    },
                )
                if self.state_machine is not None:
                    self._apply_machine_transition(
                        self.state_machine.on_error(reason="phone_stream_tts_attempt_failed")
                    )
        self.pending_marks.clear()
        self.assistant_playing = False
        if last_error is not None:
            self.service.trace_phone_event(
                "phone_stream_tts_failed",
                caller=self.caller,
                call_sid=self.call_sid,
                stream_sid=self.stream_sid,
                mode=self.mode,
                error=str(last_error),
            )
            logger.exception(
                "Phone speech failed after hosted and local attempts",
                exc_info=last_error,
                extra={
                    "call_sid": self.call_sid,
                    "stream_sid": self.stream_sid,
                    "mode": self.mode,
                },
            )
            if self.state_machine is not None:
                self._apply_machine_transition(self.state_machine.on_error(reason="phone_stream_tts_failed"))

    async def _speak_text_once(self, text: str, *, prefer_local: bool) -> None:
        voice_result = await synthesize_speech(
            chat_upload_dir(self.service.runtime),
            text=text,
            prefer_local=prefer_local,
            expressive=self.service.phone_tts_expressive() and not prefer_local,
            state_dir=self.service.runtime.ctx.config.state_dir,
            config=self.service.config,
        )
        audio_path = Path(str(voice_result.audio_attachment.get("path") or "")).expanduser()
        mulaw = await asyncio.to_thread(transcode_audio_file_to_mulaw, audio_path)
        self.service.trace_phone_event(
            "phone_stream_tts_sent",
            caller=self.caller,
            call_sid=self.call_sid,
            stream_sid=self.stream_sid,
            mode=self.mode,
            provider="local" if prefer_local else "hosted",
            audio_path=str(audio_path),
        )
        if self.state_machine is not None:
            self._apply_machine_transition(self.state_machine.on_tts_sent())
        await self._send_mulaw_audio(mulaw, mark_name=f"speech-{uuid.uuid4().hex}")

    async def _send_mulaw_audio(self, payload: bytes, *, mark_name: str) -> None:
        if not self.stream_sid or not payload:
            return
        self.assistant_playing = True
        chunk_size = 1600
        for offset in range(0, len(payload), chunk_size):
            chunk = payload[offset : offset + chunk_size]
            await self.websocket.send_text(
                json.dumps(
                    {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": base64.b64encode(chunk).decode("ascii")},
                    }
                )
            )
        self.pending_marks.add(mark_name)
        await self.websocket.send_text(
            json.dumps(
                {
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": mark_name},
                }
            )
        )

    async def _interrupt_for_barge_in(self) -> None:
        await self._cancel_task(self.waiting_task)
        self.waiting_task = None
        await self._cancel_task(self.processing_task)
        self.processing_task = None
        await self._clear_audio()

    async def _clear_audio(self) -> None:
        if not self.stream_sid:
            return
        self.pending_marks.clear()
        self.assistant_playing = False
        await self.websocket.send_text(
            json.dumps({"event": "clear", "streamSid": self.stream_sid})
        )

    async def _cancel_task(self, task: Optional[asyncio.Task[Any]]) -> None:
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _close_websocket(self) -> None:
        try:
            await self.websocket.close()
        except Exception:
            pass

    async def _wait_for_playback_completion(self, *, timeout_seconds: float = 6.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        while self.pending_marks and asyncio.get_running_loop().time() < deadline and not self.closed:
            await asyncio.sleep(0.05)

    def _chunk_contains_speech(self, chunk: bytes) -> bool:
        try:
            pcm = mulaw_to_pcm(chunk)
            if not pcm:
                return False
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
            rms = int(np.sqrt(np.mean(samples * samples))) if samples.size else 0
            return rms >= self._speech_rms_threshold()
        except Exception:
            return False

    def _is_menu_mode(self) -> bool:
        return self.mode in {"screening", "owner_menu"} and bool(self.active_menu_key)

    def _initial_menu_key(self) -> str:
        if self.mode == "owner_menu":
            return self.service.owner_menu_key() or self.service.default_menu_key()
        if self.mode == "screening":
            return self.service.default_menu_key()
        return ""

    def _append_preroll(self, chunk: bytes) -> None:
        max_bytes = self._preroll_max_bytes()
        if max_bytes <= 0:
            return
        self.pre_speech_buffer.extend(chunk)
        if len(self.pre_speech_buffer) > max_bytes:
            del self.pre_speech_buffer[: len(self.pre_speech_buffer) - max_bytes]

    def _preroll_max_bytes(self) -> int:
        return max(0, int((_SAMPLE_RATE * self.service.config.phone_preroll_ms) / 1000))

    def _min_utterance_bytes(self) -> int:
        return int(self.service.config.phone_min_utterance_bytes)

    def _silence_gap_seconds(self) -> float:
        return float(self.service.config.phone_silence_gap_seconds)

    def _speech_rms_threshold(self) -> int:
        return int(self.service.config.phone_speech_rms_threshold)

    def _combined_recorded_audio(self) -> bytes | None:
        if not self.recorded_utterances:
            return None
        separator = pcm_to_mulaw(_silence_pcm(0.16))
        chunks: list[bytes] = []
        for index, payload in enumerate(self.recorded_utterances):
            if index > 0 and separator:
                chunks.append(separator)
            chunks.append(payload)
        return b"".join(chunks)

    @staticmethod
    def _normalize_spoken_text(text: str) -> str:
        normalized = str(text or "").replace("\r", "")
        normalized = re.sub(
            r"```[\s\S]*?```",
            " There are technical details behind that, and I can explain them in plain English or go deeper if you want. ",
            normalized,
        )
        normalized = re.sub(r"`([^`]+)`", r"\1", normalized)
        lines: list[str] = []
        for line in normalized.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            stripped = re.sub(r"^#{1,6}\s*", "", stripped)
            stripped = re.sub(r"^[-*]\s+", "", stripped)
            stripped = re.sub(r"^\d+\.\s+", "", stripped)
            lines.append(stripped)
        spoken = " ".join(lines)
        return re.sub(r"\s+", " ", spoken).strip()


def mulaw_to_pcm(payload: bytes) -> bytes:
    frames = bytearray()
    for raw in payload:
        sample = _mulaw_byte_to_pcm(raw)
        frames.extend(int(sample).to_bytes(2, byteorder="little", signed=True))
    return bytes(frames)


def pcm_to_mulaw(pcm_bytes: bytes) -> bytes:
    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:-1]
    encoded = bytearray()
    for offset in range(0, len(pcm_bytes), 2):
        sample = int.from_bytes(pcm_bytes[offset : offset + 2], byteorder="little", signed=True)
        encoded.append(_pcm_to_mulaw_byte(sample))
    return bytes(encoded)


def _mulaw_byte_to_pcm(value: int) -> int:
    value = (~value) & 0xFF
    sign = value & 0x80
    exponent = (value >> 4) & 0x07
    mantissa = value & 0x0F
    sample = ((mantissa << 3) + 0x84) << exponent
    sample -= 0x84
    return -sample if sign else sample


def _pcm_to_mulaw_byte(sample: int) -> int:
    bias = 0x84
    clip = 32635
    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample
    if sample > clip:
        sample = clip
    sample += bias
    exponent = 7
    exp_mask = 0x4000
    while exponent > 0 and not (sample & exp_mask):
        exponent -= 1
        exp_mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


OwnerPhoneMediaStreamSession = PhoneMediaStreamSession
