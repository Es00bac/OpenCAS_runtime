"""Owner-only live Twilio Media Streams support for the phone bridge."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
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

_SAMPLE_RATE = 8000
_SAMPLE_WIDTH = 2
_CHANNELS = 1
_MIN_UTTERANCE_BYTES = 1200
_SILENCE_GAP_SECONDS = 0.85
_SPEECH_RMS_THRESHOLD = 260
_WAIT_TONE_REPEAT_SECONDS = 1.8


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
class OwnerPhoneMediaStreamSession:
    """One live owner phone session over Twilio bidirectional Media Streams."""

    websocket: WebSocket
    service: Any
    caller: Any
    call_sid: str
    intro_message: str = ""
    call_token: Optional[str] = None
    stream_sid: str = ""
    input_buffer: bytearray = field(default_factory=bytearray)
    heard_speech: bool = False
    last_speech_at: float = 0.0
    assistant_playing: bool = False
    pending_marks: set[str] = field(default_factory=set)
    closed: bool = False
    monitor_task: Optional[asyncio.Task[None]] = None
    processing_task: Optional[asyncio.Task[None]] = None
    waiting_task: Optional[asyncio.Task[None]] = None

    async def run(self) -> None:
        await self.websocket.accept()
        self.monitor_task = asyncio.create_task(self._monitor_turn_completion())
        try:
            while True:
                message = await self.websocket.receive_text()
                event = json.loads(message)
                event_type = str(event.get("event") or "").strip().lower()
                if event_type == "start":
                    await self._handle_start(event)
                elif event_type == "media":
                    await self._handle_media(event)
                elif event_type == "mark":
                    self._handle_mark(event)
                elif event_type == "stop":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            self.closed = True
            await self._cancel_task(self.waiting_task)
            await self._cancel_task(self.processing_task)
            await self._cancel_task(self.monitor_task)
            await self._close_websocket()

    async def _handle_start(self, event: Mapping[str, Any]) -> None:
        start = event.get("start") if isinstance(event, Mapping) else None
        if isinstance(start, Mapping):
            self.stream_sid = str(start.get("streamSid") or self.stream_sid or "").strip()
            call_sid = str(start.get("callSid") or "").strip()
            if call_sid:
                self.call_sid = call_sid
        greeting = self.intro_message or self.service._default_greeting(self.caller)
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

        if is_speech or self.heard_speech:
            self.input_buffer.extend(chunk)
        if is_speech:
            self.heard_speech = True
            self.last_speech_at = asyncio.get_running_loop().time()

    def _handle_mark(self, event: Mapping[str, Any]) -> None:
        mark = event.get("mark") if isinstance(event, Mapping) else None
        if not isinstance(mark, Mapping):
            return
        name = str(mark.get("name") or "").strip()
        if name:
            self.pending_marks.discard(name)
        if not self.pending_marks:
            self.assistant_playing = False

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
            if len(self.input_buffer) < _MIN_UTTERANCE_BYTES:
                continue
            if (loop.time() - self.last_speech_at) < _SILENCE_GAP_SECONDS:
                continue
            utterance = bytes(self.input_buffer)
            self.input_buffer.clear()
            self.heard_speech = False
            self.processing_task = asyncio.create_task(self._process_utterance(utterance))

    async def _process_utterance(self, payload: bytes) -> None:
        transcript = await self._transcribe(payload)
        if not transcript:
            return

        self.waiting_task = asyncio.create_task(self._play_wait_tone_loop())
        try:
            response_text = await self.service.generate_owner_live_reply(
                caller=self.caller,
                transcript=transcript,
                call_sid=self.call_sid,
            )
        finally:
            await self._cancel_task(self.waiting_task)
            self.waiting_task = None
            await self._clear_audio()

        if response_text:
            await self._speak_text(response_text)

    async def _transcribe(self, payload: bytes) -> str:
        wav_bytes = mulaw_to_wav_bytes(payload)
        result = await transcribe_audio(
            chat_upload_dir(self.service.runtime),
            audio_bytes=wav_bytes,
            filename=f"phone_input_{uuid.uuid4().hex}.wav",
            media_type="audio/wav",
            prefer_local=False,
            language_code="en",
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
        voice_result = await synthesize_speech(
            chat_upload_dir(self.service.runtime),
            text=normalized,
            prefer_local=False,
            expressive=True,
        )
        audio_path = Path(str(voice_result.audio_attachment.get("path") or "")).expanduser()
        mulaw = await asyncio.to_thread(transcode_audio_file_to_mulaw, audio_path)
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

    @staticmethod
    def _chunk_contains_speech(chunk: bytes) -> bool:
        try:
            pcm = mulaw_to_pcm(chunk)
            if not pcm:
                return False
            samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
            rms = int(np.sqrt(np.mean(samples * samples))) if samples.size else 0
            return rms >= _SPEECH_RMS_THRESHOLD
        except Exception:
            return False


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
