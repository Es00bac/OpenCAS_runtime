"""Voice input/output helpers for dashboard chat."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

_ELEVENLABS_ENV_PATH = Path(
    os.environ.get("OPENCAS_ELEVENLABS_ENV_FILE")
    or os.environ.get("ELEVENLABS_ENV_FILE")
    or (Path.home() / ".opencasenv" / ".env")
).expanduser()
_ELEVENLABS_STT_MODEL = "scribe_v2"
_ELEVENLABS_FAST_MODEL = "eleven_flash_v2_5"
_ELEVENLABS_EXPRESSIVE_MODEL = "eleven_v3"
_EDGE_TTS_PREFERRED_VOICE = "Aira"
_EDGE_TTS_VOICE_ALIASES = {
    "aira": "en-US-AriaNeural",
    "aria": "en-US-AriaNeural",
}


@dataclass
class VoiceStatus:
    elevenlabs_available: bool
    local_stt_available: bool
    local_tts_available: bool
    elevenlabs_voice_id: str
    local_voice_name: str
    local_voice_resolved: str
    expressive_supported: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elevenlabs_available": self.elevenlabs_available,
            "local_stt_available": self.local_stt_available,
            "local_tts_available": self.local_tts_available,
            "elevenlabs_voice_id": self.elevenlabs_voice_id,
            "local_voice_name": self.local_voice_name,
            "local_voice_resolved": self.local_voice_resolved,
            "expressive_supported": self.expressive_supported,
        }


@dataclass
class VoiceTranscriptionResult:
    text: str
    provider: str
    mode: str
    model: str
    audio_attachment: Dict[str, Any]
    warning: Optional[str] = None

    def to_meta(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "mode": self.mode,
            "model": self.model,
            "warning": self.warning,
            "audio": self.audio_attachment,
        }


@dataclass
class VoiceSynthesisResult:
    provider: str
    mode: str
    model: str
    expressive: bool
    audio_attachment: Dict[str, Any]
    voice_id: Optional[str] = None
    voice_name: Optional[str] = None
    warning: Optional[str] = None

    def to_meta(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "mode": self.mode,
            "model": self.model,
            "expressive": self.expressive,
            "voice_id": self.voice_id,
            "voice_name": self.voice_name,
            "warning": self.warning,
            **self.audio_attachment,
        }


def _extract_env_value(key: str, path: Path = _ELEVENLABS_ENV_PATH) -> Optional[str]:
    import os

    direct = os.environ.get(key)
    if direct:
        return direct.strip().strip('"').strip("'")
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*\"?([^\"\n]+)", text)
    if not match:
        return None
    return match.group(1).strip().strip('"').strip("'")


def _resolve_edge_voice(name: str = _EDGE_TTS_PREFERRED_VOICE) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        return _EDGE_TTS_VOICE_ALIASES["aira"]
    return _EDGE_TTS_VOICE_ALIASES.get(cleaned.lower(), cleaned)


def _configured_elevenlabs_voice_id() -> str:
    return (
        _extract_env_value("OPENCAS_ELEVENLABS_VOICE_ID")
        or _extract_env_value("ELEVENLABS_VOICE_ID")
        or ""
    )


def voice_status() -> VoiceStatus:
    return VoiceStatus(
        elevenlabs_available=bool(_extract_env_value("ELEVENLABS_API_KEY")),
        local_stt_available=shutil.which("whisper") is not None and shutil.which("ffmpeg") is not None,
        local_tts_available=shutil.which("edge-tts") is not None,
        elevenlabs_voice_id=_configured_elevenlabs_voice_id(),
        local_voice_name=_EDGE_TTS_PREFERRED_VOICE,
        local_voice_resolved=_resolve_edge_voice(),
        expressive_supported=True,
    )


def _store_generated_bytes(
    upload_dir: Path,
    *,
    filename: str,
    media_type: str,
    payload: bytes,
) -> Dict[str, Any]:
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / (filename or "voice.bin")
    counter = 1
    original_dest = dest
    while dest.exists():
        dest = upload_dir / f"{original_dest.stem}_{counter}{original_dest.suffix}"
        counter += 1
    with dest.open("wb") as handle:
        handle.write(payload)
    resolved_media_type = media_type or mimetypes.guess_type(dest.name)[0] or "application/octet-stream"
    return {
        "filename": dest.name,
        "path": str(dest),
        "url": f"/api/chat/uploads/{dest.name}",
        "media_type": resolved_media_type,
        "size_bytes": dest.stat().st_size,
    }


async def transcribe_audio(
    upload_dir: Path,
    *,
    audio_bytes: bytes,
    filename: str,
    media_type: Optional[str] = None,
    prefer_local: bool = False,
    language_code: Optional[str] = None,
) -> VoiceTranscriptionResult:
    stored_audio = _store_generated_bytes(
        upload_dir,
        filename=filename or f"voice_input_{uuid.uuid4().hex}.webm",
        media_type=media_type or mimetypes.guess_type(filename or "voice_input.webm")[0] or "application/octet-stream",
        payload=audio_bytes,
    )
    status = voice_status()
    remote_error: Optional[str] = None
    if not prefer_local and status.elevenlabs_available:
        try:
            return await _transcribe_with_elevenlabs(Path(stored_audio["path"]), stored_audio, language_code=language_code)
        except Exception as exc:  # pragma: no cover - exercised with mocks/live env
            remote_error = str(exc)
    if status.local_stt_available:
        result = await _transcribe_with_whisper(Path(stored_audio["path"]), stored_audio, language_code=language_code)
        if remote_error:
            result.warning = f"ElevenLabs fallback triggered: {remote_error}"
        return result
    raise HTTPException(
        status_code=503,
        detail="Voice transcription is unavailable: ElevenLabs failed or is unconfigured, and local Whisper fallback is not available.",
    )


async def synthesize_speech(
    upload_dir: Path,
    *,
    text: str,
    prefer_local: bool = False,
    expressive: bool = False,
) -> VoiceSynthesisResult:
    status = voice_status()
    remote_error: Optional[str] = None
    normalized = (text or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="text is required for speech synthesis")
    if not prefer_local and status.elevenlabs_available:
        try:
            return await _synthesize_with_elevenlabs(upload_dir, normalized, expressive=expressive)
        except Exception as exc:  # pragma: no cover - exercised with mocks/live env
            remote_error = str(exc)
    if status.local_tts_available:
        result = await _synthesize_with_edge_tts(upload_dir, normalized)
        if remote_error:
            result.warning = f"ElevenLabs fallback triggered: {remote_error}"
        return result
    raise HTTPException(
        status_code=503,
        detail="Speech synthesis is unavailable: ElevenLabs failed or is unconfigured, and local Edge TTS fallback is not available.",
    )


async def _transcribe_with_elevenlabs(
    audio_path: Path,
    audio_attachment: Dict[str, Any],
    *,
    language_code: Optional[str] = None,
) -> VoiceTranscriptionResult:
    api_key = _extract_env_value("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")
    data: Dict[str, Any] = {
        "model_id": _ELEVENLABS_STT_MODEL,
        "timestamps_granularity": "none",
        "diarize": "false",
        "tag_audio_events": "true",
    }
    if language_code:
        data["language_code"] = language_code
    media_type = mimetypes.guess_type(audio_path.name)[0] or "application/octet-stream"
    timeout = httpx.Timeout(120.0, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with audio_path.open("rb") as handle:
            response = await client.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": api_key},
                data=data,
                files={"file": (audio_path.name, handle, media_type)},
            )
    response.raise_for_status()
    payload = response.json()
    text = str(payload.get("text") or "").strip()
    if not text:
        raise RuntimeError("ElevenLabs did not return transcript text")
    return VoiceTranscriptionResult(
        text=text,
        provider="elevenlabs",
        mode="hosted",
        model=_ELEVENLABS_STT_MODEL,
        audio_attachment=audio_attachment,
    )


async def _transcribe_with_whisper(
    audio_path: Path,
    audio_attachment: Dict[str, Any],
    *,
    language_code: Optional[str] = None,
) -> VoiceTranscriptionResult:
    def _run() -> str:
        with tempfile.TemporaryDirectory(prefix="opencas-whisper-") as tmpdir:
            cmd = [
                "whisper",
                str(audio_path),
                "--model",
                "base",
                "--task",
                "transcribe",
                "--output_dir",
                tmpdir,
                "--output_format",
                "json",
                "--verbose",
                "False",
                "--fp16",
                "False",
            ]
            if language_code:
                cmd.extend(["--language", language_code])
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "whisper failed")
            output_path = Path(tmpdir) / f"{audio_path.stem}.json"
            if not output_path.exists():
                raise RuntimeError("whisper did not emit a transcript file")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            return str(payload.get("text") or "").strip()

    text = await asyncio.to_thread(_run)
    if not text:
        raise RuntimeError("Whisper did not return transcript text")
    return VoiceTranscriptionResult(
        text=text,
        provider="whisper",
        mode="local",
        model="whisper-base",
        audio_attachment=audio_attachment,
    )


async def _synthesize_with_elevenlabs(upload_dir: Path, text: str, *, expressive: bool) -> VoiceSynthesisResult:
    api_key = _extract_env_value("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")
    voice_id = _configured_elevenlabs_voice_id()
    if not voice_id:
        raise RuntimeError("ELEVENLABS_VOICE_ID is not configured")
    model_id = _ELEVENLABS_EXPRESSIVE_MODEL if expressive and len(text) <= 5000 else _ELEVENLABS_FAST_MODEL
    timeout = httpx.Timeout(120.0, connect=15.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            params={"output_format": "mp3_44100_128"},
            json={"text": text, "model_id": model_id},
        )
    response.raise_for_status()
    attachment = _store_generated_bytes(
        upload_dir,
        filename=f"voice_output_{uuid.uuid4().hex}.mp3",
        media_type="audio/mpeg",
        payload=response.content,
    )
    return VoiceSynthesisResult(
        provider="elevenlabs",
        mode="hosted",
        model=model_id,
        expressive=bool(expressive and model_id == _ELEVENLABS_EXPRESSIVE_MODEL),
        audio_attachment=attachment,
        voice_id=voice_id,
        voice_name="Configured ElevenLabs voice",
    )


async def _synthesize_with_edge_tts(upload_dir: Path, text: str) -> VoiceSynthesisResult:
    resolved_voice = _resolve_edge_voice()

    def _run() -> bytes:
        with tempfile.TemporaryDirectory(prefix="opencas-edge-tts-") as tmpdir:
            input_path = Path(tmpdir) / "speech.txt"
            output_path = Path(tmpdir) / "speech.mp3"
            input_path.write_text(text, encoding="utf-8")
            cmd = [
                "edge-tts",
                "--file",
                str(input_path),
                "--voice",
                resolved_voice,
                "--write-media",
                str(output_path),
            ]
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "edge-tts failed")
            return output_path.read_bytes()

    payload = await asyncio.to_thread(_run)
    attachment = _store_generated_bytes(
        upload_dir,
        filename=f"voice_output_{uuid.uuid4().hex}.mp3",
        media_type="audio/mpeg",
        payload=payload,
    )
    return VoiceSynthesisResult(
        provider="edge-tts",
        mode="local",
        model="edge-tts",
        expressive=False,
        audio_attachment=attachment,
        voice_name=f"{_EDGE_TTS_PREFERRED_VOICE} ({resolved_voice})",
    )
