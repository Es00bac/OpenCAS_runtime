from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import opencas.api.voice_service as voice_service


@pytest.mark.asyncio
async def test_transcribe_audio_turns_no_transcript_into_client_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _raise_no_transcript(*args, **kwargs):
        raise RuntimeError("Whisper did not return transcript text")

    monkeypatch.setattr(
        voice_service,
        "voice_status",
        lambda: SimpleNamespace(
            elevenlabs_available=False,
            local_stt_available=True,
            local_tts_available=True,
            elevenlabs_voice_id="voice-id",
            local_voice_name="Aira",
            local_voice_resolved="en-US-AriaNeural",
            expressive_supported=True,
        ),
    )
    monkeypatch.setattr(voice_service, "_transcribe_with_whisper", _raise_no_transcript)

    with pytest.raises(HTTPException) as exc_info:
        await voice_service.transcribe_audio(
            tmp_path,
            audio_bytes=b"fake-audio",
            filename="voice.webm",
            media_type="audio/webm",
            prefer_local=True,
            language_code="en",
        )

    assert exc_info.value.status_code == 422
    assert "No speech was detected" in exc_info.value.detail
