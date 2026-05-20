import asyncio
import gc
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio


@pytest.fixture(autouse=True)
def _suppress_real_body_double_audio(monkeypatch: pytest.MonkeyPatch):
    """Prevent tests from generating or playing Body Double audio on the operator desktop."""

    def _fake_audio_player(path: Path | str) -> dict[str, Any]:
        return {
            "played": False,
            "reason": "pytest_audio_suppressed",
            "path": str(path),
            "audio_client_name": "OpenCAS Body Double",
        }

    async def _fake_speech_synthesizer(self: Any, text: str) -> dict[str, Any]:
        return {
            "provider": "pytest",
            "model": "suppressed",
            "text": text,
            "path": None,
        }

    monkeypatch.setattr("opencas.desktop_context.service.play_audio_file", _fake_audio_player)
    monkeypatch.setattr(
        "opencas.desktop_context.service.DesktopContextService._default_speech_synthesizer",
        _fake_speech_synthesizer,
    )


@pytest_asyncio.fixture(autouse=True)
async def _drain_aiosqlite():
    """Yield control so aiosqlite worker threads finish before loop closes."""
    yield
    gc.collect()
    await asyncio.sleep(0)
    await asyncio.sleep(0.1)
