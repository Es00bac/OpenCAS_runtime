from pathlib import Path


def test_dashboard_voice_playback_normalizes_audio_urls():
    html = Path("opencas/dashboard/static/index.html").read_text(encoding="utf-8")
    assert "const audioUrl = http.resolveUrl(voiceOutput.url);" in html
    assert "new Audio(`${audioUrl}${audioUrl.includes('?') ? '&' : '?'}t=${Date.now()}`);" in html
