from pathlib import Path


def test_anime_presentation_uses_slide_relative_panel_layout() -> None:
    body = Path(
        "bulma_audit_report/ultimate_anime_presentation/index.html"
    ).read_text(encoding="utf-8")

    assert "width: 100%;" in body
    assert "height: 100%;" in body
    assert "width: min(42%, 760px);" in body
    assert "width: min(70%, 1200px);" in body
    assert "width: 100vw;" not in body.split(".slide-wrapper", 1)[1].split(".glass-panel", 1)[0]
