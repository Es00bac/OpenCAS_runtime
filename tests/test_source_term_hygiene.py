from __future__ import annotations

from pathlib import Path

FORBIDDEN = "chronic" + "le"
SOURCE_ROOTS = [Path("opencas"), Path("tests")]


def test_forbidden_creative_project_term_is_not_hard_coded_in_source() -> None:
    offenders: list[str] = []
    for root in SOURCE_ROOTS:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if FORBIDDEN in text.lower():
                offenders.append(str(path))

    assert offenders == []
