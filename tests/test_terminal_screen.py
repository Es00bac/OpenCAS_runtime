from __future__ import annotations

from opencas.execution.terminal_screen import render_terminal_screen


def test_render_terminal_screen_handles_clear_home_and_newlines() -> None:
    screen = render_terminal_screen("\x1b[2J\x1b[HAlpha\nBeta", rows=4, cols=10)

    assert screen["rows"] == 4
    assert screen["cols"] == 10
    assert screen["lines"][0].rstrip() == "Alpha"
    assert screen["lines"][1].rstrip() == "Beta"
    assert screen["text"].splitlines()[:2] == ["Alpha", "Beta"]
    assert screen["cursor"] == {"row": 2, "col": 5}


def test_render_terminal_screen_overwrites_after_carriage_return() -> None:
    screen = render_terminal_screen("Hello\rYo", rows=2, cols=10)

    assert screen["lines"][0].rstrip() == "Yollo"
    assert screen["cursor"] == {"row": 1, "col": 3}


def test_render_terminal_screen_honors_cursor_position_and_clear_line() -> None:
    screen = render_terminal_screen("abcdef\r\x1b[Kxy\x1b[2;4HHi", rows=3, cols=12)

    assert screen["lines"][0].rstrip() == "xy"
    assert screen["lines"][1].rstrip() == "   Hi"
    assert screen["cursor"] == {"row": 2, "col": 6}
