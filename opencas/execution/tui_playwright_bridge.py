"""Shared contract for Playwright-targetable PTY browser sessions."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from opencas.execution.terminal_screen import render_terminal_screen

DEFAULT_SCOPE_KEY = "tui-playwright"
DEFAULT_ROWS = 30
DEFAULT_COLS = 100
DEFAULT_IDLE_SECONDS = 0.25
DEFAULT_MAX_WAIT_SECONDS = 1.5

KEY_SEQUENCES: dict[str, str] = {
    "enter": "\r",
    "return": "\r",
    "tab": "\t",
    "escape": "\x1b",
    "esc": "\x1b",
    "backspace": "\x7f",
    "delete": "\x1b[3~",
    "up": "\x1b[A",
    "arrowup": "\x1b[A",
    "down": "\x1b[B",
    "arrowdown": "\x1b[B",
    "right": "\x1b[C",
    "arrowright": "\x1b[C",
    "left": "\x1b[D",
    "arrowleft": "\x1b[D",
    "home": "\x1b[H",
    "end": "\x1b[F",
    "pageup": "\x1b[5~",
    "pagedown": "\x1b[6~",
}


def default_cwd(runtime: Any) -> str:
    """Return the runtime workspace root used for new PTY-backed TUI sessions."""
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    method = getattr(config, "primary_workspace_root", None)
    if callable(method):
        return str(method())
    return "."


def normalize_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize PTY supervisor output into the browser/tool bridge shape."""
    cleaned_output = str(
        raw.get("cleaned_combined_output")
        or raw.get("cleaned_output")
        or raw.get("last_cleaned_output")
        or ""
    )
    raw_output = str(raw.get("combined_output") or raw.get("output") or "")
    rows = _int_value(raw.get("rows"), DEFAULT_ROWS)
    cols = _int_value(raw.get("cols"), DEFAULT_COLS)
    terminal_screen = raw.get("terminal_screen")
    if not isinstance(terminal_screen, dict):
        terminal_screen = render_terminal_screen(
            raw_output or cleaned_output,
            rows=rows,
            cols=cols,
        )
    return {
        "found": bool(raw.get("found", False)),
        "running": raw.get("running"),
        "returncode": raw.get("returncode"),
        "rows": rows,
        "cols": cols,
        "cleaned_output": cleaned_output,
        "raw_output": raw_output,
        "terminal_screen": terminal_screen,
        "screen_state": raw.get("screen_state", {}) or {},
        "idle_reached": bool(raw.get("idle_reached", False)),
        "timed_out": bool(raw.get("timed_out", False)),
        "elapsed_ms": raw.get("elapsed_ms"),
    }


def _int_value(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def observe_snapshot(
    supervisor: Any,
    *,
    scope_key: str,
    session_id: str,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    max_bytes_per_poll: int = 4096,
) -> dict[str, Any]:
    """Observe a PTY until quiet and return a normalized snapshot."""
    raw = supervisor.observe_until_quiet(
        scope_key,
        session_id,
        idle_seconds=idle_seconds,
        max_wait_seconds=max_wait_seconds,
        max_bytes_per_poll=max_bytes_per_poll,
    )
    return normalize_snapshot(raw)


def playwright_url(*, session_id: str, scope_key: str = DEFAULT_SCOPE_KEY) -> str:
    """Return the relative browser URL for a PTY-backed TUI session."""
    return "/tui/playwright?" + urlencode(
        {"session_id": session_id, "scope_key": scope_key}
    )


def browser_url(
    *,
    session_id: str,
    scope_key: str = DEFAULT_SCOPE_KEY,
    base_url: str | None = None,
) -> str | None:
    """Return an absolute URL when the live API base URL is known."""
    if not base_url:
        return None
    return base_url.rstrip("/") + playwright_url(
        session_id=session_id,
        scope_key=scope_key,
    )


def playwright_selectors() -> dict[str, str]:
    """Stable selectors intended for browser automation against the TUI page."""
    return {
        "page": '[data-testid="tui-playwright-page"]',
        "terminal": '[data-testid="tui-terminal-output"]',
        "screen_grid": '[data-testid="tui-screen-grid"]',
        "input": '[data-testid="tui-terminal-input"]',
        "command": '[data-testid="tui-command"]',
        "start": '[data-testid="tui-start-button"]',
        "refresh": '[data-testid="tui-refresh-button"]',
        "status": '[data-testid="tui-status"]',
        "screen_state": '[data-testid="tui-screen-state"]',
        "send": '[data-testid="tui-send-button"]',
        "snapshot_json": '[data-testid="tui-snapshot-json"]',
        "raw_output": '[data-testid="tui-raw-output"]',
        "rows": '[data-testid="tui-rows"]',
        "cols": '[data-testid="tui-cols"]',
        "resize": '[data-testid="tui-resize-button"]',
        "close": '[data-testid="tui-close-button"]',
        "ctrl_c": '[data-testid="tui-ctrl-c-button"]',
        "escape": '[data-testid="tui-escape-button"]',
        "enter": '[data-testid="tui-enter-button"]',
        "auto_refresh": '[data-testid="tui-auto-refresh"]',
    }


def key_sequence(keys: list[str]) -> str:
    """Translate browser/tool key names into terminal control sequences."""
    output: list[str] = []
    for key in keys:
        cleaned = str(key or "").strip()
        lowered = cleaned.lower().replace(" ", "").replace("_", "-")
        if lowered.startswith("ctrl-") and len(lowered) == 6:
            code = ord(lowered[-1].upper()) - 64
            if 0 < code < 32:
                output.append(chr(code))
                continue
        lookup = lowered.replace("-", "")
        if lookup in KEY_SEQUENCES:
            output.append(KEY_SEQUENCES[lookup])
            continue
        if len(cleaned) == 1:
            output.append(cleaned)
            continue
        raise ValueError(f"Unsupported TUI key: {key}")
    return "".join(output)


def session_payload(
    *,
    session_id: str,
    scope_key: str,
    snapshot: dict[str, Any] | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Build the common API/tool payload for a browser-targetable TUI session."""
    payload = {
        "session_id": session_id,
        "scope_key": scope_key,
        "playwright_url": playwright_url(session_id=session_id, scope_key=scope_key),
        "selectors": playwright_selectors(),
        "snapshot": snapshot or {},
    }
    absolute = browser_url(
        session_id=session_id,
        scope_key=scope_key,
        base_url=base_url,
    )
    if absolute:
        payload["browser_url"] = absolute
    return payload
