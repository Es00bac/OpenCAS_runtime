"""Playwright-targetable browser surface for PTY-backed TUI sessions."""

from __future__ import annotations

from html import escape
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from opencas.execution.tui_playwright_bridge import (
    DEFAULT_COLS,
    DEFAULT_IDLE_SECONDS,
    DEFAULT_MAX_WAIT_SECONDS,
    DEFAULT_ROWS,
    DEFAULT_SCOPE_KEY,
    default_cwd,
    key_sequence,
    normalize_snapshot,
    observe_snapshot,
    session_payload,
)


class TuiPlaywrightSessionRequest(BaseModel):
    command: str = Field(..., min_length=1)
    cwd: str | None = None
    scope_key: str = DEFAULT_SCOPE_KEY
    rows: int = Field(DEFAULT_ROWS, ge=10, le=80)
    cols: int = Field(DEFAULT_COLS, ge=40, le=240)
    observe: bool = True
    idle_seconds: float = Field(DEFAULT_IDLE_SECONDS, ge=0.05, le=5.0)
    max_wait_seconds: float = Field(DEFAULT_MAX_WAIT_SECONDS, ge=0.1, le=30.0)


class TuiPlaywrightInputRequest(BaseModel):
    input: str
    observe: bool = True
    idle_seconds: float = Field(DEFAULT_IDLE_SECONDS, ge=0.05, le=5.0)
    max_wait_seconds: float = Field(DEFAULT_MAX_WAIT_SECONDS, ge=0.1, le=30.0)


class TuiPlaywrightKeysRequest(BaseModel):
    keys: list[str] = Field(..., min_length=1)
    observe: bool = True
    idle_seconds: float = Field(DEFAULT_IDLE_SECONDS, ge=0.05, le=5.0)
    max_wait_seconds: float = Field(DEFAULT_MAX_WAIT_SECONDS, ge=0.1, le=30.0)


class TuiPlaywrightResizeRequest(BaseModel):
    rows: int = Field(DEFAULT_ROWS, ge=10, le=80)
    cols: int = Field(DEFAULT_COLS, ge=40, le=240)


def build_tui_playwright_router(runtime: Any) -> APIRouter:
    router = APIRouter(tags=["tui-playwright"])

    @router.get("/tui/playwright")
    async def tui_playwright_page(
        session_id: str | None = None,
        scope_key: str = DEFAULT_SCOPE_KEY,
    ) -> HTMLResponse:
        return HTMLResponse(_render_page(session_id=session_id, scope_key=scope_key))

    @router.get("/api/tui-playwright/sessions")
    async def list_sessions(
        request: Request,
        scope_key: str = DEFAULT_SCOPE_KEY,
    ) -> dict[str, Any]:
        supervisor = _pty_supervisor(runtime)
        snapshot = supervisor.snapshot(scope_key=scope_key, sample_limit=200)
        base_url = _request_base_url(request)
        sessions = []
        for entry in snapshot.get("entries", []):
            session_id = str(entry.get("session_id") or "")
            if not session_id:
                continue
            session = session_payload(
                session_id=session_id,
                scope_key=scope_key,
                snapshot=_snapshot_from_entry(entry),
                base_url=base_url,
            )
            session["command"] = entry.get("command")
            session["cwd"] = entry.get("cwd")
            session["running"] = entry.get("running")
            session["rows"] = entry.get("rows")
            session["cols"] = entry.get("cols")
            sessions.append(session)
        return {
            "scope_key": scope_key,
            "sessions": sessions,
            "total_count": snapshot.get("total_count", len(sessions)),
            "running_count": snapshot.get("running_count"),
            "completed_count": snapshot.get("completed_count"),
        }

    @router.post("/api/tui-playwright/sessions")
    async def start_session(
        request: Request,
        payload: TuiPlaywrightSessionRequest,
    ) -> dict[str, Any]:
        supervisor = _pty_supervisor(runtime)
        cwd = payload.cwd or default_cwd(runtime)
        session_id = supervisor.start(
            payload.scope_key,
            payload.command,
            cwd=cwd,
            rows=payload.rows,
            cols=payload.cols,
        )
        snapshot = {}
        if payload.observe:
            snapshot = observe_snapshot(
                supervisor,
                scope_key=payload.scope_key,
                session_id=session_id,
                idle_seconds=payload.idle_seconds,
                max_wait_seconds=payload.max_wait_seconds,
            )
        return session_payload(
            session_id=session_id,
            scope_key=payload.scope_key,
            snapshot=snapshot,
            base_url=_request_base_url(request),
        )

    @router.get("/api/tui-playwright/sessions/{session_id}")
    async def get_session(
        session_id: str,
        scope_key: str = DEFAULT_SCOPE_KEY,
        refresh: bool = True,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    ) -> dict[str, Any]:
        supervisor = _pty_supervisor(runtime)
        if refresh:
            snapshot = observe_snapshot(
                supervisor,
                scope_key=scope_key,
                session_id=session_id,
                idle_seconds=idle_seconds,
                max_wait_seconds=max_wait_seconds,
            )
            return {"found": snapshot.get("found", False), "snapshot": snapshot}
        snapshot = supervisor.snapshot(scope_key=scope_key, sample_limit=500)
        entry = next(
            (
                item
                for item in snapshot.get("entries", [])
                if item.get("session_id") == session_id
            ),
            None,
        )
        return {"found": entry is not None, "snapshot": entry or {}}

    @router.post("/api/tui-playwright/sessions/{session_id}/input")
    async def send_input(
        session_id: str,
        payload: TuiPlaywrightInputRequest,
        scope_key: str = DEFAULT_SCOPE_KEY,
    ) -> dict[str, Any]:
        supervisor = _pty_supervisor(runtime)
        ok = supervisor.write(scope_key, session_id, payload.input)
        if not ok:
            return {"found": False, "ok": False, "snapshot": {}}
        snapshot = {}
        if payload.observe:
            snapshot = observe_snapshot(
                supervisor,
                scope_key=scope_key,
                session_id=session_id,
                idle_seconds=payload.idle_seconds,
                max_wait_seconds=payload.max_wait_seconds,
            )
        return {
            "found": bool(snapshot.get("found", True)),
            "ok": True,
            "input": payload.input,
            "snapshot": snapshot,
        }

    @router.post("/api/tui-playwright/sessions/{session_id}/keys")
    async def send_keys(
        session_id: str,
        payload: TuiPlaywrightKeysRequest,
        scope_key: str = DEFAULT_SCOPE_KEY,
    ) -> dict[str, Any]:
        try:
            input_text = key_sequence(payload.keys)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        supervisor = _pty_supervisor(runtime)
        ok = supervisor.write(scope_key, session_id, input_text)
        if not ok:
            return {"found": False, "ok": False, "input": input_text, "snapshot": {}}
        snapshot = {}
        if payload.observe:
            snapshot = observe_snapshot(
                supervisor,
                scope_key=scope_key,
                session_id=session_id,
                idle_seconds=payload.idle_seconds,
                max_wait_seconds=payload.max_wait_seconds,
            )
        return {
            "found": bool(snapshot.get("found", True)),
            "ok": True,
            "input": input_text,
            "snapshot": snapshot,
        }

    @router.post("/api/tui-playwright/sessions/{session_id}/resize")
    async def resize_session(
        session_id: str,
        payload: TuiPlaywrightResizeRequest,
        scope_key: str = DEFAULT_SCOPE_KEY,
    ) -> dict[str, Any]:
        supervisor = _pty_supervisor(runtime)
        ok = supervisor.resize(
            scope_key,
            session_id,
            rows=payload.rows,
            cols=payload.cols,
        )
        return {
            "found": bool(ok),
            "ok": bool(ok),
            "rows": payload.rows,
            "cols": payload.cols,
        }

    @router.delete("/api/tui-playwright/sessions/{session_id}")
    async def close_session(
        session_id: str,
        scope_key: str = DEFAULT_SCOPE_KEY,
    ) -> dict[str, Any]:
        supervisor = _pty_supervisor(runtime)
        ok = supervisor.remove(scope_key, session_id)
        return {
            "found": bool(ok),
            "ok": bool(ok),
            "session_id": session_id,
        }

    return router


def _pty_supervisor(runtime: Any) -> Any:
    supervisor = getattr(runtime, "pty_supervisor", None)
    if supervisor is None:
        raise HTTPException(status_code=503, detail="PTY supervisor is not available")
    return supervisor


def _request_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _snapshot_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return normalize_snapshot(
        {
            "found": True,
            "running": entry.get("running"),
            "returncode": entry.get("returncode"),
            "rows": entry.get("rows"),
            "cols": entry.get("cols"),
            "cleaned_output": entry.get("last_cleaned_output") or "",
            "screen_state": entry.get("last_screen_state") or {},
        }
    )


def _render_page(*, session_id: str | None, scope_key: str) -> str:
    safe_session_id = escape(session_id or "", quote=True)
    safe_scope = escape(scope_key or "tui-playwright", quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>OpenCAS TUI</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101214;
      color: #e5e7eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #101214;
    }}
    main {{
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-height: 100vh;
    }}
    header, footer {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      padding: 10px 12px;
      background: #171a1f;
      border-bottom: 1px solid #2a3038;
    }}
    footer {{
      border-top: 1px solid #2a3038;
      border-bottom: 0;
    }}
    input, textarea, button {{
      border: 1px solid #3a424d;
      background: #111827;
      color: #f3f4f6;
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
    }}
    input {{
      min-width: 180px;
      flex: 1 1 240px;
    }}
    input.size {{
      min-width: 0;
      width: 72px;
      flex: 0 0 72px;
    }}
    button {{
      cursor: pointer;
      background: #253247;
    }}
    button:hover, button:focus {{
      background: #30405a;
      outline: 2px solid #5b8def;
      outline-offset: 1px;
    }}
    .terminal-wrap {{
      min-height: 0;
      display: flex;
      flex-direction: column;
      padding: 12px;
    }}
    .visually-hidden {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }}
    label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #cbd5e1;
      font-size: 13px;
    }}
    label input[type="checkbox"] {{
      min-width: 0;
      flex: 0 0 auto;
    }}
    pre {{
      flex: 1 1 auto;
      min-height: 55vh;
      margin: 0;
      padding: 12px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid #2d3748;
      border-radius: 6px;
      background: #05080c;
      color: #d1fae5;
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 14px;
      line-height: 1.45;
    }}
    .screen-grid {{
      min-height: 0;
      max-height: 28vh;
      margin-top: 10px;
      color: #bae6fd;
    }}
    textarea {{
      width: 100%;
      min-height: 64px;
      resize: vertical;
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
    }}
    .status {{
      min-width: 220px;
      color: #a7f3d0;
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 13px;
    }}
    .meta {{
      color: #cbd5e1;
      font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <main data-testid="tui-playwright-page" data-session-id="{safe_session_id}" data-scope-key="{safe_scope}">
    <header>
      <input data-testid="tui-command" id="command" value="bash" autocomplete="off">
      <input data-testid="tui-rows" id="rows" class="size" value="30" inputmode="numeric" aria-label="Rows">
      <input data-testid="tui-cols" id="cols" class="size" value="100" inputmode="numeric" aria-label="Columns">
      <button data-testid="tui-start-button" id="start" type="button">Start</button>
      <button data-testid="tui-resize-button" id="resize" type="button">Resize</button>
      <button data-testid="tui-refresh-button" id="refresh" type="button">Refresh</button>
      <button data-testid="tui-close-button" id="close" type="button">Close</button>
      <button data-testid="tui-ctrl-c-button" id="ctrlC" type="button">Ctrl-C</button>
      <button data-testid="tui-escape-button" id="escape" type="button">Esc</button>
      <button data-testid="tui-enter-button" id="enter" type="button">Enter</button>
      <label><input data-testid="tui-auto-refresh" id="autoRefresh" type="checkbox">Auto</label>
      <span data-testid="tui-status" id="status" class="status">idle</span>
      <span data-testid="tui-screen-state" id="screenState" class="meta"></span>
    </header>
    <section class="terminal-wrap">
      <pre data-testid="tui-terminal-output" id="terminal" role="log" aria-live="polite" tabindex="0"></pre>
      <pre data-testid="tui-screen-grid" id="screenGrid" class="screen-grid" aria-label="Rendered terminal screen"></pre>
      <pre data-testid="tui-raw-output" id="rawOutput" class="visually-hidden"></pre>
      <pre data-testid="tui-snapshot-json" id="snapshotJson" class="visually-hidden">{{}}</pre>
    </section>
    <footer>
      <textarea data-testid="tui-terminal-input" id="terminalInput" autocomplete="off" spellcheck="false"></textarea>
      <button data-testid="tui-send-button" id="send" type="button">Send</button>
    </footer>
  </main>
  <script>
    const root = document.querySelector('[data-testid="tui-playwright-page"]');
    const terminal = document.querySelector('[data-testid="tui-terminal-output"]');
    const screenGrid = document.querySelector('[data-testid="tui-screen-grid"]');
    const input = document.querySelector('[data-testid="tui-terminal-input"]');
    const command = document.querySelector('[data-testid="tui-command"]');
    const rowsInput = document.querySelector('[data-testid="tui-rows"]');
    const colsInput = document.querySelector('[data-testid="tui-cols"]');
    const statusNode = document.querySelector('[data-testid="tui-status"]');
    const stateNode = document.querySelector('[data-testid="tui-screen-state"]');
    const rawOutput = document.querySelector('[data-testid="tui-raw-output"]');
    const snapshotJson = document.querySelector('[data-testid="tui-snapshot-json"]');
    const autoRefresh = document.querySelector('[data-testid="tui-auto-refresh"]');
    const scopeKey = root.dataset.scopeKey || 'tui-playwright';
    let sessionId = root.dataset.sessionId || '';
    let autoRefreshTimer = null;

    function setStatus(text) {{
      statusNode.textContent = text;
    }}

    function renderSnapshot(snapshot) {{
      snapshot = snapshot || {{}};
      const renderedScreen = snapshot.terminal_screen || {{}};
      const screenText = renderedScreen.text || '';
      terminal.textContent = screenText || snapshot.cleaned_output || '';
      screenGrid.textContent = screenText;
      rawOutput.textContent = snapshot.raw_output || '';
      snapshotJson.textContent = JSON.stringify(snapshot);
      const state = snapshot.screen_state || {{}};
      stateNode.textContent = [state.app, state.mode].filter(Boolean).join(' ');
      root.dataset.screenMode = state.mode || '';
      root.dataset.running = String(snapshot.running);
      root.dataset.returncode = snapshot.returncode == null ? '' : String(snapshot.returncode);
      root.dataset.rows = String(snapshot.rows || renderedScreen.rows || '');
      root.dataset.cols = String(snapshot.cols || renderedScreen.cols || '');
      terminal.scrollTop = terminal.scrollHeight;
    }}

    function numericValue(node, fallback) {{
      const parsed = Number.parseInt(node.value || '', 10);
      return Number.isFinite(parsed) ? parsed : fallback;
    }}

    function sessionUrl() {{
      return `/api/tui-playwright/sessions/${{encodeURIComponent(sessionId)}}?scope_key=${{encodeURIComponent(scopeKey)}}&refresh=true`;
    }}

    async function startSession() {{
      setStatus('starting');
      const response = await fetch('/api/tui-playwright/sessions', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          command: command.value || 'bash',
          scope_key: scopeKey,
          rows: numericValue(rowsInput, 30),
          cols: numericValue(colsInput, 100)
        }})
      }});
      const payload = await response.json();
      sessionId = payload.session_id || '';
      root.dataset.sessionId = sessionId;
      if (payload.playwright_url) {{
        history.replaceState(null, '', payload.playwright_url);
      }}
      renderSnapshot(payload.snapshot);
      setStatus(sessionId ? `session ${{sessionId}}` : 'no session');
      input.focus();
    }}

    async function refreshSession() {{
      if (!sessionId) {{
        setStatus('no session');
        return;
      }}
      const response = await fetch(sessionUrl());
      const payload = await response.json();
      renderSnapshot(payload.snapshot);
      setStatus(payload.found ? `session ${{sessionId}}` : 'not found');
    }}

    async function sendText(text) {{
      if (!sessionId || !text) {{
        return;
      }}
      setStatus('sending');
      const response = await fetch(`/api/tui-playwright/sessions/${{encodeURIComponent(sessionId)}}/input?scope_key=${{encodeURIComponent(scopeKey)}}`, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{input: text, observe: true}})
      }});
      const payload = await response.json();
      renderSnapshot(payload.snapshot);
      setStatus(payload.ok ? `session ${{sessionId}}` : 'write failed');
    }}

    async function sendKeys(keys) {{
      if (!sessionId) {{
        setStatus('no session');
        return;
      }}
      setStatus('sending keys');
      const response = await fetch(`/api/tui-playwright/sessions/${{encodeURIComponent(sessionId)}}/keys?scope_key=${{encodeURIComponent(scopeKey)}}`, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{keys, observe: true}})
      }});
      const payload = await response.json();
      renderSnapshot(payload.snapshot);
      setStatus(payload.ok ? `session ${{sessionId}}` : 'key failed');
    }}

    async function resizeSession() {{
      if (!sessionId) {{
        setStatus('no session');
        return;
      }}
      const rows = numericValue(rowsInput, 30);
      const cols = numericValue(colsInput, 100);
      const response = await fetch(`/api/tui-playwright/sessions/${{encodeURIComponent(sessionId)}}/resize?scope_key=${{encodeURIComponent(scopeKey)}}`, {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{rows, cols}})
      }});
      const payload = await response.json();
      root.dataset.rows = String(rows);
      root.dataset.cols = String(cols);
      setStatus(payload.ok ? `resized ${{rows}}x${{cols}}` : 'resize failed');
    }}

    async function closeSession() {{
      if (!sessionId) {{
        setStatus('no session');
        return;
      }}
      const response = await fetch(`/api/tui-playwright/sessions/${{encodeURIComponent(sessionId)}}?scope_key=${{encodeURIComponent(scopeKey)}}`, {{
        method: 'DELETE'
      }});
      const payload = await response.json();
      if (payload.ok) {{
        sessionId = '';
        root.dataset.sessionId = '';
        renderSnapshot({{}});
      }}
      setStatus(payload.ok ? 'closed' : 'close failed');
    }}

    function syncAutoRefresh() {{
      if (autoRefreshTimer) {{
        clearInterval(autoRefreshTimer);
        autoRefreshTimer = null;
      }}
      if (autoRefresh.checked) {{
        autoRefreshTimer = setInterval(refreshSession, 1000);
      }}
    }}

    function keyToTerminalInput(event) {{
      if (event.ctrlKey && event.key && event.key.length === 1) {{
        const code = event.key.toUpperCase().charCodeAt(0) - 64;
        if (code > 0 && code < 32) {{
          return String.fromCharCode(code);
        }}
      }}
      if (event.key === 'Enter') return '\\r';
      if (event.key === 'Backspace') return '\\x7f';
      if (event.key === 'Tab') return '\\t';
      if (event.key === 'Escape') return '\\x1b';
      if (event.key === 'ArrowUp') return '\\x1b[A';
      if (event.key === 'ArrowDown') return '\\x1b[B';
      if (event.key === 'ArrowRight') return '\\x1b[C';
      if (event.key === 'ArrowLeft') return '\\x1b[D';
      if (event.key && event.key.length === 1 && !event.altKey && !event.metaKey) {{
        return event.key;
      }}
      return '';
    }}

    document.querySelector('[data-testid="tui-start-button"]').addEventListener('click', startSession);
    document.querySelector('[data-testid="tui-refresh-button"]').addEventListener('click', refreshSession);
    document.querySelector('[data-testid="tui-resize-button"]').addEventListener('click', resizeSession);
    document.querySelector('[data-testid="tui-close-button"]').addEventListener('click', closeSession);
    document.querySelector('[data-testid="tui-ctrl-c-button"]').addEventListener('click', () => sendKeys(['Ctrl-C']));
    document.querySelector('[data-testid="tui-escape-button"]').addEventListener('click', () => sendKeys(['Escape']));
    document.querySelector('[data-testid="tui-enter-button"]').addEventListener('click', () => sendKeys(['Enter']));
    autoRefresh.addEventListener('change', syncAutoRefresh);
    document.querySelector('[data-testid="tui-send-button"]').addEventListener('click', async () => {{
      const text = input.value;
      input.value = '';
      await sendText(text.endsWith('\\r') || text.endsWith('\\n') ? text : text + '\\r');
      input.focus();
    }});
    input.addEventListener('keydown', async (event) => {{
      const text = keyToTerminalInput(event);
      if (!text) return;
      event.preventDefault();
      await sendText(text);
    }});
    terminal.addEventListener('click', () => input.focus());
    if (sessionId) {{
      refreshSession();
    }}
  </script>
</body>
</html>"""
