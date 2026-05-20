from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opencas.api.routes.tui_playwright import build_tui_playwright_router
from opencas.tools.adapters.tui_playwright import TuiPlaywrightToolAdapter


class _FakePtySupervisor:
    def __init__(self) -> None:
        self.started: list[dict[str, object]] = []
        self.writes: list[tuple[str, str, str]] = []
        self.resizes: list[tuple[str, str, int, int]] = []
        self.removed: list[tuple[str, str]] = []
        self._session_id = "pty-playwright-001"

    def start(self, scope_key: str, command: str, *, cwd: str, rows: int, cols: int) -> str:
        self.started.append(
            {
                "scope_key": scope_key,
                "command": command,
                "cwd": cwd,
                "rows": rows,
                "cols": cols,
            }
        )
        return self._session_id

    def observe_until_quiet(
        self,
        scope_key: str,
        session_id: str,
        *,
        idle_seconds: float = 0.25,
        max_wait_seconds: float = 1.5,
        max_bytes_per_poll: int = 4096,
    ) -> dict[str, object]:
        return {
            "found": True,
            "session_id": session_id,
            "running": True,
            "rows": 32,
            "cols": 100,
            "combined_output": "\x1b[2J\x1b[HPrompt ready",
            "cleaned_combined_output": "Prompt ready",
            "screen_state": {
                "app": "bash",
                "mode": "shell_prompt",
                "ready_for_input": True,
            },
        }

    def write(self, scope_key: str, session_id: str, input_text: str) -> bool:
        self.writes.append((scope_key, session_id, input_text))
        return True

    def resize(self, scope_key: str, session_id: str, *, rows: int, cols: int) -> bool:
        self.resizes.append((scope_key, session_id, rows, cols))
        return True

    def remove(self, scope_key: str, session_id: str) -> bool:
        self.removed.append((scope_key, session_id))
        return True

    def snapshot(self, scope_key: str | None = None, sample_limit: int = 10) -> dict[str, object]:
        entries = [
            {
                "session_id": self._session_id,
                "scope_key": "tui",
                "command": "bash",
                "cwd": "/tmp/opencas-test-root",
                "running": True,
                "rows": 32,
                "cols": 100,
                "last_cleaned_output": "Prompt ready",
                "last_screen_state": {
                    "app": "bash",
                    "mode": "shell_prompt",
                    "ready_for_input": True,
                },
            }
        ]
        if scope_key is not None:
            entries = [entry for entry in entries if entry["scope_key"] == scope_key]
        return {
            "total_count": len(entries),
            "running_count": len(entries),
            "completed_count": 0,
            "entries": entries[:sample_limit],
        }


def _client() -> tuple[TestClient, _FakePtySupervisor]:
    supervisor = _FakePtySupervisor()
    runtime = SimpleNamespace(
        pty_supervisor=supervisor,
        ctx=SimpleNamespace(
            config=SimpleNamespace(
                primary_workspace_root=lambda: "/tmp/opencas-test-root",
            )
        ),
    )
    app = FastAPI()
    app.include_router(build_tui_playwright_router(runtime))
    return TestClient(app), supervisor


def test_tui_playwright_page_exposes_stable_playwright_targets() -> None:
    client, _ = _client()

    response = client.get("/tui/playwright?session_id=pty-playwright-001&scope_key=tui")

    assert response.status_code == 200
    html = response.text
    assert 'rel="icon"' in html
    assert 'data-testid="tui-playwright-page"' in html
    assert 'data-testid="tui-terminal-output"' in html
    assert 'data-testid="tui-screen-grid"' in html
    assert 'data-testid="tui-terminal-input"' in html
    assert 'data-testid="tui-snapshot-json"' in html
    assert 'data-testid="tui-raw-output"' in html
    assert 'data-testid="tui-rows"' in html
    assert 'data-testid="tui-cols"' in html
    assert 'data-testid="tui-resize-button"' in html
    assert 'data-testid="tui-close-button"' in html
    assert 'data-testid="tui-ctrl-c-button"' in html
    assert 'data-testid="tui-escape-button"' in html
    assert 'data-testid="tui-enter-button"' in html
    assert 'data-testid="tui-auto-refresh"' in html
    assert "/api/tui-playwright/sessions/" in html


def test_start_tui_playwright_session_returns_browser_url_and_snapshot() -> None:
    client, supervisor = _client()

    response = client.post(
        "/api/tui-playwright/sessions",
        json={"command": "bash", "scope_key": "tui", "rows": 32, "cols": 100},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "pty-playwright-001"
    assert payload["playwright_url"] == "/tui/playwright?session_id=pty-playwright-001&scope_key=tui"
    assert payload["browser_url"] == "http://testserver/tui/playwright?session_id=pty-playwright-001&scope_key=tui"
    assert payload["selectors"]["terminal"] == '[data-testid="tui-terminal-output"]'
    assert payload["selectors"]["screen_grid"] == '[data-testid="tui-screen-grid"]'
    assert payload["selectors"]["snapshot_json"] == '[data-testid="tui-snapshot-json"]'
    assert payload["snapshot"]["screen_state"]["mode"] == "shell_prompt"
    assert payload["snapshot"]["terminal_screen"]["lines"][0].rstrip() == "Prompt ready"
    assert payload["snapshot"]["terminal_screen"]["cursor"] == {"row": 1, "col": 13}
    assert supervisor.started == [
        {
            "scope_key": "tui",
            "command": "bash",
            "cwd": "/tmp/opencas-test-root",
            "rows": 32,
            "cols": 100,
        }
    ]


def test_tui_playwright_session_list_returns_reattach_targets() -> None:
    client, _ = _client()

    response = client.get("/api/tui-playwright/sessions?scope_key=tui")

    assert response.status_code == 200
    payload = response.json()
    assert payload["scope_key"] == "tui"
    assert payload["sessions"][0]["session_id"] == "pty-playwright-001"
    assert payload["sessions"][0]["playwright_url"] == "/tui/playwright?session_id=pty-playwright-001&scope_key=tui"
    assert payload["sessions"][0]["browser_url"] == "http://testserver/tui/playwright?session_id=pty-playwright-001&scope_key=tui"
    assert payload["sessions"][0]["snapshot"]["screen_state"]["mode"] == "shell_prompt"
    assert payload["sessions"][0]["snapshot"]["terminal_screen"]["text"] == "Prompt ready"


def test_tui_playwright_input_endpoint_writes_to_pty_and_returns_snapshot() -> None:
    client, supervisor = _client()

    response = client.post(
        "/api/tui-playwright/sessions/pty-playwright-001/input?scope_key=tui",
        json={"input": "ls -la\r", "observe": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["found"] is True
    assert payload["ok"] is True
    assert payload["snapshot"]["cleaned_output"] == "Prompt ready"
    assert payload["snapshot"]["terminal_screen"]["text"] == "Prompt ready"
    assert supervisor.writes == [("tui", "pty-playwright-001", "ls -la\r")]


def test_tui_playwright_key_endpoint_writes_named_control_sequences() -> None:
    client, supervisor = _client()

    response = client.post(
        "/api/tui-playwright/sessions/pty-playwright-001/keys?scope_key=tui",
        json={"keys": ["Ctrl-C", "Enter"], "observe": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["input"] == "\x03\r"
    assert supervisor.writes == [("tui", "pty-playwright-001", "\x03\r")]


def test_tui_playwright_resize_and_close_endpoints_control_pty() -> None:
    client, supervisor = _client()

    resize = client.post(
        "/api/tui-playwright/sessions/pty-playwright-001/resize?scope_key=tui",
        json={"rows": 40, "cols": 120},
    )
    close = client.delete("/api/tui-playwright/sessions/pty-playwright-001?scope_key=tui")

    assert resize.status_code == 200
    assert resize.json() == {"found": True, "ok": True, "rows": 40, "cols": 120}
    assert close.status_code == 200
    assert close.json() == {"found": True, "ok": True, "session_id": "pty-playwright-001"}
    assert supervisor.resizes == [("tui", "pty-playwright-001", 40, 120)]
    assert supervisor.removed == [("tui", "pty-playwright-001")]


def test_tui_playwright_tool_adapter_starts_session_and_returns_browser_target() -> None:
    supervisor = _FakePtySupervisor()
    runtime = SimpleNamespace(
        pty_supervisor=supervisor,
        server_base_url="http://127.0.0.1:32147",
        ctx=SimpleNamespace(
            config=SimpleNamespace(
                primary_workspace_root=lambda: "/tmp/opencas-test-root",
            )
        ),
    )
    adapter = TuiPlaywrightToolAdapter(runtime)

    result = adapter("tui_playwright_open", {"command": "vim notes.md", "scope_key": "tui"})

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["session_id"] == "pty-playwright-001"
    assert payload["playwright_url"] == "/tui/playwright?session_id=pty-playwright-001&scope_key=tui"
    assert payload["browser_url"] == "http://127.0.0.1:32147/tui/playwright?session_id=pty-playwright-001&scope_key=tui"
    assert payload["next_tools"] == ["browser_start", "browser_navigate", "browser_snapshot"]
    assert payload["selectors"]["input"] == '[data-testid="tui-terminal-input"]'
    assert supervisor.started[0]["command"] == "vim notes.md"


def test_tui_playwright_tool_adapter_manages_existing_session() -> None:
    supervisor = _FakePtySupervisor()
    runtime = SimpleNamespace(
        pty_supervisor=supervisor,
        server_base_url="http://127.0.0.1:32147",
        ctx=SimpleNamespace(
            config=SimpleNamespace(
                primary_workspace_root=lambda: "/tmp/opencas-test-root",
            )
        ),
    )
    adapter = TuiPlaywrightToolAdapter(runtime)

    sent = adapter(
        "tui_playwright_input",
        {"session_id": "pty-playwright-001", "scope_key": "tui", "input": "i"},
    )
    keys = adapter(
        "tui_playwright_keys",
        {"session_id": "pty-playwright-001", "scope_key": "tui", "keys": ["Escape", "Ctrl-C"]},
    )
    resized = adapter(
        "tui_playwright_resize",
        {"session_id": "pty-playwright-001", "scope_key": "tui", "rows": 40, "cols": 120},
    )
    snapshot = adapter(
        "tui_playwright_snapshot",
        {"session_id": "pty-playwright-001", "scope_key": "tui"},
    )
    closed = adapter(
        "tui_playwright_close",
        {"session_id": "pty-playwright-001", "scope_key": "tui"},
    )

    assert sent.success is True
    assert keys.success is True
    assert resized.success is True
    assert snapshot.success is True
    assert closed.success is True
    assert supervisor.writes == [
        ("tui", "pty-playwright-001", "i"),
        ("tui", "pty-playwright-001", "\x1b\x03"),
    ]
    assert supervisor.resizes == [("tui", "pty-playwright-001", 40, 120)]
    assert supervisor.removed == [("tui", "pty-playwright-001")]


def test_tui_playwright_tool_adapter_rejects_empty_key_lists() -> None:
    supervisor = _FakePtySupervisor()
    runtime = SimpleNamespace(pty_supervisor=supervisor)
    adapter = TuiPlaywrightToolAdapter(runtime)

    result = adapter(
        "tui_playwright_keys",
        {"session_id": "pty-playwright-001", "keys": []},
    )

    assert result.success is False
    assert "keys is required" in result.output
    assert supervisor.writes == []
