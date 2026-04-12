"""Tests for PTY-backed terminal sessions."""

import json
import time

from opencas.execution.pty_supervisor import PtySupervisor, infer_screen_state, strip_ansi
from opencas.tools.adapters.pty import PtyToolAdapter


def test_pty_supervisor_can_run_and_poll() -> None:
    supervisor = PtySupervisor()
    session_id = supervisor.start("test", "printf 'hello from pty\\n'")
    try:
        output = ""
        deadline = time.time() + 3.0
        while time.time() < deadline:
            result = supervisor.poll("test", session_id)
            output += result.get("output", "")
            if "hello from pty" in output:
                break
            time.sleep(0.05)
        assert "hello from pty" in output
    finally:
        supervisor.remove("test", session_id)


def test_pty_supervisor_can_write_and_resize() -> None:
    supervisor = PtySupervisor()
    session_id = supervisor.start("test", "cat")
    try:
        assert supervisor.resize("test", session_id, rows=40, cols=120) is True
        assert supervisor.write("test", session_id, "hello interactive\n") is True
        output = ""
        deadline = time.time() + 3.0
        while time.time() < deadline:
            result = supervisor.poll("test", session_id)
            output += result.get("output", "")
            if "hello interactive" in output:
                break
            time.sleep(0.05)
        assert "hello interactive" in output
        latest = supervisor.poll("test", session_id)
        assert latest["rows"] == 40
        assert latest["cols"] == 120
    finally:
        supervisor.remove("test", session_id)


def test_pty_tool_adapter_surface() -> None:
    supervisor = PtySupervisor()
    adapter = PtyToolAdapter(supervisor, default_cwd=".")
    started = adapter("pty_start", {"command": "printf 'adapter ok\\n'"})
    assert started.success is True
    session_id = json.loads(started.output)["session_id"]
    try:
        deadline = time.time() + 3.0
        output = ""
        while time.time() < deadline:
            polled = adapter("pty_poll", {"session_id": session_id})
            assert polled.success is True
            output += json.loads(polled.output).get("output", "")
            if "adapter ok" in output:
                break
            time.sleep(0.05)
        assert "adapter ok" in output
    finally:
        adapter("pty_remove", {"session_id": session_id})


def test_pty_supervisor_observe_until_quiet() -> None:
    supervisor = PtySupervisor()
    session_id = supervisor.start("test", "printf 'line one\\n'; sleep 0.1; printf 'line two\\n'")
    try:
        observed = supervisor.observe_until_quiet(
            "test",
            session_id,
            idle_seconds=0.15,
            max_wait_seconds=3.0,
        )
        assert observed["found"] is True
        assert "line one" in observed["combined_output"]
        assert "line two" in observed["combined_output"]
        assert observed.get("idle_reached") is True or observed["running"] is False
    finally:
        supervisor.remove("test", session_id)


def test_pty_tool_adapter_observe_surface() -> None:
    supervisor = PtySupervisor()
    adapter = PtyToolAdapter(supervisor, default_cwd=".")
    started = adapter("pty_start", {"command": "printf 'observe ok\\n'"})
    session_id = json.loads(started.output)["session_id"]
    try:
        observed = adapter(
            "pty_observe",
            {"session_id": session_id, "idle_seconds": 0.1, "max_wait_seconds": 2.0},
        )
        assert observed.success is True
        payload = json.loads(observed.output)
        assert "observe ok" in payload["combined_output"]
    finally:
        adapter("pty_remove", {"session_id": session_id})


def test_pty_supervisor_interact_starts_and_observes() -> None:
    supervisor = PtySupervisor()
    observed = supervisor.interact(
        "test",
        command="printf 'interact ok\\n'",
        idle_seconds=0.1,
        max_wait_seconds=2.0,
    )
    session_id = observed["session_id"]
    try:
        assert observed["found"] is True
        assert observed["started"] is True
        assert "interact ok" in observed["combined_output"]
    finally:
        supervisor.remove("test", session_id)


def test_pty_tool_adapter_interact_surface() -> None:
    supervisor = PtySupervisor()
    adapter = PtyToolAdapter(supervisor, default_cwd=".")
    started = adapter(
        "pty_interact",
        {"command": "printf 'adapter interact\\n'", "idle_seconds": 0.1, "max_wait_seconds": 2.0},
    )
    assert started.success is True
    payload = json.loads(started.output)
    session_id = payload["session_id"]
    try:
        assert payload["started"] is True
        assert "adapter interact" in payload["combined_output"]
    finally:
        adapter("pty_remove", {"session_id": session_id})


def test_pty_tool_adapter_uses_default_cwd(tmp_path) -> None:
    supervisor = PtySupervisor()
    adapter = PtyToolAdapter(supervisor, default_cwd=str(tmp_path))
    started = adapter("pty_start", {"command": "pwd"})
    session_id = json.loads(started.output)["session_id"]
    try:
        deadline = time.time() + 3.0
        output = ""
        while time.time() < deadline:
            polled = adapter("pty_poll", {"session_id": session_id})
            output += json.loads(polled.output).get("output", "")
            if str(tmp_path) in output:
                break
            time.sleep(0.05)
        assert str(tmp_path) in output
        snapshot = supervisor.snapshot()
        assert snapshot["entries"][0]["cwd"] == str(tmp_path)
    finally:
        adapter("pty_remove", {"session_id": session_id})


def test_pty_supervisor_snapshot_surfaces_entries() -> None:
    supervisor = PtySupervisor()
    session_id = supervisor.start("scope_a", "sleep 1", rows=30, cols=100)
    try:
        snapshot = supervisor.snapshot()
        assert snapshot["total_count"] == 1
        assert snapshot["running_count"] == 1
        assert snapshot["scope_count"] == 1
        assert snapshot["entries"][0]["session_id"] == session_id
        assert snapshot["entries"][0]["rows"] == 30
        assert snapshot["entries"][0]["cols"] == 100
        assert snapshot["entries"][0]["cwd"]
    finally:
        supervisor.remove("scope_a", session_id)


# --- ANSI stripping and cleaned output tests ---


def test_strip_ansi_removes_csi_sequences() -> None:
    raw = "\x1b[31mred text\x1b[0m normal"
    assert strip_ansi(raw) == "red text normal"


def test_strip_ansi_removes_osc_sequences() -> None:
    raw = "\x1b]0;window title\x07some content"
    assert strip_ansi(raw) == "some content"


def test_strip_ansi_removes_cursor_movement() -> None:
    raw = "\x1b[2J\x1b[H\x1b[?25lhello\x1b[?25h"
    assert strip_ansi(raw) == "hello"


def test_strip_ansi_preserves_plain_text() -> None:
    plain = "hello world\nline two\n"
    assert strip_ansi(plain) == plain


def test_strip_ansi_removes_dec_private_and_dcs_sequences() -> None:
    raw = (
        "\x1b[>7u"          # DEC private CSI >
        "\x1b[>0q"          # DEC private CSI >
        "\x1b[>4;2m"        # DEC private CSI > with params
        "\x1b[<u"           # DEC private CSI <
        "\x1bPzz\x1b\\"    # DCS sequence
        "visible"
    )
    assert strip_ansi(raw) == "visible"


def test_strip_ansi_handles_complex_terminal_output() -> None:
    raw = (
        "\x1b[?2004h"           # Bracketed paste mode
        "\x1b[1;32muser@host\x1b[0m"  # Colored prompt
        ":\x1b[1;34m~/project\x1b[0m$ "  # Colored path
        "echo hello\r\n"
        "hello\r\n"
        "\x1b[?2004l"           # End bracketed paste
    )
    cleaned = strip_ansi(raw)
    assert "user@host" in cleaned
    assert "~/project" in cleaned
    assert "echo hello" in cleaned
    assert "hello" in cleaned
    assert "\x1b" not in cleaned


def test_pty_poll_includes_cleaned_output() -> None:
    supervisor = PtySupervisor()
    session_id = supervisor.start(
        "test",
        "printf '\\x1b[31mcolored\\x1b[0m plain\\n'",
    )
    try:
        output = ""
        cleaned = ""
        deadline = time.time() + 3.0
        while time.time() < deadline:
            result = supervisor.poll("test", session_id)
            output += result.get("output", "")
            cleaned += result.get("cleaned_output", "")
            if "plain" in cleaned:
                break
            time.sleep(0.05)
        assert "colored" in cleaned
        assert "plain" in cleaned
        assert "\x1b" not in cleaned
    finally:
        supervisor.remove("test", session_id)


def test_pty_observe_includes_cleaned_combined_output() -> None:
    supervisor = PtySupervisor()
    session_id = supervisor.start(
        "test",
        "printf '\\x1b[1mbold text\\x1b[0m\\n'",
    )
    try:
        observed = supervisor.observe_until_quiet(
            "test",
            session_id,
            idle_seconds=0.15,
            max_wait_seconds=3.0,
        )
        assert observed["found"] is True
        assert "bold text" in observed["cleaned_combined_output"]
        assert "\x1b" not in observed["cleaned_combined_output"]
        assert "\x1b" in observed["combined_output"] or "bold text" in observed["combined_output"]
        assert observed["screen_state"]["app"] == "printf"
    finally:
        supervisor.remove("test", session_id)


def test_pty_interact_includes_cleaned_combined_output() -> None:
    supervisor = PtySupervisor()
    observed = supervisor.interact(
        "test",
        command="printf '\\x1b[32mgreen\\x1b[0m ok\\n'",
        idle_seconds=0.1,
        max_wait_seconds=2.0,
    )
    session_id = observed["session_id"]
    try:
        assert observed["found"] is True
        assert "green" in observed["cleaned_combined_output"]
        assert "ok" in observed["cleaned_combined_output"]
        assert "\x1b" not in observed["cleaned_combined_output"]
        assert observed["screen_state"]["ready_for_input"] is False
    finally:
        supervisor.remove("test", session_id)


def test_infer_screen_state_detects_vim_insert_mode() -> None:
    state = infer_screen_state("vim notes.md", "-- INSERT --\n", running=True)

    assert state["app"] == "vim"
    assert state["mode"] == "insert"
    assert state["ready_for_input"] is True
    assert "editor" in state["indicators"]


def test_infer_screen_state_detects_shell_prompt() -> None:
    state = infer_screen_state("/bin/bash", "user@host:~/repo$ ", running=True)

    assert state["app"] == "bash"
    assert state["mode"] == "shell_prompt"
    assert state["ready_for_input"] is True
    assert "shell_prompt" in state["indicators"]


def test_infer_screen_state_detects_auth_gate() -> None:
    state = infer_screen_state("kilocode", "Please sign in to continue", running=True)

    assert state["app"] == "kilocode"
    assert state["mode"] == "auth_required"
    assert state["blocked"] is True
    assert "auth_required" in state["indicators"]


def test_infer_screen_state_detects_vim_write_error() -> None:
    state = infer_screen_state(
        "vim nested/missing/note.md",
        "E212: Can't open file for writing",
        running=True,
    )

    assert state["app"] == "vim"
    assert state["mode"] == "error_prompt"
    assert state["ready_for_input"] is True
    assert state["needs_input"] is True
    assert "vim_write_error" in state["indicators"]
    assert "editor" in state["indicators"]


def test_infer_screen_state_prefers_vim_write_error_over_insert_mode() -> None:
    state = infer_screen_state(
        "vim nested/missing/note.md",
        "-- INSERT --\nE212: Can't open file for writing\nPress ENTER or type command to continue",
        running=True,
    )

    assert state["app"] == "vim"
    assert state["mode"] == "error_prompt"
    assert state["ready_for_input"] is True
    assert state["needs_input"] is True
    assert "vim_write_error" in state["indicators"]
    assert "vim_insert" not in state["indicators"]


def test_infer_screen_state_detects_fragmented_vim_write_error() -> None:
    state = infer_screen_state(
        "vim nested/missing/note.md",
        '-- INSERT --0,1All:w\r"nested/missing/note.md" E212: Can\'t open file for writi\r\r\nng\b\r\nPress ENTER or type command to continue',
        running=True,
    )

    assert state["app"] == "vim"
    assert state["mode"] == "error_prompt"
    assert "vim_write_error" in state["indicators"]


def test_snapshot_includes_last_observed_state() -> None:
    supervisor = PtySupervisor()
    observed = supervisor.interact(
        "test",
        command="printf 'hello from snapshot\\n'",
        idle_seconds=0.1,
        max_wait_seconds=2.0,
    )
    session_id = observed["session_id"]
    try:
        snapshot = supervisor.snapshot()
        entry = next(item for item in snapshot["entries"] if item["session_id"] == session_id)
        assert entry["last_cleaned_output"]
        assert "hello from snapshot" in entry["last_cleaned_output"]
        assert entry["last_screen_state"]["app"] == "printf"
        assert entry["last_observed_at"] is not None
    finally:
        supervisor.remove("test", session_id)
