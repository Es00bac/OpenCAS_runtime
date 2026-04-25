from __future__ import annotations

import signal
from unittest.mock import patch

from scripts.sweep_operator_processes import (
    ProcessEntry,
    find_target_processes,
    kill_processes,
)


def test_find_target_processes_defaults_exclude_recent_and_nonmatches() -> None:
    rows = [
        ProcessEntry(pid=100, etimes=120, command="python scripts/run_live_debug_validation.py"),
        ProcessEntry(pid=101, etimes=10, command="python scripts/run_live_debug_validation.py"),
        ProcessEntry(pid=102, etimes=300, command="python app.py"),
    ]

    targets = find_target_processes(
        rows,
        patterns=("run_live_debug_validation.py",),
        older_than_seconds=30,
    )

    assert [item.pid for item in targets] == [100]


def test_find_target_processes_can_include_local_tools() -> None:
    rows = [
        ProcessEntry(pid=200, etimes=90, command=".venv/bin/python -m pytest -q"),
        ProcessEntry(pid=201, etimes=90, command="mpv clip.mp3"),
    ]

    targets = find_target_processes(
        rows,
        patterns=(),
        older_than_seconds=30,
        include_local_test_tools=True,
    )

    assert [item.pid for item in targets] == [200, 201]


def test_find_target_processes_excludes_selected_pids() -> None:
    rows = [
        ProcessEntry(pid=300, etimes=90, command="kilocode"),
        ProcessEntry(pid=301, etimes=90, command="kilo run"),
    ]

    targets = find_target_processes(
        rows,
        patterns=("kilocode", "kilo run"),
        older_than_seconds=30,
        exclude_pids=(300,),
    )

    assert [item.pid for item in targets] == [301]


def test_kill_processes_ignores_missing_processes() -> None:
    rows = [
        ProcessEntry(pid=400, etimes=90, command="kilocode"),
        ProcessEntry(pid=401, etimes=90, command="kilo run"),
    ]

    def fake_kill(pid: int, sig: int) -> None:
        assert sig == signal.SIGTERM
        if pid == 401:
            raise ProcessLookupError

    with patch("scripts.sweep_operator_processes.os.kill", side_effect=fake_kill) as mocked:
        killed = kill_processes(rows)

    assert killed == [400]
    assert mocked.call_count == 2
