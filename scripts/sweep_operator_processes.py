#!/usr/bin/env python3
"""Inspect and optionally kill stale OpenCAS operator/test processes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence


DEFAULT_PATTERNS: tuple[str, ...] = (
    "run_live_debug_validation.py",
    "run_qualification_cycle.py",
    "kilocode",
    "kilo run",
)

LOCAL_PATTERNS: tuple[str, ...] = (
    "pytest",
    "mpv",
    "edge-tts",
)


@dataclass
class ProcessEntry:
    pid: int
    etimes: int
    command: str


def _ps_rows() -> List[ProcessEntry]:
    proc = subprocess.run(
        ["ps", "-eo", "pid=,etimes=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: List[ProcessEntry] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            etimes = int(parts[1])
        except ValueError:
            continue
        rows.append(ProcessEntry(pid=pid, etimes=etimes, command=parts[2]))
    return rows


def _matches_any(command: str, patterns: Sequence[str]) -> bool:
    return any(pattern in command for pattern in patterns)


def find_target_processes(
    rows: Iterable[ProcessEntry],
    *,
    patterns: Sequence[str],
    older_than_seconds: int,
    include_local_test_tools: bool = False,
    exclude_pids: Sequence[int] = (),
) -> List[ProcessEntry]:
    active_patterns = list(patterns)
    if include_local_test_tools:
        active_patterns.extend(LOCAL_PATTERNS)
    excluded = set(exclude_pids)
    targets: List[ProcessEntry] = []
    for row in rows:
        if row.pid in excluded:
            continue
        if row.etimes < older_than_seconds:
            continue
        if not _matches_any(row.command, active_patterns):
            continue
        targets.append(row)
    return targets


def kill_processes(rows: Iterable[ProcessEntry], *, sig: int = signal.SIGTERM) -> List[int]:
    killed: List[int] = []
    for row in rows:
        try:
            os.kill(row.pid, sig)
        except ProcessLookupError:
            continue
        killed.append(row.pid)
    return killed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and optionally kill stale OpenCAS operator/test processes."
    )
    parser.add_argument(
        "--older-than-seconds",
        type=int,
        default=30,
        help="Only target matching processes at least this old.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Additional substring pattern to match. Repeatable.",
    )
    parser.add_argument(
        "--include-local-test-tools",
        action="store_true",
        help="Also target local-only helpers such as pytest, mpv, and edge-tts.",
    )
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Send SIGTERM to matched processes instead of only reporting them.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _ps_rows()
    patterns = list(DEFAULT_PATTERNS)
    patterns.extend(args.pattern)
    current_pid = os.getpid()
    parent_pid = os.getppid()
    targets = find_target_processes(
        rows,
        patterns=patterns,
        older_than_seconds=max(0, args.older_than_seconds),
        include_local_test_tools=bool(args.include_local_test_tools),
        exclude_pids=(current_pid, parent_pid),
    )
    killed: List[int] = []
    if args.kill:
        killed = kill_processes(targets)

    payload = {
        "count": len(targets),
        "killed": killed,
        "targets": [asdict(item) for item in targets],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        if not targets:
            print("No matching stale processes found.")
        else:
            for item in targets:
                action = "terminated" if item.pid in killed else "matched"
                print(f"{action}: pid={item.pid} age={item.etimes}s cmd={item.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
