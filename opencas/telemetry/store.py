"""JSONL append-only telemetry store."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from .models import EventKind, TelemetryEvent


class TelemetryStore:
    """Append-only JSONL store for telemetry events.

    Events are written to a rotating set of daily files under the base path.
    Reads are performed by scanning files in reverse chronological order.
    """

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _file_for_date(self, date_str: str) -> Path:
        return self.base_path / f"{date_str}.jsonl"

    def append(self, event: TelemetryEvent) -> None:
        """Append a single event to the store durably."""
        date_str = event.timestamp.strftime("%Y-%m-%d")
        filepath = self._file_for_date(date_str)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(event.to_jsonl())
            f.flush()
            os.fsync(f.fileno())

    def _all_files(self) -> List[Path]:
        """Return daily JSONL files sorted newest first."""
        files = sorted(self.base_path.glob("*.jsonl"), reverse=True)
        return files

    def query(
        self,
        kinds: Optional[List[EventKind]] = None,
        session_id: Optional[str] = None,
        span_id: Optional[str] = None,
        limit: int = 1000,
        predicate: Optional[Callable[[TelemetryEvent], bool]] = None,
    ) -> List[TelemetryEvent]:
        """Query events with optional filters.

        Scans files newest-first and returns up to *limit* matching events
        in chronological order.
        """
        results: List[TelemetryEvent] = []
        kind_set = set(kinds) if kinds else None

        for filepath in self._all_files():
            if len(results) >= limit:
                break
            for event in self._read_file(filepath):
                if kind_set and event.kind not in kind_set:
                    continue
                if session_id and event.session_id != session_id:
                    continue
                if span_id and event.span_id != span_id:
                    continue
                if predicate and not predicate(event):
                    continue
                results.append(event)
                if len(results) >= limit:
                    break

        # Return chronologically sorted
        results.sort(key=lambda e: e.timestamp)
        return results

    def iter_all(self) -> Iterator[TelemetryEvent]:
        """Iterate all events in chronological order."""
        for filepath in reversed(self._all_files()):
            yield from self._read_file(filepath)

    @staticmethod
    def _read_file(filepath: Path) -> Iterator[TelemetryEvent]:
        if not filepath.exists():
            return
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield TelemetryEvent.from_jsonl(line)
                except (json.JSONDecodeError, ValueError):
                    # Skip corrupt lines rather than crash
                    continue
