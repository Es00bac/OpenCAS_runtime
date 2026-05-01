"""JSONL append-only telemetry store."""

from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Callable, Iterator, List, Optional

from .models import EventKind, TelemetryEvent

EventSubscriber = Callable[[TelemetryEvent], None]


class TelemetryStore:
    """Append-only JSONL store for telemetry events.

    Events are written to a rotating set of daily files under the base path.
    Reads are performed by scanning files in reverse chronological order.
    """

    def __init__(self, base_path: Path | str) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._subscribers: List[EventSubscriber] = []

    def _file_for_date(self, date_str: str) -> Path:
        return self.base_path / f"{date_str}.jsonl"

    def subscribe(self, subscriber: EventSubscriber) -> None:
        """Register a callback that receives every appended event in real time."""
        self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: EventSubscriber) -> None:
        """Remove a previously registered callback."""
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def append(self, event: TelemetryEvent) -> None:
        """Append a single event to the store durably and notify subscribers."""
        date_str = event.timestamp.strftime("%Y-%m-%d")
        filepath = self._file_for_date(date_str)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(event.to_jsonl())
            f.flush()
            os.fsync(f.fileno())
        for subscriber in list(self._subscribers):
            try:
                subscriber(event)
            except Exception:
                pass

    def _all_files(self) -> List[Path]:
        """Return daily JSONL files sorted newest first."""
        files = sorted(self.base_path.glob("*.jsonl"), reverse=True)
        return files

    def prune_old_files(self, max_age_days: int = 30, *, now: Optional[datetime] = None) -> int:
        """Delete telemetry files older than *max_age_days* and return deletion count."""
        max_age_days = int(max_age_days)
        if max_age_days < 0:
            return 0
        cutoff = (now or datetime.now()).date() - timedelta(days=max_age_days)
        removed = 0
        for filepath in self._all_files():
            file_date = _parse_date_from_filename(filepath.name)
            if file_date is None:
                continue
            if file_date < cutoff:
                filepath.unlink(missing_ok=True)
                removed += 1
        return removed

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


def _parse_date_from_filename(filename: str) -> Optional[datetime.date]:
    stem = Path(filename).stem
    try:
        return datetime.strptime(stem, "%Y-%m-%d").date()
    except ValueError:
        return None
