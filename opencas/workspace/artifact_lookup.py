"""Unified provenance lookup for workspace artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from opencas.workspace.scanner import sha256_file_sync


@dataclass(frozen=True)
class TimelineEntry:
    timestamp: datetime
    source: str
    event_kind: str
    summary: str
    ref_id: str
    evidence_payload: dict[str, Any] = field(default_factory=dict)
    source_table: str = ""

    @property
    def kind(self) -> str:
        return self.event_kind

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.astimezone(timezone.utc).isoformat(timespec="seconds")
        return payload


@dataclass(frozen=True)
class ArtifactLookupResult:
    query: dict[str, Any]
    current: dict[str, Any]
    sibling_paths: list[str]
    timeline: list[TimelineEntry]
    summary_counts: dict[str, int]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "current": self.current,
            "sibling_paths": self.sibling_paths,
            "summary_counts": self.summary_counts,
            "notes": self.notes,
            "timeline": [entry.to_dict() for entry in self.timeline],
        }


class ArtifactLookupService:
    """Join workspace, memory, workflow, and provenance evidence by path/checksum."""

    def __init__(
        self,
        *,
        workspace_store: Any,
        memory_store: Any,
        schedule_store: Optional[Any] = None,
        task_store: Optional[Any] = None,
        receipt_store: Optional[Any] = None,
        commitment_store: Optional[Any] = None,
        plan_store: Optional[Any] = None,
        provenance_path: Optional[Path] = None,
    ) -> None:
        self.workspace_store = workspace_store
        self.memory_store = memory_store
        self.schedule_store = schedule_store
        self.task_store = task_store
        self.receipt_store = receipt_store
        self.commitment_store = commitment_store
        self.plan_store = plan_store
        self.provenance_path = Path(provenance_path) if provenance_path else None

    async def lookup(
        self,
        *,
        path: Optional[str] = None,
        checksum: Optional[str] = None,
        limit: int = 50,
    ) -> ArtifactLookupResult:
        query_path = self._resolve_path(path)
        checksum = str(checksum or "").strip() or None
        notes: list[str] = []
        entries: list[TimelineEntry] = []

        if query_path and query_path.is_file():
            live_checksum = sha256_file_sync(query_path)
            if live_checksum != "ERROR_READING_FILE":
                checksum = checksum or live_checksum

        indexed_record = None
        if query_path:
            try:
                indexed_record = await self.workspace_store.get_path_record(query_path)
            except Exception as exc:
                notes.append(f"workspace path lookup failed: {type(exc).__name__}: {exc}")
        if indexed_record and not checksum:
            checksum = getattr(indexed_record, "current_checksum", None)

        sibling_paths = await self._sibling_paths(checksum)
        if not query_path and sibling_paths:
            query_path = Path(sibling_paths[0]).expanduser().resolve()

        current = await self._current_state(query_path, checksum, indexed_record)
        if current.get("exists_on_disk"):
            entries.append(
                TimelineEntry(
                    timestamp=self._timestamp_from_mtime(current.get("mtime")),
                    source="filesystem",
                    source_table="filesystem",
                    event_kind="CURRENT",
                    summary=f"Current file exists at {current.get('path')}",
                    ref_id=str(current.get("path") or query_path or checksum or "artifact"),
                    evidence_payload=current,
                )
            )

        needles = self._needles(query_path, checksum)
        if needles:
            entries.extend(await self._memory_entries(needles))
            entries.extend(await self._schedule_entries(needles))
            entries.extend(await self._task_entries(needles))
            entries.extend(await self._receipt_entries(needles))
            entries.extend(await self._commitment_entries(needles))
            entries.extend(await self._plan_entries(needles))
            entries.extend(self._provenance_entries(needles))
        else:
            notes.append("No path or checksum was available to join operational stores.")

        entries = self._dedupe_entries(entries)
        entries.sort(key=lambda entry: entry.timestamp)
        if len(entries) > limit:
            notes.append(f"timeline truncated from {len(entries)} to {limit} entries")
            entries = entries[:limit]

        counts = Counter(entry.source for entry in entries)
        if not entries:
            notes.append("no timeline entries found")
        if checksum and not sibling_paths:
            notes.append("no workspace_paths rows currently match the checksum")

        return ArtifactLookupResult(
            query={"path": str(query_path) if query_path else path, "checksum": checksum},
            current=current,
            sibling_paths=sibling_paths,
            timeline=entries,
            summary_counts=dict(counts),
            notes=notes,
        )

    @staticmethod
    def _resolve_path(path: Optional[str]) -> Optional[Path]:
        value = str(path or "").strip()
        if not value:
            return None
        return Path(value).expanduser().resolve()

    async def _sibling_paths(self, checksum: Optional[str]) -> list[str]:
        if not checksum:
            return []
        try:
            records = await self.workspace_store.get_paths_by_checksum(checksum, limit=25)
        except Exception:
            return []
        return [str(record.abs_path) for record in records]

    async def _current_state(
        self,
        path: Optional[Path],
        checksum: Optional[str],
        indexed_record: Any,
    ) -> dict[str, Any]:
        exists = bool(path and path.is_file())
        stat = None
        if exists and path is not None:
            try:
                stat = path.stat()
            except OSError:
                stat = None
        current_checksum = checksum
        if exists and path is not None and not current_checksum:
            live = sha256_file_sync(path)
            current_checksum = None if live == "ERROR_READING_FILE" else live
        gist = None
        if path is not None:
            try:
                lookup = await self.workspace_store.get_gist_for_path(path)
                gist = getattr(lookup, "gist_text", None) if lookup else None
            except Exception:
                gist = None
        return {
            "path": str(path) if path else None,
            "exists_on_disk": exists,
            "size_bytes": stat.st_size if stat else getattr(indexed_record, "size_bytes", None),
            "mtime": stat.st_mtime if stat else None,
            "mtime_ns": stat.st_mtime_ns if stat else getattr(indexed_record, "mtime_ns", None),
            "checksum": current_checksum,
            "indexed": indexed_record is not None,
            "indexed_checksum": getattr(indexed_record, "current_checksum", None),
            "gist": gist,
        }

    @staticmethod
    def _needles(path: Optional[Path], checksum: Optional[str]) -> list[str]:
        out: list[str] = []
        if path is not None:
            out.append(str(path))
            out.append(path.as_posix())
            out.append(path.name)
            if len(path.parts) >= 2:
                out.append(path.parent.name)
        if checksum:
            out.append(checksum)
        seen = set()
        unique: list[str] = []
        for item in out:
            if item and item not in seen:
                unique.append(item)
                seen.add(item)
        return unique

    async def _memory_entries(self, needles: list[str]) -> list[TimelineEntry]:
        db = getattr(self.memory_store, "_db", None)
        if db is None:
            return []
        clauses = " OR ".join("content LIKE ?" for _ in needles)
        cursor = await db.execute(
            f"""
            SELECT episode_id, created_at, kind, session_id, content, payload
            FROM episodes
            WHERE {clauses}
            ORDER BY created_at DESC
            LIMIT 75
            """,
            tuple(f"%{needle}%" for needle in needles),
        )
        rows = await cursor.fetchall()
        entries = []
        for row in rows:
            kind = str(row["kind"] or "episode").upper()
            entries.append(
                TimelineEntry(
                    timestamp=self._parse_ts(row["created_at"]),
                    source="memory",
                    source_table="episodes",
                    event_kind=kind,
                    summary=self._compact(row["content"]),
                    ref_id=str(row["episode_id"]),
                    evidence_payload={
                        "episode_id": row["episode_id"],
                        "kind": row["kind"],
                        "session_id": row["session_id"],
                        "payload": self._json_obj(row["payload"]),
                    },
                )
            )
        return entries

    async def _schedule_entries(self, needles: list[str]) -> list[TimelineEntry]:
        store = self.schedule_store
        db = getattr(store, "_db", None)
        if db is None:
            return []
        clauses, params = self._like_clause(("title", "description", "objective", "meta"), needles)
        cursor = await db.execute(
            f"""
            SELECT schedule_id, created_at, updated_at, title, objective, meta
            FROM schedules
            WHERE {clauses}
            ORDER BY updated_at DESC
            LIMIT 25
            """,
            params,
        )
        entries = [
            TimelineEntry(
                timestamp=self._parse_ts(row["updated_at"] or row["created_at"]),
                source="schedule",
                source_table="schedules",
                event_kind="SCHEDULE",
                summary=self._compact(row["objective"] or row["title"]),
                ref_id=str(row["schedule_id"]),
                evidence_payload=dict(row),
            )
            for row in await cursor.fetchall()
        ]
        run_clauses, run_params = self._like_clause(("meta", "task_id", "error"), needles)
        cursor = await db.execute(
            f"""
            SELECT run_id, schedule_id, scheduled_for, started_at, finished_at, status, task_id, meta
            FROM schedule_runs
            WHERE {run_clauses}
            ORDER BY started_at DESC
            LIMIT 25
            """,
            run_params,
        )
        for row in await cursor.fetchall():
            entries.append(
                TimelineEntry(
                    timestamp=self._parse_ts(row["started_at"]),
                    source="schedule",
                    source_table="schedule_runs",
                    event_kind="RUN",
                    summary=self._compact(f"schedule {row['schedule_id']} run {row['status']} task={row['task_id']}"),
                    ref_id=str(row["run_id"]),
                    evidence_payload=dict(row),
                )
            )
        return entries

    async def _task_entries(self, needles: list[str]) -> list[TimelineEntry]:
        store = self.task_store
        db = getattr(store, "_db", None)
        if db is None:
            return []
        columns = ("objective", "artifacts", "meta", "result_output")
        clauses, params = self._like_clause(columns, needles)
        cursor = await db.execute(
            f"""
            SELECT task_id, created_at, updated_at, objective, status, artifacts, result_output, result_timestamp
            FROM tasks
            WHERE {clauses}
            ORDER BY COALESCE(result_timestamp, updated_at) DESC
            LIMIT 25
            """,
            params,
        )
        return [
            TimelineEntry(
                timestamp=self._parse_ts(row["result_timestamp"] or row["updated_at"] or row["created_at"]),
                source="task",
                source_table="tasks",
                event_kind="TASK",
                summary=self._compact(row["result_output"] or row["objective"]),
                ref_id=str(row["task_id"]),
                evidence_payload=dict(row),
            )
            for row in await cursor.fetchall()
        ]

    async def _receipt_entries(self, needles: list[str]) -> list[TimelineEntry]:
        store = self.receipt_store
        db = getattr(store, "_db", None)
        if db is None:
            return []
        columns = ("objective", "plan", "phases", "output")
        clauses, params = self._like_clause(columns, needles)
        cursor = await db.execute(
            f"""
            SELECT receipt_id, task_id, objective, plan, created_at, completed_at, success, output
            FROM receipts
            WHERE {clauses}
            ORDER BY COALESCE(completed_at, created_at) DESC
            LIMIT 25
            """,
            params,
        )
        return [
            TimelineEntry(
                timestamp=self._parse_ts(row["completed_at"] or row["created_at"]),
                source="receipt",
                source_table="receipts",
                event_kind="RECEIPT",
                summary=self._compact(row["output"] or row["objective"]),
                ref_id=str(row["receipt_id"]),
                evidence_payload=dict(row),
            )
            for row in await cursor.fetchall()
        ]

    async def _commitment_entries(self, needles: list[str]) -> list[TimelineEntry]:
        store = self.commitment_store
        db = getattr(store, "_db", None)
        if db is None:
            return []
        columns = ("content", "linked_work_ids", "linked_task_ids", "tags", "meta")
        clauses, params = self._like_clause(columns, needles)
        cursor = await db.execute(
            f"""
            SELECT commitment_id, created_at, updated_at, content, status, meta
            FROM commitments
            WHERE {clauses}
            ORDER BY updated_at DESC
            LIMIT 25
            """,
            params,
        )
        return [
            TimelineEntry(
                timestamp=self._parse_ts(row["updated_at"] or row["created_at"]),
                source="commitment",
                source_table="commitments",
                event_kind="COMMITMENT",
                summary=self._compact(row["content"]),
                ref_id=str(row["commitment_id"]),
                evidence_payload=dict(row),
            )
            for row in await cursor.fetchall()
        ]

    async def _plan_entries(self, needles: list[str]) -> list[TimelineEntry]:
        store = self.plan_store
        db = getattr(store, "_db", None)
        if db is None:
            return []
        clauses, params = self._like_clause(("content",), needles)
        cursor = await db.execute(
            f"""
            SELECT plan_id, created_at, updated_at, status, content, project_id, task_id
            FROM plans
            WHERE {clauses}
            ORDER BY updated_at DESC
            LIMIT 25
            """,
            params,
        )
        entries = [
            TimelineEntry(
                timestamp=self._parse_ts(row["updated_at"] or row["created_at"]),
                source="plan",
                source_table="plans",
                event_kind="PLAN",
                summary=self._compact(row["content"]),
                ref_id=str(row["plan_id"]),
                evidence_payload=dict(row),
            )
            for row in await cursor.fetchall()
        ]
        columns = ("args", "result_summary")
        clauses, params = self._like_clause(columns, needles)
        cursor = await db.execute(
            f"""
            SELECT action_id, plan_id, tool_name, args, result_summary, success, timestamp
            FROM plan_actions
            WHERE {clauses}
            ORDER BY timestamp DESC
            LIMIT 25
            """,
            params,
        )
        for row in await cursor.fetchall():
            entries.append(
                TimelineEntry(
                    timestamp=self._parse_ts(row["timestamp"]),
                    source="plan",
                    source_table="plan_actions",
                    event_kind="PLAN_ACTION",
                    summary=self._compact(f"{row['tool_name']}: {row['result_summary']}"),
                    ref_id=str(row["action_id"]),
                    evidence_payload=dict(row),
                )
            )
        return entries

    def _provenance_entries(self, needles: list[str]) -> list[TimelineEntry]:
        path = self.provenance_path
        if path is None or not path.exists():
            return []
        entries = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not any(needle in line for needle in needles):
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    timestamp = (
                        data.get("created_at")
                        or data.get("recorded_at")
                        or data.get("timestamp")
                        or data.get("ts")
                        or data.get("occurred_at")
                    )
                    details = data.get("details") if isinstance(data.get("details"), dict) else {}
                    summary = (
                        data.get("why")
                        or data.get("summary")
                        or data.get("triggering_artifact")
                        or data.get("target_entity")
                        or details.get("title")
                        or details.get("target_entity")
                        or details.get("source_artifact")
                        or line
                    )
                    ref_id = str(
                        data.get("transition_id")
                        or data.get("event_id")
                        or data.get("source_link")
                        or data.get("target_entity")
                        or len(entries)
                    )
                    entries.append(
                        TimelineEntry(
                            timestamp=self._parse_ts(timestamp),
                            source="provenance",
                            source_table="provenance.transitions.jsonl",
                            event_kind=str(
                                data.get("event_type")
                                or data.get("kind")
                                or data.get("type")
                                or "TRANSITION"
                            ),
                            summary=self._compact(summary),
                            ref_id=ref_id,
                            evidence_payload=self._truncate_payload(data),
                        )
                    )
                    if len(entries) >= 25:
                        break
        except OSError:
            return []
        return entries

    @staticmethod
    def _like_clause(columns: tuple[str, ...], needles: list[str]) -> tuple[str, tuple[str, ...]]:
        clauses: list[str] = []
        params: list[str] = []
        for column in columns:
            for needle in needles:
                clauses.append(f"COALESCE({column}, '') LIKE ?")
                params.append(f"%{needle}%")
        return " OR ".join(clauses), tuple(params)

    @staticmethod
    def _dedupe_entries(entries: list[TimelineEntry]) -> list[TimelineEntry]:
        seen: set[tuple[str, str, str]] = set()
        out: list[TimelineEntry] = []
        for entry in entries:
            key = (entry.source_table, entry.ref_id, entry.event_kind)
            if key in seen:
                continue
            seen.add(key)
            out.append(entry)
        return out

    @staticmethod
    def _parse_ts(value: Any) -> datetime:
        text = str(value or "").strip()
        if not text:
            return datetime.fromtimestamp(0, tz=timezone.utc)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return datetime.fromtimestamp(0, tz=timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _timestamp_from_mtime(value: Any) -> datetime:
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return datetime.fromtimestamp(0, tz=timezone.utc)

    @staticmethod
    def _compact(value: Any, limit: int = 260) -> str:
        text = " ".join(str(value or "").split())
        return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _json_obj(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str) or not value.strip():
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict) and "payload" in parsed and isinstance(parsed["payload"], dict):
            return parsed["payload"]
        return parsed if isinstance(parsed, dict) else {}

    @classmethod
    def _truncate_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, default=str)
        if len(encoded) <= 1200:
            return payload
        out: dict[str, Any] = {}
        for key, value in payload.items():
            out[key] = cls._compact(value, limit=220)
        return out
