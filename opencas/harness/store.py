"""SQLite store for research notebooks, entries, and objective loops."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

from .models import (
    DeliverableSchema,
    NotebookEntry,
    NotebookEntryKind,
    ObjectiveLoop,
    ObjectiveStatus,
    ResearchNotebook,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_notebooks (
    notebook_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    deliverable_schema TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_notebooks_updated_at ON research_notebooks(updated_at);

CREATE TABLE IF NOT EXISTS notebook_entries (
    entry_id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    source_episode_ids TEXT NOT NULL DEFAULT '[]',
    source_task_ids TEXT NOT NULL DEFAULT '[]',
    meta TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (notebook_id) REFERENCES research_notebooks(notebook_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_entries_notebook_id ON notebook_entries(notebook_id);
CREATE INDEX IF NOT EXISTS idx_entries_kind ON notebook_entries(kind);

CREATE TABLE IF NOT EXISTS objective_loops (
    loop_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    notebook_id TEXT,
    generated_task_ids TEXT NOT NULL DEFAULT '[]',
    completion_criteria TEXT NOT NULL DEFAULT '[]',
    meta TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (notebook_id) REFERENCES research_notebooks(notebook_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_loops_status ON objective_loops(status);
CREATE INDEX IF NOT EXISTS idx_loops_updated_at ON objective_loops(updated_at);
"""


class HarnessStore:
    """Async SQLite store for harness models."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "HarnessStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    # Notebooks

    async def save_notebook(self, notebook: ResearchNotebook) -> None:
        assert self._db is not None
        deliverable_schema = (
            json.dumps(notebook.deliverable_schema.model_dump(mode="json"))
            if notebook.deliverable_schema else None
        )
        await self._db.execute(
            """
            INSERT INTO research_notebooks (
                notebook_id, created_at, updated_at, title, description,
                deliverable_schema, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(notebook_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                title = excluded.title,
                description = excluded.description,
                deliverable_schema = excluded.deliverable_schema,
                meta = excluded.meta
            """,
            (
                str(notebook.notebook_id),
                notebook.created_at.isoformat(),
                notebook.updated_at.isoformat(),
                notebook.title,
                notebook.description,
                deliverable_schema,
                json.dumps(notebook.meta),
            ),
        )
        await self._db.commit()

    async def get_notebook(self, notebook_id: str) -> Optional[ResearchNotebook]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM research_notebooks WHERE notebook_id = ?", (notebook_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        entries = await self._list_entries_for_notebook(notebook_id)
        return self._row_to_notebook(row, entries)

    async def list_notebooks(self, limit: int = 100, offset: int = 0) -> List[ResearchNotebook]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM research_notebooks
            ORDER BY updated_at DESC LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = await cursor.fetchall()
        result: List[ResearchNotebook] = []
        for row in rows:
            entries = await self._list_entries_for_notebook(row["notebook_id"])
            result.append(self._row_to_notebook(row, entries))
        return result

    # Entries

    async def add_entry(self, notebook_id: str, entry: NotebookEntry) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO notebook_entries (
                entry_id, notebook_id, created_at, kind, content,
                source_episode_ids, source_task_ids, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(entry.entry_id),
                notebook_id,
                entry.created_at.isoformat(),
                entry.kind.value,
                entry.content,
                json.dumps(entry.source_episode_ids),
                json.dumps(entry.source_task_ids),
                json.dumps(entry.meta),
            ),
        )
        await self._db.commit()

    async def _list_entries_for_notebook(self, notebook_id: str) -> List[NotebookEntry]:
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT * FROM notebook_entries
            WHERE notebook_id = ?
            ORDER BY created_at DESC
            """,
            (notebook_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_entry(r) for r in rows]

    # Objective loops

    async def save_loop(self, loop: ObjectiveLoop) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO objective_loops (
                loop_id, created_at, updated_at, status, title, description,
                notebook_id, generated_task_ids, completion_criteria, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(loop_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                status = excluded.status,
                title = excluded.title,
                description = excluded.description,
                notebook_id = excluded.notebook_id,
                generated_task_ids = excluded.generated_task_ids,
                completion_criteria = excluded.completion_criteria,
                meta = excluded.meta
            """,
            (
                str(loop.loop_id),
                loop.created_at.isoformat(),
                loop.updated_at.isoformat(),
                loop.status.value,
                loop.title,
                loop.description,
                loop.notebook_id,
                json.dumps(loop.generated_task_ids),
                json.dumps(loop.completion_criteria),
                json.dumps(loop.meta),
            ),
        )
        await self._db.commit()

    async def get_loop(self, loop_id: str) -> Optional[ObjectiveLoop]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM objective_loops WHERE loop_id = ?", (loop_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_loop(row)

    async def list_loops(
        self,
        status: Optional[ObjectiveStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ObjectiveLoop]:
        assert self._db is not None
        if status:
            cursor = await self._db.execute(
                """
                SELECT * FROM objective_loops
                WHERE status = ?
                ORDER BY updated_at DESC LIMIT ? OFFSET ?
                """,
                (status.value, limit, offset),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT * FROM objective_loops
                ORDER BY updated_at DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        rows = await cursor.fetchall()
        return [self._row_to_loop(r) for r in rows]

    # Helpers

    @staticmethod
    def _row_to_notebook(row: aiosqlite.Row, entries: List[NotebookEntry]) -> ResearchNotebook:
        deliverable_schema = None
        if row["deliverable_schema"]:
            schema_data = json.loads(row["deliverable_schema"])
            deliverable_schema = DeliverableSchema.model_validate(schema_data)
        return ResearchNotebook(
            notebook_id=row["notebook_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            title=row["title"],
            description=row["description"],
            entries=entries,
            deliverable_schema=deliverable_schema,
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )

    @staticmethod
    def _row_to_entry(row: aiosqlite.Row) -> NotebookEntry:
        return NotebookEntry(
            entry_id=row["entry_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            kind=NotebookEntryKind(row["kind"]),
            content=row["content"],
            source_episode_ids=json.loads(row["source_episode_ids"]) if row["source_episode_ids"] else [],
            source_task_ids=json.loads(row["source_task_ids"]) if row["source_task_ids"] else [],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )

    @staticmethod
    def _row_to_loop(row: aiosqlite.Row) -> ObjectiveLoop:
        return ObjectiveLoop(
            loop_id=row["loop_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            status=ObjectiveStatus(row["status"]),
            title=row["title"],
            description=row["description"],
            notebook_id=row["notebook_id"],
            generated_task_ids=json.loads(row["generated_task_ids"]) if row["generated_task_ids"] else [],
            completion_criteria=json.loads(row["completion_criteria"]) if row["completion_criteria"] else [],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )
