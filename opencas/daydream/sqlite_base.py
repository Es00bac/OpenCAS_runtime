"""Shared SQLite lifecycle helpers for daydream persistence modules."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import aiosqlite


class SqliteBackedStore:
    """Minimal async SQLite store shell used by the daydream subsystem."""

    SCHEMA = ""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        if self.SCHEMA:
            await self._db.executescript(self.SCHEMA)
        await self._migrate()
        await self._db.commit()
        return self

    async def _migrate(self) -> None:
        """Allow subclasses to apply incremental schema changes."""

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None
        return self._db

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
