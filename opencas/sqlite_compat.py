"""Compatibility shims for sqlite async access on Python 3.14."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional


class _CompatCursor:
    """Minimal async cursor wrapper backed by sqlite3 + asyncio.to_thread."""

    def __init__(
        self,
        connection: "_CompatConnection",
        cursor: sqlite3.Cursor,
    ) -> None:
        self._connection = connection
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    @property
    def lastrowid(self) -> int | None:
        return self._cursor.lastrowid

    async def fetchone(self) -> Any:
        async with self._connection._lock:
            return self._cursor.fetchone()

    async def fetchall(self) -> list[Any]:
        async with self._connection._lock:
            return self._cursor.fetchall()

    async def close(self) -> None:
        async with self._connection._lock:
            self._cursor.close()


class _CompatExecuteContext:
    """Allow both ``await conn.execute(...)`` and ``async with conn.execute(...)``."""

    def __init__(
        self,
        factory,
    ) -> None:
        self._factory = factory
        self._cursor: _CompatCursor | None = None

    async def _resolve(self) -> _CompatCursor:
        if self._cursor is None:
            self._cursor = await self._factory()
        return self._cursor

    def __await__(self):
        return self._resolve().__await__()

    async def __aenter__(self) -> _CompatCursor:
        return await self._resolve()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._cursor is not None:
            await self._cursor.close()


class _CompatConnection:
    """Minimal aiosqlite-compatible connection wrapper."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._lock = asyncio.Lock()

    @property
    def row_factory(self) -> Any:
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self._conn.row_factory = value

    def execute(
        self,
        sql: str,
        parameters: Optional[Iterable[Any]] = None,
    ) -> _CompatExecuteContext:
        async def _run() -> _CompatCursor:
            async with self._lock:
                cursor = self._conn.execute(sql, tuple(parameters or ()))
            return _CompatCursor(self, cursor)

        return _CompatExecuteContext(_run)

    async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> None:
        async with self._lock:
            self._conn.executemany(sql, list(parameters))

    async def executescript(self, sql: str) -> None:
        async with self._lock:
            self._conn.executescript(sql)

    async def commit(self) -> None:
        async with self._lock:
            self._conn.commit()

    async def rollback(self) -> None:
        async with self._lock:
            self._conn.rollback()

    async def close(self) -> None:
        async with self._lock:
            self._conn.close()


def patch_aiosqlite_for_python314() -> None:
    """Patch aiosqlite connect on Python 3.14.

    In this environment, aiosqlite's worker-thread handoff can stall under
    Python 3.14 during connection setup. OpenCAS only relies on a small, stable
    subset of the aiosqlite API, so we replace ``connect`` with a
    `sqlite3`-backed async compatibility layer that preserves the calling
    contract used by the project.
    """

    if sys.version_info < (3, 14):
        return

    try:
        import aiosqlite
    except Exception:
        return

    if getattr(aiosqlite, "_opencas_py314_patch", False):
        return

    async def _connect(database: str | Path, **kwargs: Any) -> _CompatConnection:
        kwargs = dict(kwargs)
        kwargs.setdefault("check_same_thread", False)
        raw = sqlite3.connect(str(database), **kwargs)
        return _CompatConnection(raw)

    aiosqlite.connect = _connect
    aiosqlite.Connection = _CompatConnection
    aiosqlite._opencas_py314_patch = True
