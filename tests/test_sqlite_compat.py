"""Regression coverage for Python 3.14 sqlite compatibility."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import aiosqlite
import pytest

import opencas  # noqa: F401 - import triggers sqlite compatibility patch


@pytest.mark.asyncio
async def test_aiosqlite_connect_remains_usable_under_python314(tmp_path: Path) -> None:
    db = await aiosqlite.connect(str(tmp_path / "compat.db"))
    db.row_factory = aiosqlite.Row

    async with db.execute("SELECT 1 AS value") as cursor:
        row = await cursor.fetchone()

    assert row["value"] == 1
    await db.close()


@pytest.mark.asyncio
async def test_python314_sqlite_compat_applies_busy_timeout(tmp_path: Path) -> None:
    db = await aiosqlite.connect(str(tmp_path / "compat-timeout.db"))
    try:
        async with db.execute("PRAGMA busy_timeout") as cursor:
            row = await cursor.fetchone()
    finally:
        await db.close()

    assert row[0] >= 30000


@pytest.mark.asyncio
async def test_python314_sqlite_compat_runs_slow_queries_off_event_loop(tmp_path: Path) -> None:
    db = await aiosqlite.connect(str(tmp_path / "compat-slow.db"))
    db.row_factory = aiosqlite.Row
    db._conn.create_function("slow", 1, lambda seconds: time.sleep(float(seconds)) or seconds)

    ticks = 0
    stop = False

    async def ticker() -> None:
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.005)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    before = ticks
    try:
        async with db.execute("SELECT slow(0.08) AS value") as cursor:
            row = await cursor.fetchone()
    finally:
        stop = True
        await task
        await db.close()

    assert row["value"] == 0.08
    assert ticks - before >= 5
