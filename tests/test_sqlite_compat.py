"""Regression coverage for Python 3.14 sqlite compatibility."""

from __future__ import annotations

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
