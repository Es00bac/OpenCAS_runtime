"""Persistent store for plugin lifecycle state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite


class PluginStore:
    """SQLite-backed store for installed/enabled plugin state."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._ensure_schema()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _ensure_schema(self) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS plugins (
                plugin_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                source TEXT,
                path TEXT,
                enabled INTEGER,
                installed_at TEXT,
                updated_at TEXT,
                manifest TEXT
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_plugins_enabled ON plugins(enabled)"
        )
        await self._db.commit()

    async def install(
        self,
        plugin_id: str,
        name: str,
        description: str,
        source: str,
        path: str,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT INTO plugins
            (plugin_id, name, description, source, path, enabled, installed_at, updated_at, manifest)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(plugin_id) DO UPDATE SET
                name=excluded.name,
                description=excluded.description,
                source=excluded.source,
                path=excluded.path,
                enabled=excluded.enabled,
                updated_at=excluded.updated_at,
                manifest=excluded.manifest
            """,
            (
                plugin_id,
                name,
                description,
                source,
                path,
                1,
                now,
                now,
                json.dumps(manifest) if manifest is not None else "{}",
            ),
        )
        await self._db.commit()

    async def uninstall(self, plugin_id: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "DELETE FROM plugins WHERE plugin_id = ?",
            (plugin_id,),
        )
        await self._db.commit()

    async def set_enabled(self, plugin_id: str, enabled: bool) -> None:
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE plugins SET enabled = ?, updated_at = ? WHERE plugin_id = ?",
            (1 if enabled else 0, now, plugin_id),
        )
        await self._db.commit()

    async def is_installed(self, plugin_id: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "SELECT 1 FROM plugins WHERE plugin_id = ?",
            (plugin_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row is not None

    async def is_enabled(self, plugin_id: str) -> bool:
        assert self._db is not None
        async with self._db.execute(
            "SELECT enabled FROM plugins WHERE plugin_id = ?",
            (plugin_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return row is not None and bool(row[0])

    async def list_installed(self) -> List[Dict[str, Any]]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT plugin_id, name, description, source, path, enabled, installed_at, updated_at, manifest FROM plugins ORDER BY plugin_id"
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    async def list_enabled(self) -> List[Dict[str, Any]]:
        assert self._db is not None
        async with self._db.execute(
            "SELECT plugin_id, name, description, source, path, enabled, installed_at, updated_at, manifest FROM plugins WHERE enabled = 1 ORDER BY plugin_id"
        ) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> Dict[str, Any]:
        manifest_raw = row[8] or "{}"
        manifest = json.loads(manifest_raw)
        return {
            "plugin_id": row[0],
            "name": row[1],
            "description": row[2],
            "source": row[3],
            "path": row[4],
            "enabled": bool(row[5]),
            "installed_at": row[6],
            "updated_at": row[7],
            "manifest": manifest,
        }
