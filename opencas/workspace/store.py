import aiosqlite
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import (
    WorkspaceChecksumRecord,
    WorkspaceGistAttempt,
    WorkspaceGistLookupResult,
    WorkspaceGistRecord,
    WorkspacePathRecord,
)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS workspace_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root_path TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    files_seen INTEGER NOT NULL DEFAULT 0,
    files_hashed INTEGER NOT NULL DEFAULT 0,
    files_reused INTEGER NOT NULL DEFAULT 0,
    files_changed INTEGER NOT NULL DEFAULT 0,
    files_deleted INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS workspace_paths (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    abs_path TEXT NOT NULL UNIQUE,
    rel_path TEXT,
    parent_dir TEXT NOT NULL,
    file_name TEXT NOT NULL,
    extension TEXT,
    exists_flag INTEGER NOT NULL DEFAULT 1,
    file_kind TEXT NOT NULL,
    size_bytes INTEGER,
    mtime_ns INTEGER,
    current_checksum TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    last_scan_id INTEGER,
    last_error TEXT,
    FOREIGN KEY(last_scan_id) REFERENCES workspace_scans(id)
);

CREATE INDEX IF NOT EXISTS idx_workspace_paths_parent_dir ON workspace_paths(parent_dir);
CREATE INDEX IF NOT EXISTS idx_workspace_paths_checksum ON workspace_paths(current_checksum);

CREATE TABLE IF NOT EXISTS workspace_checksums (
    checksum TEXT PRIMARY KEY,
    size_bytes INTEGER NOT NULL,
    file_kind TEXT NOT NULL,
    mime_type TEXT,
    content_text_status TEXT NOT NULL,
    content_preview TEXT,
    content_embedding_ref TEXT,
    content_embedding_model TEXT,
    content_embedding_dim INTEGER,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_gists (
    checksum TEXT PRIMARY KEY,
    gist_text TEXT NOT NULL,
    gist_json TEXT,
    llm_model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    gist_embedding_ref TEXT,
    gist_embedding_model TEXT,
    gist_embedding_dim INTEGER,
    cosine_similarity REAL,
    drift_score REAL,
    accepted_flag INTEGER NOT NULL DEFAULT 0,
    needs_further_reading INTEGER NOT NULL DEFAULT 0,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(checksum) REFERENCES workspace_checksums(checksum)
);

CREATE TABLE IF NOT EXISTS workspace_gist_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checksum TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    gist_text TEXT NOT NULL,
    gist_json TEXT,
    cosine_similarity REAL,
    drift_score REAL,
    accepted_flag INTEGER NOT NULL DEFAULT 0,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(checksum) REFERENCES workspace_checksums(checksum)
);

CREATE INDEX IF NOT EXISTS idx_workspace_gist_attempts_checksum ON workspace_gist_attempts(checksum);

CREATE TABLE IF NOT EXISTS workspace_index_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

class WorkspaceStore:
    """Async SQLite store for workspace paths, checksums, and gists."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "WorkspaceStore":
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

    def utcnow(self) -> datetime:
        return datetime.now(timezone.utc)

    # Scans
    async def create_scan(self, root_path: str) -> int:
        assert self._db is not None
        now = self.utcnow().isoformat()
        cursor = await self._db.execute(
            "INSERT INTO workspace_scans (root_path, started_at, status) VALUES (?, ?, ?)",
            (root_path, now, "running")
        )
        await self._db.commit()
        return cursor.lastrowid

    async def complete_scan(self, scan_id: int, status: str, stats: dict) -> None:
        assert self._db is not None
        now = self.utcnow().isoformat()
        await self._db.execute(
            """UPDATE workspace_scans 
               SET finished_at = ?, status = ?, 
                   files_seen = ?, files_hashed = ?, files_reused = ?, 
                   files_changed = ?, files_deleted = ?, errors = ?
               WHERE id = ?""",
            (now, status, 
             stats.get("seen", 0), stats.get("hashed", 0), stats.get("reused", 0),
             stats.get("changed", 0), stats.get("deleted", 0), stats.get("errors", 0),
             scan_id)
        )
        await self._db.commit()

    # Paths
    async def get_paths_under_root(self, root: Path) -> List[WorkspacePathRecord]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM workspace_paths WHERE exists_flag = 1 AND abs_path LIKE ?",
            (f"{str(root)}%",)
        )
        rows = await cursor.fetchall()
        out = []
        for r in rows:
            out.append(WorkspacePathRecord(
                abs_path=Path(r["abs_path"]),
                rel_path=Path(r["rel_path"]) if r["rel_path"] else None,
                parent_dir=Path(r["parent_dir"]),
                file_name=r["file_name"],
                extension=r["extension"],
                exists_flag=bool(r["exists_flag"]),
                file_kind=r["file_kind"],
                size_bytes=r["size_bytes"],
                mtime_ns=r["mtime_ns"],
                current_checksum=r["current_checksum"],
                first_seen_at=datetime.fromisoformat(r["first_seen_at"]),
                last_seen_at=datetime.fromisoformat(r["last_seen_at"]),
                last_scan_id=r["last_scan_id"],
                last_error=r["last_error"],
            ))
        return out

    async def get_path_record(self, abs_path: Path) -> Optional[WorkspacePathRecord]:
        assert self._db is not None
        cursor = await self._db.execute("SELECT * FROM workspace_paths WHERE abs_path = ?", (str(abs_path),))
        r = await cursor.fetchone()
        if not r:
            return None
        return WorkspacePathRecord(
            abs_path=Path(r["abs_path"]),
            rel_path=Path(r["rel_path"]) if r["rel_path"] else None,
            parent_dir=Path(r["parent_dir"]),
            file_name=r["file_name"],
            extension=r["extension"],
            exists_flag=bool(r["exists_flag"]),
            file_kind=r["file_kind"],
            size_bytes=r["size_bytes"],
            mtime_ns=r["mtime_ns"],
            current_checksum=r["current_checksum"],
            first_seen_at=datetime.fromisoformat(r["first_seen_at"]),
            last_seen_at=datetime.fromisoformat(r["last_seen_at"]),
            last_scan_id=r["last_scan_id"],
            last_error=r["last_error"],
        )

    async def upsert_path_snapshot(self, snapshot, checksum: Optional[str] = None, scan_id: Optional[int] = None) -> None:
        assert self._db is not None
        now = self.utcnow().isoformat()
        await self._db.execute(
            """INSERT INTO workspace_paths 
               (abs_path, rel_path, parent_dir, file_name, extension, exists_flag, file_kind, 
                size_bytes, mtime_ns, current_checksum, first_seen_at, last_seen_at, last_scan_id)
               VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(abs_path) DO UPDATE SET
               exists_flag = 1,
               size_bytes = excluded.size_bytes,
               mtime_ns = excluded.mtime_ns,
               file_kind = excluded.file_kind,
               current_checksum = excluded.current_checksum,
               last_seen_at = excluded.last_seen_at,
               last_scan_id = excluded.last_scan_id
            """,
            (str(snapshot.abs_path), str(snapshot.rel_path), str(snapshot.parent_dir),
             snapshot.file_name, snapshot.extension, snapshot.file_kind, snapshot.size_bytes,
             snapshot.mtime_ns, checksum, now, now, scan_id)
        )
        await self._db.commit()

    async def mark_paths_deleted(self, paths: List[Path]) -> None:
        assert self._db is not None
        if not paths:
            return
        path_strs = [(str(p),) for p in paths]
        await self._db.executemany("UPDATE workspace_paths SET exists_flag = 0 WHERE abs_path = ?", path_strs)
        await self._db.commit()

    # Checksums
    async def checksum_exists(self, checksum: str) -> bool:
        assert self._db is not None
        cursor = await self._db.execute("SELECT 1 FROM workspace_checksums WHERE checksum = ?", (checksum,))
        return bool(await cursor.fetchone())

    async def upsert_checksum(
        self, checksum: str, size_bytes: int, file_kind: str, mime_type: Optional[str],
        content_text_status: str, content_preview: Optional[str], content_embedding_ref: Optional[str],
        content_embedding_model: Optional[str], content_embedding_dim: Optional[int]
    ) -> None:
        assert self._db is not None
        now = self.utcnow().isoformat()
        await self._db.execute(
            """INSERT INTO workspace_checksums 
               (checksum, size_bytes, file_kind, mime_type, content_text_status, 
                content_preview, content_embedding_ref, content_embedding_model, 
                content_embedding_dim, first_seen_at, last_seen_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(checksum) DO UPDATE SET
               last_seen_at = excluded.last_seen_at
            """,
            (checksum, size_bytes, file_kind, mime_type, content_text_status,
             content_preview, content_embedding_ref, content_embedding_model,
             content_embedding_dim, now, now)
        )
        await self._db.commit()

    # Gists
    async def get_gist(self, checksum: str) -> Optional[WorkspaceGistRecord]:
        assert self._db is not None
        cursor = await self._db.execute("SELECT * FROM workspace_gists WHERE checksum = ?", (checksum,))
        r = await cursor.fetchone()
        if not r:
            return None
        return WorkspaceGistRecord(**dict(r))

    async def upsert_gist(self, record: WorkspaceGistRecord) -> None:
        assert self._db is not None
        await self._db.execute(
            """INSERT INTO workspace_gists 
               (checksum, gist_text, gist_json, llm_model, prompt_version, 
                gist_embedding_ref, gist_embedding_model, gist_embedding_dim, 
                cosine_similarity, drift_score, accepted_flag, needs_further_reading, 
                attempt_count, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(checksum) DO UPDATE SET
               gist_text = excluded.gist_text,
               gist_json = excluded.gist_json,
               llm_model = excluded.llm_model,
               prompt_version = excluded.prompt_version,
               gist_embedding_ref = excluded.gist_embedding_ref,
               gist_embedding_model = excluded.gist_embedding_model,
               gist_embedding_dim = excluded.gist_embedding_dim,
               cosine_similarity = excluded.cosine_similarity,
               drift_score = excluded.drift_score,
               accepted_flag = excluded.accepted_flag,
               needs_further_reading = excluded.needs_further_reading,
               attempt_count = excluded.attempt_count,
               updated_at = excluded.updated_at
            """,
            (record.checksum, record.gist_text, record.gist_json, record.llm_model, record.prompt_version,
             record.gist_embedding_ref, record.gist_embedding_model, record.gist_embedding_dim,
             record.cosine_similarity, record.drift_score, int(record.accepted_flag), 
             int(record.needs_further_reading), record.attempt_count, record.updated_at.isoformat())
        )
        await self._db.commit()

    async def insert_gist_attempt(self, checksum: str, attempt_no: int, gist_text: str,
                                  gist_json: Optional[str], cosine_similarity: Optional[float],
                                  drift_score: Optional[float], accepted_flag: bool,
                                  failure_reason: Optional[str]) -> None:
        assert self._db is not None
        now = self.utcnow().isoformat()
        await self._db.execute(
            """INSERT INTO workspace_gist_attempts 
               (checksum, attempt_no, gist_text, gist_json, cosine_similarity, drift_score, 
                accepted_flag, failure_reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (checksum, attempt_no, gist_text, gist_json, cosine_similarity, drift_score,
             int(accepted_flag), failure_reason, now)
        )
        await self._db.commit()

    # Lookups
    async def get_gist_for_path(self, abs_path: Path) -> Optional[WorkspaceGistLookupResult]:
        assert self._db is not None
        cursor = await self._db.execute(
            """SELECT p.abs_path, p.current_checksum, p.file_kind, p.size_bytes, p.mtime_ns,
                      g.gist_text, g.cosine_similarity, g.needs_further_reading
               FROM workspace_paths p
               LEFT JOIN workspace_gists g ON p.current_checksum = g.checksum
               WHERE p.abs_path = ? AND p.exists_flag = 1""",
            (str(abs_path),)
        )
        r = await cursor.fetchone()
        if not r:
            return None
        return WorkspaceGistLookupResult(
            abs_path=Path(r["abs_path"]),
            checksum=r["current_checksum"],
            gist_text=r["gist_text"],
            cosine_similarity=r["cosine_similarity"],
            needs_further_reading=bool(r["needs_further_reading"] if r["needs_further_reading"] is not None else False),
            file_kind=r["file_kind"],
            size_bytes=r["size_bytes"],
            mtime_ns=r["mtime_ns"],
        )

    async def get_gists_for_dir(self, parent_dir: Path) -> List[WorkspaceGistLookupResult]:
        assert self._db is not None
        cursor = await self._db.execute(
            """SELECT p.abs_path, p.current_checksum, p.file_kind, p.size_bytes, p.mtime_ns,
                      g.gist_text, g.cosine_similarity, g.needs_further_reading
               FROM workspace_paths p
               LEFT JOIN workspace_gists g ON p.current_checksum = g.checksum
               WHERE p.parent_dir = ? AND p.exists_flag = 1""",
            (str(parent_dir),)
        )
        rows = await cursor.fetchall()
        out = []
        for r in rows:
            out.append(WorkspaceGistLookupResult(
                abs_path=Path(r["abs_path"]),
                checksum=r["current_checksum"],
                gist_text=r["gist_text"],
                cosine_similarity=r["cosine_similarity"],
                needs_further_reading=bool(r["needs_further_reading"] if r["needs_further_reading"] is not None else False),
                file_kind=r["file_kind"],
                size_bytes=r["size_bytes"],
                mtime_ns=r["mtime_ns"],
            ))
        return out
