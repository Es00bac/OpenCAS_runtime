from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional

from opencas.api import LLMClient
from opencas.embeddings import EmbeddingService
from .store import WorkspaceStore
from .scanner import FileSnapshot, classify_file, walk_workspace, sha256_file
from .gisting import generate_validated_gist

class WorkspaceIndexService:
    def __init__(
        self,
        *,
        store: WorkspaceStore,
        embeddings_client: EmbeddingService,
        llm_client: LLMClient,
        workspace_roots: List[Path],
        llm_model: str,
        embedding_model: str,
        max_hash_concurrency: int = 8,
    ):
        self.store = store
        self.embeddings = embeddings_client
        self.llm = llm_client
        self.roots = workspace_roots
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.max_hash_concurrency = max_hash_concurrency
        self._bg_task: Optional[asyncio.Task] = None
        self._scan_task: Optional[asyncio.Task] = None
        self._queue = asyncio.Queue()

    async def start(self) -> None:
        await self.store.connect()
        self._bg_task = asyncio.create_task(self._bg_worker())
        # Track the initial scan so shutdown can cancel it deterministically.
        self._scan_task = asyncio.create_task(self.full_scan(force=False))

    async def stop(self) -> None:
        if self._bg_task:
            self._bg_task.cancel()
            try:
                await self._bg_task
            except asyncio.CancelledError:
                pass
            self._bg_task = None
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
            self._scan_task = None
        await self.store.close()

    async def _bg_worker(self) -> None:
        while True:
            try:
                paths = []
                # Drain queue
                paths.append(await self._queue.get())
                while not self._queue.empty():
                    paths.append(self._queue.get_nowait())
                
                # Debounce logic can go here, but for now just refresh immediately
                await self.refresh_paths(paths, force=False)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Workspace worker error: {e}")

    async def full_scan(self, force: bool = False) -> None:
        for root in self.roots:
            scan_id = await self.store.create_scan(str(root))
            snapshots = await walk_workspace(root)
            known_paths = await self.store.get_paths_under_root(root)
            known_map = {p.abs_path: p for p in known_paths}

            seen_paths = set()
            sem = asyncio.Semaphore(self.max_hash_concurrency)

            stats = {"seen": len(snapshots), "hashed": 0, "reused": 0, "changed": 0, "deleted": 0, "errors": 0}

            async def process(snapshot) -> None:
                seen_paths.add(snapshot.abs_path)
                existing = known_map.get(snapshot.abs_path)

                needs_hash = force or existing is None
                if existing and not force:
                    needs_hash = (
                        existing.size_bytes != snapshot.size_bytes or
                        existing.mtime_ns != snapshot.mtime_ns or
                        not existing.current_checksum
                    )

                checksum = existing.current_checksum if existing else None
                if needs_hash:
                    async with sem:
                        checksum = await sha256_file(snapshot.abs_path)
                        stats["hashed"] += 1
                        if existing:
                            stats["changed"] += 1
                else:
                    stats["reused"] += 1

                if checksum == "ERROR_READING_FILE":
                    stats["errors"] += 1
                    return

                await self.store.upsert_path_snapshot(snapshot, checksum=checksum, scan_id=scan_id)

                if checksum is None:
                    return

                checksum_exists = await self.store.checksum_exists(checksum)
                if not checksum_exists or needs_hash:
                    await self._ensure_checksum_materialized(snapshot.abs_path, checksum, snapshot.file_kind)

            # process in chunks to not explode memory, or just gather
            await asyncio.gather(*(process(s) for s in snapshots))

            deleted = [p for p in known_map if p not in seen_paths]
            if deleted:
                stats["deleted"] = len(deleted)
                await self.store.mark_paths_deleted(deleted)
            
            await self.store.complete_scan(scan_id, "complete", stats)

    async def refresh_paths(self, paths: list[Path], force: bool = False) -> None:
        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                await self.store.mark_paths_deleted([path])
                continue

            try:
                stat = path.stat()
            except OSError:
                continue

            root = self._root_for_path(path)
            snapshot = FileSnapshot(
                abs_path=path,
                rel_path=path.relative_to(root) if root else Path(path.name),
                parent_dir=path.parent,
                file_name=path.name,
                extension=path.suffix.lower() or None,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                file_kind=classify_file(path),
            )
            existing = await self.store.get_path_record(path)
            needs_hash = force or existing is None
            if existing and not force:
                needs_hash = (
                    existing.size_bytes != snapshot.size_bytes
                    or existing.mtime_ns != snapshot.mtime_ns
                    or not existing.current_checksum
                )
            checksum = existing.current_checksum if existing and not needs_hash else await sha256_file(path)
            if checksum == "ERROR_READING_FILE":
                continue
            await self.store.upsert_path_snapshot(snapshot, checksum=checksum, scan_id=None)
            if force or not await self.store.checksum_exists(checksum):
                await self._ensure_checksum_materialized(path, checksum, snapshot.file_kind)

    def _root_for_path(self, path: Path) -> Path | None:
        for root in self.roots:
            resolved = Path(root).expanduser().resolve()
            if path.is_relative_to(resolved):
                return resolved
        return None

    async def _ensure_checksum_materialized(self, path: Path, checksum: str, file_kind: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(100_000) # Read up to 100kb for gist context
        except UnicodeDecodeError:
            content = "Binary file."
            file_kind = "binary"
        except Exception:
            return

        await generate_validated_gist(
            path=path,
            checksum=checksum,
            file_kind=file_kind,
            content_excerpt=content[:5000],
            extracted_text=content,
            llm_client=self.llm,
            embeddings_client=self.embeddings,
            store=self.store,
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
        )

    async def get_gist_for_path(self, path: Path, refresh_if_stale: bool = False):
        return await self.store.get_gist_for_path(path)

    async def list_directory(self, dir_path: Path):
        return await self.store.get_gists_for_dir(dir_path)

    async def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        query_embedding = await self.embeddings.embed(
            query,
            task_type="workspace_gist_query",
        )
        scored = await self.embeddings.cache.search_similar(
            query_embedding.vector,
            limit=max(limit * 4, limit),
            query_text=query,
        )

        results: list[dict[str, Any]] = []
        for record, score in scored:
            metadata = record.meta or {}
            if not str(record.source_hash).startswith("workspace:gist:"):
                continue
            results.append(
                {
                    "path": metadata.get("path"),
                    "score": round(float(score), 4),
                    "checksum": metadata.get("checksum"),
                    "fallback": bool(metadata.get("fallback", False)),
                }
            )
            if len(results) >= limit:
                break
        return results
