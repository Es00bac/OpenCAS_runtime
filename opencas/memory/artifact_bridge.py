"""Bridge durable authored artifacts into grounded episodic memory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import NAMESPACE_URL, uuid5

from opencas.embeddings import EmbeddingService

from .models import Episode, EpisodeKind, Memory
from .store import MemoryStore


class ArtifactMemoryBridge:
    """Ingest text artifacts into episodic memory with explicit provenance."""

    TEXT_SUFFIXES = {
        ".md",
        ".txt",
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".rst",
    }
    SKIP_DIRS = {".git", ".hg", ".svn", ".venv", "__pycache__", "node_modules"}

    def __init__(
        self,
        state_dir: Path | str,
        memory: MemoryStore,
        embeddings: EmbeddingService,
        *,
        chunk_chars: int = 1600,
        overlap_chars: int = 200,
        max_bytes: int = 250_000,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.memory = memory
        self.embeddings = embeddings
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        self.max_bytes = max_bytes

    async def sync(self) -> Dict[str, int]:
        """Sync authored artifacts from `.opencas/plans` into memory."""
        root = self.state_dir / "plans"
        return await self.sync_directory(root)

    async def sync_directory(self, root: Path | str) -> Dict[str, int]:
        """Sync authored artifacts from an arbitrary directory into memory."""
        root = Path(root)
        if not root.exists():
            return {"artifacts": 0, "episodes_created": 0, "episodes_updated": 0, "episodes_deleted": 0, "memories_upserted": 0}

        artifacts = 0
        episodes_created = 0
        episodes_updated = 0
        episodes_deleted = 0
        memories_upserted = 0
        for path in self._iter_artifact_paths(root):
            result = await self._sync_artifact(path, relative_to=root)
            artifacts += 1
            episodes_created += result["episodes_created"]
            episodes_updated += result["episodes_updated"]
            episodes_deleted += result["episodes_deleted"]
            memories_upserted += result["memories_upserted"]
        return {
            "artifacts": artifacts,
            "episodes_created": episodes_created,
            "episodes_updated": episodes_updated,
            "episodes_deleted": episodes_deleted,
            "memories_upserted": memories_upserted,
        }

    def _iter_artifact_paths(self, root: Path) -> Iterable[Path]:
        if root.is_file():
            candidates = [root]
        else:
            candidates = root.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            if any(part in self.SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() not in self.TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > self.max_bytes:
                    continue
            except OSError:
                continue
            yield path

    async def _sync_artifact(self, path: Path, relative_to: Optional[Path] = None) -> Dict[str, int]:
        text = self._read_text(path)
        if not text.strip():
            return {"episodes_created": 0, "episodes_updated": 0, "episodes_deleted": 0, "memories_upserted": 0}

        relative_path = self._relative_artifact_path(path, relative_to=relative_to)
        artifact_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        title = self._title_for(path, text)
        chunks = self._chunk_text(text)
        existing = await self.memory.list_artifact_episodes(relative_path)
        existing_by_id = {str(item.episode_id): item for item in existing}
        keep_ids: List[str] = []
        batch: List[Episode] = []
        created = 0
        updated = 0

        for index, chunk in enumerate(chunks):
            episode_id = str(uuid5(NAMESPACE_URL, f"artifact:{relative_path}:{index}"))
            keep_ids.append(episode_id)
            payload = {
                "artifact": {
                    "path": relative_path,
                    "title": title,
                    "kind": path.suffix.lower().lstrip("."),
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "artifact_sha256": artifact_hash,
                    "content_sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
                }
            }
            content = (
                f"Artifact memory from {relative_path}\n"
                f"Title: {title}\n"
                f"Chunk {index + 1} of {len(chunks)}\n\n"
                f"{chunk}"
            )
            current = existing_by_id.get(episode_id)
            if current is not None:
                artifact_meta = (current.payload or {}).get("artifact", {})
                if (
                    current.content == content
                    and artifact_meta.get("artifact_sha256") == artifact_hash
                    and artifact_meta.get("chunk_count") == len(chunks)
                ):
                    continue
                updated += 1
            else:
                created += 1
            embedding = await self.embeddings.embed(content, task_type="artifact_episode")
            batch.append(
                Episode(
                    episode_id=episode_id,
                    kind=EpisodeKind.ARTIFACT,
                    session_id=None,
                    content=content,
                    embedding_id=embedding.source_hash,
                    salience=1.6,
                    payload=payload,
                )
            )

        if batch:
            await self.memory.save_episodes_batch(batch)

        stale_ids = [
            str(item.episode_id)
            for item in existing
            if str(item.episode_id) not in set(keep_ids)
        ]
        if stale_ids:
            await self.memory.delete_episodes(stale_ids)

        memory_id = str(uuid5(NAMESPACE_URL, f"artifact-memory:{relative_path}"))
        excerpt = chunks[0][:800].strip()
        memory_content = (
            f"Artifact authored or saved at {relative_path}. "
            f"Title: {title}. Type: {path.suffix.lower().lstrip('.')}. "
            f"Content excerpt: {excerpt}"
        )
        memory_embedding = await self.embeddings.embed(memory_content, task_type="artifact_memory")
        await self.memory.save_memory(
            Memory(
                memory_id=memory_id,
                content=memory_content,
                embedding_id=memory_embedding.source_hash,
                source_episode_ids=keep_ids,
                tags=["artifact", f"artifact_path:{relative_path}", f"artifact_title:{title}"],
                salience=1.8,
            )
        )
        return {
            "episodes_created": created,
            "episodes_updated": updated,
            "episodes_deleted": len(stale_ids),
            "memories_upserted": 1,
        }

    def _relative_artifact_path(self, path: Path, relative_to: Optional[Path] = None) -> str:
        resolved = path.resolve()
        project_root = self.state_dir.parent.resolve()
        try:
            return str(resolved.relative_to(project_root))
        except Exception:
            pass
        if relative_to is not None:
            try:
                return str(resolved.relative_to(relative_to.resolve()))
            except Exception:
                pass
        return str(path)

    @staticmethod
    def _title_for(path: Path, text: str) -> str:
        if path.suffix.lower() == ".md":
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if line.startswith("#"):
                    return line.lstrip("#").strip() or path.stem
        return path.stem.replace("-", " ").replace("_", " ").strip() or path.name

    def _chunk_text(self, text: str) -> List[str]:
        cleaned = text.strip()
        if len(cleaned) <= self.chunk_chars:
            return [cleaned]
        chunks: List[str] = []
        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + self.chunk_chars)
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(cleaned):
                break
            start = max(end - self.overlap_chars, start + 1)
        return chunks

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        if b"\x00" in data:
            return ""
        return data.decode("utf-8", errors="ignore")
