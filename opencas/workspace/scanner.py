from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path

CHUNK_SIZE = 1024 * 1024  # 1 MB

@dataclass
class FileSnapshot:
    abs_path: Path
    rel_path: Path
    parent_dir: Path
    file_name: str
    extension: str | None
    size_bytes: int
    mtime_ns: int
    file_kind: str

def classify_file(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        if mime.startswith("text/"):
            return "text"
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("audio/"):
            return "audio"
        if "zip" in mime or "tar" in mime:
            return "archive"
        if "sqlite" in mime or "database" in mime:
            return "db"
    if path.suffix.lower() in {
        ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
        ".js", ".ts", ".tsx", ".jsx", ".css", ".html", ".xml", ".sql",
        ".sh", ".bash", ".zsh", ".cfg", ".conf", ".rst", ".jsonc"
    }:
        return "text"
    return "binary"

def should_exclude_dir(name: str) -> bool:
    return name in {
        ".git", ".venv", "venv", "__pycache__", "node_modules",
        "dist", "build", ".cache", ".opencas", ".claude", ".omc"
    }

def walk_workspace_sync(root: Path) -> list[FileSnapshot]:
    out: list[FileSnapshot] = []
    root_resolved = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root_resolved):
        dirnames[:] = [d for d in dirnames if not should_exclude_dir(d)]
        dpath = Path(dirpath)
        for filename in filenames:
            if filename.startswith("."): # ignore hidden files like .DS_Store
                continue
            abs_path = dpath / filename
            if not abs_path.is_file():
                continue
            try:
                stat = abs_path.stat()
                out.append(
                    FileSnapshot(
                        abs_path=abs_path,
                        rel_path=abs_path.relative_to(root_resolved),
                        parent_dir=abs_path.parent,
                        file_name=filename,
                        extension=abs_path.suffix.lower() or None,
                        size_bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        file_kind=classify_file(abs_path),
                    )
                )
            except OSError:
                pass
    return out

async def walk_workspace(root: Path) -> list[FileSnapshot]:
    return await asyncio.to_thread(walk_workspace_sync, root)

def sha256_file_sync(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "ERROR_READING_FILE"

async def sha256_file(path: Path) -> str:
    return await asyncio.to_thread(sha256_file_sync, path)
