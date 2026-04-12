"""Cutover helpers for moving Bulma state into OpenCAS."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field

from .loader import stream_jsonl
from .models import BulmaEpisode, BulmaMemoryEdge

SECRET_KEY_PARTS = ("token", "api_key", "apikey", "secret", "password", "credential")


class CutoverManifestEntry(BaseModel):
    """One source item considered during Bulma cutover."""

    category: str
    source_path: str
    disposition: str
    imported_path: Optional[str] = None
    size_bytes: int = 0
    sha256: Optional[str] = None
    note: Optional[str] = None


class CutoverManifest(BaseModel):
    """Audit manifest for the one-way Bulma cutover."""

    source_system: str = "openbulma-v4"
    target_system: str = "opencas"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    entries: List[CutoverManifestEntry] = Field(default_factory=list)

    def add_path(
        self,
        path: Path,
        *,
        category: str,
        disposition: str,
        imported_path: Optional[Path] = None,
        note: Optional[str] = None,
        include_hash: bool = True,
    ) -> None:
        size = path.stat().st_size if path.exists() and path.is_file() else 0
        digest = sha256_file(path) if include_hash and path.exists() and path.is_file() else None
        self.entries.append(
            CutoverManifestEntry(
                category=category,
                source_path=str(path),
                disposition=disposition,
                imported_path=str(imported_path) if imported_path else None,
                size_bytes=size,
                sha256=digest,
                note=note,
            )
        )

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


class BulmaPreflightReport(BaseModel):
    """Read-only report describing whether Bulma state is safe to import."""

    success: bool = True
    episodes_seen: int = 0
    episodes_parse_failed: int = 0
    duplicate_episode_ids: int = 0
    duplicate_timestamps: int = 0
    edges_seen: int = 0
    orphan_edge_candidates: int = 0
    secret_bearing_files: List[str] = Field(default_factory=list)
    clutter_categories: Dict[str, int] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_bulma_state(bulma_state_dir: Path) -> BulmaPreflightReport:
    """Inspect Bulma state without writing OpenCAS or Bulma files."""
    root = Path(bulma_state_dir)
    report = BulmaPreflightReport()

    episode_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    timestamps: Counter[float] = Counter()
    episodes_path = root / "memory" / "episodes.jsonl"
    if episodes_path.exists():
        for raw in stream_jsonl(episodes_path):
            try:
                episode = BulmaEpisode.model_validate(raw)
            except Exception:
                report.episodes_parse_failed += 1
                continue
            report.episodes_seen += 1
            if episode.id in episode_ids:
                duplicate_ids.add(episode.id)
            episode_ids.add(episode.id)
            timestamps[episode.timestampMs] += 1

    report.duplicate_episode_ids = len(duplicate_ids)
    report.duplicate_timestamps = sum(count for count in timestamps.values() if count > 1)

    edges_path = root / "memory" / "edges.jsonl"
    if edges_path.exists():
        for raw in stream_jsonl(edges_path):
            try:
                edge = BulmaMemoryEdge.model_validate(raw)
            except Exception:
                continue
            report.edges_seen += 1
            if edge.sourceId not in episode_ids or edge.targetId not in episode_ids:
                report.orphan_edge_candidates += 1

    for path in _walk_files(root):
        rel = path.relative_to(root)
        if any(part in rel.name.lower() for part in SECRET_KEY_PARTS):
            report.secret_bearing_files.append(str(rel))

    clutter_roots = (
        "backups",
        "logs",
        "migration_runs",
        "root_owned_quarantine",
        "tool-result-spill",
        "foreground-artifacts",
        "foreground-workbench",
        "document-drafts",
        "deliverable-schemas",
        "heartbeat",
        "runtime-hooks",
        "webhooks",
    )
    for name in clutter_roots:
        path = root / name
        if path.exists():
            report.clutter_categories[name] = sum(1 for _ in _walk_files(path))

    if report.episodes_parse_failed or report.orphan_edge_candidates:
        report.success = False
    if report.duplicate_episode_ids:
        report.notes.append("Duplicate episode IDs are present; importer will retain one canonical memory per Bulma ID.")
    if report.duplicate_timestamps:
        report.notes.append("Duplicate timestamps are present; importer must not derive IDs from timestamps.")
    return report


def copy_curated_legacy_workspace(
    curated_root: Path,
    target_root: Path,
    manifest: CutoverManifest,
) -> int:
    """Copy the user-curated legacy workspace into OpenCAS-owned storage."""
    curated_root = Path(curated_root)
    if not curated_root.exists():
        return 0
    target_root = Path(target_root)
    copied = 0
    for source in _walk_files(curated_root):
        relative = source.relative_to(curated_root)
        if source.is_symlink():
            manifest.add_path(
                source,
                category="workspace",
                disposition="skipped-symlink",
                note="Symlink was not copied during cutover to avoid external workspace dependencies.",
                include_hash=False,
            )
            continue
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.add_path(
            source,
            category="workspace",
            disposition="copied",
            imported_path=target,
        )
        copied += 1
    return copied


def record_retired_categories(
    bulma_state_dir: Path,
    manifest: CutoverManifest,
    categories: Iterable[str],
) -> None:
    """Record clutter categories as deliberately retired, without copying them."""
    root = Path(bulma_state_dir)
    for category in categories:
        path = root / category
        if not path.exists():
            continue
        count = sum(1 for _ in _walk_files(path))
        manifest.entries.append(
            CutoverManifestEntry(
                category=category,
                source_path=str(path),
                disposition="retired_by_cutover",
                note=f"{count} files left in inactive OpenBulma state; not used by OpenCAS.",
            )
        )


def redact_secrets(payload: Any) -> Any:
    """Return a JSON-compatible object with obvious secret values redacted."""
    if isinstance(payload, dict):
        result: Dict[str, Any] = {}
        for key, value in payload.items():
            if any(part in str(key).lower() for part in SECRET_KEY_PARTS):
                result[key] = "***" if value else value
            else:
                result[key] = redact_secrets(value)
        return result
    if isinstance(payload, list):
        return [redact_secrets(item) for item in payload]
    return payload


def load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _walk_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path
