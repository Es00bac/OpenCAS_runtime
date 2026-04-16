"""Workspace and work-product importer for Bulma state directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from opencas.autonomy.models import WorkObject, WorkStage
from opencas.autonomy.work_store import WorkStore

from .models import BulmaWorkspaceManifest
from .mapper import map_bulma_workspace


def _discover_files(directory: Path) -> List[str]:
    files: List[str] = []
    for p in directory.rglob("*"):
        if p.is_file() and p.name not in {".DS_Store", "Thumbs.db"}:
            files.append(str(p.relative_to(directory)))
    return files


def _load_manifest(directory: Path) -> Optional[Dict[str, Any]]:
    for name in ("manifest.json", "metadata.json", "project.json"):
        path = directory / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _scan_workspace_directory(source_dir: Path) -> BulmaWorkspaceManifest:
    manifest_data = _load_manifest(source_dir)
    files = _discover_files(source_dir)
    has_subdirectories = any(p.is_dir() for p in source_dir.iterdir() if p.name not in {".git"})

    if manifest_data:
        return BulmaWorkspaceManifest(
            project_name=manifest_data.get("project_name", manifest_data.get("name", source_dir.name)),
            created_at=manifest_data.get("created_at"),
            updated_at=manifest_data.get("updated_at"),
            status=manifest_data.get("status", "imported"),
            source_dir=str(source_dir),
            files=files,
            meta={"has_subdirectories": has_subdirectories, **manifest_data.get("meta", {})},
        )

    return BulmaWorkspaceManifest(
        project_name=source_dir.name,
        source_dir=str(source_dir),
        files=files,
        meta={"has_subdirectories": has_subdirectories, "synthesized": True},
    )


async def import_workspaces(
    workspaces_root: Path,
    work_store: WorkStore,
) -> int:
    """Walk Bulma workspaces/ and create clean WorkObjects.

    Returns the number of WorkObjects created.
    """
    count = 0
    if not workspaces_root.exists():
        return count

    for entry in workspaces_root.iterdir():
        if not entry.is_dir():
            continue
        manifest = _scan_workspace_directory(entry)
        work = map_bulma_workspace(manifest)
        await work_store.save(work)
        count += 1
    return count


async def import_work_products(
    work_products_root: Path,
    work_store: WorkStore,
) -> int:
    """Walk Bulma work-products/ and create WorkObjects.

    Returns the number of WorkObjects created.
    """
    count = 0
    if not work_products_root.exists():
        return count

    for entry in work_products_root.iterdir():
        if not entry.is_file():
            continue
        # Skip non-content files
        if entry.suffix not in {".md", ".txt", ".json", ".jsonl", ".py", ".ts", ".js", ".html", ".css"}:
            continue
        content_preview = ""
        try:
            content_preview = entry.read_text(encoding="utf-8")[:500]
        except (UnicodeDecodeError, OSError):
            pass

        work = WorkObject(
            stage=WorkStage.ARTIFACT,
            content=f"Artifact: {entry.name}\n{content_preview}",
            meta={
                "bulma_source_path": str(entry),
                "bulma_type": "work_product",
            },
        )
        await work_store.save(work)
        count += 1
    return count
