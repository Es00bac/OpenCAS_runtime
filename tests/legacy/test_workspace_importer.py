"""Tests for workspace and work-product importers."""

from pathlib import Path

import pytest
import pytest_asyncio

from opencas.autonomy.work_store import WorkStore
from opencas.autonomy.models import WorkStage
from opencas.legacy.workspace_importer import import_workspaces, import_work_products


@pytest_asyncio.fixture
async def work_store(tmp_path: Path):
    store = WorkStore(tmp_path / "work.db")
    await store.connect()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_import_workspaces_creates_projects(work_store: WorkStore, tmp_path: Path) -> None:
    ws_root = tmp_path / "workspaces"
    project_dir = ws_root / "openbulma-v3"
    project_dir.mkdir(parents=True)
    (project_dir / "readme.md").write_text("# Project")
    subdir = project_dir / "content_engine"
    subdir.mkdir()
    (subdir / "note.md").write_text("note")

    count = await import_workspaces(ws_root, work_store)
    assert count == 1

    works = await work_store.list_all(limit=10)
    assert len(works) == 1
    assert works[0].stage == WorkStage.PROJECT
    assert "openbulma-v3" in works[0].content


@pytest.mark.asyncio
async def test_import_work_products_creates_artifacts(work_store: WorkStore, tmp_path: Path) -> None:
    wp_root = tmp_path / "work-products"
    wp_root.mkdir()
    (wp_root / "report.md").write_text("# Report\nDetails here.")
    (wp_root / "script.py").write_text("print('hello')")
    (wp_root / "binary.bin").write_bytes(b"\x00\x01\x02")

    count = await import_work_products(wp_root, work_store)
    assert count == 2  # .md and .py only

    arts = [w for w in await work_store.list_all(limit=10) if w.stage == WorkStage.ARTIFACT]
    assert len(arts) == 2


@pytest.mark.asyncio
async def test_import_workspaces_empty_returns_zero(work_store: WorkStore, tmp_path: Path) -> None:
    ws_root = tmp_path / "empty_workspaces"
    ws_root.mkdir()
    count = await import_workspaces(ws_root, work_store)
    assert count == 0
