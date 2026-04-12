"""Tests for PluginStore."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.plugins.store import PluginStore


@pytest_asyncio.fixture
async def store(tmp_path: Path):
    db = PluginStore(tmp_path / "plugins.db")
    await db.connect()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_install_and_is_installed(store: PluginStore) -> None:
    await store.install("p1", "Plugin One", "Does one thing.", "installed", "/tmp/p1")
    assert await store.is_installed("p1")
    assert not await store.is_installed("p2")


@pytest.mark.asyncio
async def test_list_installed(store: PluginStore) -> None:
    await store.install("p1", "One", "", "builtin", "/tmp/p1")
    await store.install("p2", "Two", "", "installed", "/tmp/p2")
    rows = await store.list_installed()
    assert len(rows) == 2
    ids = {r["plugin_id"] for r in rows}
    assert ids == {"p1", "p2"}


@pytest.mark.asyncio
async def test_enable_disable(store: PluginStore) -> None:
    await store.install("p1", "One", "", "installed", "/tmp/p1")
    assert await store.is_enabled("p1")
    await store.set_enabled("p1", False)
    assert not await store.is_enabled("p1")
    await store.set_enabled("p1", True)
    assert await store.is_enabled("p1")


@pytest.mark.asyncio
async def test_list_enabled(store: PluginStore) -> None:
    await store.install("p1", "One", "", "builtin", "/tmp/p1")
    await store.install("p2", "Two", "", "installed", "/tmp/p2")
    await store.set_enabled("p2", False)
    rows = await store.list_enabled()
    assert len(rows) == 1
    assert rows[0]["plugin_id"] == "p1"


@pytest.mark.asyncio
async def test_uninstall(store: PluginStore) -> None:
    await store.install("p1", "One", "", "installed", "/tmp/p1")
    assert await store.is_installed("p1")
    await store.uninstall("p1")
    assert not await store.is_installed("p1")


@pytest.mark.asyncio
async def test_install_persists_manifest(store: PluginStore) -> None:
    manifest = {"version": "1.0.0", "skills": ["s1.py"]}
    await store.install("p1", "One", "", "installed", "/tmp/p1", manifest=manifest)
    rows = await store.list_installed()
    assert rows[0]["manifest"] == manifest
