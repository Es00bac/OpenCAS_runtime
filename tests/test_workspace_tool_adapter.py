"""Tests for workspace indexing tool adapter result contracts."""

import json
from pathlib import Path

import pytest

from opencas.workspace.models import WorkspaceGistLookupResult
from opencas.workspace.tool_adapter import WorkspaceIndexerToolAdapter


class _FakeWorkspaceIndex:
    def __init__(self) -> None:
        self.force = None

    async def full_scan(self, *, force: bool = False) -> None:
        self.force = force

    async def get_gist_for_path(self, path, *, refresh_if_stale: bool = False):
        return None

    async def search(self, query: str, *, limit: int = 8):
        return []

    async def list_directory(self, path):
        return []

    async def list_directory_with_status(self, path):
        return {
            "directory": str(path),
            "indexed_files": [],
            "disk_listing": [
                {
                    "name": "probe.md",
                    "kind": "text",
                    "indexed": False,
                    "gist_pending": True,
                    "size_bytes": 5,
                }
            ],
            "index_status": {
                "last_scan_age_seconds": None,
                "indexed_count": 0,
                "disk_count": 1,
                "fallback_used": True,
                "live_listing_complete": True,
                "scan_root": None,
            },
        }


@pytest.mark.asyncio
async def test_workspace_refresh_index_returns_tool_result_contract() -> None:
    service = _FakeWorkspaceIndex()
    adapter = WorkspaceIndexerToolAdapter(service)

    result = await adapter("workspace_refresh_index", {"force": True})

    assert result.success is True
    assert result.output == "Workspace index refresh triggered."
    assert result.metadata == {}
    assert service.force is True


@pytest.mark.asyncio
async def test_workspace_get_missing_gist_returns_tool_result_contract(tmp_path) -> None:
    adapter = WorkspaceIndexerToolAdapter(_FakeWorkspaceIndex())

    result = await adapter(
        "workspace_get_file_gist",
        {"abs_path": str(tmp_path / "missing.md")},
    )

    assert result.success is False
    assert "No workspace_paths row and no file on disk" in result.output
    assert result.metadata["path"].endswith("missing.md")


@pytest.mark.asyncio
async def test_workspace_list_directory_gists_reports_live_fallback_status(tmp_path: Path) -> None:
    adapter = WorkspaceIndexerToolAdapter(_FakeWorkspaceIndex())

    result = await adapter(
        "workspace_list_directory_gists",
        {"dir_path": str(tmp_path)},
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["directory"] == str(tmp_path.resolve())
    assert payload["indexed_files"] == []
    assert payload["disk_listing"][0]["name"] == "probe.md"
    assert payload["index_status"]["indexed_count"] == 0
    assert payload["index_status"]["fallback_used"] is True
    assert payload["index_status"]["live_listing_complete"] is True


@pytest.mark.asyncio
async def test_workspace_get_file_gist_returns_pending_for_indexed_file_without_gist(tmp_path: Path) -> None:
    path = tmp_path / "probe.md"

    class _IndexedNoGist(_FakeWorkspaceIndex):
        async def get_gist_for_path(self, path, *, refresh_if_stale: bool = False):
            return WorkspaceGistLookupResult(
                abs_path=path,
                checksum="a" * 64,
                gist_text=None,
                cosine_similarity=None,
                needs_further_reading=False,
                file_kind="text",
                size_bytes=12,
                mtime_ns=123,
            )

    result = await WorkspaceIndexerToolAdapter(_IndexedNoGist())(
        "workspace_get_file_gist",
        {"abs_path": str(path)},
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["status"] == "gist_pending"
    assert payload["checksum"] == "a" * 64
    assert result.metadata["gist_pending"] is True


@pytest.mark.asyncio
async def test_workspace_get_file_gist_returns_pending_for_live_unindexed_file(tmp_path: Path) -> None:
    path = tmp_path / "probe.md"
    path.write_text("hello", encoding="utf-8")
    result = await WorkspaceIndexerToolAdapter(_FakeWorkspaceIndex())(
        "workspace_get_file_gist",
        {"abs_path": str(path)},
    )

    assert result.success is True
    payload = json.loads(result.output)
    assert payload["status"] == "gist_pending"
    assert payload["exists_on_disk"] is True
    assert payload["size_bytes"] == 5
    assert len(payload["checksum"]) == 64
    assert result.metadata["gist_pending"] is True
