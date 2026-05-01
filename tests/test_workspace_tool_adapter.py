"""Tests for workspace indexing tool adapter result contracts."""

import pytest

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
    assert "No gist found" in result.output
    assert result.metadata["path"].endswith("missing.md")
