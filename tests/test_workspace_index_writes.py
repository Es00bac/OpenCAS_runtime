from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.autonomy.models import ActionRiskTier, ApprovalLevel
from opencas.runtime.provenance_hooks import register_runtime_provenance_hooks
from opencas.runtime.tool_runtime import execute_runtime_tool
from opencas.tools import FileSystemToolAdapter, ToolRegistry
from opencas.tools.adapters.edit import EditToolAdapter
from opencas.workspace.service import WorkspaceIndexService
from opencas.workspace.store import WorkspaceStore


class _FakeSomatic:
    def bump_from_work(self, intensity: float = 0.1, success: bool = True) -> None:
        pass

    async def emit_appraisal_event(self, *args, **kwargs) -> None:
        pass


class _FakeApproval:
    def evaluate(self, request):
        return SimpleNamespace(
            level=ApprovalLevel.CAN_DO_NOW,
            reasoning="approved",
            confidence=0.9,
            score=0.9,
            action_id=request.action_id,
        )

    async def maybe_record(self, decision, request, score):
        pass


async def _no_goals(_output: str):
    return []


def _runtime(tmp_path: Path, workspace_index: WorkspaceIndexService):
    from opencas.infra import HookBus

    hook_bus = HookBus()
    ctx = SimpleNamespace(
        config=SimpleNamespace(
            session_id="session-1",
            state_dir=tmp_path / ".opencas",
            agent_workspace_root=lambda: tmp_path / "workspace",
        ),
        hook_bus=hook_bus,
        somatic=_FakeSomatic(),
        web_trust=None,
        plan_store=None,
        work_store=None,
        workspace_index=workspace_index,
    )
    runtime = SimpleNamespace(
        ctx=ctx,
        approval=_FakeApproval(),
        tools=ToolRegistry(hook_bus=hook_bus),
        executive=SimpleNamespace(check_goal_resolution=_no_goals),
        _sync_executive_snapshot=lambda: None,
        _trace=lambda *args, **kwargs: None,
    )
    register_runtime_provenance_hooks(runtime)
    root = tmp_path / "workspace"
    runtime.tools.register("fs_write_file", "Write file", FileSystemToolAdapter([str(root)]), ActionRiskTier.WORKSPACE_WRITE)
    runtime.tools.register("edit_file", "Edit file", EditToolAdapter([str(root)]), ActionRiskTier.WORKSPACE_WRITE)
    return runtime


@pytest.mark.asyncio
async def test_fs_write_file_updates_workspace_index_synchronously(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    store = await WorkspaceStore(tmp_path / ".opencas" / "workspace.db").connect()
    service = WorkspaceIndexService(
        store=store,
        embeddings_client=SimpleNamespace(),
        llm_client=SimpleNamespace(),
        workspace_roots=[root],
        llm_model="test",
        embedding_model="test",
    )
    runtime = _runtime(tmp_path, service)
    target = root / "notes" / "probe.md"
    content = "alpha\n"

    result = await execute_runtime_tool(
        runtime,
        "fs_write_file",
        {"file_path": str(target), "content": content},
    )

    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert result["success"] is True
    assert result["metadata"]["checksum"] == checksum
    row = await store.get_path_record(target.resolve())
    assert row is not None
    assert row.abs_path == target.resolve()
    assert row.parent_dir == target.parent.resolve()
    assert row.current_checksum == checksum
    assert row.size_bytes == len(content.encode("utf-8"))
    assert await store.checksum_exists(checksum)
    await store.close()


@pytest.mark.asyncio
async def test_edit_file_updates_workspace_index_synchronously(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "probe.md"
    target.write_text("alpha\n", encoding="utf-8")
    store = await WorkspaceStore(tmp_path / ".opencas" / "workspace.db").connect()
    service = WorkspaceIndexService(
        store=store,
        embeddings_client=SimpleNamespace(),
        llm_client=SimpleNamespace(),
        workspace_roots=[root],
        llm_model="test",
        embedding_model="test",
    )
    runtime = _runtime(tmp_path, service)

    result = await execute_runtime_tool(
        runtime,
        "edit_file",
        {"file_path": str(target), "old_string": "alpha", "new_string": "beta"},
    )

    content = "beta\n"
    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert result["success"] is True
    assert result["metadata"]["checksum"] == checksum
    row = await store.get_path_record(target.resolve())
    assert row is not None
    assert row.current_checksum == checksum
    assert row.size_bytes == len(content.encode("utf-8"))
    assert await store.checksum_exists(checksum)
    await store.close()
