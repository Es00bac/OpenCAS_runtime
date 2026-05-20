from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.memory.models import EpisodeKind
from opencas.tools.context import ToolUseContext
from opencas.tools.loop import ToolUseLoop


class _Runtime:
    def __init__(self, path: Path, body: str) -> None:
        self.path = path
        self.body = body
        self.episodes = []
        self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir=path.parent))

    async def execute_tool(self, name, args, *, session_id=None, task_id=None, audit_only=False):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.body, encoding="utf-8")
        checksum = hashlib.sha256(self.body.encode("utf-8")).hexdigest()
        return {
            "success": True,
            "output": "ok",
            "metadata": {"path": str(self.path), "checksum": checksum},
        }

    async def _record_episode(self, **kwargs):
        self.episodes.append(kwargs)


@pytest.mark.asyncio
async def test_file_write_action_episode_is_compact_and_payload_keeps_args(tmp_path: Path) -> None:
    body = "x" * 5000
    target = tmp_path / "probe.md"
    runtime = _Runtime(target, body)
    loop = ToolUseLoop(llm=SimpleNamespace(), tools=SimpleNamespace(), approval=SimpleNamespace())

    result = await loop._execute_tool_call(
        {
            "name": "fs_write_file",
            "args": {"file_path": str(target), "content": body},
        },
        ToolUseContext(runtime=runtime, session_id="s1"),
    )

    assert result["success"] is True
    episode = runtime.episodes[0]
    assert episode["kind"] is EpisodeKind.ACTION
    assert episode["content"].startswith(f"tool fs_write_file path={target}")
    assert "x" * 100 not in episode["content"]
    assert len(episode["content"]) < 256
    assert episode["payload"]["tool_name"] == "fs_write_file"
    assert episode["payload"]["args"]["content"] == body
    assert episode["payload"]["result_metadata"]["checksum"] == hashlib.sha256(body.encode()).hexdigest()


@pytest.mark.asyncio
async def test_audit_only_tool_call_does_not_write_action_episode(tmp_path: Path) -> None:
    target = tmp_path / "probe.md"
    runtime = _Runtime(target, "audit body")
    loop = ToolUseLoop(llm=SimpleNamespace(), tools=SimpleNamespace(), approval=SimpleNamespace())

    result = await loop._execute_tool_call(
        {
            "name": "fs_write_file",
            "args": {"file_path": str(target), "content": "audit body"},
        },
        ToolUseContext(runtime=runtime, session_id="audit-session", audit_only=True),
    )

    assert result["success"] is True
    assert runtime.episodes == []
