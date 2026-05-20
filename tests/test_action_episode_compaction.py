from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.memory.models import EpisodeKind
from opencas.tools.context import ToolUseContext
from opencas.tools.loop import ToolUseLoop


class _Runtime:
    def __init__(self, metadata: dict | None = None, output: str = "ok") -> None:
        self.episodes = []
        self.metadata = metadata or {}
        self.output = output
        self.ctx = SimpleNamespace(config=SimpleNamespace(state_dir=Path("/tmp")))

    async def execute_tool(self, name, args, *, session_id=None, task_id=None):
        return {
            "success": True,
            "output": self.output,
            "metadata": self.metadata,
        }

    async def _record_episode(self, **kwargs):
        self.episodes.append(kwargs)


@pytest.mark.asyncio
async def test_long_non_file_tool_args_record_compact_action_head() -> None:
    payload = "owned research note " * 200
    args = {
        "query": "trace generic OpenCAS recall authorship",
        "path": "workspace/research/action-authorship.md",
        "notes": payload,
    }
    runtime = _Runtime(metadata={"status": "stored"})
    loop = ToolUseLoop(llm=SimpleNamespace(), tools=SimpleNamespace(), approval=SimpleNamespace())

    result = await loop._execute_tool_call(
        {"name": "web_research_archive", "args": args},
        ToolUseContext(runtime=runtime, session_id="s1"),
    )

    serialized = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected_digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    episode = runtime.episodes[0]

    assert result["success"] is True
    assert episode["kind"] is EpisodeKind.ACTION
    assert episode["content"] == (
        "tool web_research_archive "
        f"args_digest={expected_digest} "
        "arg_keys=notes,path,query "
        f"arg_bytes={len(serialized.encode('utf-8'))} "
        "artifact=workspace/research/action-authorship.md"
    )
    assert payload[:120] not in episode["content"]
    assert episode["payload"]["args"] == args
    assert episode["payload"]["result_metadata"] == {"status": "stored"}
    assert episode["payload"]["args_digest"] == expected_digest


@pytest.mark.asyncio
async def test_short_non_file_tool_args_stay_readable() -> None:
    args = {"expression": "21 * 2"}
    runtime = _Runtime()
    loop = ToolUseLoop(llm=SimpleNamespace(), tools=SimpleNamespace(), approval=SimpleNamespace())

    await loop._execute_tool_call(
        {"name": "calculate", "args": args},
        ToolUseContext(runtime=runtime, session_id="s1"),
    )

    episode = runtime.episodes[0]
    assert episode["content"] == 'tool calculate: {"expression": "21 * 2"}'
    assert episode["payload"]["args"] == args


@pytest.mark.asyncio
async def test_file_write_action_episode_still_compacts_and_keeps_full_args(tmp_path: Path) -> None:
    body = "x" * 5000
    target = tmp_path / "probe.md"
    checksum = hashlib.sha256(body.encode("utf-8")).hexdigest()
    runtime = _Runtime(metadata={"path": str(target), "checksum": checksum})
    loop = ToolUseLoop(llm=SimpleNamespace(), tools=SimpleNamespace(), approval=SimpleNamespace())

    await loop._execute_tool_call(
        {
            "name": "fs_write_file",
            "args": {"file_path": str(target), "content": body},
        },
        ToolUseContext(runtime=runtime, session_id="s1"),
    )

    episode = runtime.episodes[0]
    assert episode["content"].startswith(f"tool fs_write_file path={target}")
    assert "x" * 100 not in episode["content"]
    assert len(episode["content"]) < 256
    assert episode["payload"]["args"]["content"] == body
    assert episode["payload"]["result_metadata"]["checksum"] == checksum


@pytest.mark.asyncio
async def test_edit_file_action_episode_still_compacts_and_keeps_full_args(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    replacement = "replacement text " * 500
    checksum = hashlib.sha256(replacement.encode("utf-8")).hexdigest()
    runtime = _Runtime(metadata={"path": str(target), "checksum": checksum})
    loop = ToolUseLoop(llm=SimpleNamespace(), tools=SimpleNamespace(), approval=SimpleNamespace())

    await loop._execute_tool_call(
        {
            "name": "edit_file",
            "args": {"file_path": str(target), "old_string": "old", "new_string": replacement},
        },
        ToolUseContext(runtime=runtime, session_id="s1"),
    )

    episode = runtime.episodes[0]
    assert episode["content"].startswith(f"tool edit_file path={target}")
    assert "replacement text replacement text" not in episode["content"]
    assert len(episode["content"]) < 256
    assert episode["payload"]["args"]["new_string"] == replacement


@pytest.mark.asyncio
async def test_fs_read_file_chunk_call_records_artifact_episode() -> None:
    target = Path("/tmp/novel.txt")
    payload = json.dumps(
        {
            "ok": True,
            "path": str(target),
            "content": "chapter text",
            "total_chars": 11,
            "returned_count": 11,
            "offset": 0,
            "limit": 11,
            "truncated": False,
            "next_offset": None,
        },
        ensure_ascii=False,
    )
    runtime = _Runtime(output=payload)
    loop = ToolUseLoop(llm=SimpleNamespace(), tools=SimpleNamespace(), approval=SimpleNamespace())

    await loop._execute_tool_call(
        {
            "name": "fs_read_file",
            "args": {
                "file_path": str(target),
                "offset": 0,
                "limit": 11,
                "read_session_id": "novel-session",
                "concept_scope": "chapter",
                "concept_label": "opening",
            },
        },
        ToolUseContext(runtime=runtime, session_id="session-read"),
    )

    assert len(runtime.episodes) == 2
    action_episode = runtime.episodes[0]
    artifact_episode = runtime.episodes[1]
    assert action_episode["kind"] is EpisodeKind.ACTION
    assert artifact_episode["kind"] is EpisodeKind.ARTIFACT
    assert "Read chunk 1/1" in artifact_episode["content"]
    assert "novel-session" in artifact_episode["content"]
    assert artifact_episode["payload"]["source_lane"] == "reflective"
    assert artifact_episode["payload"]["origin_context_lane"] == "reflective"
    assert artifact_episode["payload"]["context_authority"] == "interpretation"
    assert artifact_episode["payload"]["context_material"] == "artifact"
    assert artifact_episode["payload"]["artifact"]["path"] == str(target)
    assert artifact_episode["payload"]["artifact"]["concept_scope"] == "chapter"
    assert artifact_episode["payload"]["artifact"]["concept_label"] == "opening"
