"""Tests for process-isolated consolidation worker orchestration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencas.bootstrap import BootstrapConfig
from opencas.embeddings import EmbeddingService
from opencas.runtime.consolidation_worker import (
    _connect_and_run_worker,
    _run_cli,
    build_consolidation_worker_command,
    consolidation_worker_result_path,
    consolidation_worker_status_path,
    run_consolidation_in_worker_process,
)
from opencas.runtime.maintenance_runtime import run_runtime_consolidation


def test_build_consolidation_worker_command_includes_paths_and_budget(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    workspace_root = tmp_path / "repo"
    config = SimpleNamespace(
        state_dir=state_dir,
        session_id="worker-session",
        default_llm_model="kimi-coding/k2p5",
        embedding_model_id="local-fallback",
        provider_config_path=None,
        provider_env_path=None,
        primary_workspace_root=lambda: workspace_root,
    )

    command = build_consolidation_worker_command(
        config,
        budget={"max_seconds": 5, "max_llm_calls": 1},
        run_id="run-1",
        python_executable="/usr/bin/python",
    )

    assert command.argv[:3] == [
        "/usr/bin/python",
        "-m",
        "opencas.runtime.consolidation_worker",
    ]
    assert command.result_path == consolidation_worker_result_path(state_dir, "run-1")
    assert command.status_path == consolidation_worker_status_path(state_dir)
    assert "--state-dir" in command.argv
    assert str(state_dir) in command.argv
    assert "--workspace-root" in command.argv
    assert str(workspace_root) in command.argv
    assert "--budget-json" in command.argv


@pytest.mark.asyncio
async def test_run_runtime_consolidation_uses_worker_process_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    activity = []
    worker_calls = []

    async def fake_worker(runtime, *, budget):
        worker_calls.append({"runtime": runtime, "budget": budget})
        return {
            "result_id": "worker-result",
            "timestamp": "2026-04-24T01:00:00+00:00",
            "worker": {"mode": "subprocess"},
        }

    async def forbidden_local_run(**kwargs):
        raise AssertionError("local consolidation should not run")

    monkeypatch.setattr(
        "opencas.runtime.maintenance_runtime.run_consolidation_in_worker_process",
        fake_worker,
        raising=False,
    )
    runtime = SimpleNamespace(
        consolidation=SimpleNamespace(run=forbidden_local_run),
        _set_activity=lambda value: activity.append(value),
        _last_consolidation_result=None,
        ctx=SimpleNamespace(
            config=SimpleNamespace(
                state_dir=tmp_path,
                consolidation_worker_enabled=True,
            )
        ),
    )

    payload = await run_runtime_consolidation(
        runtime,
        budget={"max_seconds": 5},
    )

    assert payload["result_id"] == "worker-result"
    assert worker_calls[0]["budget"] == {"max_seconds": 5}
    assert runtime._last_consolidation_result == payload
    assert activity == ["consolidating", "idle"]


@pytest.mark.asyncio
async def test_run_consolidation_in_worker_process_kills_timed_out_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class SlowProcess:
        returncode = None

        def __init__(self) -> None:
            self.killed = False
            self.pid = 1234

        async def communicate(self):
            await asyncio.sleep(1)
            return b"", b""

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode or -9

    process = SlowProcess()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    config = SimpleNamespace(
        state_dir=tmp_path,
        session_id="worker-session",
        default_llm_model="kimi-coding/k2p5",
        embedding_model_id="local-fallback",
        provider_config_path=None,
        provider_env_path=None,
        primary_workspace_root=lambda: tmp_path,
    )
    runtime = SimpleNamespace(ctx=SimpleNamespace(config=config))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    payload = await run_consolidation_in_worker_process(
        runtime,
        budget={"max_seconds": 0.01, "worker_grace_seconds": 0},
    )

    assert process.killed is True
    assert payload["budget_exhausted"] is True
    assert payload["budget_reason"] == "worker_timeout"
    assert payload["worker"]["mode"] == "subprocess"


@pytest.mark.asyncio
async def test_run_consolidation_in_worker_process_kills_cancelled_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class SlowProcess:
        returncode = None

        def __init__(self) -> None:
            self.killed = False
            self.pid = 1234

        async def communicate(self):
            await asyncio.sleep(60)
            return b"", b""

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode or -9

    process = SlowProcess()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return process

    config = SimpleNamespace(
        state_dir=tmp_path,
        session_id="worker-session",
        default_llm_model="kimi-coding/k2p5",
        embedding_model_id="local-fallback",
        provider_config_path=None,
        provider_env_path=None,
        primary_workspace_root=lambda: tmp_path,
    )
    runtime = SimpleNamespace(ctx=SimpleNamespace(config=config))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    task = asyncio.create_task(
        run_consolidation_in_worker_process(
            runtime,
            budget={"max_seconds": 60},
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed is True


@pytest.mark.asyncio
async def test_worker_uses_canonical_gemma_embedding_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured = {}

    class FakeGemma:
        async def embed(self, text: str):
            return [0.1] * 768

        async def embed_batch(self, texts):
            return [[0.1] * 768 for _ in texts]

    async def fake_get_local_gemma(self):
        self._local_gemma = FakeGemma()
        return self._local_gemma

    class FakeConsolidationResult:
        def model_dump(self, *, mode: str):
            return {
                "result_id": "fake-consolidation",
                "timestamp": "2026-04-27T00:00:00+00:00",
            }

    class FakeNightlyConsolidationEngine:
        def __init__(self, **kwargs):
            self.embeddings = kwargs["embeddings"]

        async def run(self, *, budget):
            records = await self.embeddings.embed_batch(["worker gemma check"])
            captured["record"] = records[0]
            return FakeConsolidationResult()

    monkeypatch.setattr(EmbeddingService, "_get_local_gemma", fake_get_local_gemma)
    monkeypatch.setattr(
        "opencas.runtime.consolidation_worker.NightlyConsolidationEngine",
        FakeNightlyConsolidationEngine,
    )
    config = BootstrapConfig(
        state_dir=tmp_path / "state",
        workspace_root=tmp_path,
        embedding_model_id="google/embeddinggemma-300m",
    ).resolve_paths()

    await _connect_and_run_worker(config, budget={"max_seconds": 1})

    record = captured["record"]
    assert record.model_id == "google/embeddinggemma-300m"
    assert record.dimension == 768
    assert "embedding_dimension_coercion" not in record.meta


@pytest.mark.asyncio
async def test_worker_cli_failure_writes_error_message_and_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_connect_and_run_worker(config, *, budget):
        raise ValueError("bad consolidation budget")

    monkeypatch.setattr(
        "opencas.runtime.consolidation_worker._connect_and_run_worker",
        fake_connect_and_run_worker,
    )
    args = SimpleNamespace(
        state_dir=str(tmp_path / "state"),
        workspace_root=str(tmp_path),
        run_id="failure-run",
        result_path=str(tmp_path / "worker-result.json"),
        status_path=str(tmp_path / "worker-status.json"),
        budget_json=json.dumps({"max_seconds": 1}),
        heartbeat_seconds=60,
        session_id="worker-session",
        default_llm_model="kimi-coding/k2p5",
        embedding_model_id="local-fallback",
        provider_config_path=None,
        provider_env_path=None,
    )

    rc = await _run_cli(args)

    assert rc == 1
    result_payload = json.loads(Path(args.result_path).read_text(encoding="utf-8"))
    status_payload = json.loads(Path(args.status_path).read_text(encoding="utf-8"))
    assert result_payload["worker"]["error_type"] == "ValueError"
    assert result_payload["worker"]["error_message"] == "bad consolidation budget"
    assert "ValueError: bad consolidation budget" in result_payload["worker"]["traceback"]
    assert status_payload["error_message"] == "bad consolidation budget"
    assert "ValueError: bad consolidation budget" in status_payload["traceback"]
