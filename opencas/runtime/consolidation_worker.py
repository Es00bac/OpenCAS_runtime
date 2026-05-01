"""Process-isolated nightly consolidation worker support."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from open_llm_auth.auth.manager import ProviderManager

from opencas.api import LLMClient
from opencas.autonomy.commitment_store import CommitmentStore
from opencas.autonomy.work_store import WorkStore
from opencas.bootstrap.config import BootstrapConfig
from opencas.bootstrap.pipeline_support import (
    resolve_embedding_dimensions,
    resolve_embedding_model,
)
from opencas.consolidation import ConsolidationCurationStore, NightlyConsolidationEngine
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.identity import IdentityManager, IdentityStore
from opencas.memory import MemoryStore
from opencas.model_routing import load_persisted_model_routing_state
from opencas.tom import TomStore


_WORKER_DIR = "consolidation_worker"
_STATUS_FILE = "status.json"
_RESULTS_DIR = "results"


@dataclass(frozen=True)
class ConsolidationWorkerCommand:
    """Subprocess command and file paths for one consolidation worker run."""

    argv: list[str]
    run_id: str
    result_path: Path
    status_path: Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def consolidation_worker_status_path(state_dir: Path | str) -> Path:
    """Return the current consolidation-worker status file path."""
    return Path(state_dir) / _WORKER_DIR / _STATUS_FILE


def consolidation_worker_result_path(state_dir: Path | str, run_id: str) -> Path:
    """Return the result file path for a specific consolidation-worker run."""
    return Path(state_dir) / _WORKER_DIR / _RESULTS_DIR / f"{run_id}.json"


def load_consolidation_worker_status(state_dir: Path | str) -> Dict[str, Any]:
    """Load the latest worker status snapshot, if present."""
    path = consolidation_worker_status_path(state_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {"status": "unreadable", "path": str(path)}
    return payload if isinstance(payload, dict) else {}


def _workspace_root_from_config(config: Any) -> Path:
    resolver = getattr(config, "primary_workspace_root", None)
    if callable(resolver):
        try:
            return Path(resolver()).expanduser().resolve()
        except Exception:
            pass
    workspace_root = getattr(config, "workspace_root", None)
    if workspace_root:
        return Path(workspace_root).expanduser().resolve()
    return Path(getattr(config, "state_dir")).expanduser().resolve().parent


def build_consolidation_worker_command(
    config: Any,
    *,
    budget: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    python_executable: Optional[str] = None,
) -> ConsolidationWorkerCommand:
    """Build the subprocess command for a bounded consolidation worker."""
    state_dir = Path(getattr(config, "state_dir")).expanduser().resolve()
    clean_run_id = run_id or str(uuid4())
    result_path = consolidation_worker_result_path(state_dir, clean_run_id)
    status_path = consolidation_worker_status_path(state_dir)
    argv = [
        python_executable or sys.executable,
        "-m",
        "opencas.runtime.consolidation_worker",
        "--state-dir",
        str(state_dir),
        "--workspace-root",
        str(_workspace_root_from_config(config)),
        "--run-id",
        clean_run_id,
        "--result-path",
        str(result_path),
        "--status-path",
        str(status_path),
        "--budget-json",
        json.dumps(dict(budget or {}), sort_keys=True),
    ]
    session_id = getattr(config, "session_id", None)
    if session_id:
        argv.extend(["--session-id", str(session_id)])
    default_model = getattr(config, "default_llm_model", None)
    if default_model:
        argv.extend(["--default-llm-model", str(default_model)])
    embedding_model = getattr(config, "embedding_model_id", None)
    if embedding_model:
        argv.extend(["--embedding-model-id", str(embedding_model)])
    provider_config_path = getattr(config, "provider_config_path", None)
    if provider_config_path:
        argv.extend(["--provider-config-path", str(provider_config_path)])
    provider_env_path = getattr(config, "provider_env_path", None)
    if provider_env_path:
        argv.extend(["--provider-env-path", str(provider_env_path)])
    return ConsolidationWorkerCommand(
        argv=argv,
        run_id=clean_run_id,
        result_path=result_path,
        status_path=status_path,
    )


def _worker_timeout_seconds(budget: Dict[str, Any]) -> Optional[float]:
    if budget.get("worker_timeout_seconds") is not None:
        try:
            return max(0.001, float(budget["worker_timeout_seconds"]))
        except (TypeError, ValueError):
            return None
    if budget.get("max_seconds") is None:
        return None
    try:
        max_seconds = float(budget["max_seconds"])
    except (TypeError, ValueError):
        return None
    try:
        grace = float(budget.get("worker_grace_seconds", 15))
    except (TypeError, ValueError):
        grace = 15.0
    return max(0.001, max_seconds + max(0.0, grace))


def _timeout_payload(command: ConsolidationWorkerCommand, process: Any) -> Dict[str, Any]:
    timestamp = _now_iso()
    return {
        "result_id": f"worker-timeout-{command.run_id}",
        "timestamp": timestamp,
        "budget_exhausted": True,
        "budget_reason": "worker_timeout",
        "worker": {
            "mode": "subprocess",
            "run_id": command.run_id,
            "pid": getattr(process, "pid", None),
            "status": "timeout_killed",
            "result_path": str(command.result_path),
            "status_path": str(command.status_path),
        },
    }


async def _kill_worker_process(process: Any) -> None:
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except Exception:
        pass


async def run_consolidation_in_worker_process(
    runtime: Any,
    *,
    budget: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run consolidation in a child process and return its result payload."""
    config = getattr(getattr(runtime, "ctx", None), "config", None)
    if config is None:
        raise RuntimeError("runtime.ctx.config is required for worker consolidation")
    command = build_consolidation_worker_command(config, budget=budget)
    worker_budget = dict(budget or {})
    try:
        process = await asyncio.create_subprocess_exec(
            *command.argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        timestamp = _now_iso()
        error_message = str(exc)
        error_traceback = traceback.format_exc()
        return {
            "result_id": f"worker-start-failed-{command.run_id}",
            "timestamp": timestamp,
            "budget_exhausted": True,
            "budget_reason": "worker_start_failed",
            "worker": {
                "mode": "subprocess",
                "run_id": command.run_id,
                "status": "start_failed",
                "error_type": type(exc).__name__,
                "error_message": error_message,
                "traceback": error_traceback,
                "result_path": str(command.result_path),
                "status_path": str(command.status_path),
            },
        }

    timeout = _worker_timeout_seconds(worker_budget)
    try:
        if timeout is None:
            await process.communicate()
        else:
            await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await _kill_worker_process(process)
        payload = _timeout_payload(command, process)
        _write_json(command.status_path, payload["worker"] | {"updated_at": payload["timestamp"]})
        return payload
    except asyncio.CancelledError:
        await _kill_worker_process(process)
        _write_json(
            command.status_path,
            {
                "mode": "subprocess",
                "run_id": command.run_id,
                "pid": getattr(process, "pid", None),
                "status": "cancelled",
                "result_path": str(command.result_path),
                "status_path": str(command.status_path),
                "updated_at": _now_iso(),
            },
        )
        raise

    result_payload = _load_result_payload(command.result_path)
    if result_payload:
        worker = dict(result_payload.get("worker") or {})
        worker.update(
            {
                "mode": "subprocess",
                "run_id": command.run_id,
                "returncode": process.returncode,
                "result_path": str(command.result_path),
                "status_path": str(command.status_path),
            }
        )
        result_payload["worker"] = worker
        return result_payload

    timestamp = _now_iso()
    return {
        "result_id": f"worker-no-result-{command.run_id}",
        "timestamp": timestamp,
        "budget_exhausted": True,
        "budget_reason": "worker_no_result" if process.returncode == 0 else "worker_failed",
        "worker": {
            "mode": "subprocess",
            "run_id": command.run_id,
            "returncode": process.returncode,
            "status": "no_result",
            "result_path": str(command.result_path),
            "status_path": str(command.status_path),
        },
    }


def _load_result_payload(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_materialized_default_model(state_dir: Path) -> Optional[str]:
    config_path = state_dir / "provider_material" / "config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("defaultModel")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _build_worker_bootstrap_config(args: argparse.Namespace) -> BootstrapConfig:
    state_dir = Path(args.state_dir).expanduser().resolve()
    config_kwargs: Dict[str, Any] = {
        "state_dir": state_dir,
        "workspace_root": Path(args.workspace_root).expanduser().resolve(),
        "session_id": args.session_id,
        "consolidation_worker_enabled": False,
    }
    persisted_model_routing = load_persisted_model_routing_state(state_dir)
    materialized_default = _read_materialized_default_model(state_dir)
    if persisted_model_routing is not None:
        config_kwargs["model_routing"] = persisted_model_routing.model_routing
        if persisted_model_routing.default_llm_model:
            config_kwargs["default_llm_model"] = persisted_model_routing.default_llm_model
    if args.default_llm_model:
        config_kwargs["default_llm_model"] = args.default_llm_model
    elif not config_kwargs.get("default_llm_model") and materialized_default:
        config_kwargs["default_llm_model"] = materialized_default
    if args.embedding_model_id:
        config_kwargs["embedding_model_id"] = args.embedding_model_id

    provider_config_path = Path(args.provider_config_path).expanduser().resolve() if args.provider_config_path else None
    provider_env_path = Path(args.provider_env_path).expanduser().resolve() if args.provider_env_path else None
    materialized_config = state_dir / "provider_material" / "config.json"
    materialized_env = state_dir / "provider_material" / ".env"
    if provider_config_path is None and materialized_config.exists():
        provider_config_path = materialized_config
    if provider_env_path is None and materialized_env.exists():
        provider_env_path = materialized_env
    if provider_config_path is not None:
        config_kwargs["provider_config_path"] = provider_config_path
    if provider_env_path is not None:
        config_kwargs["provider_env_path"] = provider_env_path
    return BootstrapConfig(**config_kwargs).resolve_paths()


async def _connect_and_run_worker(
    config: BootstrapConfig,
    *,
    budget: Dict[str, Any],
) -> Dict[str, Any]:
    memory = MemoryStore(config.memory_db)
    embedding_cache = EmbeddingCache(config.embedding_db)
    curation_store = ConsolidationCurationStore(config.state_dir / "curation.db")
    commitment_store = CommitmentStore(config.state_dir / "commitments.db")
    work_store = WorkStore(config.work_db)
    tom_store = TomStore(config.tom_db)
    stores = [memory, embedding_cache, curation_store, commitment_store, work_store, tom_store]
    try:
        await memory.connect()
        await embedding_cache.connect()
        await curation_store.connect()
        await commitment_store.connect()
        await work_store.connect()
        await tom_store.connect()

        identity = IdentityManager(IdentityStore(config.state_dir / "identity"))
        identity.load()
        provider_manager = ProviderManager(
            config_path=config.provider_config_path,
            env_path=config.provider_env_path,
        )
        llm = LLMClient(
            provider_manager=provider_manager,
            default_model=config.default_llm_model,
            model_routing=config.model_routing,
        )
        embed_model = resolve_embedding_model(config, llm)
        embed_dimensions = resolve_embedding_dimensions(embed_model)
        embed_fn = None
        # Match normal bootstrap: Gemma is served by the local embedder at
        # its native 768-dimensional text embedding size.
        if embed_model not in {"local-fallback", "google/embeddinggemma-300m"}:
            embed_fn = lambda text: llm.embed(
                text,
                model=embed_model,
                dimensions=embed_dimensions,
            )
        embeddings = EmbeddingService(
            cache=embedding_cache,
            embed_fn=embed_fn,
            expected_dimension=embed_dimensions,
            model_id=embed_model,
            store=memory,
        )
        engine = NightlyConsolidationEngine(
            memory=memory,
            embeddings=embeddings,
            llm=llm,
            identity=identity,
            curation_store=curation_store,
            commitment_store=commitment_store,
            work_store=work_store,
            tom_store=tom_store,
        )
        result = await engine.run(budget=budget)
        return result.model_dump(mode="json")
    finally:
        for store in reversed(stores):
            close = getattr(store, "close", None)
            if callable(close):
                try:
                    maybe_result = close()
                    if hasattr(maybe_result, "__await__"):
                        await maybe_result
                except Exception:
                    pass


async def _heartbeat(
    status_path: Path,
    *,
    run_id: str,
    heartbeat_seconds: float,
) -> None:
    started_at = _now_iso()
    while True:
        _write_json(
            status_path,
            {
                "mode": "subprocess",
                "run_id": run_id,
                "pid": os.getpid(),
                "status": "running",
                "started_at": started_at,
                "heartbeat_at": _now_iso(),
            },
        )
        await asyncio.sleep(max(0.25, heartbeat_seconds))


async def _run_cli(args: argparse.Namespace) -> int:
    run_id = args.run_id
    result_path = Path(args.result_path).expanduser().resolve()
    status_path = Path(args.status_path).expanduser().resolve()
    try:
        budget = json.loads(args.budget_json or "{}")
        if not isinstance(budget, dict):
            budget = {}
    except json.JSONDecodeError:
        budget = {}

    _write_json(
        status_path,
        {
            "mode": "subprocess",
            "run_id": run_id,
            "pid": os.getpid(),
            "status": "starting",
            "started_at": _now_iso(),
        },
    )
    heartbeat_task = asyncio.create_task(
        _heartbeat(status_path, run_id=run_id, heartbeat_seconds=args.heartbeat_seconds)
    )
    try:
        config = _build_worker_bootstrap_config(args)
        payload = await _connect_and_run_worker(config, budget=budget)
        worker_meta = dict(payload.get("worker") or {})
        worker_meta.update(
            {
                "mode": "subprocess",
                "run_id": run_id,
                "pid": os.getpid(),
                "status": "completed",
                "result_path": str(result_path),
                "status_path": str(status_path),
            }
        )
        payload["worker"] = worker_meta
        _write_json(result_path, payload)
        _write_json(status_path, worker_meta | {"updated_at": _now_iso()})
        return 0
    except Exception as exc:
        timestamp = _now_iso()
        error_message = str(exc)
        error_traceback = traceback.format_exc()
        payload = {
            "result_id": f"worker-failed-{run_id}",
            "timestamp": timestamp,
            "budget_exhausted": True,
            "budget_reason": "worker_failed",
            "worker": {
                "mode": "subprocess",
                "run_id": run_id,
                "pid": os.getpid(),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": error_message,
                "traceback": error_traceback,
                "result_path": str(result_path),
                "status_path": str(status_path),
            },
        }
        _write_json(result_path, payload)
        _write_json(status_path, payload["worker"] | {"updated_at": timestamp})
        return 1
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint for `python -m opencas.runtime.consolidation_worker`."""
    parser = argparse.ArgumentParser(description="Run one OpenCAS consolidation worker.")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--budget-json", default="{}")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--default-llm-model", default=None)
    parser.add_argument("--embedding-model-id", default=None)
    parser.add_argument("--provider-config-path", default=None)
    parser.add_argument("--provider-env-path", default=None)
    parser.add_argument("--heartbeat-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    raise SystemExit(main())
