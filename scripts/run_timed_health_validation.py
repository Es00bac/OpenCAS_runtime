#!/usr/bin/env python3
"""Timed live validation for OpenCAS health and Qdrant embedding contract."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.embeddings.qdrant_backend import QdrantVectorBackend


async def _get_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"payload": payload}


async def _qdrant_collection(url: str, collection: str) -> dict[str, Any]:
    return await _get_json(f"{url.rstrip('/')}/collections/{collection}")


async def _embedding_probe(
    *,
    state_dir: Path,
    qdrant_url: str,
    collection: str,
    model_id: str,
    expected_dimension: int,
    sample_id: int,
) -> dict[str, Any]:
    backend = QdrantVectorBackend(
        url=qdrant_url,
        collection=collection,
        dimension=expected_dimension,
    )
    await backend.connect()
    cache = EmbeddingCache(state_dir / "embeddings.db", vector_backend=backend)
    await cache.connect()
    service = EmbeddingService(
        cache,
        model_id=model_id,
        expected_dimension=expected_dimension,
    )
    try:
        text = (
            "OpenCAS timed health validation probe "
            f"{sample_id} {datetime.now(timezone.utc).isoformat()}"
        )
        record = await service.embed(
            text,
            task_type="health_probe",
            meta={"validation_probe": "timed_health"},
        )
        hits = await cache.search_similar(
            record.vector,
            limit=3,
            model_id=model_id,
        )
        found = any(hit.source_hash == record.source_hash for hit, _score in hits)
        recent_path = cache._search_history[-1]["path"] if cache._search_history else None
        return {
            "record_source_hash": record.source_hash,
            "record_dimension": record.dimension,
            "record_model_id": record.model_id,
            "qdrant_found": found,
            "search_path": recent_path,
        }
    finally:
        await cache.close()


async def run(args: argparse.Namespace) -> int:
    state_dir = args.state_dir.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sample = 0
    failures = 0
    with out_path.open("a", encoding="utf-8") as fh:
        while True:
            sample += 1
            now = datetime.now(timezone.utc).isoformat()
            entry: dict[str, Any] = {"sample": sample, "timestamp": now}
            try:
                entry["health"] = await _get_json(f"{args.base_url.rstrip('/')}/health")
                monitor = await _get_json(f"{args.base_url.rstrip('/')}/api/monitor/health")
                entry["monitor_overall"] = monitor.get("overall")
                entry["monitor_failures"] = monitor.get("failures")
                entry["monitor_warnings"] = monitor.get("warnings")
                embeddings = await _get_json(f"{args.base_url.rstrip('/')}/api/monitor/embeddings")
                entry["embedding_model_id"] = embeddings.get("model_id")
                entry["embedding_total_records"] = embeddings.get("total_records")
                entry["embedding_probe"] = await _embedding_probe(
                    state_dir=state_dir,
                    qdrant_url=args.qdrant_url,
                    collection=args.qdrant_collection,
                    model_id=args.model_id,
                    expected_dimension=args.expected_dimension,
                    sample_id=sample,
                )
                collection = await _qdrant_collection(args.qdrant_url, args.qdrant_collection)
                params = (
                    collection.get("result", {})
                    .get("config", {})
                    .get("params", {})
                    .get("vectors", {})
                )
                entry["qdrant_collection_status"] = collection.get("result", {}).get("status")
                entry["qdrant_vector_size"] = params.get("size")
                if (
                    entry["embedding_probe"]["record_dimension"] != args.expected_dimension
                    or not entry["embedding_probe"]["qdrant_found"]
                    or entry["embedding_probe"]["search_path"] != "qdrant"
                ):
                    failures += 1
                    entry["status"] = "fail"
                else:
                    entry["status"] = "pass"
            except Exception as exc:
                failures += 1
                entry["status"] = "error"
                entry["error"] = f"{type(exc).__name__}: {exc}"
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
            fh.flush()
            print(json.dumps(entry, sort_keys=True))
            if time.monotonic() - started >= args.duration_seconds:
                break
            await asyncio.sleep(args.interval_seconds)
    return 0 if failures == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ROOT / ".opencas")
    parser.add_argument("--base-url", default="http://127.0.0.1:32147")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--qdrant-collection", default="episodes_embed_gemma_768_v1")
    parser.add_argument("--model-id", default="google/embeddinggemma-300m")
    parser.add_argument("--expected-dimension", type=int, default=768)
    parser.add_argument("--duration-seconds", type=float, default=120.0)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".opencas" / "timed_health_validation.jsonl",
    )
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()
