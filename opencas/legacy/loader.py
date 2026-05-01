"""Loaders for OpenBulma v4 JSONL, JSON, and Qdrant state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from opencas.telemetry import EventKind


def stream_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield decoded JSON objects from a JSONL file, skipping bad lines."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a single JSON object from a file, or None if missing/invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError):
        return None


def load_qdrant_collection(
    qdrant_dir: Path,
    collection_name: str,
) -> Iterator[Tuple[str, List[float], Dict[str, Any]]]:
    """Yield (point_id, vector, payload) tuples from a local Qdrant collection.

    Uses qdrant_client in local mode so no server needs to be running.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required to import Bulma embeddings") from exc

    client = QdrantClient(path=str(qdrant_dir))
    try:
        offset = None
        while True:
            result = client.scroll(
                collection_name=collection_name,
                offset=offset,
                limit=1000,
                with_vectors=True,
                with_payload=True,
            )
            points, next_offset = result
            for point in points:
                yield (
                    str(point.id),
                    list(point.vector) if point.vector is not None else [],
                    dict(point.payload or {}),
                )
            if next_offset is None:
                break
            offset = next_offset
    finally:
        client.close()
