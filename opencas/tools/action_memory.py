"""Compact, evidence-preserving ACTION episode helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

COMPACT_ARG_BYTE_THRESHOLD = 1024
ARTIFACT_HINT_KEYS = (
    "file_path",
    "path",
    "output_path",
    "artifact_path",
    "canonical_artifact_path",
)


@dataclass(frozen=True)
class CompactActionHead:
    content: str
    digest: str
    arg_keys: list[str]
    arg_bytes: int
    artifact: str | None = None


def serialize_action_args(args: Mapping[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def action_args_fingerprint(args: Mapping[str, Any]) -> tuple[str, list[str], int]:
    serialized = serialize_action_args(args)
    encoded = serialized.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), sorted(str(key) for key in args.keys()), len(encoded)


def artifact_hint_from_mapping(args: Mapping[str, Any]) -> str | None:
    for key in ARTIFACT_HINT_KEYS:
        value = args.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def compact_action_head(
    tool_name: str,
    args: Mapping[str, Any],
    *,
    threshold: int = COMPACT_ARG_BYTE_THRESHOLD,
) -> CompactActionHead | None:
    digest, arg_keys, arg_bytes = action_args_fingerprint(args)
    if arg_bytes <= threshold:
        return None
    artifact = artifact_hint_from_mapping(args)
    parts = [
        f"tool {tool_name}",
        f"args_digest={digest}",
        f"arg_keys={','.join(arg_keys)}",
        f"arg_bytes={arg_bytes}",
    ]
    if artifact:
        parts.append(f"artifact={artifact}")
    return CompactActionHead(
        content=" ".join(parts),
        digest=digest,
        arg_keys=arg_keys,
        arg_bytes=arg_bytes,
        artifact=artifact,
    )


def compact_action_payload_metadata(head: CompactActionHead) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "args_digest": head.digest,
        "arg_keys": head.arg_keys,
        "arg_bytes": head.arg_bytes,
    }
    if head.artifact:
        metadata["artifact"] = head.artifact
    return metadata
