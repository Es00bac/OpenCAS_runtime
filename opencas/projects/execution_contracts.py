"""Execution-contract guards for project work.

These helpers keep project-return metadata actionable at the execution and
completion boundaries. They are intentionally generic: a "new project" may be
a novel, app, tool, or any other artifact, but it must not be completed by
silently materializing a sibling project's files into the requested root.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


_COPY_ACTION_MARKERS = (
    "copy2",
    "copyfile",
    "copytree",
    "copied",
    "copying",
    "materialize",
    "materialized",
    "preserve source project",
    "preserved as project.source",
    "rsync",
    "shutil.copy",
    "source project context copy",
)
_SHELL_COPY_COMMAND_RE = re.compile(r"(?<![a-z0-9_])(?:cp|mv|rsync)\b", re.IGNORECASE)
_COPY_EVIDENCE_MARKERS = (
    "copied directories",
    "copied from",
    "copied the",
    "from that sibling",
    "from the sibling",
    "materialized manuscript",
    "materialized the",
    "preserved as `project.source",
    "preserved as project.source",
    "repairs sibling-project",
    "repairs the earlier sibling",
    "sibling project",
    "source project context copy",
)
_PATH_RE = re.compile(r"(?P<path>(?:~|/)[^\s`'\"<>]+|(?:\.?/)?workspace/[^\s`'\"<>]+)")


def is_new_project_contract(meta: dict[str, Any] | None) -> bool:
    """Return true when task/commitment metadata represents a new project root."""

    if not isinstance(meta, dict):
        return False
    contract = meta.get("project_start_contract")
    if isinstance(contract, dict) and contract.get("new_project") is True:
        return True
    return str(meta.get("requested_workspace_kind") or "").strip().lower() == "parent"


def new_project_tool_block_reason(
    meta: dict[str, Any] | None,
    *,
    tool_name: str,
    args: dict[str, Any] | None,
) -> str | None:
    """Return a block reason when a tool would copy a sibling into a new project."""

    if not is_new_project_contract(meta):
        return None
    args = args or {}
    target_root = _target_root(meta)
    if not target_root:
        return None
    payload = _tool_payload_text(tool_name=tool_name, args=args)
    if not payload:
        return None
    lowered = payload.lower()
    if not _mentions_copy_action(lowered):
        return None
    if not _mentions_path(lowered, target_root):
        return None
    source = _mentioned_forbidden_source(meta, payload)
    if source is None:
        source = _mentioned_sibling_path(meta, payload)
    if source is None:
        return None
    return (
        "new-project contract forbids satisfying the task by copying or "
        f"materializing sibling project artifacts into the target root ({source})"
    )


def new_project_completion_rejection_reason(
    meta: dict[str, Any] | None,
    evidence_text: str | None,
) -> str | None:
    """Reject completion evidence that closes a new project via sibling copy."""

    if not is_new_project_contract(meta):
        return None
    text = str(evidence_text or "")
    if not text.strip():
        return None
    lowered = text.lower()
    if not any(marker in lowered for marker in _COPY_EVIDENCE_MARKERS):
        return None
    if _mentioned_forbidden_source(meta, text) is None and _mentioned_sibling_path(meta, text) is None:
        return None
    return (
        "Completion evidence describes copying/materializing a sibling or source project into a "
        "new project root. New-project work must produce artifacts for the new project itself; "
        "premise/reference continuity is allowed, but sibling artifact materialization is not completion."
    )


def new_project_execution_rejection_reason(
    meta: dict[str, Any] | None,
    *,
    output: str | None,
    tool_calls: Iterable[dict[str, Any]] | None,
) -> str | None:
    """Reject a completed tool loop whose trace violates the new-project contract."""

    if not is_new_project_contract(meta):
        return None
    combined_parts = [str(output or "")]
    for call in tool_calls or ():
        try:
            combined_parts.append(json.dumps(call, sort_keys=True))
        except TypeError:
            combined_parts.append(str(call))
    combined = "\n".join(combined_parts)
    if not combined.strip():
        return None
    if any(marker in combined.lower() for marker in _COPY_EVIDENCE_MARKERS + _COPY_ACTION_MARKERS):
        return new_project_completion_rejection_reason(meta, combined) or new_project_tool_block_reason(
            meta,
            tool_name="bash_run_command",
            args={"command": combined},
        )
    return None


def _tool_payload_text(*, tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "bash_run_command":
        return str(args.get("command") or "")
    if tool_name in {"fs_write_file", "edit_file", "write_file"}:
        return "\n".join(
            str(args.get(key) or "")
            for key in ("file_path", "path", "content", "old_text", "new_text")
        )
    return ""


def _mentions_copy_action(lowered_payload: str) -> bool:
    return _SHELL_COPY_COMMAND_RE.search(lowered_payload) is not None or any(
        marker in lowered_payload for marker in _COPY_ACTION_MARKERS
    )


def _target_root(meta: dict[str, Any] | None) -> str:
    meta = meta or {}
    contract = meta.get("project_start_contract") if isinstance(meta.get("project_start_contract"), dict) else {}
    return _normalize_path_text(
        str(
            contract.get("target_workspace_abs_path")
            or meta.get("workspace_abs_path")
            or meta.get("workspace_rel_path")
            or ""
        )
    )


def _requested_parent(meta: dict[str, Any] | None) -> str:
    meta = meta or {}
    contract = meta.get("project_start_contract") if isinstance(meta.get("project_start_contract"), dict) else {}
    return _normalize_path_text(
        str(
            contract.get("requested_parent_abs_path")
            or meta.get("requested_workspace_abs_path")
            or ""
        )
    )


def _forbidden_source_paths(meta: dict[str, Any] | None) -> list[str]:
    meta = meta or {}
    contract = meta.get("project_start_contract") if isinstance(meta.get("project_start_contract"), dict) else {}
    values = contract.get("forbidden_source_paths") or meta.get("forbidden_source_paths") or []
    if not isinstance(values, list):
        return []
    return [_normalize_path_text(str(value)) for value in values if str(value or "").strip()]


def _mentioned_forbidden_source(meta: dict[str, Any] | None, text: str) -> str | None:
    lowered = str(text or "").lower()
    for source_path in _forbidden_source_paths(meta):
        if source_path and source_path.lower() in lowered:
            return source_path
    contract = (meta or {}).get("project_start_contract")
    terms = contract.get("forbidden_source_terms") if isinstance(contract, dict) else []
    if isinstance(terms, list):
        for term in terms:
            cleaned = " ".join(str(term or "").lower().split())
            if cleaned and cleaned in " ".join(lowered.split()):
                return cleaned
    return None


def _mentioned_sibling_path(meta: dict[str, Any] | None, text: str) -> str | None:
    parent = _requested_parent(meta)
    target = _target_root(meta)
    if not parent or not target:
        return None
    for candidate in _path_mentions(text):
        normalized = _normalize_path_text(candidate)
        if not normalized.lower().startswith(parent.lower().rstrip("/") + "/"):
            continue
        if normalized.lower() == target.lower() or normalized.lower().startswith(target.lower().rstrip("/") + "/"):
            continue
        return normalized
    return None


def _path_mentions(text: str) -> list[str]:
    return [_clean_path_token(match.group("path")) for match in _PATH_RE.finditer(str(text or ""))]


def _mentions_path(text: str, path: str) -> bool:
    normalized_text = str(text or "").replace("\\", "/").lower()
    normalized_path = path.replace("\\", "/").lower().rstrip("/")
    return bool(normalized_path and normalized_path in normalized_text)


def _normalize_path_text(path: str) -> str:
    text = _clean_path_token(path).replace("\\", "/")
    if text.startswith("file://"):
        text = text[7:]
    try:
        if text.startswith(("~", "/")):
            return Path(text).expanduser().resolve(strict=False).as_posix()
    except OSError:
        pass
    return text.rstrip("/")


def _clean_path_token(raw: str) -> str:
    return str(raw or "").strip().rstrip(".,;:!?)\"'")
