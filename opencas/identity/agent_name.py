"""Helpers for resolving the active agent's self-name without hard-coding personas."""

from __future__ import annotations

from typing import Any


def resolve_agent_name(
    *,
    runtime: Any = None,
    identity: Any = None,
    context: Any = None,
    default: str = "OpenCAS",
    max_length: int = 80,
) -> str:
    """Return the best available live agent name.

    OpenCAS can host freshly bootstrapped agents as well as migrated agents.
    Runtime prompts should preserve the specific self-model name when present
    and use a neutral product fallback only when no self-name is available.
    """

    candidates = (
        _identity_name(identity),
        _identity_name(getattr(context, "identity", None)),
        _identity_name(getattr(getattr(runtime, "ctx", None), "identity", None)),
        _identity_name(getattr(runtime, "identity", None)),
        getattr(runtime, "agent_name", None),
    )
    for candidate in candidates:
        name = " ".join(str(candidate or "").split())
        if name:
            return name[: max(1, int(max_length))]
    fallback = " ".join(str(default or "").split())
    return fallback[: max(1, int(max_length))] or "OpenCAS"


def _identity_name(identity: Any) -> str:
    self_model = getattr(identity, "self_model", None)
    return str(getattr(self_model, "name", "") or "")
