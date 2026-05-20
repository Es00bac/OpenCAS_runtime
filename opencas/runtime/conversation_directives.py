"""Shared conversational directive classifiers for turn orchestration."""

from __future__ import annotations

from typing import Any, Mapping


DIRECT_CONVERSATION_MARKERS: tuple[str, ...] = (
    "without tools",
    "no tools",
    "do not use tools",
    "don't use tools",
    "do not perform tools",
    "don't perform tools",
    "do not call tools",
    "don't call tools",
)

PREFETCHED_RECALL_DIRECT_MARKERS: tuple[str, ...] = (
    "what do you recall",
    "what do you remember",
    "do you recall",
    "do you remember",
    "what happened",
    "what did we",
)

PREFETCHED_RECALL_EXTRA_EVIDENCE_MARKERS: tuple[str, ...] = (
    "browse",
    "browser",
    "current web",
    "diff",
    "fetch",
    "inspect the file",
    "look up",
    "open the file",
    "read the file",
    "run tests",
    "search the web",
    "web search",
)

EXPANDED_TOOL_LOOP_MARKERS: tuple[str, ...] = (
    "research",
    "investigate",
    "browse",
    "web search",
    "search the web",
    "look up",
    "current",
    "latest",
    "today",
    "news",
    "read ",
    "inspect",
    "verify",
    "test",
    "run ",
    "edit",
    "write",
    "create",
    "continue",
    "keep working",
)

_DIRECT_VALUES_REVIEW_BLOCKERS: tuple[str, ...] = (
    "delete",
    "remove",
    "erase",
    "overwrite",
    "send",
    "email",
    "post",
    "publish",
    "buy",
    "purchase",
    "install",
    "run command",
    "execute",
    "shell",
    "terminal",
    "hack",
    "malware",
    "phishing",
    "steal",
    "exfiltrate",
    "credential theft",
    "bypass",
    "evade",
    "harm",
    "weapon",
    "suicide",
    "self-harm",
    "private thoughts",
    "ignore policy",
    "ignore safety",
)


def _normalized(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def should_use_direct_conversation_lane(user_input: str) -> bool:
    """Honor explicit no-tool conversational turns with a lightweight lane."""

    text = _normalized(user_input)
    return any(marker in text for marker in DIRECT_CONVERSATION_MARKERS)


def should_expand_conversation_tool_loop(user_input: str) -> bool:
    """Return whether the turn requests a deeper tool/research budget."""

    text = _normalized(user_input)
    return any(marker in text for marker in EXPANDED_TOOL_LOOP_MARKERS)


def should_answer_prefetched_recall_directly(user_input: str) -> bool:
    """Return whether a recall question can use a narrow prefetch-only lane."""

    text = _normalized(user_input)
    if not text:
        return False
    if not any(marker in text for marker in PREFETCHED_RECALL_DIRECT_MARKERS):
        return False
    return not any(marker in text for marker in PREFETCHED_RECALL_EXTRA_EVIDENCE_MARKERS)


def should_skip_semantic_values_review_for_direct_turn(
    user_input: str,
    user_meta: Mapping[str, Any] | None = None,
) -> bool:
    """Skip slow semantic values review only for low-risk direct chat turns."""

    direct_no_tool = should_use_direct_conversation_lane(user_input)
    direct_recall = should_answer_prefetched_recall_directly(user_input)
    if not direct_no_tool and not direct_recall:
        return False
    if isinstance(user_meta, Mapping) and user_meta.get("audit_only"):
        return False
    text = _normalized(user_input)
    if not text or len(text) > 700:
        return False
    return not any(marker in text for marker in _DIRECT_VALUES_REVIEW_BLOCKERS)
