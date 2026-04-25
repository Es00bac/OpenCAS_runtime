"""Data models for context management and prompt assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from opencas.memory import Episode, Memory


class MessageRole(str, Enum):
    """Roles for messages in the LLM conversation context."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    MEMORY = "memory"
    TOOL = "tool"


class MessageEntry(BaseModel):
    """A single message in the conversation context."""

    message_id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    meta: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """A single retrieved memory or episode snippet for context injection."""

    source_type: str  # "memory" or "episode"
    source_id: str
    content: str
    score: float = 0.0
    episode: Optional[Any] = None
    memory: Optional[Any] = None
    embedding: Optional[List[float]] = None


class ContextManifest(BaseModel):
    """Assembled prompt context ready for LLM consumption."""

    system: Optional[MessageEntry] = None
    history: List[MessageEntry] = Field(default_factory=list)
    retrieved: List[MessageEntry] = Field(default_factory=list)
    token_estimate: Optional[int] = None

    def to_message_list(self) -> List[Dict[str, Any]]:
        """Convert manifest to OpenAI-style message list."""
        messages: List[Dict[str, Any]] = []

        system_content = []
        if self.system:
            system_content.append(self.system.content)
        for entry in self.retrieved:
            system_content.append(f"[{entry.role.upper()}] {entry.content}")

        # Merge mid-history system messages (e.g., compaction summaries) into the
        # top-level system prompt. Anthropic-format APIs reject system messages
        # that appear after user/assistant/tool messages in the list.
        history_system_content: List[str] = []
        for entry in self.history:
            if entry.meta.get("hidden"):
                continue
            if entry.role == MessageRole.SYSTEM:
                history_system_content.append(entry.content)

        if history_system_content:
            system_content.append("\n\n".join(history_system_content))

        if system_content:
            messages.append({"role": "system", "content": "\n\n".join(system_content)})

        for entry in self.history:
            if entry.meta.get("hidden"):
                continue
            if entry.role == MessageRole.SYSTEM:
                continue
            msg: Dict[str, Any] = {
                "role": entry.role.value,
                "content": self._render_entry_content(entry),
            }
            if entry.role == MessageRole.TOOL:
                msg["tool_call_id"] = entry.meta.get("tool_call_id", "")
                msg["name"] = entry.meta.get("name", "")
            if entry.role == MessageRole.ASSISTANT and entry.meta.get("tool_calls"):
                msg["tool_calls"] = entry.meta["tool_calls"]
            messages.append(msg)
        return repair_tool_message_sequence(messages)

    @staticmethod
    def _render_entry_content(entry: MessageEntry) -> str:
        if entry.role != MessageRole.USER:
            return entry.content
        attachments = entry.meta.get("attachments") or []
        if not attachments:
            return entry.content

        parts: List[str] = []
        if entry.content:
            parts.append(entry.content)
        for attachment in attachments:
            filename = attachment.get("filename") or "attachment"
            media_type = attachment.get("media_type") or "application/octet-stream"
            text_content = attachment.get("text_content")
            truncated = bool(attachment.get("text_truncated"))
            if text_content:
                header = f"[Attached file: {filename} ({media_type})"
                if truncated:
                    header += " — truncated"
                header += "]"
                parts.append(
                    "\n".join(
                        [
                            header,
                            "--- Begin attachment content ---",
                            text_content,
                            "--- End attachment content ---",
                        ]
                    )
                )
                continue
            location = attachment.get("url") or attachment.get("path") or filename
            parts.append(f"[Attached file: {filename} ({media_type}) available at {location}]")
        return "\n\n".join(part for part in parts if part)


def repair_tool_message_sequence(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip dangling tool calls/results from a message sequence.

    Provider APIs require every assistant ``tool_calls`` entry to be matched by a
    subsequent ``tool`` message with the same ``tool_call_id``. Guarded tool
    loops can leave partially-fulfilled assistant calls in persisted history, so
    repair the sequence before replaying it to the model.
    """

    tool_result_ids = {
        msg.get("tool_call_id")
        for msg in messages
        if msg.get("role") == "tool" and msg.get("tool_call_id")
    }

    repaired: List[Dict[str, Any]] = []
    kept_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            repaired.append(msg)
            continue

        filtered_calls = [
            tc for tc in (msg.get("tool_calls") or []) if tc.get("id") in tool_result_ids
        ]
        if not filtered_calls and not msg.get("content"):
            continue

        repaired_msg = dict(msg)
        if filtered_calls:
            repaired_msg["tool_calls"] = filtered_calls
            kept_call_ids.update(tc.get("id") for tc in filtered_calls if tc.get("id"))
        else:
            repaired_msg.pop("tool_calls", None)
        repaired.append(repaired_msg)

    final_messages: List[Dict[str, Any]] = []
    for msg in repaired:
        if msg.get("role") == "tool" and msg.get("tool_call_id") not in kept_call_ids:
            continue
        final_messages.append(msg)
    return final_messages
