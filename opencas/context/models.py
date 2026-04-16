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
        return messages

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
