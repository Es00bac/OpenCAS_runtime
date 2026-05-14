"""Data models for context management and prompt assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

MAX_CONTINUATION_HANDLES_PER_MESSAGE = 8
MAX_RETRIEVAL_CUES_PER_CONTINUATION_HANDLE = 6


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


class ContinuationHandle(BaseModel):
    """Exact continuation handle preserved across compaction boundaries."""

    tool_name: Optional[str] = None
    kind: Optional[str] = None
    session_id: Optional[str] = None
    path: Optional[str] = None
    checksum: Optional[str] = None
    task_id: Optional[str] = None
    schedule_id: Optional[str] = None
    receipt_id: Optional[str] = None
    plan_id: Optional[str] = None
    source_episode_id: Optional[str] = None
    episode_id: Optional[str] = None
    retrieval_cues: List[str] = Field(default_factory=list)


class ContinuationPacket(BaseModel):
    """Structured continuation data that must be rendered back into prompts."""

    version: int = 1
    source_episode_count: Optional[int] = None
    handles: List[ContinuationHandle] = Field(default_factory=list)


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
    token_budget: Optional[int] = None
    context_window: Optional[int] = None
    context_budget: Dict[str, Any] = Field(default_factory=dict)

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
                history_system_content.append(self.render_entry_content_for_prompt(entry))

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
                "content": self.render_entry_content_for_prompt(entry),
            }
            if entry.role == MessageRole.TOOL:
                msg["tool_call_id"] = entry.meta.get("tool_call_id", "")
                msg["name"] = entry.meta.get("name", "")
            if entry.role == MessageRole.ASSISTANT and entry.meta.get("tool_calls"):
                msg["tool_calls"] = entry.meta["tool_calls"]
            messages.append(msg)
        return repair_tool_message_sequence(messages)

    @staticmethod
    def render_entry_content_for_prompt(entry: MessageEntry) -> str:
        content = entry.content
        if entry.role == MessageRole.USER:
            content = ContextManifest._render_user_content(entry)
        continuation_block = ContextManifest._render_continuation_packet(
            entry.meta.get("continuation_packet")
        )
        if continuation_block and "Continuation handles:" not in content:
            content = f"{content}\n{continuation_block}" if content else continuation_block
        return content

    @staticmethod
    def _render_user_content(entry: MessageEntry) -> str:
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

    @staticmethod
    def _render_continuation_packet(packet_data: Any) -> str:
        packet = ContextManifest._parse_continuation_packet(packet_data)
        if packet is None or not packet.handles:
            return ""
        lines = ["Continuation handles:"]
        ranked_handles = ContextManifest._rank_continuation_handles(packet.handles)
        rendered_handles = ranked_handles[:MAX_CONTINUATION_HANDLES_PER_MESSAGE]
        for handle in rendered_handles:
            parts: List[str] = []
            for key in (
                "path",
                "checksum",
                "source_episode_id",
                "episode_id",
                "tool_name",
                "kind",
                "session_id",
                "task_id",
                "schedule_id",
                "receipt_id",
                "plan_id",
            ):
                value = getattr(handle, key)
                if value:
                    parts.append(f"{key}={value}")
            if handle.retrieval_cues:
                parts.append(
                    "retrieval_cues="
                    + ", ".join(
                        ContextManifest._rank_continuation_retrieval_cues(
                            handle.retrieval_cues
                        )[:MAX_RETRIEVAL_CUES_PER_CONTINUATION_HANDLE]
                    )
                )
            if parts:
                lines.append(f"- {'; '.join(parts)}")
        if len(packet.handles) > len(rendered_handles):
            lines.append(
                f"- ... {len(packet.handles) - len(rendered_handles)} more handle(s) in continuation_packet metadata"
            )
        return "\n".join(lines)

    @staticmethod
    def _rank_continuation_handles(
        handles: List[ContinuationHandle],
    ) -> List[ContinuationHandle]:
        indexed = list(enumerate(handles))
        indexed.sort(
            key=lambda item: (
                -ContextManifest._continuation_handle_priority(item[1]),
                item[0],
            )
        )
        return [handle for _index, handle in indexed]

    @staticmethod
    def _continuation_handle_priority(handle: ContinuationHandle) -> int:
        score = 0
        if handle.path:
            score += 8
        if handle.checksum:
            score += 6
        if handle.source_episode_id or handle.episode_id:
            score += 5
        cues = handle.retrieval_cues or []
        if any(str(cue).startswith("path:") for cue in cues):
            score += 4
        if any(str(cue).startswith("checksum:") for cue in cues):
            score += 3
        if any(str(cue).startswith("episode:") for cue in cues):
            score += 3
        return score

    @staticmethod
    def _rank_continuation_retrieval_cues(cues: List[str]) -> List[str]:
        priority_prefixes = ("path:", "checksum:", "episode:")

        def cue_priority(item: tuple[int, str]) -> tuple[int, int]:
            index, cue = item
            for priority, prefix in enumerate(priority_prefixes):
                if cue.startswith(prefix):
                    return (priority, index)
            return (len(priority_prefixes), index)

        return [cue for _index, cue in sorted(enumerate(cues), key=cue_priority)]

    @staticmethod
    def _parse_continuation_packet(packet_data: Any) -> Optional[ContinuationPacket]:
        if isinstance(packet_data, ContinuationPacket):
            return packet_data
        if not isinstance(packet_data, dict):
            return None

        handles: List[ContinuationHandle] = []
        for raw_handle in list(packet_data.get("handles") or []):
            if not isinstance(raw_handle, dict):
                continue
            source_episode_id = raw_handle.get("source_episode_id") or raw_handle.get("episode_id")
            retrieval_cues = raw_handle.get("retrieval_cues") or []
            if not isinstance(retrieval_cues, list):
                retrieval_cues = [str(retrieval_cues)]
            handles.append(
                ContinuationHandle(
                    tool_name=_string_or_none(raw_handle.get("tool_name")),
                    kind=_string_or_none(raw_handle.get("kind")),
                    session_id=_string_or_none(raw_handle.get("session_id")),
                    path=_string_or_none(raw_handle.get("path")),
                    checksum=_string_or_none(raw_handle.get("checksum")),
                    task_id=_string_or_none(raw_handle.get("task_id")),
                    schedule_id=_string_or_none(raw_handle.get("schedule_id")),
                    receipt_id=_string_or_none(raw_handle.get("receipt_id")),
                    plan_id=_string_or_none(raw_handle.get("plan_id")),
                    source_episode_id=_string_or_none(source_episode_id),
                    episode_id=_string_or_none(raw_handle.get("episode_id")),
                    retrieval_cues=[
                        cue for cue in (_string_or_none(cue) for cue in retrieval_cues) if cue
                    ],
                )
            )
        if not handles:
            return None
        source_episode_count = packet_data.get("source_episode_count")
        try:
            source_episode_count = int(source_episode_count) if source_episode_count is not None else None
        except (TypeError, ValueError):
            source_episode_count = None
        return ContinuationPacket(
            version=int(packet_data.get("version") or 1),
            source_episode_count=source_episode_count,
            handles=handles,
        )


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


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
