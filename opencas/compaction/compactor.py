"""Conversation compactor for OpenCAS."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from opencas.api import LLMClient
from opencas.identity.agent_name import resolve_agent_name
from opencas.memory import CompactionRecord, Episode, Memory, MemoryStore
from opencas.telemetry import EventKind, Tracer


MAX_COMPACTION_SOURCE_ITEMS = 80
MAX_COMPACTION_ITEM_CHARS = 1200
MAX_COMPACTION_PROMPT_CHARS = 32_000
MAX_COMPACTION_SUMMARY_CHARS = 5_000
MAX_NARRATIVE_BRIDGE_CHARS = 1_200


class ConversationCompactor:
    """Summarizes old episodes into a compact Memory record."""

    def __init__(
        self,
        memory: MemoryStore,
        llm: LLMClient,
        tracer: Optional[Tracer] = None,
        context_store: Optional[Any] = None,
        identity: Optional[Any] = None,
        embeddings: Optional[Any] = None,
    ) -> None:
        self.memory = memory
        self.llm = llm
        self.tracer = tracer
        self.context_store = context_store
        self.identity = identity
        self.embeddings = embeddings

    async def compact_session(
        self,
        session_id: str,
        tail_size: int = 10,
        min_removed_count: int = 1,
    ) -> Optional[CompactionRecord]:
        """Compact old episodes for a session, keeping the most recent *tail_size*."""
        episodes = await self.memory.list_non_compacted_episodes(
            session_id=session_id, limit=1000
        )
        if len(episodes) <= tail_size:
            return None

        to_compact = episodes[: len(episodes) - tail_size]
        if len(to_compact) < max(1, min_removed_count):
            return None
        continuation_packet = self._build_continuation_packet(to_compact)

        if self.context_store is not None and not self._episodes_have_structured_content(to_compact):
            messages = await self._recent_messages_for_summary(session_id)
            summary = await self._summarize_messages(messages) if messages else await self._summarize_episodes(to_compact)
        else:
            summary = await self._summarize_episodes(to_compact)

        avg_confidence = sum(e.confidence_score for e in to_compact) / len(to_compact) if to_compact else 0.8
        embedding_id = None
        if self.embeddings is not None:
            try:
                record = await self.embeddings.embed(
                    summary,
                    task_type="memory_compaction",
                    meta={
                        "source": "compaction",
                        "session_id": session_id,
                        "removed_count": len(to_compact),
                    },
                )
                embedding_id = record.source_hash
            except Exception as exc:
                if self.tracer:
                    self.tracer.log(
                        EventKind.MEMORY_COMPACT,
                        "Compaction summary embedding failed",
                        {
                            "session_id": session_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )

        memory = Memory(
            content=summary,
            embedding_id=embedding_id,
            source_episode_ids=[str(e.episode_id) for e in to_compact],
            tags=["compaction", f"session:{session_id}"],
            confidence_score=round(avg_confidence, 4),
        )
        await self.memory.save_memory(memory)

        episode_ids = [str(e.episode_id) for e in to_compact]
        await self.memory.mark_compacted(episode_ids)

        # Clean up orphaned edges for compacted episodes
        for ep_id in episode_ids:
            await self.memory.delete_edges_for(ep_id)

        record = CompactionRecord(
            episode_ids=episode_ids,
            summary=summary,
            removed_count=len(to_compact),
        )
        await self.memory.record_compaction(record)
        if self.identity is not None:
            self.identity.record_compaction(session_id=session_id)

        if self.tracer:
            self.tracer.log(
                EventKind.MEMORY_COMPACT,
                f"Compacted session {session_id}",
                {
                    "session_id": session_id,
                    "removed_count": len(to_compact),
                    "memory_id": str(memory.memory_id),
                },
            )

        await self._inject_narrative_bridge(
            session_id,
            summary,
            continuation_packet=continuation_packet,
        )

        return record

    async def _recent_messages_for_summary(self, session_id: str) -> List[Dict[str, Any]]:
        list_recent = getattr(self.context_store, "list_recent", None)
        if not callable(list_recent):
            return []
        try:
            entries = await list_recent(session_id)
        except TypeError:
            entries = await list_recent(session_id=session_id)
        messages: List[Dict[str, Any]] = []
        for entry in entries or []:
            role = getattr(getattr(entry, "role", None), "value", getattr(entry, "role", ""))
            message = {
                "role": str(role or ""),
                "content": str(getattr(entry, "content", "") or ""),
            }
            meta = dict(getattr(entry, "meta", {}) or {})
            if "tool_calls" in meta:
                message["tool_calls"] = meta["tool_calls"]
            if "tool_call_id" in meta:
                message["tool_call_id"] = meta["tool_call_id"]
            messages.append(message)
        return messages

    @staticmethod
    def _episodes_have_structured_content(episodes: List[Episode]) -> bool:
        return all(
            isinstance(getattr(getattr(episode, "kind", None), "value", None), str)
            and isinstance(getattr(episode, "content", None), str)
            for episode in episodes
        )

    async def _inject_narrative_bridge(
        self,
        session_id: str,
        summary: str,
        continuation_packet: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generate a first-person narrative bridge from the system's perspective.

        Falls back to the legacy metadata injection if the LLM call fails.
        """
        if self.context_store is None:
            return
        from opencas.context.models import MessageRole

        bounded_summary = self._limit_summary(summary)
        bridge_prefix = f"[Context: earlier conversation was compacted. Summary: {bounded_summary}]"
        bridge = await self._generate_narrative_bridge(bounded_summary, session_id)
        content = bridge_prefix if not bridge else f"{bridge_prefix}\n{bridge}"
        handles_block = self._render_continuation_handles(continuation_packet)
        if handles_block:
            content = f"{content}\n{handles_block}"

        meta: Dict[str, Any] = {
            "synthetic": True,
            "source": "compaction_narrative_bridge",
            "session_id": session_id,
        }
        if continuation_packet and continuation_packet.get("handles"):
            meta["continuation_packet"] = continuation_packet

        await self.context_store.append(session_id, MessageRole.SYSTEM, content, meta=meta)

    @staticmethod
    def _build_continuation_packet(episodes: List[Episode]) -> Dict[str, Any]:
        """Extract exact handles that prose summaries routinely lose."""
        handles: List[Dict[str, Any]] = []
        for episode in episodes:
            payload = dict(getattr(episode, "payload", {}) or {})
            args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
            metadata = payload.get("result_metadata") if isinstance(payload.get("result_metadata"), dict) else {}
            content = str(getattr(episode, "content", "") or "")
            path = (
                args.get("file_path")
                or args.get("path")
                or metadata.get("path")
                or metadata.get("artifact_path")
                or ConversationCompactor._regex_value(r"\bpath=([^\s]+)", content)
            )
            checksum = (
                metadata.get("checksum")
                or args.get("checksum")
                or ConversationCompactor._regex_value(r"\bchecksum=([^\s]+)", content)
            )
            episode_id = str(getattr(episode, "episode_id", ""))
            kind = str(getattr(getattr(episode, "kind", ""), "value", getattr(episode, "kind", "")))
            handle = {
                "episode_id": episode_id,
                "source_episode_id": episode_id,
                "kind": kind,
                "session_id": str(getattr(episode, "session_id", "")),
                "tool_name": payload.get("tool_name"),
                "path": str(path) if path else None,
                "checksum": str(checksum) if checksum else None,
                "task_id": metadata.get("task_id") or payload.get("task_id"),
                "schedule_id": metadata.get("schedule_id") or payload.get("schedule_id"),
                "receipt_id": metadata.get("receipt_id") or payload.get("receipt_id"),
                "plan_id": metadata.get("plan_id") or payload.get("plan_id"),
            }
            handle["retrieval_cues"] = ConversationCompactor._build_retrieval_cues(handle)
            if any(handle.get(key) for key in ("path", "checksum", "task_id", "schedule_id", "receipt_id", "plan_id")):
                handles.append({key: value for key, value in handle.items() if value})
            if len(handles) >= 20:
                break
        return {
            "version": 1,
            "source_episode_count": len(episodes),
            "handles": handles,
        }

    @staticmethod
    def _render_continuation_handles(packet: Optional[Dict[str, Any]]) -> str:
        handles = list((packet or {}).get("handles", []) or [])
        if not handles:
            return ""
        lines = ["Continuation handles:"]
        for handle in handles[:8]:
            parts = []
            for key in (
                "tool_name",
                "kind",
                "session_id",
                "path",
                "checksum",
                "task_id",
                "schedule_id",
                "receipt_id",
                "plan_id",
                "source_episode_id",
            ):
                value = handle.get(key)
                if value:
                    parts.append(f"{key}={value}")
            retrieval_cues = list(handle.get("retrieval_cues") or [])
            if retrieval_cues:
                parts.append("retrieval_cues=" + ", ".join(str(cue) for cue in retrieval_cues[:10]))
            if parts:
                lines.append(f"- {'; '.join(parts)}")
        if len(handles) > 8:
            lines.append(f"- ... {len(handles) - 8} more handle(s) in continuation_packet metadata")
        return "\n".join(lines)

    @staticmethod
    def _regex_value(pattern: str, text: str) -> Optional[str]:
        match = re.search(pattern, text)
        return match.group(1) if match else None

    @staticmethod
    def _build_retrieval_cues(handle: Dict[str, Any]) -> List[str]:
        """Store encoding-specific cues that should later retrieve the handle."""
        cues: List[str] = []

        def add(prefix: str, value: Any) -> None:
            if value is None:
                return
            text = str(value)
            if not text:
                return
            cue = f"{prefix}:{text}"
            if cue not in cues:
                cues.append(cue)

        add("tool", handle.get("tool_name"))
        add("kind", handle.get("kind"))
        add("session", handle.get("session_id"))
        add("episode", handle.get("source_episode_id") or handle.get("episode_id"))
        path = handle.get("path")
        add("path", path)
        if path:
            artifact_path = Path(str(path))
            add("basename", artifact_path.name)
            if artifact_path.parent.name:
                add("parent", artifact_path.parent.name)
        add("checksum", handle.get("checksum"))
        add("task", handle.get("task_id"))
        add("schedule", handle.get("schedule_id"))
        add("receipt", handle.get("receipt_id"))
        add("plan", handle.get("plan_id"))
        return cues

    async def _generate_narrative_bridge(self, summary: str, session_id: str) -> Optional[str]:
        """Use the LLM to generate a grounded continuity bridge."""
        agent_name = resolve_agent_name(identity=self.identity)
        bounded_summary = self._limit_summary(summary)
        prompt = (
            f"Summarize this conversation as an evidence-grounded continuity bridge for {agent_name}. "
            "Include what mattered, what state or evidence should carry forward, and any explicit uncertainty. "
            "Do not invent feelings, identity facts, relationship claims, or preferences; only mention internal "
            "state when the supplied summary directly supports it. Write in first person, 2-3 sentences.\n\n"
            f"Conversation summary: {bounded_summary}"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are {agent_name}, writing a compact continuity bridge after memory compaction. "
                    "Ground every claim in the supplied summary."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.llm.chat_completion(
                messages,
                complexity="standard",
                source="compaction_bridge",
            )
            choices = response.get("choices", [])
            if choices:
                return self._clip_middle(
                    choices[0].get("message", {}).get("content", "").strip(),
                    MAX_NARRATIVE_BRIDGE_CHARS,
                    marker="\n... [bridge truncated]\n",
                )
        except Exception:
            pass
        return None

    async def _inject_continuation_message_legacy(
        self,
        session_id: str,
        summary: str,
    ) -> None:
        """Legacy injection: thin metadata tag (pre-Phase 3 behavior)."""
        if self.context_store is None:
            return
        from opencas.context.models import MessageRole
        content = (
            f"[Context: earlier conversation was compacted. Summary: {summary}]"
        )
        await self.context_store.append(
            session_id,
            MessageRole.SYSTEM,
            content,
            meta={"synthetic": True, "source": "compaction_continuation"},
        )

    @staticmethod
    def _repair_tool_pairing(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove orphaned tool result messages whose tool_call_id has no matching call.

        This helper is designed for OpenAI-style message lists where assistant
        messages may contain ``tool_calls`` and subsequent ``tool`` role messages
        provide the results. If a ``tool`` message lacks a paired call, it is
        removed to keep the message history structurally valid.
        """
        call_ids = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    call_id = tc.get("id")
                    if call_id:
                        call_ids.add(call_id)
        repaired: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "tool":
                if msg.get("tool_call_id") in call_ids:
                    repaired.append(msg)
                # else: orphaned tool result → drop
            else:
                repaired.append(msg)
        return repaired

    @staticmethod
    def _strip_tool_details(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strip verbose ``details`` from tool result messages before summarization.

        Large tool outputs (e.g. stack traces, file listings) can blow the LLM
        context budget. This replaces deep detail fields with a short placeholder.
        """
        stripped: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 800:
                    # Truncate long tool outputs
                    msg = dict(msg)
                    msg["content"] = content[:400] + "\n... [output truncated]\n" + content[-100:]
                stripped.append(msg)
            else:
                stripped.append(msg)
        return stripped

    @staticmethod
    def _truncate_episode_content(content: str, max_chars: int = 2000) -> str:
        """Truncate a single episode content if it exceeds the compaction budget."""
        return ConversationCompactor._clip_middle(
            content,
            max_chars,
            marker="\n... [episode content truncated]\n",
        )

    async def _summarize_episodes(self, episodes: List[Episode]) -> str:
        """Use the LLM to condense a batch of episodes into a summary."""
        if not episodes:
            return ""
        lines = []
        sampled = self._sample_for_summary(episodes, MAX_COMPACTION_SOURCE_ITEMS)
        omitted = max(0, len(episodes) - len(sampled))
        if omitted:
            lines.append(f"[omitted] {omitted} middle episode(s) omitted to keep compaction bounded.")
        for ep in sampled:
            prefix = f"[{ep.kind.value}]"
            if ep.session_id:
                prefix += f" ({ep.session_id})"
            content = self._truncate_episode_content(ep.content or "", max_chars=MAX_COMPACTION_ITEM_CHARS)
            lines.append(f"{prefix} {content}")
        prompt = self._clip_middle(
            "Summarize the following conversation episodes into a concise paragraph. "
            "Preserve key facts, decisions, and user intent.\n\n"
            + "\n".join(lines),
            MAX_COMPACTION_PROMPT_CHARS,
            marker="\n... [compaction source truncated]\n",
        )
        messages = [
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.llm.chat_completion(
                messages,
                complexity="light",
                source="compaction",
            )
            choices = response.get("choices", [])
            if choices:
                return self._limit_summary(choices[0].get("message", {}).get("content", "").strip())
        except Exception:
            pass
        # Fallback: concatenate truncated contents
        return self._limit_summary(" | ".join((e.content or "")[:200] for e in sampled))

    async def _summarize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Use the LLM to condense a list of conversation messages into a summary."""
        filtered = [m for m in messages if not self._is_compaction_bridge_message(m)]
        if not filtered:
            return ""
        lines = []
        filtered = self._repair_tool_pairing(filtered)
        filtered = self._strip_tool_details(filtered)
        sampled = self._sample_for_summary(filtered, MAX_COMPACTION_SOURCE_ITEMS)
        omitted = max(0, len(filtered) - len(sampled))
        if omitted:
            lines.append(f"[omitted] {omitted} middle message(s) omitted to keep compaction bounded.")
        for msg in sampled:
            role = msg.get("role", "unknown")
            content = self._truncate_episode_content(str(msg.get("content", "")), max_chars=MAX_COMPACTION_ITEM_CHARS)
            lines.append(f"[{role}] {content}")
        prompt = self._clip_middle(
            "Summarize the following conversation messages into a concise paragraph. "
            "Preserve key facts, decisions, and user intent.\n\n"
            + "\n".join(lines),
            MAX_COMPACTION_PROMPT_CHARS,
            marker="\n... [compaction source truncated]\n",
        )
        llm_messages = [
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.llm.chat_completion(
                llm_messages,
                complexity="light",
                source="compaction",
            )
            choices = response.get("choices", [])
            if choices:
                return self._limit_summary(choices[0].get("message", {}).get("content", "").strip())
        except Exception:
            pass
        # Fallback: concatenate truncated contents
        return self._limit_summary(" | ".join(str(m.get("content", ""))[:200] for m in sampled))

    @staticmethod
    def _sample_for_summary(items: List[Any], limit: int) -> List[Any]:
        if len(items) <= limit:
            return list(items)
        head_count = max(1, limit // 2)
        tail_count = max(1, limit - head_count)
        return [*items[:head_count], *items[-tail_count:]]

    @staticmethod
    def _clip_middle(text: str, max_chars: int, *, marker: str) -> str:
        text = str(text or "")
        if len(text) <= max_chars:
            return text
        if max_chars <= len(marker) + 20:
            return text[:max_chars]
        available = max_chars - len(marker)
        head = available // 2
        tail = available - head
        return text[:head] + marker + text[-tail:]

    @classmethod
    def _limit_summary(cls, summary: str) -> str:
        return cls._clip_middle(
            summary,
            MAX_COMPACTION_SUMMARY_CHARS,
            marker="\n... [summary truncated]\n",
        )

    @staticmethod
    def _is_compaction_bridge_message(message: Dict[str, Any]) -> bool:
        if str(message.get("role") or "").lower() != "system":
            return False
        content = str(message.get("content") or "").lstrip()
        return content.startswith("[Context: earlier conversation was compacted.")
