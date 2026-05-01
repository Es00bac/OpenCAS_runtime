"""Conversation compactor for OpenCAS."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from opencas.api import LLMClient
from opencas.memory import CompactionRecord, Episode, Memory, MemoryStore
from opencas.telemetry import EventKind, Tracer


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

        # Prefer reconstructed conversation messages for summarization
        if self.context_store is not None:
            messages = await self.context_store.list_recent(session_id, limit=2000)
            if messages:
                message_dicts = []
                for m in messages:
                    msg: Dict[str, Any] = {"role": m.role.value, "content": m.content}
                    if m.meta.get("tool_calls"):
                        msg["tool_calls"] = m.meta["tool_calls"]
                    if m.meta.get("tool_call_id"):
                        msg["tool_call_id"] = m.meta["tool_call_id"]
                    if m.meta.get("name"):
                        msg["name"] = m.meta["name"]
                    message_dicts.append(msg)
                message_dicts = self._repair_tool_pairing(message_dicts)
                message_dicts = self._strip_tool_details(message_dicts)
                summary = await self._summarize_messages(message_dicts)
            else:
                summary = await self._summarize_episodes(to_compact)
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

        await self._inject_narrative_bridge(session_id, summary)

        return record

    async def _inject_narrative_bridge(
        self,
        session_id: str,
        summary: str,
    ) -> None:
        """Generate a first-person narrative bridge from the system's perspective.

        Falls back to the legacy metadata injection if the LLM call fails.
        """
        if self.context_store is None:
            return
        from opencas.context.models import MessageRole

        bridge_prefix = f"[Context: earlier conversation was compacted. Summary: {summary}]"
        bridge = await self._generate_narrative_bridge(summary, session_id)
        content = bridge_prefix if not bridge else f"{bridge_prefix}\n{bridge}"

        await self.context_store.append(
            session_id,
            MessageRole.SYSTEM,
            content,
            meta={
                "synthetic": True,
                "source": "compaction_narrative_bridge",
                "session_id": session_id,
            },
        )

    async def _generate_narrative_bridge(self, summary: str, session_id: str) -> Optional[str]:
        """Use the LLM to generate a first-person narrative bridge."""
        prompt = (
            "Summarize this conversation as a narrative bridge from your perspective as the OpenCAS agent. "
            "Include: what mattered, how you felt, and what continuity thread carries forward. "
            "Write in first person, 2-3 sentences. Be emotionally honest but not melodramatic.\n\n"
            f"Conversation summary: {summary}"
        )
        messages = [
            {"role": "system", "content": "You are the OpenCAS agent writing a continuity bridge for yourself after memory compaction."},
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
                return choices[0].get("message", {}).get("content", "").strip()
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
        if len(content) <= max_chars:
            return content
        half = max_chars // 2
        return content[:half] + "\n... [episode content truncated]\n" + content[-half:]

    async def _summarize_episodes(self, episodes: List[Episode]) -> str:
        """Use the LLM to condense a batch of episodes into a summary."""
        if not episodes:
            return ""
        lines = []
        for ep in episodes:
            prefix = f"[{ep.kind.value}]"
            if ep.session_id:
                prefix += f" ({ep.session_id})"
            content = self._truncate_episode_content(ep.content or "", max_chars=2000)
            lines.append(f"{prefix} {content}")
        prompt = (
            "Summarize the following conversation episodes into a concise paragraph. "
            "Preserve key facts, decisions, and user intent.\n\n"
            + "\n".join(lines)
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
                return choices[0].get("message", {}).get("content", "").strip()
        except Exception:
            pass
        # Fallback: concatenate truncated contents
        return " | ".join(e.content[:200] for e in episodes)

    async def _summarize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Use the LLM to condense a list of conversation messages into a summary."""
        if not messages:
            return ""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = self._truncate_episode_content(str(msg.get("content", "")), max_chars=2000)
            lines.append(f"[{role}] {content}")
        prompt = (
            "Summarize the following conversation messages into a concise paragraph. "
            "Preserve key facts, decisions, and user intent.\n\n"
            + "\n".join(lines)
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
                return choices[0].get("message", {}).get("content", "").strip()
        except Exception:
            pass
        # Fallback: concatenate truncated contents
        return " | ".join(str(m.get("content", ""))[:200] for m in messages)
