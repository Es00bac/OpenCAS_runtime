"""Tests for compaction tool-pairing repair and detail stripping."""

import pytest

from opencas.compaction.compactor import ConversationCompactor


class TestRepairToolPairing:
    def test_keeps_paired_tool_results(self):
        messages = [
            {"role": "assistant", "content": "ok", "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
        ]
        repaired = ConversationCompactor._repair_tool_pairing(messages)
        assert len(repaired) == 2

    def test_removes_orphaned_tool_results(self):
        messages = [
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "tool_call_id": "tc1", "content": "result"},
        ]
        repaired = ConversationCompactor._repair_tool_pairing(messages)
        assert len(repaired) == 1
        assert repaired[0]["role"] == "assistant"

    def test_mixed_paired_and_orphaned(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "good"},
            {"role": "tool", "tool_call_id": "tc2", "content": "orphan"},
        ]
        repaired = ConversationCompactor._repair_tool_pairing(messages)
        assert len(repaired) == 2
        assert repaired[1]["tool_call_id"] == "tc1"


class TestStripToolDetails:
    def test_leaves_short_tool_content_alone(self):
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": "short"},
        ]
        stripped = ConversationCompactor._strip_tool_details(messages)
        assert stripped[0]["content"] == "short"

    def test_truncates_long_tool_content(self):
        long_content = "x" * 1000
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": long_content},
        ]
        stripped = ConversationCompactor._strip_tool_details(messages)
        assert "truncated" in stripped[0]["content"]
        assert len(stripped[0]["content"]) < len(long_content)

    def test_preserves_non_tool_messages(self):
        messages = [
            {"role": "user", "content": "hello"},
        ]
        stripped = ConversationCompactor._strip_tool_details(messages)
        assert stripped[0]["content"] == "hello"


class TestTruncateEpisodeContent:
    def test_short_content_unchanged(self):
        assert ConversationCompactor._truncate_episode_content("hi") == "hi"

    def test_long_content_truncated(self):
        long_text = "x" * 3000
        truncated = ConversationCompactor._truncate_episode_content(long_text, max_chars=2000)
        assert "truncated" in truncated
        assert len(truncated) < len(long_text)


class TestCompactSessionUsesHelpers:
    @pytest.mark.asyncio
    async def test_compact_session_uses_message_repair_and_strip(self):
        from unittest.mock import AsyncMock, MagicMock
        from opencas.context.models import MessageEntry, MessageRole
        from opencas.memory import CompactionRecord

        memory = MagicMock()
        # Create 12 non-compacted episodes so compaction triggers
        fake_episodes = [MagicMock(compacted=False) for _ in range(12)]
        memory.list_non_compacted_episodes = AsyncMock(return_value=fake_episodes)
        memory.save_memory = AsyncMock()
        memory.mark_compacted = AsyncMock()
        memory.record_compaction = AsyncMock()

        # Context store returns messages including an orphaned tool result
        context_store = MagicMock()
        context_store.list_recent = AsyncMock(
            return_value=[
                MessageEntry(role=MessageRole.ASSISTANT, content="ok", meta={"tool_calls": [{"id": "tc1"}]}),
                MessageEntry(role=MessageRole.TOOL, content="result", meta={"tool_call_id": "tc1"}),
                MessageEntry(role=MessageRole.TOOL, content="orphan", meta={"tool_call_id": "tc2"}),
            ]
        )
        context_store.append = AsyncMock()

        llm = MagicMock()
        llm.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "summary"}}]}
        )

        compactor = ConversationCompactor(
            memory=memory, llm=llm, context_store=context_store
        )
        record = await compactor.compact_session("s1", tail_size=10)

        assert isinstance(record, CompactionRecord)
        # Because _repair_tool_pairing should drop the orphaned tc2 message,
        # the prompt should only contain the paired messages.
        prompt = llm.chat_completion.call_args[0][0][1]["content"]
        assert "[assistant]" in prompt
        assert "[tool]" in prompt
        assert "orphan" not in prompt
