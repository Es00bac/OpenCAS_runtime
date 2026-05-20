from __future__ import annotations

from datetime import datetime, timezone

from opencas.context.builder_support import to_memory_entries
from opencas.context.models import RetrievalResult
from opencas.memory import Episode, EpisodeKind


def test_context_builder_self_label_for_retrieved_self_episodes() -> None:
    created = datetime(2026, 5, 5, 12, tzinfo=timezone.utc)
    action = Episode(
        created_at=created,
        kind=EpisodeKind.ACTION,
        session_id="s1",
        content="tool fs_write_file path=/tmp/a.md bytes=1 checksum=abc",
        payload={"tool_name": "fs_write_file"},
    )
    assistant_turn = Episode(
        created_at=created,
        kind=EpisodeKind.TURN,
        session_id="s1",
        content="I revised the manuscript.",
        payload={"role": "assistant"},
    )
    user_turn = Episode(
        created_at=created,
        kind=EpisodeKind.TURN,
        session_id="s1",
        content="Did you revise the manuscript?",
        payload={"role": "user"},
    )

    entries = to_memory_entries(
        [
            RetrievalResult(
                source_type="episode",
                source_id=str(action.episode_id),
                content=action.content,
                episode=action,
            ),
            RetrievalResult(
                source_type="episode",
                source_id=str(assistant_turn.episode_id),
                content=assistant_turn.content,
                episode=assistant_turn,
            ),
            RetrievalResult(
                source_type="episode",
                source_id=str(user_turn.episode_id),
                content=user_turn.content,
                episode=user_turn,
            ),
        ]
    )

    assert entries[0].content.startswith("[SELF]")
    assert "[ACTION][tool=fs_write_file]" in entries[0].content
    assert entries[1].content.startswith("[SELF]")
    assert "[TURN][role=assistant]" in entries[1].content
    assert entries[2].content.startswith("[EPISODE]")
    assert "[TURN][role=user]" in entries[2].content
