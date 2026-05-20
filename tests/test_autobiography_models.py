from __future__ import annotations

from datetime import datetime, timezone

from opencas.memory.autobiography.models import SessionAnchor


def test_session_anchor_defaults_are_compact_and_structured() -> None:
    anchor = SessionAnchor(
        session_id="s1",
        created_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
        agent_name="OpenCAS",
    )

    assert anchor.episode_kind_tally == {}
    assert anchor.recall_failure_episode_ids == []
    assert anchor.recall_recovery_episode_ids == []
    assert anchor.gist is None
    assert anchor.gist_version == 0
    assert anchor.version == 1
