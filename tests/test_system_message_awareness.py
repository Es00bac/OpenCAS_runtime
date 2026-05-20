from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.context.models import MessageRole
from opencas.memory import EpisodeKind
from opencas.runtime.system_message_awareness import record_agent_visible_system_message


class _ContextStore:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    async def append(self, session_id: str, role: MessageRole, content: str, meta: dict | None = None) -> None:
        self.entries.append(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "meta": meta or {},
            }
        )


@pytest.mark.asyncio
async def test_agent_visible_system_message_uses_runtime_episode_path_with_salience() -> None:
    context_store = _ContextStore()
    recorded: list[dict] = []

    async def _record_episode(
        content: str,
        kind: EpisodeKind,
        *,
        session_id: str,
        role: str | None = None,
        payload: dict | None = None,
        salience: float | None = None,
    ) -> None:
        recorded.append(
            {
                "content": content,
                "kind": kind,
                "session_id": session_id,
                "role": role,
                "payload": payload or {},
                "salience": salience,
            }
        )

    runtime = SimpleNamespace(
        ctx=SimpleNamespace(
            context_store=context_store,
            config=SimpleNamespace(session_id="runtime-session"),
        ),
        _record_episode=_record_episode,
    )

    result = await record_agent_visible_system_message(
        runtime,
        content="Recent project work memory: I completed a background writing task on a manuscript.",
        event_kind="baa_task_activity",
        status="completed",
        source="baa",
        source_id="task-1",
        salience=7.5,
        payload={"project_title": "The Test Novel"},
    )

    assert result["context_recorded"] is True
    assert result["memory_recorded"] is True
    [entry] = context_store.entries
    assert entry["session_id"] == "runtime-session"
    assert entry["role"] == MessageRole.SYSTEM
    [episode] = recorded
    assert episode["kind"] == EpisodeKind.OBSERVATION
    assert episode["session_id"] == "runtime-session"
    assert episode["role"] == "system"
    assert episode["salience"] == 7.5
    assert episode["payload"]["payload"]["project_title"] == "The Test Novel"
