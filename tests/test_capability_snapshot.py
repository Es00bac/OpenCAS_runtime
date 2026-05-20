"""Capability-drift detection across runtime restarts.

These tests pin the contract that prevents the gaslighting loop where Bulma
"remembers" tools she had last boot but can no longer call this boot — without a
drift signal she keeps claiming she has them and falls back to bash/web/browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from opencas.runtime.capability_snapshot import (
    CapabilityDriftReport,
    detect_capability_drift,
    record_drift_episode,
    render_capability_drift_warning,
)
from opencas.runtime.capability_context import build_runtime_capability_context


def test_first_boot_has_no_drift_and_writes_snapshot(tmp_path: Path) -> None:
    """First-ever boot has no previous snapshot — neither lost nor gained."""
    report = detect_capability_drift(
        state_dir=tmp_path,
        current_tools=["search_memories", "bash_run_command"],
    )
    assert not report.has_drift
    assert report.lost == ()
    assert report.gained == ()
    snapshot = json.loads((tmp_path / "capability_snapshot.json").read_text())
    assert sorted(snapshot["tools"]) == ["bash_run_command", "search_memories"]


def test_subsequent_boot_reports_lost_and_gained(tmp_path: Path) -> None:
    """A second boot with a different tool set surfaces the diff."""
    detect_capability_drift(
        state_dir=tmp_path,
        current_tools=[
            "search_memories",
            "google_workspace_gmail_headlines",
            "google_workspace_calendar_schedule",
        ],
    )
    report = detect_capability_drift(
        state_dir=tmp_path,
        current_tools=["search_memories", "bash_run_command"],
    )
    assert report.lost == (
        "google_workspace_calendar_schedule",
        "google_workspace_gmail_headlines",
    )
    assert report.gained == ("bash_run_command",)
    assert report.has_drift


def test_drift_telemetry_logs_warning_event(tmp_path: Path) -> None:
    detect_capability_drift(
        state_dir=tmp_path,
        current_tools=["google_workspace_gmail_headlines", "search_memories"],
    )

    tracer = MagicMock()
    detect_capability_drift(
        state_dir=tmp_path,
        current_tools=["search_memories"],
        tracer=tracer,
    )
    tracer.log.assert_called_once()
    call_args = tracer.log.call_args
    payload = call_args.args[2] if len(call_args.args) > 2 else call_args.kwargs.get("payload")
    # log is called as log(event_kind, message, payload-dict)
    assert "google_workspace_gmail_headlines" in payload["lost"]


def test_no_telemetry_when_no_drift(tmp_path: Path) -> None:
    detect_capability_drift(state_dir=tmp_path, current_tools=["a", "b"])
    tracer = MagicMock()
    detect_capability_drift(state_dir=tmp_path, current_tools=["a", "b"], tracer=tracer)
    tracer.log.assert_not_called()


def test_render_warning_returns_none_when_no_lost() -> None:
    assert render_capability_drift_warning(None) is None
    assert render_capability_drift_warning(CapabilityDriftReport()) is None
    gained_only = CapabilityDriftReport(gained=("new_tool",))
    assert render_capability_drift_warning(gained_only) is None


def test_render_warning_lists_lost_tools_and_directive() -> None:
    report = CapabilityDriftReport(
        lost=("google_workspace_gmail_headlines", "google_workspace_calendar_schedule"),
    )
    rendered = render_capability_drift_warning(report)
    assert rendered is not None
    assert "capability_drift" in rendered
    assert "google_workspace_gmail_headlines" in rendered
    assert "google_workspace_calendar_schedule" in rendered
    # directive that breaks the gaslighting loop
    assert "Do not claim to call them" in rendered


def test_capability_context_includes_drift_warning() -> None:
    """The prompt the LLM actually sees must include the drift warning at top."""
    runtime = SimpleNamespace(
        capability_drift_report=CapabilityDriftReport(
            lost=("google_workspace_gmail_headlines",),
        ),
        # Minimal stubs so build_runtime_capability_context doesn't blow up
        tools=None,
        capability_registry=None,
        ctx=SimpleNamespace(
            identity=None,
            capability_registry=None,
            skill_registry=None,
        ),
        skill_registry=None,
        somatic=None,
        desktop_context=None,
    )
    rendered = build_runtime_capability_context(runtime)
    assert "Runtime capability evidence:" in rendered
    assert "WARNING capability_drift" in rendered
    assert "google_workspace_gmail_headlines" in rendered


@pytest.mark.asyncio
async def test_record_drift_episode_writes_observation(tmp_path: Path) -> None:
    saved: list[Any] = []

    class FakeMemory:
        async def save_episodes_batch(self, episodes):
            saved.extend(episodes)

    class FakeEmbeddings:
        async def embed(self, content, *, task_type):
            return SimpleNamespace(source_hash=f"sha-{hash(content) & 0xFFFF:x}")

    report = CapabilityDriftReport(
        lost=("google_workspace_gmail_headlines",),
        previous_recorded_at="2026-05-01T00:00:00+00:00",
        current_recorded_at="2026-05-10T00:00:00+00:00",
    )
    await record_drift_episode(
        memory=FakeMemory(),
        embeddings=FakeEmbeddings(),
        report=report,
    )
    assert len(saved) == 1
    episode = saved[0]
    assert episode.kind.value == "observation"
    assert "google_workspace_gmail_headlines" in episode.content
    assert episode.payload["capability_drift"]["lost"] == ["google_workspace_gmail_headlines"]
    assert episode.salience == 2.0
    assert episode.embedding_id is not None


@pytest.mark.asyncio
async def test_record_drift_episode_skips_when_nothing_lost() -> None:
    fake_memory = AsyncMock()
    await record_drift_episode(
        memory=fake_memory,
        embeddings=None,
        report=CapabilityDriftReport(),
    )
    fake_memory.save_episodes_batch.assert_not_called()


@pytest.mark.asyncio
async def test_record_drift_episode_succeeds_without_embeddings() -> None:
    saved: list[Any] = []

    class FakeMemory:
        async def save_episodes_batch(self, episodes):
            saved.extend(episodes)

    report = CapabilityDriftReport(lost=("some_tool",))
    await record_drift_episode(memory=FakeMemory(), embeddings=None, report=report)
    assert len(saved) == 1
    assert saved[0].embedding_id is None
