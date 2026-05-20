"""Focused daydream API visibility tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from opencas.api.routes.daydream import build_daydream_router, _tail_text_lines
from opencas.context import ContextLane, ContextProposal, ContextProposalStore, ProposalStatus
from opencas.daydream.association_memory import DAYDREAM_ASSOCIATION_TAG
from opencas.memory import Memory


def _runtime_with_state_dir(state_dir: Path, *, executive=None) -> SimpleNamespace:
    return SimpleNamespace(
        ctx=SimpleNamespace(
            config=SimpleNamespace(state_dir=state_dir),
            daydream_store=None,
            conflict_store=None,
            work_store=None,
            daydream_signal_store=None,
            context_proposal_store=None,
        ),
        tracer=None,
        memory=None,
        executive=executive,
    )


class _FakeMemoryStore:
    def __init__(self, memories: list[Memory]) -> None:
        self.memories = memories

    async def list_memories_by_tag(
        self,
        tag: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        return [
            item
            for item in self.memories
            if tag in list(item.tags or [])
        ][offset : offset + limit]


def _write_scheduler_event(
    telemetry_dir: Path,
    *,
    timestamp: str,
    event_name: str,
    payload: dict[str, object],
) -> None:
    day = timestamp[:10]
    event = {
        "timestamp": timestamp,
        "kind": "tool_call",
        "message": f"AgentScheduler: {event_name}",
        "payload": payload,
    }
    with (telemetry_dir / f"{day}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def test_daydream_tail_text_lines_returns_recent_lines(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    path.write_text("".join(f"line-{index}\n" for index in range(500)), encoding="utf-8")

    lines = _tail_text_lines(path, max_lines=12, block_size=64)

    assert lines == [f"line-{index}" for index in range(488, 500)]


@pytest.mark.asyncio
async def test_daydream_summary_surfaces_persisted_scheduler_skips(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    _write_scheduler_event(
        telemetry_dir,
        timestamp=f"{today}T01:00:00+00:00",
        event_name="daydream_complete",
        payload={"reflections": 1, "keepers": 0, "daydream_memories_created": 0},
    )
    _write_scheduler_event(
        telemetry_dir,
        timestamp=f"{today}T02:00:00+00:00",
        event_name="daydream_skipped",
        payload={
            "reason": "executive_recommended_pause",
            "motivation": 0.14,
            "cooldown_seconds_remaining": 480,
        },
    )
    _write_scheduler_event(
        telemetry_dir,
        timestamp=f"{today}T03:00:00+00:00",
        event_name="daydream_skipped",
        payload={"scheduler_skip_reason": "conversation_not_quiet"},
    )
    app = FastAPI()
    app.include_router(build_daydream_router(_runtime_with_state_dir(tmp_path)))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/daydream/summary?window_days=1")

    assert response.status_code == 200
    data = response.json()
    assert [item["event"] for item in data["recent_runs"]] == [
        "daydream_complete",
        "daydream_skipped",
        "daydream_skipped",
    ]
    assert data["recent_runs"][1]["skip_reason"] == "executive_recommended_pause"
    assert data["recent_runs"][1]["motivation"] == 0.14
    assert data["recent_runs"][1]["cooldown_seconds_remaining"] == 480
    assert data["recent_runs"][2]["scheduler_skip_reason"] == "conversation_not_quiet"
    assert data["summary"]["consecutive_skip_count"] == 2
    assert data["summary"]["skip_diagnostic"]["status"] == "repeated_skips"
    assert "force" not in data["summary"]["skip_diagnostic"]["action"].lower()


@pytest.mark.asyncio
async def test_daydream_summary_counts_context_proposal_runs_as_useful(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    today = datetime.now(timezone.utc).date().isoformat()
    _write_scheduler_event(
        telemetry_dir,
        timestamp=f"{today}T01:00:00+00:00",
        event_name="daydream_complete",
        payload={
            "reflections": 0,
            "keepers": 0,
            "daydream_memories_created": 0,
            "daydream_association_memories_created": 0,
            "daydream_context_proposals_created": 1,
        },
    )
    app = FastAPI()
    app.include_router(build_daydream_router(_runtime_with_state_dir(tmp_path)))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/daydream/summary?window_days=1")

    assert response.status_code == 200
    data = response.json()
    assert data["recent_runs"][0]["context_proposals_created"] == 1
    assert data["recent_runs"][0]["quality_status"] == "useful"
    assert data["recent_runs"][0]["had_activity"] is True
    assert data["summary"]["last_successful_run_at"] == data["recent_runs"][0]["timestamp"]


@pytest.mark.asyncio
async def test_daydream_summary_separates_keeper_and_association_memories(tmp_path: Path) -> None:
    runtime = _runtime_with_state_dir(tmp_path)
    runtime.memory = _FakeMemoryStore(
        [
            Memory(
                content="Keeper daydream memory",
                tags=["daydream", "keeper"],
                salience=8.0,
            ),
            Memory(
                content="Background association that can be recalled later.",
                tags=["daydream", "daydream_association", "daydream_non_keeper"],
                salience=2.0,
            ),
        ]
    )

    app = FastAPI()
    app.include_router(build_daydream_router(runtime))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/daydream/summary?window_days=1")

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["keeper_memory_count"] == 1
    assert data["summary"]["association_memory_count"] == 1
    assert data["recent_association_memories"][0]["content"] == (
        "Background association that can be recalled later."
    )


@pytest.mark.asyncio
async def test_daydream_summary_compacts_details_by_default(tmp_path: Path) -> None:
    long_text = "This daydream detail is useful but too large for a polling summary. " * 80

    class _ReflectionStore:
        async def list_recent(self, limit=10, keeper_only=None):
            return [
                SimpleNamespace(
                    reflection_id="reflection-compact",
                    created_at=datetime(2026, 5, 12, 10, 0, tzinfo=timezone.utc),
                    spark_content=long_text,
                    recollection=long_text,
                    interpretation=long_text,
                    synthesis=long_text,
                    open_question=long_text,
                    changed_self_view=long_text,
                    tension_hints=["latency", "observability"],
                    alignment_score=0.8,
                    novelty_score=0.7,
                    keeper=True,
                    thoughts=[
                        {
                            "content": long_text,
                            "inner_dialogue": [long_text, long_text],
                        }
                    ],
                )
            ][:limit]

        async def get_summary(self, window_days=7):
            return {
                "total_reflections": 1,
                "total_keepers": 1,
                "window_days": window_days,
                "window_reflections": 1,
                "window_keepers": 1,
                "latest_reflection_at": "2026-05-12T10:00:00+00:00",
            }

    runtime = _runtime_with_state_dir(tmp_path)
    runtime.ctx.daydream_store = _ReflectionStore()
    runtime.memory = _FakeMemoryStore(
        [
            Memory(content=long_text, tags=["daydream", "keeper"], salience=3.0),
            Memory(content=long_text, tags=["daydream", DAYDREAM_ASSOCIATION_TAG], salience=2.0),
        ]
    )

    app = FastAPI()
    app.include_router(build_daydream_router(runtime))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/daydream/summary?window_days=1")

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["details_compacted"] is True
    reflection = data["recent_reflections"][0]
    assert reflection["thoughts_included"] is False
    assert "thoughts" not in reflection
    assert len(reflection["spark_content"]) <= 420
    assert len(reflection["synthesis"]) <= 420
    assert len(data["recent_keeper_memories"][0]["content"]) <= 420
    assert len(data["recent_association_memories"][0]["content"]) <= 420


@pytest.mark.asyncio
async def test_daydream_summary_and_promotions_surface_context_proposals(tmp_path: Path) -> None:
    proposal_store = await ContextProposalStore(tmp_path / "context_proposals.db").connect()
    proposal = ContextProposal(
        source_lane=ContextLane.REFLECTIVE,
        source_snapshot_id="truth:3:abc",
        source_epoch=3,
        proposal_kind="bad_idea_to_avoid",
        project_id="writing project 4246",
        content="writing project 4246 chapter 3 should not repeat the unsupported hallway idea.",
        evidence_refs=["daydream_reflection:chapter-3"],
        status=ProposalStatus.REJECTED,
    )
    await proposal_store.save(proposal)
    runtime = _runtime_with_state_dir(tmp_path)
    runtime.ctx.context_proposal_store = proposal_store

    app = FastAPI()
    app.include_router(build_daydream_router(runtime))

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            summary_response = await client.get("/api/daydream/summary?window_days=1")
            promotions_response = await client.get("/api/daydream/promotions?limit=10")
            proposals_response = await client.get("/api/daydream/proposals?search=Writing Project%204246%20chapter%203")
    finally:
        await proposal_store.close()

    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["summary"]["context_proposal_count"] == 1
    assert summary["summary"]["context_proposal_status_counts"]["rejected"] == 1
    assert summary["recent_context_proposals"][0]["proposal_id"] == proposal.proposal_id

    assert promotions_response.status_code == 200
    promotions = promotions_response.json()
    assert promotions["context_proposal_count"] == 1
    assert promotions["context_proposals"][0]["content_preview"].startswith("writing project 4246")

    assert proposals_response.status_code == 200
    proposals = proposals_response.json()
    assert proposals["count"] == 1
    assert proposals["items"][0]["proposal_id"] == proposal.proposal_id


@pytest.mark.asyncio
async def test_daydream_summary_counts_skipped_complete_runs_as_consecutive_skips(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    for index in range(16):
        _write_scheduler_event(
            telemetry_dir,
            timestamp=f"{today}T{index:02d}:00:00+00:00",
            event_name="daydream_complete",
            payload={
                "reflections": 0,
                "keepers": 0,
                "daydream_memories_created": 0,
                "quality_status": "skipped",
            },
        )
    app = FastAPI()
    app.include_router(build_daydream_router(_runtime_with_state_dir(tmp_path)))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/daydream/summary?window_days=1")

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["recent_run_count"] == 16
    assert data["summary"]["consecutive_skip_count"] == 16
    assert data["summary"]["last_successful_run_at"] is None


@pytest.mark.asyncio
async def test_daydream_summary_merges_live_tracer_runs_with_persisted_runs(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    _write_scheduler_event(
        telemetry_dir,
        timestamp=f"{today}T01:00:00+00:00",
        event_name="daydream_complete",
        payload={"quality_status": "skipped", "reflections": 0, "keepers": 0},
    )

    live_event = SimpleNamespace(
        message="AgentScheduler: daydream_complete",
        timestamp=now.replace(hour=2, minute=0, second=0, microsecond=0),
        payload={"reflections": 1, "keepers": 1, "quality_status": "useful"},
    )
    runtime = _runtime_with_state_dir(tmp_path)
    runtime.tracer = SimpleNamespace(
        store=SimpleNamespace(query=lambda **_kwargs: [live_event])
    )
    app = FastAPI()
    app.include_router(build_daydream_router(runtime))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/daydream/summary?window_days=1")

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["recent_run_count"] == 2
    assert data["summary"]["latest_active_run"]["quality_status"] == "useful"
    assert data["summary"]["latest_active_run"]["timestamp"].endswith("T02:00:00+00:00")


@pytest.mark.asyncio
async def test_daydream_summary_uses_skip_reason_from_skipped_complete_runs(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    _write_scheduler_event(
        telemetry_dir,
        timestamp=f"{today}T01:00:00+00:00",
        event_name="daydream_skipped",
        payload={"reason": "executive_recommended_pause"},
    )
    _write_scheduler_event(
        telemetry_dir,
        timestamp=f"{today}T02:00:00+00:00",
        event_name="daydream_complete",
        payload={
            "quality_status": "skipped",
            "reflections": 0,
            "keepers": 0,
            "skip_reason": "motivation_below_threshold",
            "skip_reasons": ["motivation_below_threshold", "cooldown"],
            "motivation": 0.14,
            "cooldown_seconds_remaining": 480,
        },
    )
    app = FastAPI()
    app.include_router(
        build_daydream_router(
            _runtime_with_state_dir(
                tmp_path,
                executive=_FakeExecutive(recommend_pause=False),
            )
        )
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/daydream/summary?window_days=1")

    assert response.status_code == 200
    data = response.json()
    latest = data["recent_runs"][-1]
    assert latest["event"] == "daydream_complete"
    assert latest["quality_status"] == "skipped"
    assert latest["skip_reason"] == "motivation_below_threshold"
    assert latest["skip_reasons"] == ["motivation_below_threshold", "cooldown"]
    assert latest["motivation"] == 0.14
    assert latest["cooldown_seconds_remaining"] == 480
    diagnostic = data["summary"]["skip_diagnostic"]
    assert diagnostic["reason"] == "motivation_below_threshold"
    assert diagnostic["status_for_operator"] == "healthy_pause"
    assert diagnostic["pause_condition"] == "motivation_below_threshold"
    assert diagnostic["condition_clears"] == "motivation rises to the daydream threshold"
    assert diagnostic["telemetry"]["motivation"] == 0.14
    assert diagnostic["telemetry"]["cooldown_seconds_remaining"] == 480


@pytest.mark.asyncio
async def test_daydream_summary_counts_skips_since_last_success(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    events = [
        ("daydream_skipped", {"reason": "baa_busy"}),
        ("daydream_skipped", {"reason": "baa_busy"}),
        ("daydream_complete", {"reflections": 1, "keepers": 0}),
        ("daydream_complete", {"quality_status": "skipped", "reflections": 0, "keepers": 0}),
        ("daydream_skipped", {"reason": "executive_recommended_pause"}),
    ]
    for index, (event_name, payload) in enumerate(events):
        _write_scheduler_event(
            telemetry_dir,
            timestamp=f"{today}T0{index}:00:00+00:00",
            event_name=event_name,
            payload=payload,
        )
    app = FastAPI()
    app.include_router(build_daydream_router(_runtime_with_state_dir(tmp_path)))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/daydream/summary?window_days=1")

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["consecutive_skip_count"] == 2
    assert data["summary"]["last_successful_run_at"].endswith("T02:00:00+00:00")


class _FakeExecutive:
    def __init__(self, *, recommend_pause: bool, pause_reason: str | None = None) -> None:
        self._recommend_pause = recommend_pause
        self._pause_reason = pause_reason

    def recommend_pause(self) -> bool:
        return self._recommend_pause

    def pause_reason(self) -> str | None:
        return self._pause_reason


@pytest.mark.asyncio
async def test_daydream_skip_diagnostic_frames_current_executive_pause_as_healthy(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    _write_scheduler_event(
        telemetry_dir,
        timestamp=f"{today}T01:00:00+00:00",
        event_name="daydream_skipped",
        payload={"reason": "executive_recommended_pause"},
    )
    app = FastAPI()
    app.include_router(
        build_daydream_router(
            _runtime_with_state_dir(
                tmp_path,
                executive=_FakeExecutive(recommend_pause=True, pause_reason="overload"),
            )
        )
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/daydream/summary?window_days=1")

    assert response.status_code == 200
    diagnostic = response.json()["summary"]["skip_diagnostic"]
    assert diagnostic["status_for_operator"] == "healthy_pause"
    assert diagnostic["pause_condition"] == "overload"


@pytest.mark.asyncio
async def test_daydream_skip_diagnostic_frames_stale_executive_pause_as_recovered(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    _write_scheduler_event(
        telemetry_dir,
        timestamp=f"{today}T01:00:00+00:00",
        event_name="daydream_skipped",
        payload={"reason": "executive_recommended_pause"},
    )
    app = FastAPI()
    app.include_router(
        build_daydream_router(
            _runtime_with_state_dir(
                tmp_path,
                executive=_FakeExecutive(recommend_pause=False),
            )
        )
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/daydream/summary?window_days=1")

    assert response.status_code == 200
    diagnostic = response.json()["summary"]["skip_diagnostic"]
    assert diagnostic["status_for_operator"] == "stale_pause_cleared"
    assert "fresh daydream attempt" in diagnostic["explanation"]


@pytest.mark.asyncio
async def test_daydream_skip_diagnostic_treats_prior_skip_as_historical_after_success(tmp_path: Path) -> None:
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    _write_scheduler_event(
        telemetry_dir,
        timestamp=f"{today}T01:00:00+00:00",
        event_name="daydream_skipped",
        payload={"reason": "executive_recommended_pause"},
    )
    _write_scheduler_event(
        telemetry_dir,
        timestamp=f"{today}T02:00:00+00:00",
        event_name="daydream_complete",
        payload={"reflections": 2, "keepers": 1},
    )
    app = FastAPI()
    app.include_router(
        build_daydream_router(
            _runtime_with_state_dir(
                tmp_path,
                executive=_FakeExecutive(recommend_pause=True, pause_reason="overload"),
            )
        )
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/daydream/summary?window_days=1")

    assert response.status_code == 200
    data = response.json()
    diagnostic = data["summary"]["skip_diagnostic"]
    assert data["summary"]["consecutive_skip_count"] == 0
    assert data["summary"]["latest_active_run"]["event"] == "daydream_complete"
    assert diagnostic["status"] == "recovered_after_skip"
    assert diagnostic["status_for_operator"] == "active_after_prior_skip"
    assert "historical" in diagnostic["explanation"]
