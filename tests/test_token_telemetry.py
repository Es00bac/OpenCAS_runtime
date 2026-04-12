"""Tests for TokenTelemetry."""

import json
from pathlib import Path

import pytest
import pytest_asyncio

from opencas.telemetry.token_telemetry import TokenTelemetry


@pytest_asyncio.fixture
async def telemetry(tmp_path: Path):
    return TokenTelemetry(tmp_path, buffer_flush_size=2)


@pytest.mark.asyncio
async def test_record_and_flush(telemetry: TokenTelemetry, tmp_path: Path) -> None:
    await telemetry.record(model="anthropic/claude-sonnet-4-6", prompt_tokens=10, completion_tokens=5)
    # Not flushed yet (buffer size = 2)
    events_file = tmp_path / "token-events.jsonl"
    assert not events_file.exists() or events_file.read_text() == ""
    await telemetry.record(model="anthropic/claude-sonnet-4-6", prompt_tokens=20, completion_tokens=10)
    # Now flushed
    assert events_file.exists()
    lines = events_file.read_text().strip().split("\n")
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_flush_explicit(telemetry: TokenTelemetry, tmp_path: Path) -> None:
    await telemetry.record(model="openai/gpt-4o", prompt_tokens=5, completion_tokens=5)
    await telemetry.flush()
    events_file = tmp_path / "token-events.jsonl"
    lines = events_file.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["model"] == "openai/gpt-4o"
    assert data["provider"] == "openai"


@pytest.mark.asyncio
async def test_infer_provider_from_model(telemetry: TokenTelemetry, tmp_path: Path) -> None:
    await telemetry.record(model="google/gemini-pro", prompt_tokens=1, completion_tokens=1)
    await telemetry.flush()
    lines = (tmp_path / "token-events.jsonl").read_text().strip().split("\n")
    data = json.loads(lines[0])
    assert data["provider"] == "google"


@pytest.mark.asyncio
async def test_session_events_and_summary(telemetry: TokenTelemetry) -> None:
    await telemetry.record(model="m1", prompt_tokens=10, completion_tokens=0, session_id="s1")
    await telemetry.record(model="m1", prompt_tokens=20, completion_tokens=0, session_id="s1")
    await telemetry.record(model="m1", prompt_tokens=5, completion_tokens=0, session_id="s2")
    await telemetry.flush()
    events = telemetry.get_session_events("s1")
    assert len(events) == 2
    summary = telemetry.get_session_summary("s1")
    assert summary.total_tokens == 30
    assert summary.total_calls == 2
    assert summary.avg_tokens_per_call == 15


@pytest.mark.asyncio
async def test_task_summary(telemetry: TokenTelemetry) -> None:
    await telemetry.record(model="m1", prompt_tokens=7, completion_tokens=3, task_id="t1")
    await telemetry.flush()
    summary = telemetry.get_task_summary("t1")
    assert summary.total_tokens == 10
    assert summary.total_calls == 1


@pytest.mark.asyncio
async def test_get_events_by_time_range(telemetry: TokenTelemetry) -> None:
    import time

    before = int(time.time() * 1000)
    await telemetry.record(model="m1", prompt_tokens=1, completion_tokens=0)
    await telemetry.flush()
    after = int(time.time() * 1000) + 1
    events = telemetry.get_events(before, after)
    assert len(events) == 1

    events_empty = telemetry.get_events(after + 1, after + 2)
    assert len(events_empty) == 0


@pytest.mark.asyncio
async def test_unknown_provider_fallback(telemetry: TokenTelemetry, tmp_path: Path) -> None:
    await telemetry.record(model="custom-model", prompt_tokens=1, completion_tokens=0)
    await telemetry.flush()
    lines = (tmp_path / "token-events.jsonl").read_text().strip().split("\n")
    data = json.loads(lines[0])
    assert data["provider"] == "custom-model"


@pytest.mark.asyncio
async def test_daily_rollup(telemetry: TokenTelemetry) -> None:
    import time
    base = int(time.time() * 1000)
    # Two events "today"
    await telemetry.record(model="m1", prompt_tokens=10, completion_tokens=0, latency_ms=100)
    await telemetry.record(model="m1", prompt_tokens=20, completion_tokens=0, latency_ms=200)
    # One event yesterday (simulate by writing raw JSONL with old ts)
    raw = json.dumps({
        "ts": base - 86_400_000,
        "provider": "p",
        "model": "m1",
        "promptTokens": 5,
        "completionTokens": 0,
        "totalTokens": 5,
        "latencyMs": 50,
        "source": "test",
    }) + "\n"
    events_file = telemetry.events_file
    with open(events_file, "a", encoding="utf-8") as f:
        f.write(raw)
    await telemetry.flush()

    rollups = telemetry.get_daily_rollup(base - 172_800_000, base + 86_400_000)
    assert len(rollups) == 2
    # Yesterday
    assert rollups[0].total_tokens == 5
    assert rollups[0].total_calls == 1
    # Today
    assert rollups[1].total_tokens == 30
    assert rollups[1].total_calls == 2


@pytest.mark.asyncio
async def test_time_series(telemetry: TokenTelemetry) -> None:
    import time
    base = int(time.time() * 1000)
    # Write events into two 1-hour buckets
    for offset in [0, 1000, 3_600_001]:
        raw = json.dumps({
            "ts": base + offset,
            "provider": "p",
            "model": "m1",
            "promptTokens": 10,
            "completionTokens": 0,
            "totalTokens": 10,
            "latencyMs": 100,
            "source": "test",
        }) + "\n"
        with open(telemetry.events_file, "a", encoding="utf-8") as f:
            f.write(raw)
    await telemetry.flush()

    series = telemetry.get_time_series(base, base + 7_200_000, bucket_ms=3_600_000)
    assert len(series) == 2
    assert series[0].total_calls == 2
    assert series[0].total_tokens == 20
    assert series[1].total_calls == 1
    assert series[1].total_tokens == 10
