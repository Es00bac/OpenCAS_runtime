"""Token usage telemetry: buffered JSONL with query helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TokenUsageEvent:
    """Single token usage record."""

    def __init__(
        self,
        ts: int,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: int,
        source: str,
        cost: Optional[float] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        execution_mode: Optional[str] = None,
    ) -> None:
        self.ts = ts
        self.provider = provider
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.latency_ms = latency_ms
        self.source = source
        self.cost = cost
        self.session_id = session_id
        self.task_id = task_id
        self.execution_mode = execution_mode

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "ts": self.ts,
            "provider": self.provider,
            "model": self.model,
            "promptTokens": self.prompt_tokens,
            "completionTokens": self.completion_tokens,
            "totalTokens": self.total_tokens,
            "latencyMs": self.latency_ms,
            "source": self.source,
        }
        if self.cost is not None:
            d["cost"] = self.cost
        if self.session_id is not None:
            d["sessionId"] = self.session_id
        if self.task_id is not None:
            d["taskId"] = self.task_id
        if self.execution_mode is not None:
            d["executionMode"] = self.execution_mode
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenUsageEvent":
        return cls(
            ts=int(data.get("ts", 0)),
            provider=str(data.get("provider", "unknown")),
            model=str(data.get("model", "unknown")),
            prompt_tokens=int(data.get("promptTokens", 0)),
            completion_tokens=int(data.get("completionTokens", 0)),
            total_tokens=int(data.get("totalTokens", 0)),
            latency_ms=int(data.get("latencyMs", 0)),
            source=str(data.get("source", "unknown")),
            cost=data.get("cost") if isinstance(data.get("cost"), (int, float)) else None,
            session_id=data.get("sessionId") if isinstance(data.get("sessionId"), str) else None,
            task_id=data.get("taskId") if isinstance(data.get("taskId"), str) else None,
            execution_mode=data.get("executionMode") if isinstance(data.get("executionMode"), str) else None,
        )


class TokenSummary:
    """Aggregated token usage summary."""

    def __init__(
        self,
        total_tokens: int = 0,
        total_calls: int = 0,
        avg_tokens_per_call: int = 0,
        avg_latency_ms: int = 0,
        cost_estimate: float = 0.0,
        by_provider: Optional[Dict[str, Dict[str, Any]]] = None,
        by_model: Optional[Dict[str, Dict[str, int]]] = None,
        top_models: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.total_tokens = total_tokens
        self.total_calls = total_calls
        self.avg_tokens_per_call = avg_tokens_per_call
        self.avg_latency_ms = avg_latency_ms
        self.cost_estimate = cost_estimate
        self.by_provider = by_provider or {}
        self.by_model = by_model or {}
        self.top_models = top_models or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "totalTokens": self.total_tokens,
            "totalCalls": self.total_calls,
            "avgTokensPerCall": self.avg_tokens_per_call,
            "avgLatencyMs": self.avg_latency_ms,
            "costEstimate": self.cost_estimate,
            "byProvider": self.by_provider,
            "byModel": self.by_model,
            "topModels": self.top_models,
        }


class DailyRollup:
    """Aggregated token usage for a single calendar day."""

    def __init__(
        self,
        date: str,
        total_tokens: int = 0,
        total_calls: int = 0,
        avg_latency_ms: int = 0,
        cost_estimate: float = 0.0,
    ) -> None:
        self.date = date
        self.total_tokens = total_tokens
        self.total_calls = total_calls
        self.avg_latency_ms = avg_latency_ms
        self.cost_estimate = cost_estimate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "totalTokens": self.total_tokens,
            "totalCalls": self.total_calls,
            "avgLatencyMs": self.avg_latency_ms,
            "costEstimate": self.cost_estimate,
        }


class TimeSeriesPoint:
    """A single point in a time-series of token usage."""

    def __init__(
        self,
        bucket_start: int,
        total_tokens: int = 0,
        total_calls: int = 0,
        avg_latency_ms: int = 0,
        cost_estimate: float = 0.0,
    ) -> None:
        self.bucket_start = bucket_start
        self.total_tokens = total_tokens
        self.total_calls = total_calls
        self.avg_latency_ms = avg_latency_ms
        self.cost_estimate = cost_estimate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucketStart": self.bucket_start,
            "totalTokens": self.total_tokens,
            "totalCalls": self.total_calls,
            "avgLatencyMs": self.avg_latency_ms,
            "costEstimate": self.cost_estimate,
        }


class TokenTelemetry:
    """Buffered JSONL store for LLM token usage events.

    Events are flushed to *state_dir*/telemetry/token-events.jsonl either
    when the in-memory buffer reaches *buffer_flush_size* or when flush() is
    called explicitly.
    """

    def __init__(
        self,
        telemetry_dir: Path | str,
        buffer_flush_size: int = 20,
    ) -> None:
        self.events_file = Path(telemetry_dir) / "token-events.jsonl"
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_flush_size = max(1, buffer_flush_size)
        self._buffer: List[TokenUsageEvent] = []
        self._lock = asyncio.Lock()

    async def record(
        self,
        *,
        model: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        latency_ms: Optional[int] = None,
        provider: Optional[str] = None,
        source: Optional[str] = None,
        cost: Optional[float] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        execution_mode: Optional[str] = None,
    ) -> None:
        """Record a token usage event."""
        norm_provider = _infer_provider(model, provider)
        pt = _norm_int(prompt_tokens, 0)
        ct = _norm_int(completion_tokens, 0)
        tt = _norm_int(total_tokens, pt + ct)
        event = TokenUsageEvent(
            ts=_now_ms(),
            provider=norm_provider,
            model=_norm_str(model, "unknown"),
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=tt,
            latency_ms=_norm_int(latency_ms, 0),
            source=_norm_str(source, "chat"),
            cost=cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
            session_id=_norm_optional_str(session_id),
            task_id=_norm_optional_str(task_id),
            execution_mode=_norm_optional_str(execution_mode),
        )
        async with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self.buffer_flush_size:
                await self._flush_unlocked()

    async def flush(self) -> None:
        """Flush any buffered events to disk."""
        async with self._lock:
            await self._flush_unlocked()

    async def prune_old_events(self, max_age_days: int = 30) -> int:
        """Prune token-events older than *max_age_days* and return removed events."""
        max_age_days = int(max_age_days)
        if max_age_days < 0:
            return 0
        cutoff_ms = _now_ms() - (max_age_days * 24 * 60 * 60 * 1000)
        if not self.events_file.exists():
            return 0

        async with self._lock:
            # Preserve buffered events first, then prune persisted rows.
            await self._flush_unlocked()
            kept: List[str] = []
            removed = 0
            with open(self.events_file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                    except (json.JSONDecodeError, TypeError):
                        removed += 1
                        continue
                    ts = _norm_int(data.get("ts", 0), 0)
                    if ts >= cutoff_ms:
                        kept.append(stripped)
                    else:
                        removed += 1

            if removed:
                text = "\n".join(kept)
                if kept:
                    text += "\n"
                with open(self.events_file, "w", encoding="utf-8") as f:
                    f.write(text)
            return removed

    async def _flush_unlocked(self) -> None:
        if not self._buffer:
            return
        lines = "\n".join(json.dumps(e.to_dict()) for e in self._buffer) + "\n"
        _append_to_file(self.events_file, lines)
        self._buffer.clear()

    def get_events(self, start_time: int, end_time: int) -> List[TokenUsageEvent]:
        """Return events with ts in [start_time, end_time]."""
        start = _norm_int(start_time, 0)
        end = _norm_int(end_time, _now_ms())
        if end < start:
            return []
        return [e for e in self._read_all_events() if start <= e.ts <= end]

    def get_session_events(self, session_id: str, limit: int = 1000) -> List[TokenUsageEvent]:
        sid = _norm_str(session_id, "")
        if not sid:
            return []
        clamped = max(1, min(10000, int(limit)))
        events = [e for e in self._read_all_events() if e.session_id == sid]
        return events[-clamped:]

    def get_session_summary(self, session_id: str) -> TokenSummary:
        return _build_summary(self.get_session_events(session_id))

    def get_task_summary(self, task_id: str) -> TokenSummary:
        tid = _norm_str(task_id, "")
        if not tid:
            return _empty_summary()
        events = [e for e in self._read_all_events() if e.task_id == tid]
        return _build_summary(events)

    def get_summary(self, start_time: int, end_time: int) -> TokenSummary:
        return _build_summary(self.get_events(start_time, end_time))

    def get_daily_rollup(
        self,
        start_time: int,
        end_time: int,
    ) -> List[DailyRollup]:
        """Return aggregated usage rolled up by calendar day."""
        events = self.get_events(start_time, end_time)
        by_day: Dict[str, List[TokenUsageEvent]] = {}
        for e in events:
            day = _ts_to_date(e.ts)
            by_day.setdefault(day, []).append(e)
        rollups = []
        for day in sorted(by_day.keys()):
            summary = _build_summary(by_day[day])
            rollups.append(
                DailyRollup(
                    date=day,
                    total_tokens=summary.total_tokens,
                    total_calls=summary.total_calls,
                    avg_latency_ms=summary.avg_latency_ms,
                    cost_estimate=summary.cost_estimate,
                )
            )
        return rollups

    def get_time_series(
        self,
        start_time: int,
        end_time: int,
        bucket_ms: int = 3_600_000,
    ) -> List[TimeSeriesPoint]:
        """Return time-series points aggregated into fixed-size buckets."""
        if bucket_ms <= 0:
            bucket_ms = 3_600_000
        events = self.get_events(start_time, end_time)
        if not events:
            return []
        buckets: Dict[int, List[TokenUsageEvent]] = {}
        for e in events:
            bucket_start = (e.ts // bucket_ms) * bucket_ms
            buckets.setdefault(bucket_start, []).append(e)
        points = []
        for bucket_start in sorted(buckets.keys()):
            summary = _build_summary(buckets[bucket_start])
            points.append(
                TimeSeriesPoint(
                    bucket_start=bucket_start,
                    total_tokens=summary.total_tokens,
                    total_calls=summary.total_calls,
                    avg_latency_ms=summary.avg_latency_ms,
                    cost_estimate=summary.cost_estimate,
                )
            )
        return points

    def get_recent_events(
        self,
        start_time: int,
        end_time: int,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clamped = max(1, min(int(limit), 1000))
        events = self.get_events(start_time, end_time)
        return [event.to_dict() for event in sorted(events, key=lambda item: item.ts, reverse=True)[:clamped]]

    def get_top_events(
        self,
        start_time: int,
        end_time: int,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        clamped = max(1, min(int(limit), 200))
        events = self.get_events(start_time, end_time)
        ranked = sorted(
            events,
            key=lambda item: (item.total_tokens, item.latency_ms, item.ts),
            reverse=True,
        )[:clamped]
        return [event.to_dict() for event in ranked]

    def get_breakdown(
        self,
        start_time: int,
        end_time: int,
        field: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        clamped = max(1, min(int(limit), 200))
        field_map = {
            "provider": "provider",
            "model": "model",
            "source": "source",
            "execution_mode": "execution_mode",
            "session_id": "session_id",
            "task_id": "task_id",
        }
        attr = field_map.get(str(field))
        if attr is None:
            raise ValueError(f"Unsupported breakdown field '{field}'")
        grouped: Dict[str, List[TokenUsageEvent]] = {}
        for event in self.get_events(start_time, end_time):
            label = getattr(event, attr, None)
            normalized = str(label).strip() if label not in (None, "") else "unknown"
            grouped.setdefault(normalized, []).append(event)
        rows: List[Dict[str, Any]] = []
        for label, events in grouped.items():
            summary = _build_summary(events)
            rows.append(
                {
                    field: label,
                    "totalTokens": summary.total_tokens,
                    "totalCalls": summary.total_calls,
                    "avgLatencyMs": summary.avg_latency_ms,
                    "costEstimate": summary.cost_estimate,
                }
            )
        rows.sort(key=lambda item: (item["totalTokens"], item["totalCalls"]), reverse=True)
        return rows[:clamped]

    def _read_all_events(self) -> List[TokenUsageEvent]:
        # Run sync file read in executor to be safe, but caller may be sync
        if not self.events_file.exists():
            return []
        with open(self.events_file, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip():
            return []
        events: List[TokenUsageEvent] = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                if not isinstance(data, dict):
                    continue
                events.append(TokenUsageEvent.from_dict(data))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return events


def _append_to_file(path: Path, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def _build_summary(events: List[TokenUsageEvent]) -> TokenSummary:
    if not events:
        return _empty_summary()
    by_provider: Dict[str, Dict[str, Any]] = {}
    by_model: Dict[str, Dict[str, int]] = {}
    total_tokens = 0
    total_latency = 0
    total_cost = 0.0
    for e in events:
        total_tokens += e.total_tokens
        total_latency += e.latency_ms
        total_cost += e.cost or 0.0
        if e.provider not in by_provider:
            by_provider[e.provider] = {"tokens": 0, "calls": 0, "cost": 0.0}
        by_provider[e.provider]["tokens"] += e.total_tokens
        by_provider[e.provider]["calls"] += 1
        by_provider[e.provider]["cost"] += e.cost or 0.0
        if e.model not in by_model:
            by_model[e.model] = {"tokens": 0, "calls": 0}
        by_model[e.model]["tokens"] += e.total_tokens
        by_model[e.model]["calls"] += 1
    top_models = sorted(
        ({"model": m, "tokens": d["tokens"]} for m, d in by_model.items()),
        key=lambda x: x["tokens"],
        reverse=True,
    )[:10]
    return TokenSummary(
        total_tokens=total_tokens,
        total_calls=len(events),
        avg_tokens_per_call=round(total_tokens / len(events)),
        avg_latency_ms=round(total_latency / len(events)),
        cost_estimate=round(total_cost, 4),
        by_provider=by_provider,
        by_model=by_model,
        top_models=top_models,
    )


def _empty_summary() -> TokenSummary:
    return TokenSummary()


def _infer_provider(model: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit.strip() or "unknown"
    normalized = str(model or "").lower().strip()
    if not normalized:
        return "unknown"
    if "/" in normalized:
        return normalized.split("/", 1)[0]
    if "kimi" in normalized:
        return "kimi"
    if "zai" in normalized or "glm" in normalized:
        return "zai"
    if "google" in normalized:
        return "google"
    if "openai" in normalized:
        return "openai"
    if "anthropic" in normalized:
        return "anthropic"
    return normalized


def _norm_str(value: Any, fallback: str) -> str:
    s = str(value).strip()
    return s if s else fallback


def _norm_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _norm_int(value: Any, fallback: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and value == int(value):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _now_ms() -> int:
    import time
    return int(time.time() * 1000)


def _ts_to_date(ts: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
