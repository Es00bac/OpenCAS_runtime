"""Doctor diagnostic runner for OpenCAS."""

from __future__ import annotations

import inspect
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

import httpx

if TYPE_CHECKING:
    from opencas.bootstrap import BootstrapContext

from opencas.runtime.consolidation_worker import (
    consolidation_worker_status_path,
    load_consolidation_worker_status,
)

from .models import CheckStatus, DiagnosticCheck, HealthReport


class Doctor:
    """Runs health and integrity checks across the OpenCAS substrate."""

    def __init__(self, context: Optional[BootstrapContext] = None) -> None:
        self.context = context

    async def run_all(self) -> HealthReport:
        """Execute all known diagnostic checks and return a report."""
        report = HealthReport()
        report.checks.append(await self.check_bootstrap_readiness())
        report.checks.append(await self.check_readiness())
        report.checks.append(await self.check_sandbox())
        report.checks.append(await self.check_qdrant_reachable())
        report.checks.append(await self.check_memory_integrity())
        report.checks.append(await self.check_embedding_index())
        report.checks.append(await self.check_continuity())
        report.checks.append(await self.check_identity_persistence())
        report.checks.append(await self.check_somatic_state())
        report.checks.append(await self.check_somatic_variance())
        report.checks.append(await self.check_telemetry_writable())
        report.checks.append(await self.check_baa_queue_depth())
        report.checks.append(await self.check_embedding_latency())
        report.checks.append(await self.check_compaction_lag())
        report.checks.append(await self.check_consolidation_worker())

        # Recompute overall from collected checks
        statuses = {c.status for c in report.checks}
        if CheckStatus.FAIL in statuses:
            report.overall = CheckStatus.FAIL
        elif CheckStatus.WARN in statuses:
            report.overall = CheckStatus.WARN
        elif statuses == {CheckStatus.SKIP}:
            report.overall = CheckStatus.SKIP
        else:
            report.overall = CheckStatus.PASS

        return report

    async def check_bootstrap_readiness(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="bootstrap_readiness",
                status=CheckStatus.SKIP,
                message="No bootstrap context provided",
            )
        return DiagnosticCheck(
            name="bootstrap_readiness",
            status=CheckStatus.PASS,
            message="Bootstrap context present",
            details={"session_id": self.context.config.session_id},
        )

    async def check_memory_integrity(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="memory_integrity",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        try:
            episodes = await self.context.memory.list_episodes(limit=1)
            return DiagnosticCheck(
                name="memory_integrity",
                status=CheckStatus.PASS,
                message="Memory store is queryable",
                details={"sample_episode_count": len(episodes)},
            )
        except Exception as exc:
            return DiagnosticCheck(
                name="memory_integrity",
                status=CheckStatus.FAIL,
                message=f"Memory query failed: {exc}",
            )

    async def check_qdrant_reachable(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="qdrant_reachable",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        config = getattr(self.context, "config", None)
        qdrant_url = getattr(config, "qdrant_url", None)
        if not qdrant_url:
            return DiagnosticCheck(
                name="qdrant_reachable",
                status=CheckStatus.SKIP,
                message="Qdrant URL is not configured",
            )
        url = f"{str(qdrant_url).rstrip('/')}/collections"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(url)
                response.raise_for_status()
            payload = response.json()
            collections = (
                payload.get("result", {}).get("collections", [])
                if isinstance(payload, dict)
                else []
            )
            status = CheckStatus.PASS if collections else CheckStatus.WARN
            return DiagnosticCheck(
                name="qdrant_reachable",
                status=status,
                message=(
                    f"Qdrant reachable with {len(collections)} collections"
                    if collections
                    else "Qdrant reachable but no collections were reported"
                ),
                details={
                    "url": str(qdrant_url),
                    "collection_count": len(collections),
                    "collections": [
                        c.get("name") for c in collections if isinstance(c, dict)
                    ][:20],
                },
            )
        except Exception as exc:
            return DiagnosticCheck(
                name="qdrant_reachable",
                status=CheckStatus.FAIL,
                message=f"Qdrant probe failed: {exc}",
                details={"url": str(qdrant_url)},
            )

    async def check_embedding_index(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="embedding_index",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        try:
            health = await self.context.embeddings.health()
            embeddings = self.context.embeddings
            model_id = getattr(embeddings, "model_id", None)
            if not isinstance(model_id, str):
                model_id = None
            expected_dimension = getattr(embeddings, "expected_dimension", None)
            if not isinstance(expected_dimension, int):
                expected_dimension = None

            window_hours = 6
            cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
            recent_records = await _recent_embedding_records_since(embeddings, cutoff)
            if not recent_records:
                recent_records = await _recent_embedding_records(embeddings)
            model_counts: dict[str, int] = {}
            dimension_counts: dict[str, int] = {}
            dimensions_by_model: dict[str, set[int]] = {}
            recent_fallback_count = 0
            active_dimension_mismatch_count = 0
            clean_active_streak = 0
            streak_open = True
            for record in recent_records:
                meta = _record_attr(record, "meta")
                if isinstance(meta, dict) and meta.get("embedding_remediated_at"):
                    continue
                recent_model = _record_attr(record, "model_id")
                recent_dimension = _record_attr(record, "dimension")
                clean_active_record = (
                    model_id is not None
                    and expected_dimension is not None
                    and recent_model == model_id
                    and recent_dimension == expected_dimension
                )
                if streak_open and clean_active_record:
                    clean_active_streak += 1
                elif streak_open:
                    streak_open = False
                if isinstance(recent_model, str):
                    model_counts[recent_model] = model_counts.get(recent_model, 0) + 1
                    if recent_model.startswith("local-"):
                        recent_fallback_count += 1
                if isinstance(recent_dimension, int):
                    dim_key = str(recent_dimension)
                    dimension_counts[dim_key] = dimension_counts.get(dim_key, 0) + 1
                    if isinstance(recent_model, str):
                        dimensions_by_model.setdefault(recent_model, set()).add(recent_dimension)
                    if (
                        model_id is not None
                        and recent_model == model_id
                        and expected_dimension is not None
                        and recent_dimension != expected_dimension
                    ):
                        active_dimension_mismatch_count += 1

            status = CheckStatus.PASS if health.total_records >= 0 else CheckStatus.WARN
            checked_count = sum(model_counts.values())
            fallback_ratio = (recent_fallback_count / checked_count) if checked_count else 0.0
            multi_dimension_models = sorted(
                model for model, dimensions in dimensions_by_model.items() if len(dimensions) > 1
            )
            active_multi_dimension_models = [
                model for model in multi_dimension_models if model == model_id
            ]
            stale_fallback_window = fallback_ratio > 0.1 and clean_active_streak >= 5
            if active_dimension_mismatch_count or active_multi_dimension_models:
                status = CheckStatus.FAIL
            elif fallback_ratio > 0.1:
                status = CheckStatus.WARN if stale_fallback_window else CheckStatus.FAIL
            return DiagnosticCheck(
                name="embedding_index",
                status=status,
                message=(
                    "Embedding index has recent fallback or dimension drift"
                    if status == CheckStatus.FAIL
                    else "Embedding index has stale fallback records but current writes are clean"
                    if status == CheckStatus.WARN and stale_fallback_window
                    else "Embedding service healthy"
                ),
                details={
                    "total_records": health.total_records,
                    "model_id": model_id,
                    "expected_dimension": expected_dimension,
                    "recent_window_hours": window_hours,
                    "recent_checked": checked_count,
                    "recent_model_counts": model_counts,
                    "recent_dimension_counts": dimension_counts,
                    "recent_fallback_count": recent_fallback_count,
                    "recent_fallback_ratio": round(fallback_ratio, 4),
                    "clean_active_streak": clean_active_streak,
                    "active_dimension_mismatch_count": active_dimension_mismatch_count,
                    "multi_dimension_model_ids": multi_dimension_models,
                    "active_multi_dimension_model_ids": active_multi_dimension_models,
                },
            )
        except Exception as exc:
            return DiagnosticCheck(
                name="embedding_index",
                status=CheckStatus.FAIL,
                message=f"Embedding health check failed: {exc}",
            )

    async def check_continuity(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="continuity",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        boot_count = self.context.identity.continuity.boot_count
        status = CheckStatus.PASS if boot_count > 0 else CheckStatus.WARN
        return DiagnosticCheck(
            name="continuity",
            status=status,
            message=f"Boot count: {boot_count}",
            details={"boot_count": boot_count},
        )

    async def check_identity_persistence(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="identity_persistence",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        self_model = self.context.identity.self_model
        return DiagnosticCheck(
            name="identity_persistence",
            status=CheckStatus.PASS,
            message="Identity model loaded",
            details={
                "name": self_model.name,
                "version": self_model.version,
            },
        )

    async def check_somatic_state(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="somatic_state",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        state = self.context.somatic.state
        return DiagnosticCheck(
            name="somatic_state",
            status=CheckStatus.PASS,
            message="Somatic state loaded",
            details={
                "arousal": state.arousal,
                "fatigue": state.fatigue,
                "tension": state.tension,
                "valence": state.valence,
            },
        )

    async def check_somatic_variance(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="somatic_variance",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        store = getattr(getattr(self.context, "somatic", None), "store", None)
        if store is None:
            return DiagnosticCheck(
                name="somatic_variance",
                status=CheckStatus.SKIP,
                message="Somatic store is not wired",
            )
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=6)
            snapshots = await store.trajectory(start=start, end=end)
            if len(snapshots) < 2:
                return DiagnosticCheck(
                    name="somatic_variance",
                    status=CheckStatus.WARN,
                    message="Not enough recent somatic snapshots to assess variance",
                    details={"snapshot_count": len(snapshots)},
                )

            dimensions = ("arousal", "fatigue", "tension", "valence", "focus", "energy", "certainty")
            ranges: dict[str, float] = {}
            flat_dimensions: list[str] = []
            for dimension in dimensions:
                values = [
                    float(value)
                    for value in (_record_attr(snapshot, dimension) for snapshot in snapshots)
                    if isinstance(value, (int, float))
                ]
                if not values:
                    continue
                span = round(max(values) - min(values), 6)
                ranges[dimension] = span
                if span <= 0.000001:
                    flat_dimensions.append(dimension)

            status = CheckStatus.PASS
            message = "Somatic snapshots show variance"
            if set(flat_dimensions) == set(dimensions):
                status = CheckStatus.FAIL
                message = "Somatic snapshots are flat across all tracked dimensions"
            elif len(snapshots) >= 5 and {"focus", "energy"}.issubset(flat_dimensions):
                status = CheckStatus.FAIL
                message = "Somatic focus and energy are flat across recent snapshots"
            elif flat_dimensions:
                status = CheckStatus.WARN
                message = "Some somatic dimensions are flat across recent snapshots"

            return DiagnosticCheck(
                name="somatic_variance",
                status=status,
                message=message,
                details={
                    "snapshot_count": len(snapshots),
                    "window_hours": 6,
                    "ranges": ranges,
                    "flat_dimensions": flat_dimensions,
                },
            )
        except Exception as exc:
            return DiagnosticCheck(
                name="somatic_variance",
                status=CheckStatus.FAIL,
                message=f"Somatic variance query failed: {exc}",
            )

    async def check_readiness(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="readiness",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        readiness = self.context.readiness
        status = CheckStatus.PASS if readiness.state.value == "ready" else CheckStatus.WARN
        return DiagnosticCheck(
            name="readiness",
            status=status,
            message=f"Readiness state: {readiness.state.value}",
            details=readiness.snapshot(),
        )

    async def check_sandbox(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="sandbox",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        sandbox = self.context.sandbox
        report = sandbox.report_isolation()
        status = CheckStatus.PASS
        if report["fallback"]:
            status = CheckStatus.WARN
        return DiagnosticCheck(
            name="sandbox",
            status=status,
            message=f"Sandbox mode: {report['mode']}, container: {report['container_detected']}",
            details=report,
        )

    async def check_telemetry_writable(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="telemetry_writable",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        try:
            from opencas.telemetry import EventKind, TelemetryEvent

            event = TelemetryEvent(
                kind=EventKind.DIAGNOSTIC_RUN,
                message="Doctor telemetry probe",
            )
            self.context.tracer.store.append(event)
            return DiagnosticCheck(
                name="telemetry_writable",
                status=CheckStatus.PASS,
                message="Telemetry store is writable",
            )
        except Exception as exc:
            return DiagnosticCheck(
                name="telemetry_writable",
                status=CheckStatus.FAIL,
                message=f"Telemetry write failed: {exc}",
            )

    async def check_baa_queue_depth(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="baa_queue_depth",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        baa = getattr(self.context.harness, "baa", None)
        if baa is None:
            return DiagnosticCheck(
                name="baa_queue_depth",
                status=CheckStatus.SKIP,
                message="BAA not wired to harness",
            )
        queue_size = baa.queue_size
        held_size = baa.held_size
        status = CheckStatus.PASS
        if queue_size > 50 or held_size > 20:
            status = CheckStatus.WARN
        if queue_size > 100 or held_size > 50:
            status = CheckStatus.FAIL
        return DiagnosticCheck(
            name="baa_queue_depth",
            status=status,
            message=f"BAA queue: {queue_size}, held: {held_size}",
            details={"queue_size": queue_size, "held_size": held_size},
        )

    async def check_embedding_latency(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="embedding_latency",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        try:
            health = await self.context.embeddings.health()
            elapsed_ms = health.avg_embed_latency_ms_1h
            if elapsed_ms is None:
                return DiagnosticCheck(
                    name="embedding_latency",
                    status=CheckStatus.SKIP,
                    message="No recent embedding activity",
                )
            status = CheckStatus.PASS
            if elapsed_ms > 2000:
                status = CheckStatus.WARN
            if elapsed_ms > 15000:
                status = CheckStatus.FAIL
            return DiagnosticCheck(
                name="embedding_latency",
                status=status,
                message=f"Embedding latency: {elapsed_ms:.1f}ms",
                details={"latency_ms": round(elapsed_ms, 2)},
            )
        except Exception as exc:
            return DiagnosticCheck(
                name="embedding_latency",
                status=CheckStatus.FAIL,
                message=f"Embedding latency probe failed: {exc}",
            )

    async def check_compaction_lag(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="compaction_lag",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        try:
            episodes = await self.context.memory.list_non_compacted_episodes(limit=1000)
            sample_size = len(episodes)
            tail_size = 10
            lag = sample_size
            total_non_compacted = sample_size
            backlog_stats = None
            counter = getattr(self.context.memory, "count_non_compacted_episodes", None)
            stats_func = getattr(self.context.memory, "compaction_backlog_stats", None)
            if callable(stats_func):
                stats = stats_func(tail_size=tail_size)
                if inspect.isawaitable(stats):
                    stats = await stats
                if isinstance(stats, dict):
                    backlog_stats = stats
                    counted_total = stats.get("total_non_compacted")
                    compactable_lag = stats.get("compactable_lag")
                    if isinstance(counted_total, int):
                        total_non_compacted = counted_total
                    if isinstance(compactable_lag, int):
                        lag = compactable_lag
            if backlog_stats is None and callable(counter):
                counted = counter()
                if inspect.isawaitable(counted):
                    counted = await counted
                if isinstance(counted, int):
                    lag = counted
                    total_non_compacted = counted

            kind_counts: dict[str, int] = {}
            session_counts: dict[str, int] = {}
            for episode in episodes:
                kind = getattr(episode, "kind", None)
                kind_key = getattr(kind, "value", kind)
                if not isinstance(kind_key, str):
                    kind_key = "unknown"
                kind_counts[kind_key] = kind_counts.get(kind_key, 0) + 1

                session_id = getattr(episode, "session_id", None)
                if not session_id:
                    session_id = "unknown"
                session_counts[str(session_id)] = session_counts.get(str(session_id), 0) + 1

            status = CheckStatus.PASS
            if lag > 500:
                status = CheckStatus.WARN
            if lag >= 1000:
                status = CheckStatus.FAIL
            return DiagnosticCheck(
                name="compaction_lag",
                status=status,
                message=(
                    f"Compactable episodes: {lag} "
                    f"(total non-compacted: {total_non_compacted})"
                    if backlog_stats is not None
                    else f"Non-compacted episodes: {lag}"
                ),
                details={
                    "lag": lag,
                    "compactable_lag": lag,
                    "total_non_compacted": total_non_compacted,
                    "tail_size": tail_size,
                    **(backlog_stats or {}),
                    "sample_size": sample_size,
                    "kind_counts_sample": dict(
                        sorted(kind_counts.items(), key=lambda item: item[1], reverse=True)
                    ),
                    "top_sessions_sample": dict(
                        sorted(session_counts.items(), key=lambda item: item[1], reverse=True)[:10]
                    ),
                },
            )
        except Exception as exc:
            return DiagnosticCheck(
                name="compaction_lag",
                status=CheckStatus.FAIL,
                message=f"Compaction lag query failed: {exc}",
            )

    async def check_consolidation_worker(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="consolidation_worker",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        config = getattr(self.context, "config", None)
        state_dir = getattr(config, "state_dir", None)
        if state_dir is None:
            return DiagnosticCheck(
                name="consolidation_worker",
                status=CheckStatus.SKIP,
                message="State directory is not configured",
            )

        status_path = consolidation_worker_status_path(state_dir)
        payload = load_consolidation_worker_status(state_dir)
        if not payload:
            return DiagnosticCheck(
                name="consolidation_worker",
                status=CheckStatus.WARN,
                message="No consolidation worker status file found",
                details={"status_path": str(status_path)},
            )

        worker_status = str(payload.get("status") or "unknown").lower()
        timestamp = _latest_status_timestamp(payload)
        age_seconds = None
        if timestamp is not None:
            age_seconds = round((datetime.now(timezone.utc) - timestamp).total_seconds(), 3)

        details = {
            "status_path": str(status_path),
            "worker_status": worker_status,
            "run_id": payload.get("run_id"),
            "age_seconds": age_seconds,
            "error_message": payload.get("error_message"),
            "reason": payload.get("reason"),
        }
        pid_alive = _worker_pid_alive(payload)
        if pid_alive is not None:
            details["pid_alive"] = pid_alive

        terminal_failure_statuses = {
            "failed",
            "error",
            "unreadable",
            "timeout_killed",
            "start_failed",
            "no_result",
        }
        if worker_status == "cancelled":
            reason = str(payload.get("reason") or "cancelled")
            return DiagnosticCheck(
                name="consolidation_worker",
                status=CheckStatus.WARN,
                message=f"Consolidation worker was cancelled intentionally: {reason}",
                details=details,
            )
        if worker_status in terminal_failure_statuses:
            if worker_status == "timeout_killed":
                reason = "worker timed out and was killed"
            else:
                reason = payload.get("error_message") or "no error message"
            return DiagnosticCheck(
                name="consolidation_worker",
                status=CheckStatus.FAIL,
                message=f"Consolidation worker status is {worker_status}: {reason}",
                details=details,
            )
        if worker_status in {"running", "started"} and pid_alive is False:
            readiness_since = _readiness_since(self.context)
            if (
                readiness_since is not None
                and timestamp is not None
                and timestamp < readiness_since
            ):
                return DiagnosticCheck(
                    name="consolidation_worker",
                    status=CheckStatus.WARN,
                    message="Consolidation worker status is stale from a previous boot",
                    details=details,
                )
            return DiagnosticCheck(
                name="consolidation_worker",
                status=CheckStatus.FAIL,
                message="Consolidation worker status is running but its pid is not alive",
                details=details,
            )
        if worker_status in {"running", "started"} and age_seconds is not None and age_seconds > 2 * 60 * 60:
            return DiagnosticCheck(
                name="consolidation_worker",
                status=CheckStatus.FAIL,
                message="Consolidation worker heartbeat is stale",
                details=details,
            )
        if age_seconds is not None and age_seconds > 36 * 60 * 60:
            return DiagnosticCheck(
                name="consolidation_worker",
                status=CheckStatus.WARN,
                message="Consolidation worker status has not updated recently",
                details=details,
            )

        return DiagnosticCheck(
            name="consolidation_worker",
            status=CheckStatus.PASS,
            message=f"Consolidation worker status is {worker_status}",
            details=details,
        )


async def _recent_embedding_records(embeddings: Any, limit: int = 200) -> list[Any]:
    cache = getattr(embeddings, "cache", None)
    recent_records = getattr(cache, "recent_records", None)
    if not callable(recent_records):
        return []
    result = recent_records(limit=limit)
    if hasattr(result, "__await__"):
        result = await result
    if isinstance(result, list):
        return result
    return []


async def _recent_embedding_records_since(
    embeddings: Any,
    since: datetime,
    limit: int = 10000,
) -> list[Any]:
    cache = getattr(embeddings, "cache", None)
    recent_records_since = getattr(cache, "recent_records_since", None)
    if not callable(recent_records_since):
        return []
    result = recent_records_since(since, limit=limit)
    if hasattr(result, "__await__"):
        result = await result
    if isinstance(result, list):
        return result
    return []


def _record_attr(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


def _latest_status_timestamp(payload: dict[str, Any]) -> Optional[datetime]:
    for key in ("heartbeat_at", "updated_at", "completed_at", "started_at", "created_at"):
        parsed = _parse_timestamp(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def _worker_pid_alive(payload: dict[str, Any]) -> Optional[bool]:
    try:
        pid = int(payload.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return None
    return True


def _readiness_since(context: Any) -> Optional[datetime]:
    readiness = getattr(context, "readiness", None)
    snapshot = getattr(readiness, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        payload = snapshot()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return _parse_timestamp(payload.get("since"))


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
