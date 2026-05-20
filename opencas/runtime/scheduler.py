"""Background scheduler for OpenCAS autonomous loops."""

from __future__ import annotations

import asyncio
import inspect
import os
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from opencas.execution.lanes import CommandLane, LaneConfig, LaneManager
from opencas.runtime.consolidation_state import (
    consolidation_delay_until_due,
    consolidation_runtime_state_payload,
    load_consolidation_runtime_state,
    persist_consolidation_runtime_state,
)
from opencas.runtime.consolidation_worker import load_consolidation_worker_status
from opencas.runtime.readiness import AgentReadiness, ReadinessState
from opencas.telemetry import EventKind, Tracer


class AgentScheduler:
    """Drives runtime cycles, consolidation, wellbeing maintenance, and heartbeat."""

    def __init__(
        self,
        runtime: Any,
        cycle_interval: int = 600,
        consolidation_interval: int = 86400,
        baa_heartbeat_interval: int = 120,
        daydream_interval: int = 720,
        schedule_interval: int = 60,
        wellbeing_interval: float = 1800,
        identity_heartbeat_interval: float = 60,
        readiness: Optional[AgentReadiness] = None,
        tracer: Optional[Tracer] = None,
        lane_manager: Optional[LaneManager] = None,
        focus_mode_timeout_seconds: int = 60,
        time_source: Optional[Callable[[], datetime]] = None,
        conversation_quiet_seconds: int = 90,
        consolidation_budget: Optional[Dict[str, Any]] = None,
        consolidation_retry_attempts: int = 3,
        consolidation_retry_base_seconds: Optional[float] = None,
        consolidation_failure_retry_seconds: float = 900.0,
        consolidation_failure_max_retry_seconds: float = 21600.0,
        initiative_contact_jitter_seconds: int = 180,
        telemetry_retention_days: int = 30,
        telemetry_prune_interval_seconds: int = 86400,
        workspace_sync_interval_seconds: int = 600,
        compaction_backlog_interval_seconds: float = 900,
        nightly_dream_check_interval_seconds: int = 300,
        nightly_dream_local_hour: int = 3,
        nightly_dream_timezone: str = "America/Denver",
        nightly_dream_mode: str = "medium",
    ) -> None:
        self.runtime = runtime
        self.cycle_interval = cycle_interval
        self.consolidation_interval = consolidation_interval
        self.baa_heartbeat_interval = baa_heartbeat_interval
        self.daydream_interval = daydream_interval
        self.schedule_interval = schedule_interval
        self.wellbeing_interval = max(0.0, float(wellbeing_interval))
        self.identity_heartbeat_interval = max(0.001, float(identity_heartbeat_interval))
        self.readiness = readiness
        self.tracer = tracer
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._recovery_task: asyncio.Task[None] | None = None
        # Focus mode: suspend daydream and cycle loops during deep tool-use work.
        self._focus_mode: bool = False
        self._focus_mode_since: Optional[datetime] = None
        self.focus_mode_timeout_seconds = focus_mode_timeout_seconds
        self._time_source = time_source or (lambda: datetime.now(timezone.utc))
        self.conversation_quiet_seconds = conversation_quiet_seconds
        self.initiative_contact_jitter_seconds = max(0, int(initiative_contact_jitter_seconds))
        self.telemetry_retention_days = max(1, int(telemetry_retention_days))
        self.telemetry_prune_interval_seconds = max(60, int(telemetry_prune_interval_seconds))
        self.workspace_sync_interval_seconds = max(60, int(workspace_sync_interval_seconds))
        self.compaction_backlog_interval_seconds = max(
            0.001,
            float(compaction_backlog_interval_seconds),
        )
        self.nightly_dream_check_interval_seconds = max(
            60,
            int(nightly_dream_check_interval_seconds),
        )
        self.nightly_dream_local_hour = max(0, min(23, int(nightly_dream_local_hour)))
        self.nightly_dream_timezone = str(nightly_dream_timezone or "America/Denver")
        self.nightly_dream_mode = str(nightly_dream_mode or "medium")
        self.consolidation_retry_attempts = max(1, int(consolidation_retry_attempts))
        self.consolidation_retry_base_seconds = consolidation_retry_base_seconds
        self.consolidation_failure_retry_seconds = max(0.0, float(consolidation_failure_retry_seconds))
        self.consolidation_failure_max_retry_seconds = max(
            self.consolidation_failure_retry_seconds,
            float(consolidation_failure_max_retry_seconds),
        )
        self.consolidation_budget = consolidation_budget or {
            "max_seconds": 120,
            "worker_timeout_seconds": 1800,
            "max_llm_calls": 12,
            "max_cluster_summaries": 6,
            "max_candidates": 100,
            "max_prompt_chars": 12000,
            "max_compaction_sessions": 8,
            "max_compaction_candidates": 1000,
            "min_compaction_session_lag": 20,
            "compaction_tail_size": 10,
        }
        self._lane_manager = lane_manager or LaneManager(
            configs={
                CommandLane.CHAT: LaneConfig(max_concurrent=1),
                CommandLane.CONSOLIDATION: LaneConfig(max_concurrent=1),
                CommandLane.BAA: LaneConfig(max_concurrent=1),
                CommandLane.CRON: LaneConfig(max_concurrent=1),
            }
        )
        # Track executive pause separately from readiness/focus gating so deferred work
        # resumes when the executive actually recovers, not only when the broader
        # scheduler gate flips back to runnable.
        self._last_executive_pause: Optional[bool] = None

    async def start(self) -> None:
        """Spawn background loops."""
        if self._running:
            return
        self._running = True
        self._trace("scheduler_start", {})
        self._last_executive_pause = self._executive_pause_active()

        # Ensure BAA worker is running
        try:
            await self.runtime.baa.start()
        except Exception as exc:
            self._trace("baa_start_error", {"error": str(exc)})

        backfill_signals = getattr(self.runtime, "backfill_daydream_signal_thread_beads", None)
        if callable(backfill_signals):
            try:
                result = await backfill_signals(limit=50)
                self._trace("daydream_signal_thread_backfill", result)
            except Exception as exc:
                self._trace("daydream_signal_thread_backfill_error", {"error": str(exc)})

        record_shadow_beads = getattr(self.runtime, "record_shadow_registry_thread_beads", None)
        if callable(record_shadow_beads):
            try:
                result = await record_shadow_beads(limit=10)
                self._trace("shadow_registry_thread_beads", result)
            except Exception as exc:
                self._trace("shadow_registry_thread_beads_error", {"error": str(exc)})

        record_capability_drift = getattr(self.runtime, "record_capability_drift_episode", None)
        if callable(record_capability_drift):
            try:
                result = await record_capability_drift()
                self._trace("capability_drift_episode_recorded", result)
            except Exception as exc:
                self._trace("capability_drift_episode_error", {"error": str(exc)})

        # Start health monitor if available
        health_monitor = getattr(self.runtime.ctx, "health_monitor", None)
        if health_monitor is not None:
            try:
                health_monitor.start()
                self._trace("health_monitor_started", {})
            except Exception as exc:
                self._trace("health_monitor_start_error", {"error": str(exc)})

        self._lane_manager.start(worker_factory=self._loop_factory)
        self._tasks = [
            worker
            for state in self._lane_manager._lanes.values()
            for worker in state.workers
        ]
        if getattr(self.runtime, "recovery_coordinator", None) is not None:
            self._recovery_task = asyncio.create_task(self._recovery_loop())

    async def stop(self) -> None:
        """Cancel background loops and drain BAA."""
        if not self._running:
            return
        self._running = False
        self._trace("scheduler_stop", {})

        for task in self._tasks:
            task.cancel()
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            await asyncio.gather(self._recovery_task, return_exceptions=True)
            self._recovery_task = None
        await self._lane_manager.stop()
        self._tasks.clear()

        try:
            await self.runtime.baa.stop()
        except Exception as exc:
            self._trace("baa_stop_error", {"error": str(exc)})

        health_monitor = getattr(self.runtime.ctx, "health_monitor", None)
        if health_monitor is not None:
            try:
                await health_monitor.stop()
                self._trace("health_monitor_stopped", {})
            except Exception as exc:
                self._trace("health_monitor_stop_error", {"error": str(exc)})

    def _loop_factory(self, lane: CommandLane) -> Any:
        """Return the background loop coroutine for a given lane."""
        if lane == CommandLane.CHAT:
            return self._cycle_loop()
        if lane == CommandLane.CONSOLIDATION:
            return self._consolidation_loop()
        if lane == CommandLane.BAA:
            return self._baa_heartbeat_loop()
        if lane == CommandLane.CRON:
            return self._cron_loop()
        raise ValueError(f"Unknown lane: {lane}")

    def enter_focus_mode(self) -> None:
        """Suspend daydream and cycle loops for high-intensity tool-use work."""
        if not self._focus_mode:
            self._focus_mode = True
            self._focus_mode_since = datetime.now(timezone.utc)
            self._trace("focus_mode_entered", {})

    def exit_focus_mode(self) -> None:
        """Resume daydream and cycle loops after tool-use work completes."""
        if self._focus_mode:
            self._focus_mode = False
            self._focus_mode_since = None
            self._trace("focus_mode_exited", {})

    @property
    def focus_mode(self) -> bool:
        """True while the agent is in focus mode (daydream/cycle suspended)."""
        return self._focus_mode

    def _should_run_cycle(self) -> bool:
        if self._focus_mode and getattr(self, "_focus_mode_since", None):
            elapsed = (datetime.now(timezone.utc) - self._focus_mode_since).total_seconds()
            if elapsed > self.focus_mode_timeout_seconds:
                self._trace("focus_mode_auto_exited", {"elapsed_seconds": elapsed})
                self.exit_focus_mode()
        ready = True
        if self.readiness is not None:
            ready = self.readiness.state == ReadinessState.READY
        return ready and not self._focus_mode

    def _readiness_degraded_by_consolidation(self) -> bool:
        if self.readiness is None or self.readiness.state != ReadinessState.DEGRADED:
            return False
        reason = str(getattr(self.readiness, "reason", "") or "").lower()
        return reason.startswith("consolidation failed:") or reason.startswith(
            "run_consolidation failed:"
        )

    def _readiness_degraded_by_schedule(self) -> bool:
        if self.readiness is None or self.readiness.state != ReadinessState.DEGRADED:
            return False
        reason = str(getattr(self.readiness, "reason", "") or "").lower()
        return reason.startswith("schedule processing failed:")

    def _should_run_consolidation(self) -> bool:
        """Allow consolidation to recover its own degraded readiness state."""
        if self._focus_mode and getattr(self, "_focus_mode_since", None):
            elapsed = (datetime.now(timezone.utc) - self._focus_mode_since).total_seconds()
            if elapsed > self.focus_mode_timeout_seconds:
                self._trace("focus_mode_auto_exited", {"elapsed_seconds": elapsed})
                self.exit_focus_mode()
        if self._focus_mode:
            return False
        if self.readiness is None:
            return True
        if self.readiness.state == ReadinessState.READY:
            return True
        return self._readiness_degraded_by_consolidation()

    def _mark_consolidation_recovered(self) -> None:
        if self.readiness is not None and self._readiness_degraded_by_consolidation():
            self.readiness.ready("consolidation_recovered")

    def _mark_schedule_recovered(self) -> None:
        if self.readiness is not None and self._readiness_degraded_by_schedule():
            self.readiness.ready("schedule_processing_recovered")

    def _focus_mode_block_reason(self) -> Optional[str]:
        if self._focus_mode and getattr(self, "_focus_mode_since", None):
            elapsed = (self._time_source() - self._focus_mode_since).total_seconds()
            if elapsed > self.focus_mode_timeout_seconds:
                self._trace("focus_mode_auto_exited", {"elapsed_seconds": elapsed})
                self.exit_focus_mode()
        if self._focus_mode:
            return "focus_mode"
        return None

    def _schedule_processing_block_reason(self) -> Optional[str]:
        focus_reason = self._focus_mode_block_reason()
        if focus_reason is not None:
            return focus_reason
        if self._desktop_context_media_commentary_foreground_active():
            return "foreground_media_commentary"
        if self.readiness is None:
            return None
        if self.readiness.state in {ReadinessState.READY, ReadinessState.DEGRADED}:
            return None
        return f"readiness_{self.readiness.state.value}"

    @staticmethod
    def _schedule_result_has_activity(result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        return any(
            int(result.get(key) or 0) > 0
            for key in ("processed", "submitted", "recorded", "skipped", "failed")
        )

    def schedule_processing_status(self, due_now: Optional[int] = None) -> Dict[str, Any]:
        """Return operator-facing state for whether due schedule work can run."""

        block_reason = self._schedule_processing_block_reason()
        readiness_payload: Dict[str, Any] | None = None
        if self.readiness is not None:
            readiness_payload = self.readiness.snapshot()
        payload: Dict[str, Any] = {
            "scheduler_running": bool(self._running),
            "can_process": bool(self._running) and block_reason is None,
            "blocked_reason": None if self._running else "scheduler_not_running",
            "focus_mode": bool(self._focus_mode),
            "readiness": readiness_payload,
        }
        if self._running and block_reason is not None:
            payload["blocked_reason"] = block_reason
        if due_now is not None:
            payload["due_now"] = int(due_now)
            payload["will_process_due_now"] = bool(payload["can_process"] and int(due_now) > 0)
        return payload

    async def _cycle_loop(self) -> None:
        while self._running:
            # Pacing adjustment based on somatic fatigue/overload
            sleep_time = self.cycle_interval
            if hasattr(self.runtime, "executive") and self.runtime.executive.recommend_pause():
                sleep_time = self.cycle_interval * 2  # Pacing: back off when fatigued or overloaded
                self._trace("cycle_backoff", {"reason": "executive_recommended_pause", "sleep_time": sleep_time})

            await asyncio.sleep(sleep_time)

            if not self._running:
                break
            executive_paused = self._executive_pause_active()
            if self._last_executive_pause is None:
                self._last_executive_pause = executive_paused
            elif self._last_executive_pause and not executive_paused:
                await self._on_cycle_resume()
                self._last_executive_pause = executive_paused
            else:
                self._last_executive_pause = executive_paused
            can_run = self._should_run_cycle()
            if not can_run:
                continue
            block_reason = self._background_llm_block_reason(require_quiet_baa=True)
            if block_reason is not None:
                self._trace("cycle_skipped", {"reason": block_reason})
                continue

            # If still severely fatigued after sleep, skip the cycle entirely
            if executive_paused and getattr(getattr(self.runtime.executive, "somatic", None), "state", None) and self.runtime.executive.somatic.state.fatigue > 0.8:
                self._trace("cycle_skipped", {"reason": "severe_fatigue"})
                continue

            try:
                result = await self.runtime.run_cycle()
                self._trace("cycle_complete", result)
            except Exception as exc:
                self._trace("cycle_error", {"error": str(exc)})
                if self.readiness:
                    self.readiness.degraded(f"run_cycle failed: {exc}")

    async def _on_cycle_resume(self) -> None:
        """Trigger deferred work restoration when the agent recovers from pause."""
        self._trace("cycle_resumed", {})
        if hasattr(self.runtime, "executive") and self.runtime.executive:
            try:
                result = await self.runtime.executive.resume_deferred_work()
                self._trace("deferred_work_resumed", result)
            except Exception as exc:
                self._trace("deferred_work_resume_error", {"error": str(exc)})

    def _executive_pause_active(self) -> bool:
        executive = getattr(self.runtime, "executive", None)
        if executive is None:
            return False
        try:
            return bool(executive.recommend_pause())
        except Exception:
            return False

    def _baa_busy(self, *, include_backlog: bool = True) -> bool:
        baa = getattr(self.runtime, "baa", None)
        if baa is None:
            return False
        queue_size = int(getattr(baa, "queue_size", 0) or 0)
        held_size = int(getattr(baa, "held_size", 0) or 0)
        active_count = int(getattr(baa, "active_count", 0) or 0)
        if include_backlog:
            return (queue_size + held_size + active_count) > 0
        return active_count > 0

    def _consolidation_worker_active(self) -> bool:
        config = getattr(getattr(self.runtime, "ctx", None), "config", None)
        state_dir = getattr(config, "state_dir", None)
        if state_dir is None:
            return False
        status = load_consolidation_worker_status(state_dir)
        if str(status.get("status") or "").lower() != "running":
            return False
        pid = status.get("pid")
        if pid is None:
            return True
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except (PermissionError, OSError, TypeError, ValueError):
            return True
        return True

    def _recent_user_activity_active(self) -> bool:
        if self.conversation_quiet_seconds <= 0:
            return False
        last_user_turn_at = getattr(self.runtime, "_last_user_turn_at", None)
        if last_user_turn_at is None:
            return False
        if isinstance(last_user_turn_at, str):
            try:
                last_user_turn_at = datetime.fromisoformat(last_user_turn_at)
            except ValueError:
                return False
        if not isinstance(last_user_turn_at, datetime):
            return False
        if last_user_turn_at.tzinfo is None:
            last_user_turn_at = last_user_turn_at.replace(tzinfo=timezone.utc)
        elapsed = (self._time_source() - last_user_turn_at).total_seconds()
        return elapsed < self.conversation_quiet_seconds

    def _background_llm_block_reason(
        self,
        *,
        require_idle: bool = False,
        require_quiet_baa: bool = False,
        require_conversation_quiet: bool = False,
        allow_executive_pause: bool = False,
        baa_blocks_on_backlog: bool = True,
    ) -> Optional[str]:
        if self._executive_pause_active() and not allow_executive_pause:
            return "executive_recommended_pause"
        if self._consolidation_worker_active():
            return "runtime_activity_consolidating"
        if require_quiet_baa and self._baa_busy(include_backlog=baa_blocks_on_backlog):
            return "baa_busy"
        if require_conversation_quiet and self._recent_user_activity_active():
            return "recent_user_activity"
        if require_idle:
            activity = str(getattr(self.runtime, "_activity", "idle") or "idle")
            if activity != "idle":
                return f"runtime_activity_{activity}"
        return None

    def _desktop_context_media_commentary_foreground_active(self) -> bool:
        service = getattr(self.runtime, "desktop_context", None)
        config = getattr(service, "config", None)
        if config is None:
            return False
        if not bool(
            getattr(config, "enabled", False)
            and getattr(config, "media_commentary_mode_enabled", False)
            and getattr(config, "proactive_video_commentary_enabled", True)
        ):
            return False

        media_state = self._desktop_context_media_state(service)
        if not self._desktop_context_media_state_is_recent(media_state, config):
            return False
        items = media_state.get("items") if isinstance(media_state.get("items"), dict) else {}
        for item in items.values():
            if not isinstance(item, dict):
                continue
            if str(item.get("status") or "").strip().lower() == "playing":
                return True
        return False

    def _desktop_context_media_state(self, service: Any) -> Dict[str, Any]:
        loader = getattr(service, "_load_media_state", None)
        if callable(loader):
            try:
                payload = loader()
                if isinstance(payload, dict):
                    return payload
            except Exception:
                return {}
        payload = getattr(service, "media_state", None)
        return payload if isinstance(payload, dict) else {}

    def _desktop_context_media_state_is_recent(self, media_state: Dict[str, Any], config: Any) -> bool:
        observed_at = media_state.get("observed_at")
        if not observed_at:
            return False
        if isinstance(observed_at, str):
            try:
                observed_at = datetime.fromisoformat(observed_at)
            except ValueError:
                return False
        if not isinstance(observed_at, datetime):
            return False
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        interval = float(getattr(config, "media_state_poll_seconds", 2) or 2)
        freshness_seconds = max(120.0, min(600.0, interval * 10.0))
        return (self._time_source() - observed_at).total_seconds() <= freshness_seconds

    async def _due_schedule_work_pending(self) -> bool:
        service = getattr(self.runtime, "schedule_service", None)
        store = getattr(getattr(self.runtime, "ctx", None), "schedule_store", None) or getattr(
            service,
            "store",
            None,
        )
        list_due = getattr(store, "list_due", None)
        if not callable(list_due):
            return False
        try:
            due = list_due(self._time_source(), limit=1)
            if inspect.isawaitable(due):
                due = await due
        except Exception as exc:
            self._trace("schedule_pending_check_error", {"error": str(exc)})
            return False
        return bool(due)

    async def _body_double_observation_block_reason(self) -> Optional[str]:
        if not self._should_run_cycle():
            return "scheduler_not_runnable"
        media_commentary_foreground = self._desktop_context_media_commentary_foreground_active()
        if not media_commentary_foreground and await self._due_schedule_work_pending():
            return "scheduled_work_pending"
        return self._background_llm_block_reason(
            require_quiet_baa=not media_commentary_foreground,
            require_conversation_quiet=not media_commentary_foreground,
            allow_executive_pause=True,
            baa_blocks_on_backlog=not media_commentary_foreground,
        )

    def _consolidation_result_requires_retry(self, result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        if self._consolidation_result_was_foreground_deferred(result):
            return False
        failure_reasons = {
            "worker_timeout",
            "worker_start_failed",
            "worker_failed",
            "worker_no_result",
        }
        reason = str(result.get("budget_reason") or "").lower()
        if reason in failure_reasons:
            return True
        worker = result.get("worker")
        if isinstance(worker, dict):
            worker_status = str(worker.get("status") or "").lower()
            if worker_status in {
                "timeout_killed",
                "start_failed",
                "failed",
                "error",
                "no_result",
                "unreadable",
                "cancelled",
            }:
                return True
        return False

    @staticmethod
    def _consolidation_result_was_foreground_deferred(result: Any) -> bool:
        if not isinstance(result, dict):
            return False
        return str(result.get("budget_reason") or "").lower() in {
            "foreground_user_turn",
            "foreground_interrupted",
        }

    async def _run_consolidation_with_retries(self, retry_delay: float) -> Dict[str, Any]:
        attempts = 0
        result: Dict[str, Any] = {}
        base_delay = (
            retry_delay
            if self.consolidation_retry_base_seconds is None
            else max(0.0, float(self.consolidation_retry_base_seconds))
        )
        while attempts < self.consolidation_retry_attempts:
            attempts += 1
            result = await self.runtime.run_consolidation(budget=self.consolidation_budget)
            if not self._consolidation_result_requires_retry(result):
                return result
            worker = result.get("worker")
            self._trace(
                "consolidation_retry_scheduled",
                {
                    "attempt": attempts,
                    "max_attempts": self.consolidation_retry_attempts,
                    "reason": result.get("budget_reason"),
                    "worker_status": worker.get("status") if isinstance(worker, dict) else None,
                },
            )
            if attempts >= self.consolidation_retry_attempts:
                break
            await asyncio.sleep(base_delay * (2 ** (attempts - 1)))
        return result

    def _persist_failed_consolidation_state(
        self,
        runtime_state_dir: Path,
        result: Dict[str, Any],
    ) -> None:
        """Remember failed consolidation attempts so the scheduler learns from them."""
        now = self._time_source().astimezone(timezone.utc)
        prior = load_consolidation_runtime_state(runtime_state_dir)
        prior_result_id = str(prior.get("last_result_id", "") or "").strip().lower()
        prior_reason = str(prior.get("budget_reason", "") or "").strip().lower()
        reason = str(result.get("budget_reason") or "").strip().lower()
        failed_prefixes = (
            "worker-timeout-",
            "worker-start-failed-",
            "worker-failed-",
            "worker-no-result-",
        )
        if prior_result_id.startswith(failed_prefixes) and prior_reason == reason:
            try:
                consecutive_failures = int(prior.get("consecutive_failures") or 0) + 1
            except (TypeError, ValueError):
                consecutive_failures = 1
        else:
            consecutive_failures = 1
        retry_seconds = self.consolidation_failure_retry_seconds * (
            2 ** max(0, consecutive_failures - 1)
        )
        retry_seconds = min(self.consolidation_failure_max_retry_seconds, retry_seconds)
        next_retry_after = now + timedelta(seconds=retry_seconds)
        worker = result.get("worker")
        worker = worker if isinstance(worker, dict) else {}
        payload = {
            "last_attempt_at": now.isoformat(),
            "last_result_id": result.get("result_id"),
            "budget_exhausted": result.get("budget_exhausted"),
            "budget_reason": result.get("budget_reason"),
            "consecutive_failures": consecutive_failures,
            "next_retry_after": next_retry_after.isoformat(),
            "worker_status": worker.get("status"),
            "worker_error_type": worker.get("error_type"),
        }
        last_run_at = prior.get("last_run_at")
        if last_run_at:
            payload["last_run_at"] = last_run_at
        persist_consolidation_runtime_state(runtime_state_dir, payload)

    async def _consolidation_loop(self) -> None:
        retry_delay = max(5.0, min(float(self.schedule_interval), 300.0))
        state_dir = getattr(getattr(self.runtime, "ctx", None), "config", None)
        runtime_state_dir = getattr(state_dir, "state_dir", None)
        while self._running:
            delay = 0.0
            if runtime_state_dir is not None:
                delay = consolidation_delay_until_due(
                    runtime_state_dir,
                    self.consolidation_interval,
                    now=self._time_source(),
                )
            else:
                delay = float(self.consolidation_interval)
            if delay > 0:
                await asyncio.sleep(delay)
            if not self._running:
                break
            if not self._should_run_consolidation():
                await asyncio.sleep(retry_delay)
                continue
            block_reason = self._background_llm_block_reason(
                require_idle=True,
                require_quiet_baa=True,
                require_conversation_quiet=True,
            )
            if block_reason is not None:
                self._trace("consolidation_skipped", {"reason": block_reason})
                await asyncio.sleep(retry_delay)
                continue
            try:
                result = await self._run_consolidation_with_retries(retry_delay)
                if self._consolidation_result_was_foreground_deferred(result):
                    self._trace("consolidation_deferred", result)
                    if self._running:
                        await asyncio.sleep(retry_delay)
                    continue
                result_failed = self._consolidation_result_requires_retry(result)
                if runtime_state_dir is not None and not result_failed:
                    persist_consolidation_runtime_state(
                        runtime_state_dir,
                        consolidation_runtime_state_payload(
                            result if isinstance(result, dict) else {},
                            fallback_timestamp=self._time_source().isoformat(),
                        ),
                    )
                if isinstance(result, dict) and result.get("budget_exhausted"):
                    self._trace(
                        "consolidation_budget_exhausted",
                        {
                            "reason": result.get("budget_reason"),
                            "budget": result.get("budget"),
                            "llm_calls_used": result.get("llm_calls_used"),
                        },
                    )
                if result_failed:
                    if runtime_state_dir is not None:
                        self._persist_failed_consolidation_state(runtime_state_dir, result)
                    self._trace("consolidation_failed", result)
                    if self.readiness:
                        reason = result.get("budget_reason") if isinstance(result, dict) else "unknown"
                        self.readiness.degraded(f"consolidation failed: {reason}")
                    if self._running:
                        await asyncio.sleep(retry_delay)
                    continue
                self._mark_consolidation_recovered()
                self._trace("consolidation_complete", result)
            except Exception as exc:
                self._trace("consolidation_error", {"error": str(exc)})
                if self.readiness:
                    self.readiness.degraded(f"run_consolidation failed: {exc}")
                if self._running:
                    await asyncio.sleep(retry_delay)

    async def _baa_heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.baa_heartbeat_interval)
            if not self._running:
                break
            # Natural somatic decay/recovery every heartbeat tick
            somatic = getattr(self.runtime.ctx, "somatic", None)
            if somatic is not None:
                try:
                    somatic.decay()
                except Exception:
                    pass
            try:
                queue_size = self.runtime.baa.queue_size
                held_size = self.runtime.baa.held_size
                released = 0
                release_held = getattr(self.runtime.baa, "try_release_held", None)
                if held_size and callable(release_held):
                    release_result = release_held()
                    if inspect.isawaitable(release_result):
                        release_result = await release_result
                    released = int(release_result or 0)
                self._trace(
                    "baa_heartbeat",
                    {
                        "queue_size": queue_size,
                        "held_size": held_size,
                        "released_held": released,
                        "lane_queue_depth": queue_size,
                    },
                )
            except Exception:
                pass

    async def _daydream_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.daydream_interval)
            if not self._running:
                break
            if not self._should_run_cycle():
                continue
            executive_paused = self._executive_pause_active()
            block_reason = self._background_llm_block_reason(
                require_idle=True,
                require_quiet_baa=True,
                require_conversation_quiet=True,
                allow_executive_pause=True,
                baa_blocks_on_backlog=False,
            )
            if block_reason is not None:
                self._trace("daydream_skipped", {"reason": block_reason})
                continue
            try:
                result = await self.runtime.run_daydream(
                    force=True,
                    reflective_only=executive_paused,
                )
                self._trace("daydream_complete", result)
            except Exception as exc:
                self._trace("daydream_error", {"error": str(exc)})
                if self.readiness:
                    self.readiness.degraded(f"run_daydream failed: {exc}")

    def _nightly_dream_zone(self) -> timezone | ZoneInfo:
        try:
            return ZoneInfo(self.nightly_dream_timezone)
        except ZoneInfoNotFoundError:
            return timezone.utc

    def _nightly_dream_target_date(self) -> date:
        zone = self._nightly_dream_zone()
        local_now = self._time_source().astimezone(zone)
        target = local_now.date()
        if local_now.hour < self.nightly_dream_local_hour:
            target = target - timedelta(days=1)
        return target

    async def _covered_nightly_dream_dates(self) -> set[str]:
        store = getattr(self.runtime, "dream_store", None)
        if store is None or not hasattr(store, "list_recent"):
            return set()
        try:
            recent = await store.list_recent(limit=400)
        except Exception:
            return set()
        zone = self._nightly_dream_zone()
        covered: set[str] = set()
        for record in recent:
            meta = getattr(record, "meta", {}) or {}
            dream_for_date = str(meta.get("dream_for_date") or "").strip()
            if dream_for_date:
                covered.add(dream_for_date[:10])
                continue
            created_at = getattr(record, "created_at", None)
            if isinstance(created_at, datetime):
                covered.add(created_at.astimezone(zone).date().isoformat())
        return covered

    def _due_nightly_dream_dates(self, covered: set[str]) -> list[str]:
        target = self._nightly_dream_target_date()
        parsed = []
        for value in covered:
            try:
                parsed.append(datetime.fromisoformat(value).date())
            except ValueError:
                continue
        if not parsed:
            return [target.isoformat()]
        latest = max(parsed)
        if latest >= target:
            return []
        due = []
        current = latest + timedelta(days=1)
        while current <= target:
            due.append(current.isoformat())
            current = current + timedelta(days=1)
        return due

    async def _run_due_nightly_dreams(self) -> dict[str, Any]:
        runner = getattr(self.runtime, "run_nightly_dream", None)
        if not callable(runner):
            return {"available": False, "reason": "nightly_dreaming_unavailable", "ran": 0}
        covered = await self._covered_nightly_dream_dates()
        due_dates = self._due_nightly_dream_dates(covered)
        results: list[dict[str, Any]] = []
        for dream_date in due_dates:
            source = dict(getattr(self.runtime, "_last_consolidation_result", None) or {})
            base = dict(source)
            source_result_id = str(source.get("result_id") or "").strip()
            base.update(
                {
                    "result_id": f"required-nightly-dream:{dream_date}",
                    "source_consolidation_result_id": source_result_id,
                    "timestamp": self._time_source().astimezone(timezone.utc).isoformat(),
                    "dream_for_date": dream_date,
                    "required_nightly_dream": True,
                    "scheduler_trigger": "required_nightly_dream",
                    "reason": "nightly dreaming is required continuity and consolidation synthesis",
                }
            )
            try:
                result = runner(mode=self.nightly_dream_mode, consolidation_result=base)
                if inspect.isawaitable(result):
                    result = await result
                payload = dict(result or {})
                payload["dream_for_date"] = dream_date
                results.append(payload)
                self._trace("required_nightly_dream_complete", payload)
            except Exception as exc:
                self._trace(
                    "required_nightly_dream_error",
                    {"dream_for_date": dream_date, "error": str(exc)},
                )
        return {
            "available": True,
            "ran": len(results),
            "due_dates": due_dates,
            "results": results,
        }

    async def _nightly_dream_loop(self) -> None:
        while self._running:
            try:
                result = await self._run_due_nightly_dreams()
                if result.get("ran"):
                    self._trace("required_nightly_dream_sweep", result)
            except Exception as exc:
                self._trace("required_nightly_dream_sweep_error", {"error": str(exc)})
            await asyncio.sleep(self.nightly_dream_check_interval_seconds)

    async def _schedule_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.schedule_interval)
            if not self._running:
                break
            service = getattr(self.runtime, "schedule_service", None)
            if service is None:
                continue
            block_reason = self._schedule_processing_block_reason()
            if block_reason is not None:
                if await self._due_schedule_work_pending():
                    self._trace("schedule_skipped", {"reason": block_reason, "due_work_pending": True})
                continue
            try:
                result = await service.process_due()
                if self._schedule_result_has_activity(result):
                    self._trace("schedule_complete", result)
                if isinstance(result, dict) and int(result.get("failed") or 0) == 0:
                    self._mark_schedule_recovered()
            except Exception as exc:
                self._trace("schedule_error", {"error": str(exc)})
                if self.readiness:
                    self.readiness.degraded(f"schedule processing failed: {exc}")

    async def _initiative_contact_loop(self) -> None:
        while self._running:
            sleep_time = float(self.schedule_interval)
            if sleep_time >= 1.0 and self.initiative_contact_jitter_seconds:
                sleep_time += random.uniform(0, self.initiative_contact_jitter_seconds)
            await asyncio.sleep(sleep_time)
            if not self._running:
                break
            if not self._should_run_cycle():
                continue
            runner = getattr(self.runtime, "maybe_run_initiative_contact", None)
            if not callable(runner):
                continue
            try:
                result = await runner()
                if isinstance(result, dict) and result.get("status") == "sent":
                    self._trace("initiative_contact_sent", result)
            except Exception as exc:
                self._trace("initiative_contact_error", {"error": str(exc)})

    async def _desktop_context_loop(self) -> None:
        while self._running:
            await asyncio.sleep(float(self.schedule_interval))
            if not self._running:
                break
            block_reason = await self._body_double_observation_block_reason()
            if block_reason is not None:
                self._trace("desktop_context_skipped", {"reason": block_reason})
                continue
            runner = getattr(self.runtime, "maybe_run_desktop_context", None)
            if not callable(runner):
                continue
            try:
                result = runner()
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, dict) and result.get("status") == "observed":
                    self._trace("desktop_context_observed", result)
            except Exception as exc:
                self._trace("desktop_context_error", {"error": str(exc)})

    async def _desktop_media_state_loop(self) -> None:
        while self._running:
            service = getattr(self.runtime, "desktop_context", None)
            config = getattr(service, "config", None)
            interval = float(getattr(config, "media_state_poll_seconds", 2) or 2)
            await asyncio.sleep(max(1.0, min(interval, 30.0)))
            if not self._running:
                break
            service = getattr(self.runtime, "desktop_context", None)
            poll = getattr(service, "poll_media_state_once", None)
            if not callable(poll):
                continue
            try:
                result = poll()
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, dict) and result.get("status") == "changed":
                    self._trace("desktop_media_state_changed", result)
                    if result.get("commentary_observation_requested"):
                        block_reason = await self._body_double_observation_block_reason()
                        if block_reason is not None:
                            self._trace("desktop_media_commentary_skipped", {"reason": block_reason})
                            continue
                        observe_once = getattr(service, "observe_once", None)
                        if callable(observe_once):
                            observation = observe_once(
                                force=True,
                                reason=str(
                                    result.get("commentary_observation_reason")
                                    or "media_commentary_mode:media_state_changed"
                                ),
                            )
                            if inspect.isawaitable(observation):
                                observation = await observation
                            if isinstance(observation, dict):
                                self._trace("desktop_media_commentary_observed", observation)
            except Exception as exc:
                self._trace("desktop_media_state_error", {"error": str(exc)})

    async def _wellbeing_maintenance_loop(self) -> None:
        while self._running:
            interval = self.wellbeing_interval or float(self.schedule_interval)
            await asyncio.sleep(interval)
            if not self._running:
                break
            if not self._should_run_cycle():
                continue
            block_reason = self._background_llm_block_reason(
                require_idle=True,
                require_quiet_baa=True,
                require_conversation_quiet=True,
                allow_executive_pause=True,
                baa_blocks_on_backlog=False,
            )
            if block_reason is not None:
                self._trace("wellbeing_maintenance_skipped", {"reason": block_reason})
                continue
            runner = getattr(self.runtime, "run_wellbeing_maintenance", None)
            if not callable(runner):
                continue
            try:
                result = runner()
                if inspect.isawaitable(result):
                    result = await result
                payload = result if isinstance(result, dict) else {"result": str(result)}
                self._trace("wellbeing_maintenance_complete", payload)
            except Exception as exc:
                self._trace("wellbeing_maintenance_error", {"error": str(exc)})
                if self.readiness:
                    self.readiness.degraded(f"wellbeing maintenance failed: {exc}")

    async def _identity_heartbeat_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.identity_heartbeat_interval)
            if not self._running:
                break
            identity = getattr(getattr(self.runtime, "ctx", None), "identity", None)
            heartbeat = getattr(identity, "record_persistence_heartbeat", None)
            if not callable(heartbeat):
                continue
            try:
                heartbeat()
                self._trace("identity_persistence_heartbeat", {})
            except Exception as exc:
                self._trace("identity_persistence_heartbeat_error", {"error": str(exc)})

    async def _cognitive_maintenance_loop(self) -> None:
        while self._running:
            interval = self.wellbeing_interval or float(self.schedule_interval)
            await asyncio.sleep(interval)
            if not self._running:
                break
            if not self._should_run_cycle():
                continue
            block_reason = self._background_llm_block_reason(
                require_quiet_baa=True,
                require_conversation_quiet=True,
                allow_executive_pause=True,
                baa_blocks_on_backlog=False,
            )
            if block_reason is not None:
                self._trace("cognitive_maintenance_skipped", {"reason": block_reason})
                continue
            runner = getattr(self.runtime, "run_cognitive_maintenance", None)
            if not callable(runner):
                continue
            try:
                result = runner()
                if inspect.isawaitable(result):
                    result = await result
                payload = result if isinstance(result, dict) else {"result": str(result)}
                self._trace("cognitive_maintenance_complete", payload)
                shadow_runner = getattr(self.runtime, "run_shadow_registry_consumer", None)
                if callable(shadow_runner):
                    shadow_result = shadow_runner(max_dismissals=20)
                    if inspect.isawaitable(shadow_result):
                        shadow_result = await shadow_result
                    shadow_payload = (
                        shadow_result
                        if isinstance(shadow_result, dict)
                        else {"result": str(shadow_result)}
                    )
                    self._trace("shadow_registry_consumer_complete", shadow_payload)
            except Exception as exc:
                self._trace("cognitive_maintenance_error", {"error": str(exc)})
                if self.readiness:
                    self.readiness.degraded(f"cognitive maintenance failed: {exc}")

    async def _compaction_backlog_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.compaction_backlog_interval_seconds)
            if not self._running:
                break
            if not self._should_run_cycle():
                continue
            block_reason = self._background_llm_block_reason(
                require_idle=True,
                require_quiet_baa=True,
                require_conversation_quiet=True,
                allow_executive_pause=True,
                baa_blocks_on_backlog=False,
            )
            if block_reason is not None:
                self._trace("compaction_backlog_skipped", {"reason": block_reason})
                continue
            runner = getattr(self.runtime, "run_compaction_backlog_maintenance", None)
            if not callable(runner):
                continue
            try:
                result = runner(
                    max_sessions=int(self.consolidation_budget.get("max_compaction_sessions", 8)),
                    min_session_lag=int(
                        self.consolidation_budget.get("min_compaction_session_lag", 20)
                    ),
                    tail_size=int(self.consolidation_budget.get("compaction_tail_size", 10)),
                    max_candidates=int(
                        self.consolidation_budget.get("max_compaction_candidates", 1000)
                    ),
                )
                if inspect.isawaitable(result):
                    result = await result
                payload = result if isinstance(result, dict) else {"result": str(result)}
                if payload.get("available"):
                    self._trace("compaction_backlog_complete", payload)
            except Exception as exc:
                self._trace("compaction_backlog_error", {"error": str(exc)})
                if self.readiness:
                    self.readiness.degraded(f"compaction backlog maintenance failed: {exc}")

    async def _telemetry_prune_loop(self) -> None:
        last_prune = 0.0
        while self._running:
            await asyncio.sleep(self.schedule_interval)
            if not self._running:
                break
            now = self._time_source()
            now_ts = now.timestamp()
            if now_ts - last_prune < float(self.telemetry_prune_interval_seconds):
                continue
            await self._run_telemetry_prune()
            last_prune = now_ts

    async def _run_telemetry_prune(self) -> None:
        if not self._running:
            return
        tracer = self.tracer or getattr(self.runtime, "tracer", None)
        if tracer is None or getattr(self.runtime, "ctx", None) is None:
            return
        runtime_ctx = self.runtime.ctx
        token_telemetry = getattr(runtime_ctx, "token_telemetry", None)
        if token_telemetry is None:
            return

        try:
            events_file_removed = 0
            daily_files_removed = 0
            if hasattr(token_telemetry, "flush"):
                await token_telemetry.flush()
            daily_files_removed = tracer.store.prune_old_files(
                self.telemetry_retention_days,
                now=self._time_source(),
            )
            events_file_removed = await token_telemetry.prune_old_events(
                self.telemetry_retention_days,
            )
            self._trace(
                "telemetry_pruned",
                {
                    "retention_days": self.telemetry_retention_days,
                    "removed_daily_files": daily_files_removed,
                    "removed_token_events": events_file_removed,
                },
            )
        except Exception as exc:
            self._trace("telemetry_prune_failed", {"error": str(exc)})

    async def _workspace_sync_loop(self) -> None:
        """Periodically sweep reflective workspace dirs into searchable memory.

        Safety net for when a writer forgets to call the bridge inline. Without it,
        any artifact written outside ``SelfWorkspaceService`` (legacy daydream-lab
        paths, manual drops, future writers) stays invisible to retrieval.
        """
        last_sync = 0.0
        while self._running:
            await asyncio.sleep(self.schedule_interval)
            if not self._running:
                break
            now_ts = self._time_source().timestamp()
            if now_ts - last_sync < float(self.workspace_sync_interval_seconds):
                continue
            await self._run_workspace_sync()
            last_sync = now_ts

    async def _run_workspace_sync(self) -> None:
        if not self._running:
            return
        block_reason = self._background_llm_block_reason(
            require_idle=True,
            require_quiet_baa=True,
            require_conversation_quiet=True,
        )
        if block_reason is not None:
            self._trace("workspace_sync_skipped", {"reason": block_reason})
            return
        ctx = getattr(self.runtime, "ctx", None)
        bridge = getattr(ctx, "artifact_bridge", None)
        if bridge is None or ctx is None:
            return
        config = getattr(ctx, "config", None)
        workspace_root = config.agent_workspace_root() if config is not None else None
        if workspace_root is None:
            return
        targets = [
            workspace_root / "self",
            workspace_root / "reflections",
            workspace_root / "daydream-lab",
        ]
        totals: Dict[str, int] = {
            "artifacts": 0,
            "episodes_created": 0,
            "episodes_updated": 0,
            "episodes_deleted": 0,
            "memories_upserted": 0,
        }
        for target in targets:
            if not target.exists():
                continue
            try:
                result = await bridge.sync_directory(target)
            except Exception as exc:
                self._trace("workspace_sync_failed", {"path": str(target), "error": str(exc)})
                continue
            for key in totals:
                totals[key] += int(result.get(key, 0) or 0)
        self._trace("workspace_sync_complete", totals)

    async def _recovery_loop(self) -> None:
        while self._running:
            await asyncio.sleep(300)
            await self._run_recovery_once()

    async def _run_recovery_once(self) -> None:
        coordinator = getattr(self.runtime, "recovery_coordinator", None)
        if coordinator is None:
            return
        if not self._should_run_cycle():
            return
        if self._background_llm_block_reason(require_quiet_baa=True) is not None:
            return
        try:
            result = await coordinator.run_once(limit_per_store=100, max_actions=5)
            self._trace("recovery_complete", getattr(result, "__dict__", {"result": str(result)}))
        except Exception as exc:
            self._trace("recovery_error", {"error": str(exc)})

    async def _run_recovery_once_for_test(self) -> None:
        await self._run_recovery_once()

    async def _cron_loop(self) -> None:
        await asyncio.gather(
            self._daydream_loop(),
            self._schedule_loop(),
            self._initiative_contact_loop(),
            self._desktop_context_loop(),
            self._desktop_media_state_loop(),
            self._wellbeing_maintenance_loop(),
            self._identity_heartbeat_loop(),
            self._cognitive_maintenance_loop(),
            self._compaction_backlog_loop(),
            self._nightly_dream_loop(),
            self._telemetry_prune_loop(),
            self._workspace_sync_loop(),
        )

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"AgentScheduler: {event}",
                payload,
            )
