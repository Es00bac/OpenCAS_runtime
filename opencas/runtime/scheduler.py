"""Background scheduler for OpenCAS autonomous loops."""

from __future__ import annotations

import asyncio
import inspect
import random
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from opencas.execution.lanes import CommandLane, LaneConfig, LaneManager
from opencas.runtime.consolidation_state import (
    consolidation_delay_until_due,
    persist_consolidation_runtime_state,
)
from opencas.runtime.readiness import AgentReadiness, ReadinessState
from opencas.telemetry import EventKind, Tracer


class AgentScheduler:
    """Drives runtime cycles, consolidation, and BAA heartbeat on intervals."""

    def __init__(
        self,
        runtime: Any,
        cycle_interval: int = 600,
        consolidation_interval: int = 86400,
        baa_heartbeat_interval: int = 120,
        daydream_interval: int = 720,
        schedule_interval: int = 60,
        readiness: Optional[AgentReadiness] = None,
        tracer: Optional[Tracer] = None,
        lane_manager: Optional[LaneManager] = None,
        focus_mode_timeout_seconds: int = 60,
        time_source: Optional[Callable[[], datetime]] = None,
        conversation_quiet_seconds: int = 90,
        consolidation_budget: Optional[Dict[str, Any]] = None,
        consolidation_retry_attempts: int = 3,
        consolidation_retry_base_seconds: Optional[float] = None,
        initiative_contact_jitter_seconds: int = 180,
        telemetry_retention_days: int = 30,
        telemetry_prune_interval_seconds: int = 86400,
    ) -> None:
        self.runtime = runtime
        self.cycle_interval = cycle_interval
        self.consolidation_interval = consolidation_interval
        self.baa_heartbeat_interval = baa_heartbeat_interval
        self.daydream_interval = daydream_interval
        self.schedule_interval = schedule_interval
        self.readiness = readiness
        self.tracer = tracer
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        # Focus mode: suspend daydream and cycle loops during deep tool-use work.
        self._focus_mode: bool = False
        self._focus_mode_since: Optional[datetime] = None
        self.focus_mode_timeout_seconds = focus_mode_timeout_seconds
        self._time_source = time_source or (lambda: datetime.now(timezone.utc))
        self.conversation_quiet_seconds = conversation_quiet_seconds
        self.initiative_contact_jitter_seconds = max(0, int(initiative_contact_jitter_seconds))
        self.telemetry_retention_days = max(1, int(telemetry_retention_days))
        self.telemetry_prune_interval_seconds = max(60, int(telemetry_prune_interval_seconds))
        self.consolidation_retry_attempts = max(1, int(consolidation_retry_attempts))
        self.consolidation_retry_base_seconds = consolidation_retry_base_seconds
        self.consolidation_budget = consolidation_budget or {
            "max_seconds": 120,
            "worker_timeout_seconds": 300,
            "max_llm_calls": 12,
            "max_cluster_summaries": 6,
            "max_candidates": 100,
            "max_prompt_chars": 12000,
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

    async def stop(self) -> None:
        """Cancel background loops and drain BAA."""
        if not self._running:
            return
        self._running = False
        self._trace("scheduler_stop", {})

        for task in self._tasks:
            task.cancel()
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

    def _baa_busy(self) -> bool:
        baa = getattr(self.runtime, "baa", None)
        if baa is None:
            return False
        queue_size = int(getattr(baa, "queue_size", 0) or 0)
        held_size = int(getattr(baa, "held_size", 0) or 0)
        active_count = int(getattr(baa, "active_count", 0) or 0)
        return (queue_size + held_size + active_count) > 0

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
    ) -> Optional[str]:
        if self._executive_pause_active():
            return "executive_recommended_pause"
        if require_quiet_baa and self._baa_busy():
            return "baa_busy"
        if require_conversation_quiet and self._recent_user_activity_active():
            return "recent_user_activity"
        if require_idle:
            activity = str(getattr(self.runtime, "_activity", "idle") or "idle")
            if activity != "idle":
                return f"runtime_activity_{activity}"
        return None

    def _consolidation_result_requires_retry(self, result: Any) -> bool:
        if not isinstance(result, dict):
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
            if not self._should_run_cycle():
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
                result_failed = self._consolidation_result_requires_retry(result)
                if runtime_state_dir is not None and not result_failed:
                    timestamp = None
                    if isinstance(result, dict):
                        timestamp = result.get("timestamp")
                    persist_consolidation_runtime_state(
                        runtime_state_dir,
                        {
                            "last_run_at": str(timestamp or self._time_source().isoformat()),
                            "last_result_id": result.get("result_id") if isinstance(result, dict) else None,
                        },
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
                    self._trace("consolidation_failed", result)
                    if self.readiness:
                        reason = result.get("budget_reason") if isinstance(result, dict) else "unknown"
                        self.readiness.degraded(f"consolidation failed: {reason}")
                    if self._running:
                        await asyncio.sleep(retry_delay)
                    continue
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
                self._trace(
                    "baa_heartbeat",
                    {"queue_size": queue_size, "held_size": held_size, "lane_queue_depth": queue_size},
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
            block_reason = self._background_llm_block_reason(
                require_idle=True,
                require_quiet_baa=True,
                require_conversation_quiet=True,
            )
            if block_reason is not None:
                self._trace("daydream_skipped", {"reason": block_reason})
                continue
            try:
                result = await self.runtime.run_daydream()
                self._trace("daydream_complete", result)
            except Exception as exc:
                self._trace("daydream_error", {"error": str(exc)})
                if self.readiness:
                    self.readiness.degraded(f"run_daydream failed: {exc}")

    async def _schedule_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.schedule_interval)
            if not self._running:
                break
            if not self._should_run_cycle():
                continue
            service = getattr(self.runtime, "schedule_service", None)
            if service is None:
                continue
            try:
                result = await service.process_due()
                if result.get("processed"):
                    self._trace("schedule_complete", result)
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
            if not self._should_run_cycle():
                continue
            block_reason = self._background_llm_block_reason(
                require_quiet_baa=True,
                require_conversation_quiet=True,
            )
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

    async def _cron_loop(self) -> None:
        await asyncio.gather(
            self._daydream_loop(),
            self._schedule_loop(),
            self._initiative_contact_loop(),
            self._desktop_context_loop(),
            self._telemetry_prune_loop(),
        )

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"AgentScheduler: {event}",
                payload,
            )
