"""Background scheduler for OpenCAS autonomous loops."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from opencas.execution.lanes import CommandLane, LaneConfig, LaneManager
from opencas.runtime.readiness import AgentReadiness, ReadinessState
from opencas.telemetry import EventKind, Tracer


class AgentScheduler:
    """Drives runtime cycles, consolidation, and BAA heartbeat on intervals."""

    def __init__(
        self,
        runtime: Any,
        cycle_interval: int = 300,
        consolidation_interval: int = 86400,
        baa_heartbeat_interval: int = 60,
        daydream_interval: int = 300,
        schedule_interval: int = 60,
        readiness: Optional[AgentReadiness] = None,
        tracer: Optional[Tracer] = None,
        lane_manager: Optional[LaneManager] = None,
        focus_mode_timeout_seconds: int = 60,
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

    async def _consolidation_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.consolidation_interval)
            if not self._running:
                break
            if not self._should_run_cycle():
                continue
            try:
                result = await self.runtime.run_consolidation()
                self._trace("consolidation_complete", result)
            except Exception as exc:
                self._trace("consolidation_error", {"error": str(exc)})
                if self.readiness:
                    self.readiness.degraded(f"run_consolidation failed: {exc}")

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

    async def _cron_loop(self) -> None:
        await asyncio.gather(self._daydream_loop(), self._schedule_loop())

    def _trace(self, event: str, payload: Dict[str, Any]) -> None:
        if self.tracer:
            self.tracer.log(
                EventKind.TOOL_CALL,
                f"AgentScheduler: {event}",
                payload,
            )
