"""Continuous health monitor for OpenCAS."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from opencas.infra import EventBus, HealthCheckEvent
from opencas.telemetry import EventKind, Tracer

from .doctor import Doctor
from .models import CheckStatus

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Runs a subset of diagnostic checks on a timer and emits events."""

    def __init__(
        self,
        doctor: Doctor,
        event_bus: Optional[EventBus] = None,
        interval_seconds: float = 60.0,
        tracer: Optional[Tracer] = None,
    ) -> None:
        self.doctor = doctor
        self.event_bus = event_bus
        self.interval_seconds = interval_seconds
        self.tracer = tracer
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False

    def start(self) -> None:
        """Begin the periodic health check loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Signal the monitor to stop and wait for the loop to finish."""
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._run_once()
            except Exception:
                logger.exception("HealthMonitor sweep failed")
            try:
                await asyncio.wait_for(
                    asyncio.sleep(self.interval_seconds),
                    timeout=self.interval_seconds + 5.0,
                )
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break

    async def _run_once(self) -> None:
        """Execute the lightweight subset of checks and optionally emit an event."""
        report = await self.doctor.run_all()
        failures = sum(1 for c in report.checks if c.status == CheckStatus.FAIL)
        warnings = sum(1 for c in report.checks if c.status == CheckStatus.WARN)
        checks = [
            {
                "name": c.name,
                "status": c.status.value,
                "message": c.message,
            }
            for c in report.checks
        ]
        event = HealthCheckEvent(
            overall=report.overall.value,
            failures=failures,
            warnings=warnings,
            checks=checks,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        if self.tracer:
            self.tracer.log(
                EventKind.DIAGNOSTIC_RUN,
                "HealthMonitor sweep",
                {
                    "health_check_failures": failures,
                    "health_check_warnings": warnings,
                    "overall": report.overall.value,
                },
            )
        if self.event_bus:
            await self.event_bus.emit(event)
