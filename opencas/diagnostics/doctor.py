"""Doctor diagnostic runner for OpenCAS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from opencas.bootstrap import BootstrapContext

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
        report.checks.append(await self.check_memory_integrity())
        report.checks.append(await self.check_embedding_index())
        report.checks.append(await self.check_continuity())
        report.checks.append(await self.check_identity_persistence())
        report.checks.append(await self.check_somatic_state())
        report.checks.append(await self.check_telemetry_writable())
        report.checks.append(await self.check_baa_queue_depth())
        report.checks.append(await self.check_embedding_latency())
        report.checks.append(await self.check_compaction_lag())

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

    async def check_embedding_index(self) -> DiagnosticCheck:
        if self.context is None:
            return DiagnosticCheck(
                name="embedding_index",
                status=CheckStatus.SKIP,
                message="No bootstrap context",
            )
        try:
            health = await self.context.embeddings.health()
            status = CheckStatus.PASS if health.total_records >= 0 else CheckStatus.WARN
            return DiagnosticCheck(
                name="embedding_index",
                status=status,
                message="Embedding service healthy",
                details={
                    "total_records": health.total_records,
                    "model_id": self.context.embeddings.model_id,
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
            if elapsed_ms > 5000:
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
            lag = len(episodes)
            status = CheckStatus.PASS
            if lag > 500:
                status = CheckStatus.WARN
            if lag > 1000:
                status = CheckStatus.FAIL
            return DiagnosticCheck(
                name="compaction_lag",
                status=status,
                message=f"Non-compacted episodes: {lag}",
                details={"lag": lag},
            )
        except Exception as exc:
            return DiagnosticCheck(
                name="compaction_lag",
                status=CheckStatus.FAIL,
                message=f"Compaction lag query failed: {exc}",
            )
