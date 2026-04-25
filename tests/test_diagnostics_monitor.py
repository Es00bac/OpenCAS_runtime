"""Tests for HealthMonitor and runtime guard."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opencas.bootstrap.pipeline import BootstrapPipeline
from opencas.diagnostics import Doctor, HealthMonitor
from opencas.diagnostics.models import CheckStatus, DiagnosticCheck, HealthReport
from opencas.infra import EventBus, HealthCheckEvent


class TestHealthMonitor:
    @pytest.mark.asyncio
    async def test_runs_doctor_and_emits_event(self):
        doctor = Doctor()
        doctor.run_all = AsyncMock(
            return_value=HealthReport(
                overall=CheckStatus.PASS,
                checks=[
                    DiagnosticCheck(
                        name="test_check", status=CheckStatus.PASS, message="ok"
                    )
                ],
            )
        )
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(HealthCheckEvent, handler)
        monitor = HealthMonitor(doctor, event_bus=bus, interval_seconds=0.01)
        monitor.start()
        await asyncio.sleep(0.05)
        await monitor.stop()

        assert len(received) >= 1
        event = received[0]
        assert isinstance(event, HealthCheckEvent)
        assert event.overall == "pass"
        assert event.failures == 0
        assert event.warnings == 0
        assert event.checks[0]["name"] == "test_check"

    @pytest.mark.asyncio
    async def test_no_event_bus_does_not_crash(self):
        doctor = Doctor()
        doctor.run_all = AsyncMock(
            return_value=HealthReport(
                overall=CheckStatus.PASS,
                checks=[],
            )
        )
        monitor = HealthMonitor(doctor, event_bus=None, interval_seconds=0.01)
        monitor.start()
        await asyncio.sleep(0.03)
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_loop(self):
        doctor = Doctor()
        doctor.run_all = AsyncMock(
            return_value=HealthReport(
                overall=CheckStatus.PASS,
                checks=[],
            )
        )
        monitor = HealthMonitor(doctor, interval_seconds=60.0)
        monitor.start()
        assert monitor._running is True
        await monitor.stop()
        assert monitor._running is False
        assert monitor._task is None


class TestRuntimeGuard:
    def _make_config(self):
        config = MagicMock()
        config.qdrant_url = None
        config.resolve_paths.return_value = config
        return config

    def test_python_version_passes(self):
        config = self._make_config()
        pipeline = BootstrapPipeline(config)
        # Should not raise on supported Python versions
        pipeline._runtime_guard()

    def test_python_version_too_old(self):
        config = self._make_config()
        pipeline = BootstrapPipeline(config)
        with patch("opencas.bootstrap.pipeline.sys") as mock_sys:
            mock_sys.version_info = (3, 10)
            mock_sys.version = "3.10.0"
            with pytest.raises(RuntimeError, match="Python >= 3.11"):
                pipeline._runtime_guard()

    def test_missing_critical_dependency(self):
        config = self._make_config()
        pipeline = BootstrapPipeline(config)

        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in ("pydantic", "open_llm_auth"):
                raise ImportError("nope")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(RuntimeError, match="Missing critical dependency"):
                pipeline._runtime_guard()


class TestDoctorNewChecks:
    @pytest.mark.asyncio
    async def test_baa_queue_depth_skip_when_no_context(self):
        doctor = Doctor(context=None)
        check = await doctor.check_baa_queue_depth()
        assert check.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_baa_queue_depth_skip_when_no_baa(self):
        context = MagicMock()
        context.harness.baa = None
        doctor = Doctor(context=context)
        check = await doctor.check_baa_queue_depth()
        assert check.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_baa_queue_depth_pass(self):
        context = MagicMock()
        baa = MagicMock()
        baa.queue_size = 5
        baa.held_size = 0
        context.harness.baa = baa
        doctor = Doctor(context=context)
        check = await doctor.check_baa_queue_depth()
        assert check.status == CheckStatus.PASS
        assert check.details["queue_size"] == 5

    @pytest.mark.asyncio
    async def test_baa_queue_depth_warn(self):
        context = MagicMock()
        baa = MagicMock()
        baa.queue_size = 60
        baa.held_size = 0
        context.harness.baa = baa
        doctor = Doctor(context=context)
        check = await doctor.check_baa_queue_depth()
        assert check.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_baa_queue_depth_fail(self):
        context = MagicMock()
        baa = MagicMock()
        baa.queue_size = 150
        baa.held_size = 60
        context.harness.baa = baa
        doctor = Doctor(context=context)
        check = await doctor.check_baa_queue_depth()
        assert check.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_compaction_lag_skip_when_no_context(self):
        doctor = Doctor(context=None)
        check = await doctor.check_compaction_lag()
        assert check.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_compaction_lag_pass(self):
        context = MagicMock()
        context.memory.list_non_compacted_episodes = AsyncMock(return_value=[])
        doctor = Doctor(context=context)
        check = await doctor.check_compaction_lag()
        assert check.status == CheckStatus.PASS
        assert check.details["lag"] == 0

    @pytest.mark.asyncio
    async def test_compaction_lag_warn(self):
        context = MagicMock()
        context.memory.list_non_compacted_episodes = AsyncMock(
            return_value=[MagicMock()] * 600
        )
        doctor = Doctor(context=context)
        check = await doctor.check_compaction_lag()
        assert check.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_compaction_lag_fail(self):
        context = MagicMock()
        context.memory.list_non_compacted_episodes = AsyncMock(
            return_value=[MagicMock()] * 1100
        )
        doctor = Doctor(context=context)
        check = await doctor.check_compaction_lag()
        assert check.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_embedding_latency_skip_when_no_context(self):
        doctor = Doctor(context=None)
        check = await doctor.check_embedding_latency()
        assert check.status == CheckStatus.SKIP

    @pytest.mark.asyncio
    async def test_embedding_latency_pass(self):
        context = MagicMock()
        context.embeddings.health = AsyncMock(
            return_value=MagicMock(avg_embed_latency_ms_1h=125.0)
        )
        doctor = Doctor(context=context)
        check = await doctor.check_embedding_latency()
        assert check.status == CheckStatus.PASS
        assert "latency_ms" in check.details

    @pytest.mark.asyncio
    async def test_embedding_latency_fail_on_exception(self):
        context = MagicMock()
        context.embeddings.health = AsyncMock(side_effect=RuntimeError("boom"))
        doctor = Doctor(context=context)
        check = await doctor.check_embedding_latency()
        assert check.status == CheckStatus.FAIL
