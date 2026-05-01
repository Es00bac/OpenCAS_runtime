"""Tests for HealthMonitor and runtime guard."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from opencas.api import provenance_store as ps
from opencas.api.routes.monitor import build_monitor_router
from opencas.bootstrap.pipeline import BootstrapPipeline
from opencas.diagnostics import Doctor, HealthMonitor
from opencas.diagnostics.models import CheckStatus, DiagnosticCheck, HealthReport
from opencas.embeddings.models import EmbeddingRecord
from opencas.infra import EventBus, HealthCheckEvent
from opencas.somatic.models import SomaticSnapshot


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
        with patch("opencas.bootstrap.pipeline_support.sys") as mock_sys:
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
    async def test_embedding_latency_warns_on_slow_provider_backed_embedding(self):
        context = MagicMock()
        context.embeddings.health = AsyncMock(
            return_value=MagicMock(avg_embed_latency_ms_1h=6500.0)
        )
        doctor = Doctor(context=context)
        check = await doctor.check_embedding_latency()
        assert check.status == CheckStatus.WARN

    @pytest.mark.asyncio
    async def test_embedding_latency_fail_on_exception(self):
        context = MagicMock()
        context.embeddings.health = AsyncMock(side_effect=RuntimeError("boom"))
        doctor = Doctor(context=context)
        check = await doctor.check_embedding_latency()
        assert check.status == CheckStatus.FAIL

    @pytest.mark.asyncio
    async def test_qdrant_reachability_pass(self, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"result": {"collections": [{"name": "episodes_semantic"}]}}

        class FakeClient:
            def __init__(self, timeout):
                self.timeout = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def get(self, url):
                assert url == "http://qdrant.local/collections"
                return FakeResponse()

        monkeypatch.setattr("opencas.diagnostics.doctor.httpx.AsyncClient", FakeClient)
        context = SimpleNamespace(config=SimpleNamespace(qdrant_url="http://qdrant.local"))
        doctor = Doctor(context=context)

        check = await doctor.check_qdrant_reachable()

        assert check.status == CheckStatus.PASS
        assert check.details["collection_count"] == 1

    @pytest.mark.asyncio
    async def test_embedding_index_fails_on_recent_fallback_dimension(self):
        now = datetime.now(timezone.utc)
        records = [
            EmbeddingRecord(
                source_hash="gemma",
                model_id="google/embeddinggemma-300m",
                dimension=3072,
                vector=[0.1] * 3072,
                updated_at=now,
            ),
            EmbeddingRecord(
                source_hash="fallback",
                model_id="local-fallback",
                dimension=768,
                vector=[0.1] * 768,
                updated_at=now,
            ),
        ]
        context = SimpleNamespace(
            embeddings=SimpleNamespace(
                model_id="google/embeddinggemma-300m",
                expected_dimension=3072,
                health=AsyncMock(return_value=SimpleNamespace(total_records=2, total_models=2)),
                cache=SimpleNamespace(recent_records_since=AsyncMock(return_value=records)),
            )
        )
        doctor = Doctor(context=context)

        check = await doctor.check_embedding_index()

        assert check.status == CheckStatus.FAIL
        assert check.details["recent_fallback_count"] == 1
        assert check.details["recent_fallback_ratio"] == 0.5
        assert check.details["active_dimension_mismatch_count"] == 0

    @pytest.mark.asyncio
    async def test_embedding_index_warns_when_current_writes_recover_from_fallback(self):
        now = datetime.now(timezone.utc)
        records = [
            EmbeddingRecord(
                source_hash=f"gemma-{i}",
                model_id="google/embeddinggemma-300m",
                dimension=3072,
                vector=[0.1] * 3072,
                updated_at=now,
            )
            for i in range(5)
        ]
        records.append(
            EmbeddingRecord(
                source_hash="fallback",
                model_id="local-fallback",
                dimension=256,
                vector=[0.1] * 256,
                updated_at=now,
            )
        )
        context = SimpleNamespace(
            embeddings=SimpleNamespace(
                model_id="google/embeddinggemma-300m",
                expected_dimension=3072,
                health=AsyncMock(return_value=SimpleNamespace(total_records=6, total_models=2)),
                cache=SimpleNamespace(recent_records_since=AsyncMock(return_value=records)),
            )
        )
        doctor = Doctor(context=context)

        check = await doctor.check_embedding_index()

        assert check.status == CheckStatus.WARN
        assert check.details["clean_active_streak"] == 5
        assert check.details["recent_fallback_count"] == 1

    @pytest.mark.asyncio
    async def test_embedding_index_fails_on_recent_active_dimension_drift(self):
        now = datetime.now(timezone.utc)
        records = [
            EmbeddingRecord(
                source_hash="gemma-old",
                model_id="google/embeddinggemma-300m",
                dimension=768,
                vector=[0.1] * 768,
                updated_at=now,
            ),
            EmbeddingRecord(
                source_hash="gemma-new",
                model_id="google/embeddinggemma-300m",
                dimension=3072,
                vector=[0.1] * 3072,
                updated_at=now,
            ),
        ]
        context = SimpleNamespace(
            embeddings=SimpleNamespace(
                model_id="google/embeddinggemma-300m",
                expected_dimension=3072,
                health=AsyncMock(return_value=SimpleNamespace(total_records=2, total_models=1)),
                cache=SimpleNamespace(recent_records_since=AsyncMock(return_value=records)),
            )
        )
        doctor = Doctor(context=context)

        check = await doctor.check_embedding_index()

        assert check.status == CheckStatus.FAIL
        assert check.details["active_dimension_mismatch_count"] == 1
        assert check.details["multi_dimension_model_ids"] == ["google/embeddinggemma-300m"]

    @pytest.mark.asyncio
    async def test_consolidation_worker_status_fails_on_failed_snapshot(self, tmp_path: Path):
        status_path = tmp_path / "consolidation_worker" / "status.json"
        status_path.parent.mkdir(parents=True)
        status_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "run_id": "run-1",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "error_message": "worker exploded",
                }
            ),
            encoding="utf-8",
        )
        context = SimpleNamespace(config=SimpleNamespace(state_dir=tmp_path))
        doctor = Doctor(context=context)

        check = await doctor.check_consolidation_worker()

        assert check.status == CheckStatus.FAIL
        assert check.details["worker_status"] == "failed"
        assert "worker exploded" in check.message

    @pytest.mark.asyncio
    async def test_consolidation_worker_status_fails_on_timeout_killed_snapshot(self, tmp_path: Path):
        status_path = tmp_path / "consolidation_worker" / "status.json"
        status_path.parent.mkdir(parents=True)
        status_path.write_text(
            json.dumps(
                {
                    "status": "timeout_killed",
                    "run_id": "run-1",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            ),
            encoding="utf-8",
        )
        context = SimpleNamespace(config=SimpleNamespace(state_dir=tmp_path))
        doctor = Doctor(context=context)

        check = await doctor.check_consolidation_worker()

        assert check.status == CheckStatus.FAIL
        assert check.details["worker_status"] == "timeout_killed"
        assert "timed out" in check.message

    @pytest.mark.asyncio
    async def test_somatic_variance_fails_flat_snapshots(self):
        snapshots = [
            SomaticSnapshot(source="test", focus=0.5, energy=0.5),
            SomaticSnapshot(source="test", focus=0.5, energy=0.5),
        ]
        store = SimpleNamespace(trajectory=AsyncMock(return_value=snapshots))
        context = SimpleNamespace(somatic=SimpleNamespace(store=store))
        doctor = Doctor(context=context)

        check = await doctor.check_somatic_variance()

        assert check.status == CheckStatus.FAIL
        assert check.details["flat_dimensions"] == [
            "arousal",
            "fatigue",
            "tension",
            "valence",
            "focus",
            "energy",
            "certainty",
        ]

    @pytest.mark.asyncio
    async def test_monitor_health_route_records_check_provenance(self, tmp_path: Path) -> None:
        first_check_id = uuid4()
        second_check_id = uuid4()
        doctor = Doctor()
        doctor.run_all = AsyncMock(
            return_value=HealthReport(
                overall=CheckStatus.PASS,
                checks=[
                    DiagnosticCheck(
                        check_id=first_check_id,
                        name="bootstrap_readiness",
                        status=CheckStatus.PASS,
                        message="ok",
                    ),
                    DiagnosticCheck(
                        check_id=second_check_id,
                        name="sandbox",
                        status=CheckStatus.WARN,
                        message="sandbox is limited",
                    ),
                ],
            )
        )
        runtime = SimpleNamespace(
            ctx=SimpleNamespace(
                doctor=doctor,
                config=SimpleNamespace(session_id="session-1", state_dir=tmp_path),
                readiness=None,
            )
        )
        app = FastAPI()
        app.include_router(build_monitor_router(runtime))
        route = next(route for route in app.router.routes if getattr(route, "path", None) == "/api/monitor/health")
        response = await route.endpoint()
        assert response.overall == "pass"
        records_path = tmp_path / "provenance.transitions.jsonl"
        records = [
            ps.parse_provenance_transition(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert [record.kind for record in records] == [
            ps.ProvenanceTransitionKind.CHECK,
            ps.ProvenanceTransitionKind.CHECK,
        ]
        assert records[0].details["source_artifact"] == "monitor|health|runtime"
        assert records[0].details["trigger_action"] == "doctor.run_all"
        assert records[0].details["target_entity"] == "bootstrap_readiness"
        assert records[0].details["origin_action_id"] == str(first_check_id)
        assert records[0].details["parent_transition_id"] == str(first_check_id)
        assert records[0].details["linked_transition_ids"] == [str(first_check_id), "bootstrap_readiness"]
        assert records[1].details["target_entity"] == "sandbox"
        assert records[1].details["origin_action_id"] == str(second_check_id)
