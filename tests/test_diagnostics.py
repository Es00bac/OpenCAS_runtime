"""Tests for the diagnostics module."""

import pytest
from pathlib import Path
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.diagnostics import Doctor


@pytest.mark.asyncio
async def test_doctor_all_checks_pass(tmp_path: Path) -> None:
    config = BootstrapConfig(state_dir=tmp_path, session_id="doc-test")
    ctx = await BootstrapPipeline(config).run()
    doctor = Doctor(ctx)
    report = await doctor.run_all()

    assert report.overall.value in ("pass", "warn")
    check_names = {c.name for c in report.checks}
    assert "bootstrap_readiness" in check_names
    assert "sandbox" in check_names
    assert "memory_integrity" in check_names
    assert "embedding_index" in check_names
    assert "continuity" in check_names
    assert "identity_persistence" in check_names
    assert "somatic_state" in check_names
    assert "telemetry_writable" in check_names

    await ctx.memory.close()
    await ctx.embeddings.cache.close()


@pytest.mark.asyncio
async def test_doctor_without_context() -> None:
    doctor = Doctor()
    report = await doctor.run_all()
    assert report.overall.value == "skip"
    for check in report.checks:
        assert check.status.value == "skip"
