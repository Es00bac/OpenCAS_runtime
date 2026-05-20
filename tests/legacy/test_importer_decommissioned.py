"""Tests for the decommissioned OpenBulma importer entry point."""

from pathlib import Path

import pytest

from opencas.runtime.agent_loop import AgentRuntime
from scripts import run_bulma_import


@pytest.mark.asyncio
async def test_import_bulma_is_decommissioned() -> None:
    with pytest.raises(RuntimeError, match="decommissioned"):
        await AgentRuntime.import_bulma(object(), Path("/unused/openbulma-v4"))


def test_import_script_fails_without_instantiating_legacy_importer() -> None:
    assert run_bulma_import.main(["--dry-run"]) == 2

    script = Path("scripts/run_bulma_import.py").read_text(encoding="utf-8")
    assert "opencas.legacy.importer" not in script
    assert "BootstrapPipeline" not in script
    assert "shutil.rmtree" not in script


def test_extract_package_review_no_longer_instructs_importer_use() -> None:
    script = Path("scripts/extract_bulma_package.py").read_text(encoding="utf-8")
    assert "from opencas.legacy.importer import BulmaImportTask" not in script
    assert "Forensic Package Only" in script
