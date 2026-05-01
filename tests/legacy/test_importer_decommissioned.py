"""Tests for the decommissioned OpenBulma importer entry point."""

from pathlib import Path

import pytest

from opencas.runtime.agent_loop import AgentRuntime


@pytest.mark.asyncio
async def test_import_bulma_is_decommissioned() -> None:
    with pytest.raises(RuntimeError, match="decommissioned"):
        await AgentRuntime.import_bulma(object(), Path("/unused/openbulma-v4"))
