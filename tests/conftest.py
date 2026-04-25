import asyncio
import gc

import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def _drain_aiosqlite():
    """Yield control so aiosqlite worker threads finish before loop closes."""
    yield
    gc.collect()
    await asyncio.sleep(0)
    await asyncio.sleep(0.1)
