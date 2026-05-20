import pytest
from types import SimpleNamespace
from opencas.tools.memory_tools import MemoryToolAdapter
from opencas.context.models import RetrievalResult

class _FakeRetriever:
    def __init__(self, results):
        self.results = results

    async def inspect(self, query: str, *, limit: int = 10, offset: int = 0):
        # Slice results
        sliced = self.results[offset : offset + limit]
        return {
            "results": sliced,
            "meta": {
                "total_count": len(self.results),
                "limit": limit,
                "offset": offset
            }
        }

@pytest.mark.asyncio
async def test_search_memories_pagination_metadata():
    results = [RetrievalResult(source_type="memory", source_id=str(i), content=f"item {i}", score=0.9-i*0.01) for i in range(15)]
    runtime = SimpleNamespace(retriever=_FakeRetriever(results))
    adapter = MemoryToolAdapter(runtime)
    
    # Test call with limit=5, offset=2
    res = await adapter("search_memories", {"query": "test", "limit": 5, "offset": 2})
    
    assert res.success
    assert "item 2" in res.output
    assert "item 6" in res.output
    assert "item 7" not in res.output
    
    meta = res.metadata
    assert meta["total_count"] == 15
    assert meta["returned_count"] == 5
    assert meta["offset"] == 2
    assert meta["limit"] == 5
    assert meta["truncated"] is True
    assert meta["next_offset"] == 7

@pytest.mark.asyncio
async def test_search_memories_no_truncation_when_at_end():
    results = [RetrievalResult(source_type="memory", source_id=str(i), content=f"item {i}", score=0.9) for i in range(5)]
    runtime = SimpleNamespace(retriever=_FakeRetriever(results))
    adapter = MemoryToolAdapter(runtime)
    
    res = await adapter("search_memories", {"query": "test", "limit": 10, "offset": 0})
    
    assert res.success
    meta = res.metadata
    assert meta["total_count"] == 5
    assert meta["returned_count"] == 5
    assert meta["truncated"] is False
    assert meta["next_offset"] is None
