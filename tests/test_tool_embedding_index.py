"""Tests for the embedding-based semantic tool router."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pytest

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.models import ToolResult
from opencas.tools.registry import ToolEntry
from opencas.tools.tool_embedding_index import (
    TOOL_SEMANTIC_HINTS,
    ToolEmbeddingIndex,
    _build_tool_semantic_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(name: str, desc: str = "stub tool") -> ToolEntry:
    return ToolEntry(
        name=name,
        description=desc,
        adapter=lambda n, a: ToolResult(success=True, output="ok", metadata={}),
        risk_tier=ActionRiskTier.READONLY,
        parameters={"type": "object"},
    )


class _FakeRecord:
    """Mimics EmbeddingRecord.vector."""

    def __init__(self, vector: List[float]):
        self.vector = vector


class _FakeEmbeddingService:
    """Deterministic fake: embeds each text as a distinct random vector.

    Supports controlled tests via ``assign_vectors``.
    """

    def __init__(self, dim: int = 64, *, seed: int = 42):
        self._rng = np.random.RandomState(seed)
        self._dim = dim
        self._assigned: Dict[str, List[float]] = {}

    def assign_vectors(self, mapping: Dict[str, List[float]]) -> None:
        """Force specific vectors for texts containing given substrings."""
        self._assigned = mapping

    async def embed(self, text: str, **kw: Any) -> _FakeRecord:
        return _FakeRecord(self._vector_for(text))

    async def embed_batch(self, texts: List[str], **kw: Any) -> List[_FakeRecord]:
        return [_FakeRecord(self._vector_for(t)) for t in texts]

    def _vector_for(self, text: str) -> List[float]:
        for substr, vec in self._assigned.items():
            if substr in text:
                return vec
        v = self._rng.randn(self._dim).astype(np.float32)
        v = v / np.linalg.norm(v)
        return v.tolist()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSemanticText:
    def test_with_hint(self):
        entry = _entry("system_status", "Show host stats.")
        text = _build_tool_semantic_text(entry)
        assert text.startswith("system_status:")
        assert "Show host stats." in text
        assert "CPU" in text  # from TOOL_SEMANTIC_HINTS

    def test_without_hint(self):
        entry = _entry("custom_tool_xyz", "Does something custom.")
        text = _build_tool_semantic_text(entry)
        assert text == "custom_tool_xyz: Does something custom."


class TestBuild:
    @pytest.mark.asyncio
    async def test_build_success(self):
        tools = [_entry("a"), _entry("b"), _entry("c")]
        svc = _FakeEmbeddingService(dim=32)
        idx = await ToolEmbeddingIndex.build(svc, tools)
        assert idx.is_ready
        assert len(idx._tool_names) == 3
        assert idx._matrix.shape == (3, 32)

    @pytest.mark.asyncio
    async def test_build_empty(self):
        idx = await ToolEmbeddingIndex.build(_FakeEmbeddingService(), [])
        assert not idx.is_ready

    @pytest.mark.asyncio
    async def test_build_failure_returns_not_ready(self):
        class _FailSvc:
            async def embed_batch(self, *a, **kw):
                raise RuntimeError("boom")
        idx = await ToolEmbeddingIndex.build(_FailSvc(), [_entry("x")])
        assert not idx.is_ready


class TestSelectTools:
    @pytest.mark.asyncio
    async def test_always_available_present(self):
        """Always-available tools are included regardless of objective."""
        tools = [
            _entry("search_memories"),
            _entry("recall_concepts"),
            _entry("web_search"),
            _entry("web_fetch"),
            _entry("system_status"),
        ]
        svc = _FakeEmbeddingService(dim=32)
        idx = await ToolEmbeddingIndex.build(svc, tools)
        # Random objective vector
        vec = np.random.randn(32).astype(np.float32)
        selected = idx.select_tools(vec, tools)
        names = {t.name for t in selected}
        assert "search_memories" in names
        assert "web_fetch" in names

    @pytest.mark.asyncio
    async def test_semantic_selection(self):
        """Objective vector similar to a tool's vector surfaces that tool."""
        dim = 32
        rng = np.random.RandomState(0)
        # Make system_status vector point in a known direction
        target_vec = rng.randn(dim).astype(np.float32)
        target_vec = target_vec / np.linalg.norm(target_vec)

        tools = [
            _entry("system_status", "Show host CPU memory disk stats."),
            _entry("note_save", "Save a note."),
            _entry("calculate", "Compute math expressions."),
        ]
        svc = _FakeEmbeddingService(dim=dim)
        svc.assign_vectors({"system_status": target_vec.tolist()})
        idx = await ToolEmbeddingIndex.build(svc, tools)

        # Use the same direction as the objective — should match system_status
        selected = idx.select_tools(target_vec, tools)
        names = [t.name for t in selected]
        assert "system_status" in names

    @pytest.mark.asyncio
    async def test_min_tools_floor(self):
        """Even below threshold, MIN_TOOLS are returned."""
        dim = 16
        tools = [_entry(f"tool_{i}") for i in range(20)]
        svc = _FakeEmbeddingService(dim=dim, seed=99)
        idx = await ToolEmbeddingIndex.build(svc, tools)
        vec = np.random.randn(dim).astype(np.float32)
        selected = idx.select_tools(vec, tools)
        # Should have at least MIN_TOOLS (8)
        assert len(selected) >= idx.MIN_TOOLS

    @pytest.mark.asyncio
    async def test_max_tools_ceiling(self):
        """Selection is capped at MAX_TOOLS."""
        dim = 16
        tools = [_entry(f"tool_{i}") for i in range(60)]
        svc = _FakeEmbeddingService(dim=dim, seed=7)
        idx = await ToolEmbeddingIndex.build(svc, tools)
        vec = np.ones(dim, dtype=np.float32)
        selected = idx.select_tools(vec, tools)
        assert len(selected) <= idx.MAX_TOOLS

    @pytest.mark.asyncio
    async def test_not_ready_returns_all(self):
        """When index not built, returns all tools (up to MAX)."""
        idx = ToolEmbeddingIndex()
        tools = [_entry(f"t{i}") for i in range(5)]
        vec = np.zeros(10, dtype=np.float32)
        selected = idx.select_tools(vec, tools)
        assert len(selected) == 5


class TestToolUseLoopIntegration:
    """Verify ToolUseLoop falls back gracefully without embedding index."""

    def test_filter_tools_keyword_fallback(self):
        """When no index, keyword routing still works."""
        from opencas.tools.context import ToolUseContext
        from opencas.tools.loop import ToolUseLoop

        tools_reg = ToolRegistry()
        tools_reg.register(
            "system_status", "Host stats",
            lambda n, a: ToolResult(success=True, output="", metadata={}),
            ActionRiskTier.READONLY, {"type": "object"},
        )
        tools_reg.register(
            "web_search", "Search web",
            lambda n, a: ToolResult(success=True, output="", metadata={}),
            ActionRiskTier.NETWORK, {"type": "object"},
        )
        tools_reg.register(
            "calculate", "Compute math",
            lambda n, a: ToolResult(success=True, output="", metadata={}),
            ActionRiskTier.READONLY, {"type": "object"},
        )

        _stub_llm = type("_L", (), {"model_routing": type("_R", (), {"auto_escalation": True})()})()
        _stub_approval = type("_A", (), {})()
        _stub_runtime = type("_RT", (), {})()

        loop = ToolUseLoop(llm=_stub_llm, tools=tools_reg, approval=_stub_approval)
        ctx = ToolUseContext(runtime=_stub_runtime, session_id="t", plan_mode=False)

        # Keyword "cpu" should surface system_status via fallback
        selected = loop._filter_tools(ctx, objective="check the cpu load")
        names = {t.name for t in selected}
        assert "system_status" in names


# Import ToolRegistry for integration test
from opencas.tools.registry import ToolRegistry
