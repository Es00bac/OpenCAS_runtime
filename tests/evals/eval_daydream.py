"""Daydream behavioral evals.

Measures whether DaydreamGenerator produces correctly staged WorkObjects
from structured LLM output. Uses a mock LLM so no API calls are needed.

Metrics:
- sparks produced per run (should be > 0 when LLM output is valid)
- all sparks start at SPARK stage (not promoted)
- reflections are created alongside sparks
- graceful handling of malformed LLM output (fallback spark created)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from opencas.autonomy.models import WorkStage
from opencas.embeddings.service import EmbeddingCache, EmbeddingService
from opencas.memory import Episode, EpisodeKind, MemoryStore
from opencas.runtime.daydream import DaydreamGenerator


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float
    notes: str
    details: dict = field(default_factory=dict)


def _mock_llm(sparks: List[str]) -> Any:
    """Return a mock LLMClient whose chat_completion returns *sparks*."""
    payload = json.dumps({
        "sparks": sparks,
        "recollection": "Mock recollection of recent events.",
        "interpretation": "Mock interpretation of patterns.",
        "synthesis": "Mock synthesis of ideas.",
        "open_question": "What does this imply?",
        "changed_self_view": "Mock self-view update.",
        "tension_hints": ["creative tension", "unresolved question"],
    })
    response = {"choices": [{"message": {"content": payload}}]}
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value=response)
    return llm


def _mock_llm_malformed() -> Any:
    """Return a mock LLM that returns non-JSON content (tests fallback path)."""
    response = {"choices": [{"message": {"content": "This is not JSON at all."}}]}
    llm = MagicMock()
    llm.chat_completion = AsyncMock(return_value=response)
    return llm


def _mock_llm_error() -> Any:
    """Return a mock LLM that raises an exception (tests graceful failure)."""
    llm = MagicMock()
    llm.chat_completion = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    return llm


async def _make_store(tmp: Path) -> MemoryStore:
    store = MemoryStore(tmp / "memory.db")
    await store.connect()
    for i in range(3):
        await store.save_episode(Episode(
            kind=EpisodeKind.TURN, session_id="eval",
            content=f"Background memory {i}: something interesting happened.",
        ))
    return store


# ---------------------------------------------------------------------------
# Eval 1: sparks generated from valid LLM output
# ---------------------------------------------------------------------------
async def eval_sparks_generated(tmp: Path) -> EvalResult:
    store = await _make_store(tmp / "sparks")
    expected_sparks = [
        "What if memory retrieval could be emotionally contextual?",
        "The relationship between somatic state and creativity is underexplored.",
        "Consider a trust-weighted graph for relational memory.",
    ]
    llm = _mock_llm(expected_sparks)
    gen = DaydreamGenerator(llm=llm, memory=store)

    work_objects, reflections = await gen.generate(tension=0.5)
    await store.close()

    count = len(work_objects)
    all_spark_stage = all(w.stage == WorkStage.SPARK for w in work_objects)
    has_reflections = len(reflections) > 0

    passed = count == len(expected_sparks) and all_spark_stage and has_reflections
    return EvalResult(
        name="daydream.sparks_generated",
        passed=passed,
        score=min(1.0, count / len(expected_sparks)) if len(expected_sparks) > 0 else 0.0,
        notes=f"Generated {count}/{len(expected_sparks)} sparks, all_spark_stage={all_spark_stage}, has_reflections={has_reflections}",
        details={
            "count": count,
            "expected": len(expected_sparks),
            "all_spark_stage": all_spark_stage,
            "has_reflections": has_reflections,
            "spark_contents": [w.content[:80] for w in work_objects],
        },
    )


# ---------------------------------------------------------------------------
# Eval 2: all sparks start at SPARK stage (not artificially promoted)
# ---------------------------------------------------------------------------
async def eval_sparks_at_correct_stage(tmp: Path) -> EvalResult:
    store = await _make_store(tmp / "stage")
    llm = _mock_llm(["Idea A", "Idea B", "Idea C", "Idea D", "Idea E"])
    gen = DaydreamGenerator(llm=llm, memory=store)

    work_objects, _ = await gen.generate(tension=0.3)
    await store.close()

    wrong_stage = [w for w in work_objects if w.stage != WorkStage.SPARK]
    passed = len(wrong_stage) == 0 and len(work_objects) > 0

    return EvalResult(
        name="daydream.sparks_at_correct_stage",
        passed=passed,
        score=1.0 if passed else (len(work_objects) - len(wrong_stage)) / max(len(work_objects), 1),
        notes=f"{len(wrong_stage)} sparks had wrong stage (expected all SPARK)",
        details={
            "total_sparks": len(work_objects),
            "wrong_stage": [{"content": w.content[:50], "stage": w.stage.value} for w in wrong_stage],
        },
    )


# ---------------------------------------------------------------------------
# Eval 3: graceful fallback on malformed LLM output
# Fallback path creates a single spark from the raw content.
# ---------------------------------------------------------------------------
async def eval_malformed_output_fallback(tmp: Path) -> EvalResult:
    store = await _make_store(tmp / "malformed")
    llm = _mock_llm_malformed()
    gen = DaydreamGenerator(llm=llm, memory=store)

    work_objects, reflections = await gen.generate(tension=0.0)
    await store.close()

    # Fallback: at least one spark created from the raw text content
    passed = len(work_objects) >= 1 and work_objects[0].stage == WorkStage.SPARK

    return EvalResult(
        name="daydream.malformed_output_fallback",
        passed=passed,
        score=1.0 if passed else 0.0,
        notes=f"Fallback produced {len(work_objects)} spark(s) from non-JSON LLM output",
        details={"sparks": [w.content[:80] for w in work_objects]},
    )


# ---------------------------------------------------------------------------
# Eval 4: graceful handling of LLM failure (no sparks, no crash)
# ---------------------------------------------------------------------------
async def eval_llm_failure_graceful(tmp: Path) -> EvalResult:
    store = await _make_store(tmp / "failure")
    llm = _mock_llm_error()
    gen = DaydreamGenerator(llm=llm, memory=store)

    try:
        work_objects, reflections = await gen.generate(tension=0.0)
        crashed = False
    except Exception as exc:
        work_objects = []
        reflections = []
        crashed = True

    await store.close()

    passed = not crashed and len(work_objects) == 0
    return EvalResult(
        name="daydream.llm_failure_graceful",
        passed=passed,
        score=1.0 if passed else 0.0,
        notes=f"LLM error: crashed={crashed}, sparks_generated={len(work_objects)}",
        details={"crashed": crashed, "sparks": len(work_objects), "reflections": len(reflections)},
    )


# ---------------------------------------------------------------------------
# Eval 5: tension metadata is recorded on sparks
# ---------------------------------------------------------------------------
async def eval_tension_metadata(tmp: Path) -> EvalResult:
    store = await _make_store(tmp / "tension")
    llm = _mock_llm(["Idea under tension"])
    gen = DaydreamGenerator(llm=llm, memory=store)

    test_tension = 0.75
    work_objects, _ = await gen.generate(tension=test_tension)
    await store.close()

    passed = (
        len(work_objects) > 0
        and work_objects[0].meta.get("origin") == "daydream"
        and work_objects[0].meta.get("tension") == test_tension
    )

    meta = work_objects[0].meta if work_objects else {}
    return EvalResult(
        name="daydream.tension_metadata",
        passed=passed,
        score=1.0 if passed else 0.0,
        notes=f"Spark meta: origin={meta.get('origin')}, tension={meta.get('tension')}",
        details={"meta": meta},
    )


async def run_all(tmp_root: Path) -> List[EvalResult]:
    tmp_root.mkdir(parents=True, exist_ok=True)
    return list(await asyncio.gather(
        eval_sparks_generated(tmp_root),
        eval_sparks_at_correct_stage(tmp_root),
        eval_malformed_output_fallback(tmp_root),
        eval_llm_failure_graceful(tmp_root),
        eval_tension_metadata(tmp_root),
    ))
