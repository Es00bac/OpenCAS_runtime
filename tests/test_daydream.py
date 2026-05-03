"""Tests for the daydream generator."""

from pathlib import Path

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.daydream import DaydreamReflection
from opencas.daydream.models import DaydreamThoughtRoute
from opencas.runtime.daydream import DaydreamGenerator


@pytest_asyncio.fixture
async def daydream(tmp_path: Path):
    config = BootstrapConfig(state_dir=tmp_path)
    ctx = await BootstrapPipeline(config).run()
    return DaydreamGenerator(
        llm=ctx.llm,
        memory=ctx.memory,
        tracer=ctx.tracer,
        identity=ctx.identity,
        somatic=ctx.somatic,
        relational=ctx.relational,
        daydream_store=ctx.daydream_store,
    )


@pytest.mark.asyncio
async def test_generate_parses_structured_json(daydream: DaydreamGenerator) -> None:
    # This test will hit the real LLM; if no key is available it may fail.
    try:
        work_objects, reflections = await daydream.generate(goals=["learn rust"], tension=0.5)
        assert isinstance(work_objects, list)
        assert isinstance(reflections, list)
        for wo in work_objects:
            assert hasattr(wo, "content")
            assert hasattr(wo, "stage")
            assert wo.stage.value == "spark"
        for r in reflections:
            assert isinstance(r, DaydreamReflection)
            assert hasattr(r, "spark_content")
    except Exception:
        pytest.skip("LLM not available for daydream generation")


@pytest.mark.asyncio
async def test_parse_structured_markdown_json(daydream: DaydreamGenerator) -> None:
    raw = (
        "```json\n"
        '{"sparks": ["spark a", "spark b"], '
        '"recollection": "recall", "interpretation": "interp", '
        '"synthesis": "synth", "open_question": "why?", '
        '"changed_self_view": "view", "tension_hints": ["hint"]}'
        "\n```"
    )
    parsed = daydream._parse_structured(raw)
    assert len(parsed) == 2
    assert parsed[0].spark_content == "spark a"
    assert parsed[1].spark_content == "spark b"
    assert parsed[0].recollection == "recall"
    assert parsed[0].tension_hints == ["hint"]


@pytest.mark.asyncio
async def test_parse_structured_daydream_thoughts(daydream: DaydreamGenerator) -> None:
    raw = (
        '{"thoughts": [{'
        '"kind": "question",'
        '"route": "deep_think",'
        '"summary": "The scheduling failure may reveal a reusable tool-use lesson.",'
        '"question": "When should a reminder become executable work?",'
        '"hypothesis": "If a project return needs output, it should route to BAA.",'
        '"possible_experiment": "Compare reminder_only with submit_baa on one writing task.",'
        '"usefulness": 0.88,'
        '"novelty": 0.66,'
        '"confidence": 0.74,'
        '"grounding": [{'
        '"kind": "learned",'
        '"source": "outcome",'
        '"claim": "A prior writing return failed as a passive reminder.",'
        '"confidence": 0.8,'
        '"evidence_ids": ["episode-1"]'
        '}]}]}'
    )

    parsed = daydream._parse_structured(raw)

    assert len(parsed) == 1
    reflection = parsed[0]
    assert reflection.spark_content == (
        "The scheduling failure may reveal a reusable tool-use lesson."
    )
    assert len(reflection.thoughts) == 1
    thought = reflection.thoughts[0]
    assert thought.route == DaydreamThoughtRoute.DEEP_THINK
    assert thought.question == "When should a reminder become executable work?"
    assert thought.possible_experiment.startswith("Compare reminder_only")
    assert thought.grounding[0].kind.value == "learned"
    assert thought.grounding[0].evidence_ids == ["episode-1"]


@pytest.mark.asyncio
async def test_parse_structured_sanitizes_fixation_terms(daydream: DaydreamGenerator) -> None:
    raw = (
        '{"sparks": ["returning to returning thread drifted"], '
        '"recollection": "A returning concern drifted toward thread", '
        '"interpretation": "thread returns to the same returning spot", '
        '"synthesis": "drifted and returning", '
        '"open_question": "why did returning happen here?", '
        '"changed_self_view": "less returning, more drifted", '
        '"tension_hints": ["returning to thread", "drifted and returning"]}'
    )
    parsed = daydream._parse_structured(raw)
    assert len(parsed) == 1
    reflection = parsed[0]
    assert "revisiting" in reflection.spark_content
    assert "path" in reflection.recollection
    assert "shifted" in reflection.synthesis
    assert reflection.open_question is not None and "revisiting" in reflection.open_question
    assert "revisiting" in reflection.changed_self_view
    assert "path" in reflection.tension_hints[0]


@pytest.mark.asyncio
async def test_parse_structured_fallback_text(daydream: DaydreamGenerator) -> None:
    raw = "just a plain text spark"
    parsed = daydream._parse_structured(raw)
    assert len(parsed) == 1
    assert parsed[0].spark_content == "just a plain text spark"


@pytest.mark.asyncio
async def test_build_prompt_includes_identity(daydream: DaydreamGenerator) -> None:
    prompt = await daydream._build_prompt(
        memory_snippets=["memory one"],
        goals=["goal a"],
        tension=0.3,
    )
    assert "goal a" in prompt
    assert "memory one" in prompt
    assert daydream.identity is not None
    # Identity fragment should include values/traits seeded by defaults
    assert len(daydream.identity.self_model.values) > 0
    for value in daydream.identity.self_model.values:
        assert value in prompt


@pytest.mark.asyncio
async def test_build_prompt_requests_grounded_thought_records(
    daydream: DaydreamGenerator,
) -> None:
    prompt = await daydream._build_prompt(
        memory_snippets=["memory one"],
        goals=["goal a"],
        tension=0.3,
    )

    assert "thoughts" in prompt
    assert "route" in prompt
    assert "grounding" in prompt


@pytest.mark.asyncio
async def test_daydream_store_preserves_structured_thoughts(
    daydream: DaydreamGenerator,
) -> None:
    raw = (
        '{"thoughts": [{'
        '"kind": "hypothesis",'
        '"route": "incubate",'
        '"summary": "A recurring failure may be a process smell.",'
        '"hypothesis": "Repeated correction requests should become learned rules.",'
        '"confidence": 0.7,'
        '"grounding": [{'
        '"kind": "observed",'
        '"source": "memory",'
        '"claim": "The operator repeatedly corrects the same behavior.",'
        '"confidence": 0.75'
        '}]}]}'
    )
    reflection = daydream._parse_structured(raw)[0]

    assert daydream.daydream_store is not None
    await daydream.daydream_store.save_reflection(reflection)
    recent = await daydream.daydream_store.list_recent(limit=1)

    assert len(recent[0].thoughts) == 1
    assert recent[0].thoughts[0].route.value == "incubate"
    assert recent[0].thoughts[0].grounding[0].source.value == "memory"
