"""Tests for the daydream generator."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.daydream import DaydreamReflection
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
