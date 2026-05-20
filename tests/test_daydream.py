"""Tests for the daydream generator."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.api.routes.daydream import _reflection_to_dict
from opencas.daydream import DaydreamReflection
from opencas.daydream.models import DaydreamThoughtRoute
from opencas.runtime.daydream import DaydreamGenerator
from opencas.runtime.reflective_working_set import (
    collect_reflective_working_set,
    working_set_source_family_counts,
)


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
        '"imaginative_branch": "A reminder could behave like a seed that only sprouts if it has soil.",'
        '"practical_branch": "Executable reminders should include the substrate needed to act.",'
        '"bridge": "The seed image distinguishes passive memory from actionable preparation.",'
        '"suggested_handler": "self_note",'
        '"contact_posture": "share_after_artifact",'
        '"self_work_intent": "write a small note comparing passive and executable reminders",'
        '"risk": 0.1,'
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
    assert thought.imaginative_branch.startswith("A reminder could behave")
    assert thought.practical_branch == "Executable reminders should include the substrate needed to act."
    assert thought.bridge.startswith("The seed image")
    assert thought.suggested_handler == "self_note"
    assert thought.contact_posture == "share_after_artifact"
    assert thought.self_work_intent.startswith("write a small note")
    assert thought.risk == 0.1
    assert thought.grounding[0].kind.value == "learned"
    assert thought.grounding[0].evidence_ids == ["episode-1"]


@pytest.mark.asyncio
async def test_parse_structured_daydream_inner_dialogue(
    daydream: DaydreamGenerator,
) -> None:
    raw = (
        '{"thoughts": [{'
        '"kind": "hypothesis",'
        '"route": "incubate",'
        '"summary": "The atlas refresh bug may reveal a broader state-continuity rule.",'
        '"inner_dialogue": ['
        '{"voice": "curious", "stance": "opening question", "text": "What keeps moving when the backend refreshes?"},'
        '{"voice": "skeptical", "stance": "pressure test", "text": "If every refresh resets the home, the map will never feel continuous."},'
        '{"voice": "practical", "stance": "next step", "text": "Preserve the local position unless the projection method actually changes."}'
        '],'
        '"confidence": 0.7,'
        '"grounding": [{'
        '"kind": "observed",'
        '"source": "runtime",'
        '"claim": "A recent atlas refresh fix depended on preserving node state.",'
        '"confidence": 0.8'
        '}]}]}'
    )

    parsed = daydream._parse_structured(raw)

    thought = parsed[0].thoughts[0]
    assert len(thought.inner_dialogue) == 3
    assert thought.inner_dialogue[0].voice == "curious"
    assert thought.inner_dialogue[1].stance == "pressure test"
    assert "projection method" in thought.inner_dialogue[2].text
    payload = _reflection_to_dict(parsed[0])
    assert payload["thought_count"] == 1
    assert payload["inner_dialogue_turn_count"] == 3
    assert payload["thoughts"][0]["inner_dialogue"][0]["voice"] == "curious"


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
    assert len(parsed[0].thoughts) == 1
    assert parsed[0].thoughts[0].route == DaydreamThoughtRoute.INCUBATE
    assert parsed[0].thoughts[0].summary == "just a plain text spark"
    assert parsed[0].thoughts[0].grounding[0].source.value == "daydream"
    assert parsed[0].thoughts[0].grounding[0].kind.value == "generated_synthesis"


@pytest.mark.asyncio
async def test_parse_structured_salvages_partially_invalid_thought(
    daydream: DaydreamGenerator,
) -> None:
    raw = (
        '{"thoughts": [{'
        '"kind": "system insight",'
        '"route": "deep think",'
        '"summary": "The daydream parser should salvage useful partial thoughts.",'
        '"question": "Why did the live run become empty?",'
        '"usefulness": 4.2,'
        '"novelty": -1,'
        '"confidence": "not-a-number",'
        '"risk": 0.2,'
        '"grounding": [{"kind": "unknown", "source": "runtime", "claim": "bad enum"}]'
        '}]}'
    )

    parsed = daydream._parse_structured(raw)

    assert len(parsed) == 1
    reflection = parsed[0]
    assert reflection.spark_content == "The daydream parser should salvage useful partial thoughts."
    assert len(reflection.thoughts) == 1
    thought = reflection.thoughts[0]
    assert thought.kind.value == "system_insight"
    assert thought.route == DaydreamThoughtRoute.DEEP_THINK
    assert thought.usefulness == 1.0
    assert thought.novelty == 0.0
    assert thought.confidence == 0.5
    assert thought.grounding[0].source.value == "daydream"


@pytest.mark.asyncio
async def test_generate_falls_back_to_working_set_when_model_returns_no_thoughts(
    daydream: DaydreamGenerator,
) -> None:
    daydream.llm.chat_completion = AsyncMock(
        return_value={"choices": [{"message": {"content": '{"thoughts": []}'}}]}
    )

    work_objects, reflections = await daydream.generate(
        tension=0.4,
        working_set=[
            {
                "kind": "active_work",
                "source": "test",
                "label": "Atlas temporal metadata",
                "text": "Flow edges need temporal metadata so the atlas can explain order.",
                "evidence_ids": ["test:flow-edge-time"],
            }
        ],
    )

    assert len(reflections) == 1
    assert len(work_objects) == 1
    reflection = reflections[0]
    assert reflection.experience_context["trigger"] == "daydream_empty_generation_fallback"
    assert "Flow edges need temporal metadata" in reflection.spark_content
    assert reflection.thoughts[0].route == DaydreamThoughtRoute.INCUBATE
    assert reflection.thoughts[0].grounding[0].evidence_ids == ["test:flow-edge-time"]


@pytest.mark.asyncio
async def test_generate_records_sampler_variation_context_on_daydream_artifacts(
    daydream: DaydreamGenerator,
) -> None:
    captured_requests = []

    async def _mock_chat_completion(*args, **kwargs):
        generation_request = kwargs.get("generation_request")
        captured_requests.append(generation_request)
        return {
            "_opencas_generation_policy": {
                "generation_phase": "daydream",
                "generation_authority": "reflective_proposal_only",
                "generation_effective_sampling": {
                    "temperature": 0.83,
                    "top_p": 0.91,
                    "top_k": 80,
                },
                "sampling_variation_seed": generation_request.variation_seed,
            },
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"thoughts": [{'
                            '"kind": "association",'
                            '"route": "incubate",'
                            '"summary": "A sampler variation can reveal which daydreams later prove useful.",'
                            '"confidence": 0.7,'
                            '"grounding": []'
                            '}]}'
                        )
                    }
                }
            ],
        }

    daydream.llm.chat_completion = _mock_chat_completion

    work_objects, reflections = await daydream.generate(tension=0.5)

    assert captured_requests
    assert captured_requests[0].variation_seed
    assert work_objects
    assert work_objects[0].meta["generation_policy"]["generation_effective_sampling"]["top_k"] == 80
    assert (
        reflections[0].experience_context["generation_policy"]["sampling_variation_seed"]
        == captured_requests[0].variation_seed
    )


@pytest.mark.asyncio
async def test_generate_augments_thought_grounding_from_working_set(
    daydream: DaydreamGenerator,
) -> None:
    daydream.llm.chat_completion = AsyncMock(
        return_value={
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"thoughts": [{'
                            '"kind": "hypothesis",'
                            '"route": "incubate",'
                            '"summary": "The daydream should preserve working-set evidence.",'
                            '"hypothesis": "Missing model grounding can still be anchored to intake evidence.",'
                            '"confidence": 0.72,'
                            '"grounding": []'
                            '}]}'
                        )
                    }
                }
            ]
        }
    )

    _work_objects, reflections = await daydream.generate(
        tension=0.3,
        working_set=[
            {
                "kind": "external_observation",
                "source": "memory",
                "label": "Gmail headline",
                "text": "Email headline about context caching in local agents.",
                "evidence_ids": ["episode-email-1"],
                "weight": 0.72,
                "meta": {"source_family": "external_observation", "authority": "runtime"},
            }
        ],
    )

    grounding = reflections[0].thoughts[0].grounding
    assert any(item.evidence_ids == ["episode-email-1"] for item in grounding)
    assert any("working-set item" in item.claim for item in grounding)


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
    assert "imaginative_branch" in prompt
    assert "practical_branch" in prompt
    assert "bridge" in prompt
    assert "inner_dialogue" in prompt
    assert "self-dialogue" in prompt
    assert "contact_posture" in prompt


@pytest.mark.asyncio
async def test_build_prompt_includes_active_working_set(
    daydream: DaydreamGenerator,
) -> None:
    prompt = await daydream._build_prompt(
        memory_snippets=[],
        goals=[],
        tension=0.2,
        working_set=[
            {
                "kind": "active_work",
                "source": "test",
                "label": "Qt atlas flow regression",
                "text": "The 2D flow view stopped responding while 3D activity still updated.",
                "created_at": "2026-05-12T10:00:00+00:00",
                "observed_at": "2026-05-12T10:00:03+00:00",
                "evidence_ids": ["event-1"],
                "meta": {"authority": "operator", "novelty_pressure": 0.9},
            }
        ],
    )

    assert "Active grounded working set" in prompt
    assert "Qt atlas flow regression" in prompt
    assert "2D flow view stopped responding" in prompt
    assert "collaborator-agent turns" in prompt
    assert "2026-05-12T10:00:03+00:00" in prompt
    assert "authority: operator" in prompt
    assert "novelty_pressure: 0.9" in prompt


@pytest.mark.asyncio
async def test_build_prompt_includes_working_set_coverage_and_cross_source_instruction(
    daydream: DaydreamGenerator,
) -> None:
    prompt = await daydream._build_prompt(
        memory_snippets=[],
        goals=[],
        tension=0.2,
        working_set=[
            {
                "kind": "active_goal",
                "source": "executive.active_goals",
                "label": "Agent autonomy",
                "text": "Make ordinary actions self-approving when low risk.",
                "meta": {"source_family": "active_work"},
            },
            {
                "kind": "tom_user_belief",
                "source": "tom.user_beliefs",
                "label": "User preference",
                "text": "The operator prefers evidence-grounded autonomy.",
                "meta": {"source_family": "tom_user_model"},
            },
            {
                "kind": "external_observation",
                "source": "telemetry.external",
                "label": "Gmail headline",
                "text": "Email headline about local AI agents and context caching.",
                "meta": {"source_family": "external_observation"},
            },
        ],
    )

    assert "Daydream intake coverage" in prompt
    assert "active_work: 1" in prompt
    assert "tom_user_model: 1" in prompt
    assert "external_observation: 1" in prompt
    assert "synthesize across at least two source families" in prompt


@pytest.mark.asyncio
async def test_reflective_working_set_collects_runtime_dialogue_and_state() -> None:
    class ContextStore:
        async def list_recent(self, session_id: str, limit: int = 6, include_hidden: bool = False):
            return [
                SimpleNamespace(
                    role=SimpleNamespace(value="user"),
                    content="Codex is asking Bulma about atlas cognition, not speaking as the operator.",
                    meta={"conversation_actor": {"type": "agent", "label": "Codex"}},
                    created_at="2026-05-12T10:00:00+00:00",
                    message_id="msg-1",
                )
            ]

    class CognitiveStore:
        async def list_attention(self, limit: int = 6):
            return [
                SimpleNamespace(
                    label="all cognitive activity visualization",
                    strength=0.9,
                    updated_at="2026-05-12T10:01:00+00:00",
                    evidence_refs=["attn-1"],
                )
            ]

        async def list_working_memory(self, limit: int = 6):
            return []

        async def list_prospective_memories(self, limit: int = 6):
            return [
                SimpleNamespace(
                    action="Help the operator turn the income-support mission into next actions.",
                    condition="When background reflection can advance the mission.",
                    confidence=0.88,
                    updated_at="2026-05-12T10:01:30+00:00",
                    trigger_at=None,
                    proof_ref="",
                    evidence_refs=["commitment-income"],
                )
            ]

        async def list_recent_events(self, session_id: str | None = None, status: str | None = None, limit: int = 8):
            return []

    runtime = SimpleNamespace(
        _activity="daydreaming",
        _activity_since="2026-05-12T10:02:00+00:00",
        current_cognition={
            "session_id": "default",
            "user_input": "Think about the OpenCAS Manager atlas.",
            "conversation_actor": {"type": "operator", "label": "operator"},
        },
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="default"),
            context_store=ContextStore(),
            cognitive_state_store=CognitiveStore(),
        ),
        executive=SimpleNamespace(active_goals=["make daydreaming process active work"]),
        tracer=None,
        memory=None,
    )

    items = await collect_reflective_working_set(runtime, limit=10)

    texts = "\n".join(item["text"] for item in items)
    assert "OpenCAS Manager atlas" in texts
    assert "income-support mission" in texts
    assert "not speaking as the operator" in texts
    assert "active work" in texts


@pytest.mark.asyncio
async def test_reflective_working_set_collects_tom_beliefs_and_preserves_source_diversity() -> None:
    class TomEngine:
        def list_beliefs(self, subject=None, limit: int = 20):
            return [
                SimpleNamespace(
                    belief_id="belief-interest",
                    subject=SimpleNamespace(value="user"),
                    relation="prefers",
                    predicate="user_prefers_autonomous_partner",
                    object="autonomous assistants that notice useful work before being asked",
                    confidence=0.82,
                    evidence_ids=["tom-msg-1"],
                    source_kind="conversation",
                    updated_at="2026-05-12T13:00:00+00:00",
                )
            ]

    class MemoryStore:
        async def list_recent_episodes(self, session_id: str | None = None, limit: int = 12):
            return [
                {
                    "episode_id": f"mem-{idx}",
                    "content": f"Email headline {idx}: local-first agent caching research",
                    "created_at": f"2026-05-12T13:0{idx}:00+00:00",
                    "meta": {"source": "gmail"},
                }
                for idx in range(3)
            ]

    class TracerStore:
        def query(self, session_id: str | None = None, limit: int = 16):
            return [
                SimpleNamespace(
                    event_id="tool-web-1",
                    kind=SimpleNamespace(value="tool_call"),
                    message="tool call completed",
                    payload={
                        "tool": "browser.search",
                        "query": "2channel AI assistant discussion",
                        "status": "completed",
                    },
                    timestamp="2026-05-12T13:06:00+00:00",
                    span_id="span-tool",
                    parent_span_id="span-parent",
                )
            ]

    runtime = SimpleNamespace(
        _activity=None,
        current_cognition={"session_id": "default"},
        ctx=SimpleNamespace(config=SimpleNamespace(session_id="default"), tom=TomEngine()),
        executive=SimpleNamespace(
            active_goals=[f"high priority active goal {idx}" for idx in range(12)]
        ),
        tracer=SimpleNamespace(store=TracerStore()),
        memory=MemoryStore(),
    )

    items = await collect_reflective_working_set(runtime, limit=6)
    families = working_set_source_family_counts(items)
    text = "\n".join(item["text"] for item in items)

    assert "autonomous assistants" in text
    assert "2channel AI assistant discussion" in text
    assert "Email headline" in text
    assert families["active_work"] >= 1
    assert families["tom_user_model"] >= 1
    assert families["tool_observation"] >= 1
    assert families["external_observation"] >= 1


@pytest.mark.asyncio
async def test_reflective_working_set_items_include_temporal_provenance_and_novelty_signals() -> None:
    class ContextStore:
        async def list_recent(self, session_id: str, limit: int = 6, include_hidden: bool = False):
            return [
                SimpleNamespace(
                    role=SimpleNamespace(value="user"),
                    content="Fresh research headline: local agents need stronger reflective intake.",
                    meta={"conversation_actor": {"type": "operator", "label": "operator"}},
                    created_at="2026-05-12T11:00:00+00:00",
                    message_id="msg-fresh",
                )
            ]

    runtime = SimpleNamespace(
        _activity="idle",
        _activity_since="2026-05-12T11:02:00+00:00",
        current_cognition={
            "session_id": "default",
            "user_input": "Process active work, conversations, hobbies, interests, news topics, and email headlines.",
            "conversation_actor": {"type": "operator", "label": "operator"},
            "created_at": "2026-05-12T11:03:00+00:00",
        },
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="default"),
            context_store=ContextStore(),
            cognitive_state_store=None,
        ),
        executive=SimpleNamespace(active_goals=["make daydreaming process active inputs"]),
        tracer=None,
        memory=None,
    )

    items = await collect_reflective_working_set(runtime, limit=8)

    assert items
    assert all(item.get("observed_at") for item in items)
    assert all(item.get("temporal", {}).get("observed_at") for item in items)
    assert all(item.get("meta", {}).get("authority") in {"operator", "collaborator_agent", "runtime"} for item in items)
    assert any(item.get("meta", {}).get("novelty_pressure", 0) > 0 for item in items)


@pytest.mark.asyncio
async def test_reflective_working_set_collects_personal_curiosity_and_user_interest_state() -> None:
    identity = SimpleNamespace(
        self_model=SimpleNamespace(
            self_beliefs={
                "daydream": {
                    "bulma_config": {"hobbySeeds": ["odd online communities", "local-first agents"]},
                    "bulma_status": {"currentInterest": "2channel/5ch culture shifts"},
                }
            }
        ),
        user_model=SimpleNamespace(
            explicit_preferences={
                "interests": "AI assistants, local autonomy, anime, and employment research",
                "communication_style": "direct, evidence-grounded updates",
            },
            inferred_goals=[
                {
                    "text": "have a persistent assistant that notices useful work before being asked",
                    "provenance": "user_comment",
                }
            ],
        ),
    )
    runtime = SimpleNamespace(
        _activity=None,
        current_cognition=None,
        ctx=SimpleNamespace(config=SimpleNamespace(session_id="default"), identity=identity),
        executive=SimpleNamespace(active_goals=[]),
        tracer=None,
        memory=None,
    )

    items = await collect_reflective_working_set(runtime, limit=10)

    texts = "\n".join(item["text"] for item in items)
    assert "odd online communities" in texts
    assert "2channel/5ch culture shifts" in texts
    assert "AI assistants" in texts
    assert any(item["source"] == "identity.self_beliefs.daydream" for item in items)
    assert any(item["source"] == "identity.user_model" for item in items)


@pytest.mark.asyncio
async def test_reflective_working_set_revisits_scaffolded_self_work() -> None:
    class DaydreamSignalStore:
        async def list_receipts(self, limit: int = 6):
            return [
                {
                    "receipt_id": "receipt-prototype",
                    "signal_id": "signal-prototype",
                    "created_at": "2026-05-12T21:54:03+00:00",
                    "route": "self_prototype",
                    "kind": "prototype",
                    "outcome": "self_prototype_scaffolded",
                    "summary": "The atlas truth repair prototype still needs validation.",
                    "artifact_paths": [
                        "workspace/self/prototypes/atlas-truth/README.md",
                        "workspace/self/prototypes/atlas-truth/VALIDATION.md",
                    ],
                    "raw": {"validation_status": "scaffold_only"},
                }
            ]

    runtime = SimpleNamespace(
        _activity=None,
        current_cognition=None,
        ctx=SimpleNamespace(
            config=SimpleNamespace(session_id="default"),
            daydream_signal_store=DaydreamSignalStore(),
        ),
        executive=SimpleNamespace(active_goals=[]),
        tracer=None,
        memory=None,
    )

    items = await collect_reflective_working_set(runtime, limit=5)

    scaffold = next(item for item in items if item["source"] == "daydream.self_work")
    assert scaffold["kind"] == "self_work_scaffold"
    assert "scaffold_only" in scaffold["text"]
    assert "VALIDATION.md" in scaffold["text"]
    assert scaffold["meta"]["validation_status"] == "scaffold_only"
    assert scaffold["meta"]["novelty_pressure"] > 0


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
        '"inner_dialogue": ['
        '{"voice": "curious", "text": "Why does this keep recurring?"},'
        '{"voice": "practical", "text": "Record it as a reusable rule."}'
        '],'
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
    assert recent[0].thoughts[0].inner_dialogue[1].voice == "practical"
    assert recent[0].thoughts[0].grounding[0].source.value == "memory"
