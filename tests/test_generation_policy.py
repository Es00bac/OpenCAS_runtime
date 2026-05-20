from types import SimpleNamespace

import pytest

from opencas.generation.policy import (
    GenerationDomain,
    GenerationPhase,
    GenerationProfile,
    LearnedGenerationPreset,
    LearnedGenerationPresetSettings,
    GenerationPolicyConfig,
    GenerationPolicyRequest,
    GenerationPolicyResolver,
)


def test_brainstorm_policy_is_creative_but_preserves_single_primary_knob() -> None:
    resolver = GenerationPolicyResolver(GenerationPolicyConfig())

    policy = resolver.resolve(
        GenerationPolicyRequest(
            phase=GenerationPhase.BRAINSTORM,
            domain=GenerationDomain.CREATIVE_WRITING,
            novelty_pressure=0.8,
        ),
        provider_id="openai-codex",
        model_ref="openai/gpt-5.5",
    )

    assert policy.profile_name == "brainstorm"
    assert policy.effective["temperature"] >= 0.85
    assert "top_p" not in policy.effective
    assert policy.telemetry["phase"] == "brainstorm"
    assert policy.telemetry["domain"] == "creative_writing"


def test_continuity_pressure_lowers_draft_entropy() -> None:
    resolver = GenerationPolicyResolver(GenerationPolicyConfig())

    loose = resolver.resolve(
        GenerationPolicyRequest(
            phase=GenerationPhase.DRAFT,
            domain=GenerationDomain.CREATIVE_WRITING,
            novelty_pressure=0.6,
            continuity_pressure=0.0,
        ),
        provider_id="openai-codex",
        model_ref="openai/gpt-5.5",
    )
    continuous = resolver.resolve(
        GenerationPolicyRequest(
            phase=GenerationPhase.DRAFT,
            domain=GenerationDomain.CREATIVE_WRITING,
            novelty_pressure=0.6,
            continuity_pressure=1.0,
            long_form=True,
        ),
        provider_id="openai-codex",
        model_ref="openai/gpt-5.5",
    )

    assert continuous.effective["temperature"] < loose.effective["temperature"]
    assert "continuity_pressure" in continuous.telemetry["adjustments"]


def test_verify_policy_clamps_somatic_arousal_and_structured_output() -> None:
    resolver = GenerationPolicyResolver(GenerationPolicyConfig())

    policy = resolver.resolve(
        GenerationPolicyRequest(
            phase=GenerationPhase.VERIFY,
            domain=GenerationDomain.SAFETY,
            structured_output=True,
            somatic=SimpleNamespace(arousal=1.0, fatigue=0.0, focus=0.0, tension=0.0, certainty=1.0),
        ),
        provider_id="openai-codex",
        model_ref="openai/gpt-5.5",
    )

    assert policy.effective["temperature"] <= 0.2
    assert "structured_output_clamp" in policy.telemetry["adjustments"]
    assert "somatic_delta" in policy.telemetry


def test_local_provider_can_receive_experimental_sampler_fields() -> None:
    config = GenerationPolicyConfig(
        local_sampler_experiments={"min_p": True, "top_h": True, "mirostat": False}
    )
    resolver = GenerationPolicyResolver(config)

    policy = resolver.resolve(
        GenerationPolicyRequest(
            phase=GenerationPhase.DIVERGE,
            domain=GenerationDomain.CREATIVE_WRITING,
            novelty_pressure=1.0,
        ),
        provider_id="local-llama",
        model_ref="local-llama/qwen-story",
    )

    assert policy.effective["temperature"] >= 0.95
    assert "min_p" in policy.effective
    assert "top_h" in policy.effective


@pytest.mark.parametrize("phase", [GenerationPhase.DAYDREAM, GenerationPhase.DREAM])
def test_reflective_phases_carry_proposal_only_authority(phase: GenerationPhase) -> None:
    resolver = GenerationPolicyResolver(GenerationPolicyConfig())

    policy = resolver.resolve(
        GenerationPolicyRequest(phase=phase, domain=GenerationDomain.AUTONOMY),
        provider_id="openai-codex",
        model_ref="openai/gpt-5.5",
    )

    assert policy.telemetry["authority"] == "reflective_proposal_only"
    assert policy.effective["temperature"] > 0.6


def test_daydream_policy_adds_bounded_sampler_variation_for_eval_trace() -> None:
    resolver = GenerationPolicyResolver(GenerationPolicyConfig())

    first = resolver.resolve(
        GenerationPolicyRequest(
            phase=GenerationPhase.DAYDREAM,
            domain=GenerationDomain.AUTONOMY,
            variation_seed="daydream-eval-a",
        ),
        provider_id="local-llama",
        model_ref="local-llama/reflective",
    )
    second = resolver.resolve(
        GenerationPolicyRequest(
            phase=GenerationPhase.DAYDREAM,
            domain=GenerationDomain.AUTONOMY,
            variation_seed="daydream-eval-b",
        ),
        provider_id="local-llama",
        model_ref="local-llama/reflective",
    )

    for policy in (first, second):
        assert 0.6 <= policy.effective["temperature"] <= 0.95
        assert 0.78 <= policy.effective["top_p"] <= 0.98
        assert 30 <= policy.effective["top_k"] <= 110
        assert "bounded_sampler_variation" in policy.telemetry["adjustments"]
        assert "sampling_variation_seed" in policy.telemetry
        assert policy.telemetry["authority"] == "reflective_proposal_only"
    first_tuple = (
        first.effective["temperature"],
        first.effective["top_p"],
        first.effective["top_k"],
    )
    second_tuple = (
        second.effective["temperature"],
        second.effective["top_p"],
        second.effective["top_k"],
    )
    assert first_tuple != second_tuple


def test_mature_learned_work_type_preset_becomes_baseline() -> None:
    config = GenerationPolicyConfig(
        learned_presets=LearnedGenerationPresetSettings(
            enabled=True,
            min_evidence_count=3,
            min_success_rate=0.6,
        ),
        learned_preset_records={
            "chronicle-scene-draft": LearnedGenerationPreset(
                preset_id="chronicle-scene-draft",
                work_type="chronicle_scene_drafting",
                phase=GenerationPhase.DRAFT,
                domain=GenerationDomain.CREATIVE_WRITING,
                profile=GenerationProfile(
                    temperature=0.72,
                    summary="Familiar Chronicle scene drafting balance.",
                ),
                evidence_count=8,
                success_count=7,
                failure_count=1,
                source="offline_eval",
            )
        },
    )
    resolver = GenerationPolicyResolver(config)

    policy = resolver.resolve(
        GenerationPolicyRequest(
            phase=GenerationPhase.DRAFT,
            domain=GenerationDomain.CREATIVE_WRITING,
            work_type="chronicle_scene_drafting",
            novelty_pressure=0.0,
        ),
        provider_id="openai-codex",
        model_ref="openai/gpt-5.5",
    )

    assert policy.profile_name == "learned:chronicle-scene-draft"
    assert policy.effective["temperature"] == 0.72
    assert policy.telemetry["learned_preset_id"] == "chronicle-scene-draft"
    assert policy.telemetry["learned_preset_success_rate"] == pytest.approx(0.875)


def test_long_form_creative_policy_declares_project_memory_focus() -> None:
    resolver = GenerationPolicyResolver(GenerationPolicyConfig())

    policy = resolver.resolve(
        GenerationPolicyRequest(
            phase=GenerationPhase.DRAFT,
            domain=GenerationDomain.CREATIVE_WRITING,
            long_form=True,
            continuity_pressure=0.9,
            work_type="series_book_three",
        ),
        provider_id="openai-codex",
        model_ref="openai/gpt-5.5",
    )

    assert "project_memory" in policy.telemetry["memory_focus"]
    assert "autobiographical_memory" in policy.telemetry["memory_focus"]
    assert "affective_writing_context" in policy.telemetry["memory_focus"]
