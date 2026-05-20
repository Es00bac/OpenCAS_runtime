from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencas.runtime import reflection_runtime


@pytest.mark.asyncio
async def test_fresh_reflective_working_set_can_trigger_daydream_below_boredom_threshold(monkeypatch) -> None:
    class FakeBoredom:
        def compute_boredom(self, now):
            return 0.1

        def compute_motivation(self, *, somatic_readiness, now):
            return 0.33

    class FakeDaydream:
        def __init__(self):
            self.calls = 0

        async def generate(self, *, goals, tension, working_set):
            self.calls += 1
            assert any(item["source"] == "context_store" for item in working_set)
            assert any(item["source"] == "telemetry" for item in working_set)
            return [], []

    async def fake_working_set(runtime):
        return [
            {
                "kind": "active_turn",
                "source": "runtime.current_cognition",
                "label": "Current operator turn",
                "text": "Make daydreaming process active work and conversations.",
                "created_at": "2026-05-12T11:00:00+00:00",
                "observed_at": "2026-05-12T11:00:05+00:00",
                "weight": 0.98,
                "meta": {"novelty_pressure": 0.95, "authority": "operator"},
            },
            {
                "kind": "recent_dialogue",
                "source": "context_store",
                "label": "Recent user",
                "text": "Include hobbies, interests, news topics, and email headlines.",
                "created_at": "2026-05-12T11:01:00+00:00",
                "observed_at": "2026-05-12T11:01:05+00:00",
                "weight": 0.82,
                "meta": {"novelty_pressure": 0.8, "authority": "operator"},
            },
            {
                "kind": "telemetry:tool_call",
                "source": "telemetry",
                "label": "web_fetch",
                "text": "Fetched a research page with new evidence.",
                "created_at": "2026-05-12T11:02:00+00:00",
                "observed_at": "2026-05-12T11:02:05+00:00",
                "weight": 0.72,
                "meta": {"novelty_pressure": 0.7, "authority": "runtime"},
            },
        ]

    monkeypatch.setattr(reflection_runtime, "collect_reflective_working_set", fake_working_set)

    fake_daydream = FakeDaydream()

    async def emit_appraisal_event(*args, **kwargs):
        return None

    runtime = SimpleNamespace(
        _last_daydream_time=None,
        boredom=FakeBoredom(),
        executive=SimpleNamespace(active_goals=["process fresh inputs"]),
        daydream=fake_daydream,
        ctx=SimpleNamespace(
            somatic=SimpleNamespace(
                state=SimpleNamespace(
                    tension=0.25,
                    fatigue=0.05,
                    energy=0.9,
                    focus=0.9,
                    arousal=0.4,
                    valence=0.1,
                    certainty=0.8,
                    somatic_tag="ready",
                ),
                emit_appraisal_event=emit_appraisal_event,
            ),
            daydream_store=None,
            identity=None,
        ),
        reflection_evaluator=SimpleNamespace(),
        conflict_registry=None,
        memory=None,
        _trace=lambda *args, **kwargs: None,
    )

    result = await reflection_runtime.run_runtime_daydream_inner(runtime)

    assert fake_daydream.calls == 1
    assert result["quality_status"] != "skipped"
    assert result["motivation"] >= result["motivation_threshold"]
    assert result["working_set_novelty_pressure"] > 0


@pytest.mark.asyncio
async def test_diverse_reflective_working_set_adds_small_motivation_bonus(monkeypatch) -> None:
    class FakeBoredom:
        def compute_boredom(self, now):
            return 0.1

        def compute_motivation(self, *, somatic_readiness, now):
            return 0.2

    class FakeDaydream:
        def __init__(self):
            self.calls = 0

        async def generate(self, *, goals, tension, working_set):
            self.calls += 1
            return [], []

    async def fake_working_set(runtime):
        return [
            {
                "kind": kind,
                "source": source,
                "label": family,
                "text": f"{family} evidence",
                "weight": 0.7,
                "meta": {
                    "source_family": family,
                    "novelty_pressure": 0.65,
                    "authority": authority,
                },
            }
            for kind, source, family, authority in [
                ("recent_dialogue", "context_store", "conversation", "operator"),
                ("active_goal", "executive.active_goals", "active_work", "runtime"),
                ("tom_user_belief", "tom.user_beliefs", "tom_user_model", "runtime"),
                ("external_observation", "memory", "external_observation", "runtime"),
                ("self_work_scaffold", "daydream.self_work", "self_work", "runtime"),
                ("attention", "cognitive_state.attention", "cognitive_state", "runtime"),
            ]
        ]

    monkeypatch.setattr(reflection_runtime, "collect_reflective_working_set", fake_working_set)

    fake_daydream = FakeDaydream()

    async def emit_appraisal_event(*args, **kwargs):
        return None

    runtime = SimpleNamespace(
        _last_daydream_time=None,
        boredom=FakeBoredom(),
        executive=SimpleNamespace(active_goals=["process diverse live intake"]),
        daydream=fake_daydream,
        ctx=SimpleNamespace(
            somatic=SimpleNamespace(
                state=SimpleNamespace(
                    tension=0.35,
                    fatigue=0.1,
                    energy=0.85,
                    focus=0.8,
                    arousal=0.4,
                    valence=0.1,
                    certainty=0.8,
                    somatic_tag="ready",
                ),
                emit_appraisal_event=emit_appraisal_event,
            ),
            daydream_store=None,
            identity=None,
        ),
        reflection_evaluator=SimpleNamespace(),
        conflict_registry=None,
        memory=None,
        _trace=lambda *args, **kwargs: None,
    )

    result = await reflection_runtime.run_runtime_daydream_inner(runtime)

    assert fake_daydream.calls == 1
    assert result["quality_status"] != "skipped"
    assert result["working_set_source_family_diversity_bonus"] > 0
    assert result["motivation"] >= result["motivation_threshold"]


@pytest.mark.asyncio
async def test_high_fatigue_still_blocks_daydream_even_with_fresh_inputs(monkeypatch) -> None:
    class FakeBoredom:
        def compute_boredom(self, now):
            return 0.1

        def compute_motivation(self, *, somatic_readiness, now):
            return 0.33

    async def fake_working_set(runtime):
        return [
            {
                "kind": "active_turn",
                "source": "runtime.current_cognition",
                "label": "Current operator turn",
                "text": "Fresh work exists.",
                "created_at": "2026-05-12T11:00:00+00:00",
                "observed_at": "2026-05-12T11:00:05+00:00",
                "weight": 0.98,
                "meta": {"novelty_pressure": 1.0, "authority": "operator"},
            }
        ]

    monkeypatch.setattr(reflection_runtime, "collect_reflective_working_set", fake_working_set)

    runtime = SimpleNamespace(
        _last_daydream_time=None,
        boredom=FakeBoredom(),
        executive=SimpleNamespace(active_goals=[]),
        daydream=SimpleNamespace(generate=lambda *args, **kwargs: None),
        ctx=SimpleNamespace(
            somatic=SimpleNamespace(
                state=SimpleNamespace(
                    tension=0.25,
                    fatigue=0.9,
                    energy=0.9,
                    focus=0.9,
                    arousal=0.4,
                    valence=0.1,
                    certainty=0.8,
                    somatic_tag="exhausted",
                )
            )
        ),
    )

    result = await reflection_runtime.run_runtime_daydream_inner(runtime)

    assert result["quality_status"] == "skipped"
    assert result["skip_reason"] == "somatic_fatigue"
    assert result["working_set_novelty_pressure"] > 0
