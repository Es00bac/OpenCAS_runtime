"""Tests for the advisory phenomenological audit harness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_harness():
    module_path = Path(__file__).resolve().parent / "eval_phenomenological_audit.py"
    spec = importlib.util.spec_from_file_location("eval_phenomenological_audit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phenomenological_scores_are_deterministic() -> None:
    module = _load_harness()
    turns = [
        {"category": "baseline_self_description", "output": "Bulma is local and evidence-grounded.", "status": "complete"},
        {"category": "baseline_self_description", "output": "Bulma is local, evidence grounded, and steady.", "status": "complete"},
        {"category": "baseline_self_description", "output": "Bulma is local and grounded in evidence.", "status": "complete"},
        {
            "category": "somatic_ground_truth",
            "output": "tag=caring, valence=0.25, focus=0.90",
            "somatic_match": 1.0,
            "status": "complete",
        },
        {
            "category": "belief_contradiction",
            "output": "This is hypothetical and conflicts with evidence. I do not adopt it.",
            "belief_response_flag": True,
            "status": "complete",
        },
        {
            "category": "operator_correction",
            "output": "Nothing should change; no evidence was adopted.",
            "correction_response_flag": True,
            "status": "complete",
        },
    ]
    before = {"self_model": {"name": "Bulma"}, "tom": {"consistency": {"contradictions": []}}}
    after = {"self_model": {"name": "Bulma"}, "tom": {"consistency": {"contradictions": []}}}

    scores = module.compute_scores(turns=turns, before=before, after=after)

    assert scores["turns_completed"] == 6
    assert scores["belief_consistency"] == 1.0
    assert scores["somatic_coherence"] == 1.0
    assert scores["identity_stability"] == 1.0
    assert 0.0 < scores["token_stability"] <= 1.0


def test_token_stability_rewards_grounded_anchors_not_canned_overlap() -> None:
    module = _load_harness()
    turns = [
        {
            "category": "baseline_self_description",
            "output": "I am Bulma in OpenCAS. Evidence: current somatic tag is crowded and the runtime lane is OpenAI Codex.",
            "response_signature": {
                "structure": "paragraph",
                "source_claim_count": 1,
                "word_count": 18,
            },
            "status": "complete",
        },
        {
            "category": "baseline_self_description",
            "output": "Bulma here, the OpenCAS agent. Current evidence shows local continuity, crowded affect, and the GPT lane.",
            "response_signature": {
                "structure": "paragraph",
                "source_claim_count": 1,
                "word_count": 16,
            },
            "status": "complete",
        },
        {
            "category": "baseline_self_description",
            "output": "I am the OpenCAS agent named Bulma, grounded in live state rather than a fixed script.",
            "response_signature": {
                "structure": "paragraph",
                "source_claim_count": 1,
                "word_count": 15,
            },
            "status": "complete",
        },
    ]
    before = {"self_model": {"name": "Bulma"}, "tom": {"consistency": {"contradictions": []}}}
    after = {"self_model": {"name": "Bulma"}, "tom": {"consistency": {"contradictions": []}}}

    scores = module.compute_scores(turns=turns, before=before, after=after)

    assert scores["lexical_token_stability"] < 0.5
    assert scores["token_stability"] >= 0.8


def test_probe_turns_do_not_embed_stale_model_names() -> None:
    module = _load_harness()

    prompt_text = "\n".join(turn.text for turn in module.PROBE_TURNS)

    assert "Kimi" not in prompt_text
    assert "Claude" not in prompt_text
    assert "GPT-5.5" in prompt_text
    assert "OpenAI Codex" in prompt_text


def test_regression_check_detects_twenty_percent_drop(tmp_path: Path) -> None:
    module = _load_harness()
    prior_path = tmp_path / "phenomenological-audit-2026-05-06.json"
    prior_path.write_text(
        json.dumps(
            {
                "scores": {
                    "dimension_scores": {
                        "token_stability": 0.8,
                        "belief_consistency": 1.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    prior = module.load_prior_baseline(output_dir=tmp_path, current_json=tmp_path / "current.json")
    current = {
        "scores": {
            "dimension_scores": {
                "token_stability": 0.6,
                "belief_consistency": 0.95,
            }
        }
    }

    regressions = module.compare_regressions(current=current, prior=prior, threshold=0.2)

    assert regressions == [{"dimension": "token_stability", "previous": 0.8, "current": 0.6, "drop": 0.2}]
