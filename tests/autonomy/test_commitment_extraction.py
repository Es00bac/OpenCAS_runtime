"""Unit tests for conversational self-commitment extraction."""

from opencas.autonomy.commitment_extraction import extract_self_commitments


def test_extract_self_commitment_normalizes_direct_object() -> None:
    commitments = extract_self_commitments(
        "I need a short rest. I'll come back to the dashboard memory atlas later."
    )

    assert len(commitments) == 1
    assert commitments[0].content == "Return to the dashboard memory atlas"
    assert commitments[0].normalization_source == "direct_object"
    assert commitments[0].confidence == 0.9


def test_extract_self_commitment_uses_prior_sentence_context_for_pronoun_deferral() -> None:
    commitments = extract_self_commitments(
        "The next step is finish the scheduler resume path. I'll come back to this after I rest."
    )

    assert len(commitments) == 1
    assert commitments[0].content == "Finish the scheduler resume path"
    assert commitments[0].normalization_source == "prior_sentence_context"
    assert commitments[0].confidence == 0.72


def test_extract_self_commitment_skips_vague_reflective_language() -> None:
    commitments = extract_self_commitments(
        "This is interesting to think about. We can revisit the broader philosophy later."
    )

    assert commitments == []
