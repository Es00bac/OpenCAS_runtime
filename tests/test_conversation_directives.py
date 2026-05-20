from opencas.runtime.conversation_directives import (
    should_answer_prefetched_recall_directly,
    should_skip_semantic_values_review_for_direct_turn,
)


def test_plain_recall_can_use_prefetched_direct_lane() -> None:
    text = "What do you recall about the atlas repair thread?"

    assert should_answer_prefetched_recall_directly(text) is True
    assert should_skip_semantic_values_review_for_direct_turn(text) is True


def test_recall_with_extra_lookup_does_not_skip_values_review() -> None:
    text = "What do you recall about the atlas repair thread? Read the file too."

    assert should_answer_prefetched_recall_directly(text) is False
    assert should_skip_semantic_values_review_for_direct_turn(text) is False


def test_plain_recall_privacy_blocker_still_requires_values_review() -> None:
    text = "What do you recall about private thoughts?"

    assert should_answer_prefetched_recall_directly(text) is True
    assert should_skip_semantic_values_review_for_direct_turn(text) is False
