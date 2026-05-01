"""Tests for identity hygiene utilities."""

from opencas.identity.text_hygiene import (
    collapse_recursive_identity_text,
    has_recursive_identity_loop,
    sanitize_identity_structure,
    sanitize_identity_text,
)


def test_collapse_recursive_identity_text_normalizes_repeating_loops() -> None:
    value = (
        "digesting the unfinished project thread around implement daydreaming: returning "
        "to returning to returning to follow the same thread"
    )
    collapsed = collapse_recursive_identity_text(value)
    assert "returning to returning" not in collapsed.lower()
    assert "returning to" in collapsed.lower()


def test_sanitize_identity_text_replaces_fixation_terms() -> None:
    value = "keep returning to thread and drifted"
    sanitized = sanitize_identity_text(value)
    assert "returning" not in sanitized.lower()
    assert "thread" not in sanitized.lower()
    assert "drifted" not in sanitized.lower()
    assert "revisiting" in sanitized.lower()
    assert "path" in sanitized.lower()
    assert "shifted" in sanitized.lower()


def test_has_recursive_identity_loop_detects_recursive_focus_labels() -> None:
    assert has_recursive_identity_loop("returning to returning to returning")
    assert has_recursive_identity_loop("thread thread")
    assert not has_recursive_identity_loop("threaded path to progress")


def test_sanitize_identity_structure_preserves_nested_shapes() -> None:
    payload = {
        "label": "returning to returning to return",
        "items": [
            "drifted drifted",
            {"term": "thread", "count": 3},
            1,
        ],
    }
    sanitized = sanitize_identity_structure(payload)
    assert isinstance(sanitized, dict)
    assert sanitized["label"] == "revisiting to return"
    assert sanitized["items"][0] == "shifted"
    assert sanitized["items"][1]["term"] == "path"
    assert sanitized["items"][2] == 1
