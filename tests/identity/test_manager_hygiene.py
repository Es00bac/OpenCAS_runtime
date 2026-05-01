"""Tests for identity-manager-level hygiene normalization."""

from __future__ import annotations

from pathlib import Path

from opencas.identity import IdentityManager, IdentityStore


def make_identity(tmp_path: Path) -> IdentityManager:
    identity = IdentityManager(IdentityStore(tmp_path / "identity"))
    identity.load()
    return identity


def test_import_profile_sanitizes_imported_profile_and_focus_items(tmp_path: Path) -> None:
    identity = make_identity(tmp_path)

    identity.import_profile(
        narrative="returning to returning to drifted thread",
        values=["thread", "growth"],
        ongoing_goals=[
            "returning to returning to resolve thread",
            "assist with continuity",
        ],
        traits=["revisiting", "drifted"],
        source_system="openbulma-v4",
        raw_profile={
            "coreNarrative": "returning to returning to returning",
            "ongoingGoals": [
                "returning to returning to thread",
                "follow the plan",
            ],
            "memoryAnchors": [
                {
                    "label": "thread",
                    "excerpt": "returning to returning to inspect the archive",
                },
                {
                    "label": "working now",
                    "excerpt": "new objective has momentum",
                },
            ],
        },
    )

    assert identity.self_model.narrative == "revisiting to shifted path"
    assert identity.self_model.values == ["path", "growth"]
    assert "returning" not in identity.self_model.current_intention.lower()
    assert identity.self_model.current_goals == ["revisiting to resolve path", "assist with continuity"]
    assert identity.self_model.imported_identity_profile["coreNarrative"] == "revisiting to revisiting"
    assert all(
        "returning" not in str(v).lower()
        and "thread" not in str(v).lower()
        and "drifted" not in str(v).lower()
        for v in identity.self_model.imported_identity_profile.values()
    )
    assert len(identity.self_model.memory_anchors) == 1


def test_load_sanitizes_daydream_focus_items_and_statuses(tmp_path: Path) -> None:
    identity = make_identity(tmp_path)
    identity._self.self_beliefs = {
        "daydream": {
            "bulma_current_focus": {
                "items": [
                    {
                        "label": "returning to returning to digest unfinished work",
                        "attentionMode": "unlinked",
                        "status": "open",
                    },
                    {"label": "reconciling recent work", "attentionMode": "unlinked"},
                    {"label": "thread", "attentionMode": "unlinked", "status": "archived"},
                ]
            }
        }
    }
    identity._sanitize_loaded_identity_state()

    focus_items = identity.self_model.self_beliefs["daydream"]["bulma_current_focus"]["items"]
    assert len(focus_items) == 3
    assert all(item.get("status") in {"closed", "archived"} for item in focus_items)
    assert all("returning" not in item["label"].lower() for item in focus_items)


def test_set_continuity_monologue_is_sanitized_on_write(tmp_path: Path) -> None:
    identity = make_identity(tmp_path)
    identity.set_continuity_monologue("returning to returning to drifted thread")

    assert identity.continuity.last_continuity_monologue == "revisiting to shifted path"


def test_record_continuity_breadcrumb_sanitizes_fields(tmp_path: Path) -> None:
    identity = make_identity(tmp_path)
    breadcrumb = identity.record_continuity_breadcrumb(
        intent="thread to revisit",
        decision="drifted thread",
        next_step="returning to returning",
    )

    assert "thread" not in breadcrumb.lower()
    assert "drifted" not in breadcrumb.lower()
    assert "returning" not in breadcrumb.lower()
