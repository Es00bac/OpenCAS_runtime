"""Tests for the identity module."""

from pathlib import Path
from opencas.identity import IdentityManager, IdentityStore


def test_identity_manager_load_and_save(tmp_path: Path) -> None:
    store = IdentityStore(tmp_path)
    mgr = IdentityManager(store)
    mgr.load()

    assert mgr.self_model.name == "OpenCAS"
    mgr.self_model.current_goals.append("learn user habits")
    mgr.save()

    mgr2 = IdentityManager(store)
    mgr2.load()
    assert mgr2.self_model.current_goals == ["learn user habits"]


def test_identity_continuity_boot(tmp_path: Path) -> None:
    store = IdentityStore(tmp_path)
    mgr = IdentityManager(store)
    mgr.load()

    mgr.record_boot(session_id="sess-1")
    assert mgr.continuity.boot_count == 1
    assert mgr.continuity.last_session_id == "sess-1"

    mgr.record_boot(session_id="sess-2")
    assert mgr.continuity.boot_count == 2


def test_identity_user_trust(tmp_path: Path) -> None:
    store = IdentityStore(tmp_path)
    mgr = IdentityManager(store)
    mgr.load()

    assert mgr.user_model.trust_level == 0.5
    mgr.adjust_trust(0.2)
    assert mgr.user_model.trust_level == 0.7
    mgr.adjust_trust(-0.5)
    assert mgr.user_model.trust_level == 0.2
    mgr.adjust_trust(2.0)
    assert mgr.user_model.trust_level == 1.0


def test_identity_beliefs_and_preferences(tmp_path: Path) -> None:
    store = IdentityStore(tmp_path)
    mgr = IdentityManager(store)
    mgr.load()

    mgr.update_self_belief("location", "local")
    assert mgr.self_model.self_beliefs["location"] == "local"

    mgr.add_user_preference("language", "english")
    assert mgr.user_model.explicit_preferences["language"] == "english"

    mgr2 = IdentityManager(store)
    mgr2.load()
    assert mgr2.self_model.self_beliefs["location"] == "local"
    assert mgr2.user_model.explicit_preferences["language"] == "english"
