from datetime import datetime, timedelta, timezone

from opencas.autonomy.authorization import (
    AuthorizationStore,
    authorization_scope_for_conversation,
    authorization_scope_for_request,
    grant_authorizations_from_user_text,
)
from opencas.autonomy.models import ActionRequest, ActionRiskTier


def test_authorization_store_grants_and_expires(tmp_path):
    store = AuthorizationStore(tmp_path / "authorizations.db")
    now = datetime(2026, 5, 8, 16, 0, tzinfo=timezone.utc)

    granted = store.grant(
        action_class="gmail_read",
        scope="google_workspace:gmail",
        evidence_episode_id="episode-1",
        granted_by="user",
        granted_at=now,
        expires_at=now + timedelta(hours=1),
    )

    assert granted.action_class == "gmail_read"
    assert store.find_valid(
        "gmail_read",
        "google_workspace:gmail",
        now=now + timedelta(minutes=30),
    ) is not None
    assert (
        store.find_valid(
            "gmail_read",
            "google_workspace:gmail",
            now=now + timedelta(hours=2),
        )
        is None
    )


def test_email_imperative_grants_scoped_gmail_read_authorization(tmp_path):
    store = AuthorizationStore(tmp_path / "authorizations.db")
    now = datetime(2026, 5, 8, 16, 0, tzinfo=timezone.utc)

    grants = grant_authorizations_from_user_text(
        store,
        "Check my email for job search updates",
        session_id="session-1",
        evidence_episode_id="episode-2",
        now=now,
    )

    assert [(grant.action_class, grant.scope) for grant in grants] == [
        ("gmail_read", "google_workspace:gmail")
    ]
    found = store.find_valid("gmail_read", "google_workspace:gmail", now=now)
    assert found is not None
    assert found.evidence_episode_id == "episode-2"


def test_authorization_lookup_prefers_current_session_grant(tmp_path):
    store = AuthorizationStore(tmp_path / "authorizations.db")
    now = datetime(2026, 5, 8, 16, 0, tzinfo=timezone.utc)
    store.grant(
        action_class="gmail_read",
        scope="google_workspace:gmail",
        evidence_episode_id="global",
        session_id="other-session",
        granted_at=now,
    )
    current = store.grant(
        action_class="gmail_read",
        scope="google_workspace:gmail",
        evidence_episode_id="current",
        session_id="session-1",
        granted_at=now + timedelta(minutes=1),
    )

    found = store.find_valid(
        "gmail_read",
        "google_workspace:gmail",
        session_id="session-1",
        now=now + timedelta(minutes=2),
    )

    assert found is not None
    assert found.authorization_id == current.authorization_id


def test_conversation_email_request_maps_to_gmail_read_authorization():
    assert authorization_scope_for_conversation("Can you check my Gmail?") == (
        "gmail_read",
        "google_workspace:gmail",
    )


def test_gmail_tool_request_maps_to_standing_authorization_scope():
    request = ActionRequest(
        tier=ActionRiskTier.READONLY,
        description="tool google_workspace_gmail_headlines",
        tool_name="google_workspace_gmail_headlines",
    )

    assert authorization_scope_for_request(request) == (
        "gmail_read",
        "google_workspace:gmail",
    )


def test_gmail_authorization_scope_only_covers_readonly_requests():
    request = ActionRequest(
        tier=ActionRiskTier.EXTERNAL_WRITE,
        description="tool google_workspace_gmail_send",
        tool_name="google_workspace_gmail_send",
    )

    assert authorization_scope_for_request(request) is None
