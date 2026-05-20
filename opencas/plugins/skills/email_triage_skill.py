"""Email triage skill metadata for OpenCAS."""

from __future__ import annotations

from opencas.plugins import SkillEntry

EMAIL_TRIAGE_TOOL_NAMES = [
    "himalaya_email_accounts",
    "himalaya_email_headlines",
    "himalaya_email_read_message",
    "google_workspace_auth_status",
    "google_workspace_gmail_headlines",
    "google_workspace_gmail_get_message",
    "google_workspace_readonly_api",
]

SKILL_ENTRY = SkillEntry(
    skill_id="email_triage_skill",
    name="Email triage with Himalaya and Google Workspace",
    description=(
        "Use the locally configured email tools when the operator asks about email, inbox, Gmail, "
        "or messages. Prefer Himalaya for local IMAP mailbox headlines and message previews; use "
        "Google Workspace/gws for Gmail-specific search, metadata, and Workspace cross-service context. "
        "If one backend is unavailable or unauthenticated, try the other read-only backend before asking "
        "the operator to reconfigure email. Do not send, delete, move, or mark messages read from this skill."
    ),
    capabilities=EMAIL_TRIAGE_TOOL_NAMES,
    meta={
        "cli_backends": ["himalaya", "gws"],
        "read_only_tools": [
            "himalaya_email_accounts",
            "himalaya_email_headlines",
            "himalaya_email_read_message",
            "google_workspace_auth_status",
            "google_workspace_gmail_headlines",
            "google_workspace_gmail_get_message",
            "google_workspace_readonly_api",
        ],
        "routing_rule": "Email/inbox/Gmail requests should surface both Himalaya and gws read-only tools when available.",
        "learning_rule": "If email handling discovers a durable backend preference, auth caveat, or query pattern, preserve it as a learned skill rather than hard-coding it.",
    },
)
