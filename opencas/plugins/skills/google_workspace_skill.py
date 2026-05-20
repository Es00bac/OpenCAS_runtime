"""Google Workspace CLI skill metadata for OpenCAS."""

from __future__ import annotations

from opencas.plugins import SkillEntry

GOOGLE_WORKSPACE_TOOL_NAMES = [
    "google_workspace_auth_status",
    "google_workspace_schema",
    "google_workspace_readonly_api",
    "google_workspace_gmail_headlines",
    "google_workspace_gmail_get_message",
    "google_workspace_calendar_schedule",
    "google_workspace_calendar_dedupe",
    "google_workspace_drive_search",
]

SKILL_ENTRY = SkillEntry(
    skill_id="google_workspace_skill",
    name="Google Workspace CLI",
    description=(
        "Use the local gws CLI through bounded Google Workspace tools for "
        "Gmail, Calendar, Drive, Docs, Sheets, Slides, and People lookups. "
        "Inspect gws schemas or help before claiming an unfamiliar command "
        "cannot be used. Calendar duplicate cleanup must use the dry-run "
        "dedupe path before applying deletions."
    ),
    capabilities=GOOGLE_WORKSPACE_TOOL_NAMES,
    meta={
        "cli": "gws",
        "scope": "read-oriented Google Workspace access plus bounded calendar duplicate cleanup",
        "learning_rule": "Use schema/help discovery for unfamiliar gws methods before asking the operator.",
    },
)
