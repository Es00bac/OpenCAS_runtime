"""Bootstrap TUI persistence and config-building helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from opencas.bootstrap import BootstrapConfig
from opencas.bootstrap.tui_state import WizardState


def compose_user_bio(state: WizardState) -> str:
    """Compose a rich user_bio from questionnaire answers if none was given."""
    if state.user_bio:
        return state.user_bio
    parts: List[str] = []
    if state.user_job:
        parts.append(f"Role: {state.user_job}")
    if state.user_interests:
        parts.append(f"Interests: {state.user_interests}")
    if state.vision_main_help:
        parts.append(f"Wants help with: {state.vision_main_help}")
    if state.vision_engagement_style:
        parts.append(f"Engagement preference: {state.vision_engagement_style.replace('_', ' ')}")
    if state.vision_success_six_months:
        parts.append(f"6-month success: {state.vision_success_six_months}")
    if state.learning_preference:
        parts.append(f"Learning preference: {state.learning_preference.replace('_', ' ')}")
    if state.feedback_style:
        parts.append(f"Feedback preference: {state.feedback_style.replace('_', ' ')}")
    if state.help_style:
        parts.append(f"Help preference: {state.help_style.replace('_', ' ')}")
    if state.happy_makers:
        parts.append(f"Energized by: {state.happy_makers}")
    if state.sad_drainers:
        parts.append(f"Drained by: {state.sad_drainers}")
    if state.angry_triggers:
        parts.append(f"Frustrated by: {state.angry_triggers}")
    if state.agent_avoid:
        parts.append(f"Avoid: {state.agent_avoid}")
    if state.bad_day_help:
        parts.append(f"Recovery: {state.bad_day_help}")
    return "\n".join(parts)


def questionnaire_payload(state: WizardState) -> Dict[str, Dict[str, object]]:
    return {
        "vision": {
            "main_help": state.vision_main_help,
            "engagement_style": state.vision_engagement_style,
            "success_six_months": state.vision_success_six_months,
            "working_notes": state.vision_working_notes,
        },
        "about_me": {
            "job": state.user_job,
            "interests": state.user_interests,
            "communication_style": state.user_comm_style,
        },
        "learning_style": {
            "preference": state.learning_preference,
            "feedback_style": state.feedback_style,
            "help_style": state.help_style,
            "collab_pair": state.collab_pair,
            "collab_async": state.collab_async,
            "collab_backforth": state.collab_backforth,
            "collab_minimal": state.collab_minimal,
        },
        "emotional_landscape": {
            "happy_makers": state.happy_makers,
            "sad_drainers": state.sad_drainers,
            "angry_triggers": state.angry_triggers,
            "agent_avoid": state.agent_avoid,
            "bad_day_help": state.bad_day_help,
        },
        "initial_goals": {
            "goal_1": state.goal_1,
            "goal_2": state.goal_2,
            "goal_3": state.goal_3,
            "timeframe": state.goal_timeframe,
        },
        "persona_theme": {
            "accent": state.persona_accent,
        },
    }


def save_questionnaire(state: WizardState, state_dir: Path) -> Path:
    path = state_dir / "bootstrap_questionnaire.json"
    path.write_text(json.dumps(questionnaire_payload(state), indent=2), encoding="utf-8")
    return path


def build_bootstrap_config(state: WizardState) -> BootstrapConfig:
    extra_roots = [r.strip() for r in state.workspace_extra.split(",") if r.strip()]
    state_dir = Path(state.state_dir)
    return BootstrapConfig(
        state_dir=state_dir,
        session_id=None,
        agent_profile_id=state.agent_profile_id,
        workspace_root=Path(state.workspace_root),
        workspace_roots=[Path(r) for r in extra_roots],
        default_llm_model=state.default_llm_model,
        embedding_model_id=state.embedding_model_id,
        provider_config_path=Path(state.provider_config_path) if state.provider_config_path else None,
        provider_env_path=Path(state.provider_env_path) if state.provider_env_path else None,
        credential_source_config_path=Path(state.credential_source_config) if state.credential_source_config else None,
        credential_source_env_path=None,
        credential_profile_ids=state.selected_profiles,
        credential_env_keys=state.credential_env_keys,
        persona_name=state.persona_name,
        user_name=state.user_name or None,
        user_bio=compose_user_bio(state) or None,
    )
