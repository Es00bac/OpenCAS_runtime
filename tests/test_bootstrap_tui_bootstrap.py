import json
from pathlib import Path

from opencas.bootstrap.tui_bootstrap import build_bootstrap_config, compose_user_bio, questionnaire_payload, save_questionnaire
from opencas.bootstrap.tui_state import WizardState


def test_compose_user_bio_uses_questionnaire_fallbacks():
    state = WizardState()
    state.user_job = "Engineer"
    state.vision_main_help = "Ship resilient systems"
    state.bad_day_help = "Be concise"

    bio = compose_user_bio(state)

    assert "Role: Engineer" in bio
    assert "Wants help with: Ship resilient systems" in bio
    assert "Recovery: Be concise" in bio


def test_questionnaire_payload_and_save_round_trip(tmp_path: Path):
    state = WizardState()
    state.goal_1 = "Finish setup"
    state.persona_accent = "teal"

    payload = questionnaire_payload(state)
    saved = save_questionnaire(state, tmp_path)

    assert payload["initial_goals"]["goal_1"] == "Finish setup"
    assert saved == tmp_path / "bootstrap_questionnaire.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["persona_theme"]["accent"] == "teal"


def test_build_bootstrap_config_uses_workspace_and_provider_fields():
    state = WizardState()
    state.state_dir = "/tmp/opencas-state"
    state.workspace_root = "/tmp/workspace"
    state.workspace_extra = "/tmp/extra-a, /tmp/extra-b"
    state.provider_config_path = "/tmp/provider-config.json"
    state.provider_env_path = "/tmp/provider.env"
    state.credential_source_config = "/tmp/source-config.json"
    state.credential_env_keys = ["OPENAI_API_KEY"]
    state.selected_profiles = ["anthropic-main"]
    state.user_name = "Casey"
    state.user_job = "Engineer"

    config = build_bootstrap_config(state)

    assert config.state_dir == Path('/tmp/opencas-state')
    assert config.workspace_root == Path('/tmp/workspace')
    assert config.workspace_roots == [Path('/tmp/extra-a'), Path('/tmp/extra-b')]
    assert config.provider_config_path == Path('/tmp/provider-config.json')
    assert config.provider_env_path == Path('/tmp/provider.env')
    assert config.credential_source_config_path == Path('/tmp/source-config.json')
    assert config.credential_env_keys == ["OPENAI_API_KEY"]
    assert config.credential_profile_ids == ["anthropic-main"]
    assert config.user_name == "Casey"
    assert "Role: Engineer" in (config.user_bio or "")
