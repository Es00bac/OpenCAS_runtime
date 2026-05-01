import json
from pathlib import Path

from opencas.model_routing import ModelRoutingMode
from opencas.sandbox.config import SandboxMode
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


def test_save_questionnaire_creates_missing_state_dir(tmp_path: Path):
    state = WizardState()
    state.goal_1 = "Finish setup"

    saved = save_questionnaire(state, tmp_path / "fresh-state-dir")

    assert saved == tmp_path / "fresh-state-dir" / "bootstrap_questionnaire.json"
    assert saved.exists()


def test_build_bootstrap_config_uses_workspace_and_provider_fields():
    state = WizardState()
    state.provider_mode = "custom"
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
    assert config.credential_source_config_path is None
    assert config.credential_env_keys == []
    assert config.credential_profile_ids == []
    assert config.user_name == "Casey"
    assert "Role: Engineer" in (config.user_bio or "")


def test_build_bootstrap_config_covers_current_bootstrap_surface():
    state = WizardState()
    state.provider_mode = "copy"
    state.state_dir = "/tmp/opencas-state"
    state.workspace_root = "/tmp/workspace"
    state.workspace_extra = "/tmp/extra-a, /tmp/extra-b"
    state.managed_workspace_root = "/tmp/workspace/managed"
    state.credential_source_config = "/tmp/source-config.json"
    state.credential_source_env_path = "/tmp/source.env"
    state.credential_env_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    state.selected_profiles = ["anthropic-main", "openai-main"]
    state.default_llm_model = "anthropic/claude-sonnet-4-6"
    state.embedding_model_id = "google/gemini-embedding-2-preview"
    state.model_routing_mode = "tiered"
    state.routing_light_model = "google/gemini-2.5-flash"
    state.routing_standard_model = "anthropic/claude-sonnet-4-6"
    state.routing_high_model = "openai/gpt-5"
    state.routing_extra_high_model = "codex-cli/gpt-5.3-codex"
    state.routing_auto_escalation = False
    state.qdrant_url = "http://localhost:6333"
    state.qdrant_api_key = "qdrant-secret"
    state.qdrant_collection = "custom_collection"
    state.hnsw_enabled = False
    state.hnsw_m = "48"
    state.hnsw_ef_construction = "512"
    state.mcp_auto_register = True
    state.mcp_servers_json = '[{"name": "filesystem", "transport": "stdio", "command": ["uvx", "mcp-server-filesystem"]}]'
    state.telegram_enabled = True
    state.telegram_bot_token = "123:abc"
    state.telegram_dm_policy = "allowlist"
    state.telegram_allow_from = "111, 222"
    state.telegram_poll_interval_seconds = "2.5"
    state.telegram_pairing_ttl_seconds = "7200"
    state.telegram_api_base_url = "https://telegram.example.com"
    state.cycle_interval = "600"
    state.daydream_interval = "720"
    state.baa_heartbeat_interval = "120"
    state.sandbox_mode = "allow-list"
    state.sandbox_allowed_roots = "/tmp/workspace, /tmp/shared"
    state.user_name = "Casey"
    state.user_job = "Engineer"

    config = build_bootstrap_config(state)

    assert config.managed_workspace_root == Path("/tmp/workspace/managed")
    assert config.credential_source_env_path == Path("/tmp/source.env")
    assert config.model_routing.mode == ModelRoutingMode.TIERED
    assert config.model_routing.light_model == "google/gemini-2.5-flash"
    assert config.model_routing.standard_model == "anthropic/claude-sonnet-4-6"
    assert config.model_routing.high_model == "openai/gpt-5"
    assert config.model_routing.extra_high_model == "codex-cli/gpt-5.3-codex"
    assert config.model_routing.auto_escalation is False
    assert config.qdrant_url == "http://localhost:6333"
    assert config.qdrant_api_key == "qdrant-secret"
    assert config.qdrant_collection == "custom_collection"
    assert config.hnsw_enabled is False
    assert config.hnsw_m == 48
    assert config.hnsw_ef_construction == 512
    assert config.mcp_auto_register is True
    assert config.mcp_servers == [
        {
            "name": "filesystem",
            "transport": "stdio",
            "command": ["uvx", "mcp-server-filesystem"],
        }
    ]
    assert config.telegram_enabled is True
    assert config.telegram_bot_token == "123:abc"
    assert config.telegram_dm_policy == "allowlist"
    assert config.telegram_allow_from == ["111", "222"]
    assert config.telegram_poll_interval_seconds == 2.5
    assert config.telegram_pairing_ttl_seconds == 7200
    assert config.telegram_api_base_url == "https://telegram.example.com"
    assert config.cycle_interval == 600
    assert config.daydream_interval == 720
    assert config.baa_heartbeat_interval == 120
    assert config.sandbox is not None
    assert config.sandbox.mode == SandboxMode.ALLOW_LIST
    assert config.sandbox.allowed_roots == [Path("/tmp/workspace"), Path("/tmp/shared")]


def test_build_bootstrap_config_ignores_stale_provider_fields_in_auto_mode():
    state = WizardState()
    state.provider_mode = "auto"
    state.provider_config_path = "/tmp/provider-config.json"
    state.provider_env_path = "/tmp/provider.env"
    state.credential_source_config = "/tmp/source-config.json"
    state.credential_source_env_path = "/tmp/source.env"
    state.credential_env_keys = ["OPENAI_API_KEY"]
    state.selected_profiles = ["anthropic-main"]

    config = build_bootstrap_config(state)

    assert config.provider_config_path is None
    assert config.provider_env_path is None
    assert config.credential_source_config_path is None
    assert config.credential_source_env_path is None
    assert config.credential_env_keys == []
    assert config.credential_profile_ids == []
