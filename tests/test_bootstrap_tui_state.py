from pathlib import Path

from opencas.bootstrap.tui_state import WizardState, discover_model_choices, scan_openllmauth_profiles


def test_scan_openllmauth_profiles_reads_auth_profiles(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cfg_dir = home / ".open_llm_auth"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.json").write_text(
        '{"authProfiles": {"anthropic-main": {"provider": "anthropic"}, "openai-main": {"provider": "openai"}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    profiles = scan_openllmauth_profiles()

    assert ("anthropic-main", "anthropic-main (anthropic)") in profiles
    assert ("openai-main", "openai-main (openai)") in profiles


def test_discover_model_choices_keeps_defaults_on_loader_failure(monkeypatch):
    state = WizardState()
    state.default_llm_model = "anthropic/custom-chat"
    state.embedding_model_id = "google/custom-embedding"

    def boom(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr("opencas.bootstrap.tui_state.load_config", boom)

    choices = discover_model_choices(state)

    assert ("anthropic/custom-chat", "anthropic/custom-chat") in choices["chat"]
    assert ("google/custom-embedding", "google/custom-embedding") in choices["embedding"]
    assert any(value == "local-fallback" for _, value in choices["embedding"])
