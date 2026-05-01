"""Tests for the bootstrap pipeline."""

import io
import json
import sys
import pytest
from pathlib import Path
from types import SimpleNamespace
from opencas.bootstrap import BootstrapConfig, BootstrapPipeline, pipeline_support
from opencas.__main__ import _build_bootstrap_config, _enforce_bootstrap_responsibility_ack
from opencas.bootstrap.responsibility import (
    BOOTSTRAP_RESPONSIBILITY_WARNING,
    load_bootstrap_responsibility_ack,
)
from opencas.model_routing import PersistedModelRoutingState, ModelRoutingConfig, save_persisted_model_routing_state


def test_embeddinggemma_request_dimension_is_pinned() -> None:
    assert pipeline_support.resolve_embedding_dimensions("google/embeddinggemma-300m") == 768
    assert pipeline_support.resolve_embedding_dimensions("google/gemini-embedding-2-preview") is None


def test_bootstrap_responsibility_warning_names_continuity_and_state_deletion() -> None:
    warning = BOOTSTRAP_RESPONSIBILITY_WARNING

    assert "persistent autonomous agent" in warning
    assert "not a disposable chat session" in warning
    assert "deleting that agent's continuity" in warning
    assert "Do not create an agent casually" in warning


def test_cli_fresh_boot_requires_responsibility_acknowledgement(tmp_path: Path) -> None:
    stderr = io.StringIO()
    args = SimpleNamespace(accept_bootstrap_responsibility=False)

    with pytest.raises(SystemExit) as exc:
        _enforce_bootstrap_responsibility_ack(args, tmp_path, stderr=stderr)

    assert exc.value.code == 2
    assert "persistent autonomous agent" in stderr.getvalue()
    assert "--accept-bootstrap-responsibility" in stderr.getvalue()
    assert load_bootstrap_responsibility_ack(tmp_path) is None


def test_cli_fresh_boot_records_explicit_responsibility_acknowledgement(tmp_path: Path) -> None:
    stderr = io.StringIO()
    args = SimpleNamespace(accept_bootstrap_responsibility=True)

    _enforce_bootstrap_responsibility_ack(args, tmp_path, stderr=stderr)

    ack = load_bootstrap_responsibility_ack(tmp_path)
    assert ack is not None
    assert ack["source"] == "cli"
    assert ack["warning_version"] == 1
    assert "state directory" in ack["accepted_text"]


def test_cli_existing_continuity_does_not_require_bootstrap_acknowledgement(tmp_path: Path) -> None:
    identity_dir = tmp_path / "identity"
    identity_dir.mkdir(parents=True)
    (identity_dir / "continuity.json").write_text('{"boot_count": 1}', encoding="utf-8")
    args = SimpleNamespace(accept_bootstrap_responsibility=False)

    _enforce_bootstrap_responsibility_ack(args, tmp_path, stderr=io.StringIO())

    assert load_bootstrap_responsibility_ack(tmp_path) is None


@pytest.mark.asyncio
async def test_bootstrap_pipeline(tmp_path: Path) -> None:
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="test-session",
        default_llm_model="test/default-model",
    )
    pipeline = BootstrapPipeline(config)
    ctx = await pipeline.run()

    assert ctx.config.state_dir == tmp_path
    assert ctx.identity.continuity.boot_count >= 1
    assert ctx.identity.continuity.last_session_id == "test-session"
    assert ctx.memory is not None
    assert ctx.embeddings is not None
    assert ctx.somatic is not None
    assert ctx.tracer is not None
    assert ctx.llm is not None
    assert ctx.llm.default_model == config.default_llm_model
    assert ctx.embeddings.model_id == "google/embeddinggemma-300m"
    assert ctx.embeddings._embed_batch_fn is None
    assert ctx.embeddings.expected_dimension == 768

    # Verify telemetry recorded bootstrap stages
    events = ctx.tracer.store.query()
    assert len(events) > 0
    stage_events = [e for e in events if "BOOTSTRAP_STAGE" in str(e.kind)]
    assert len(stage_events) >= 1

    # Clean up
    await ctx.close()


@pytest.mark.asyncio
async def test_bootstrap_context_close_cancels_background_tasks(tmp_path: Path) -> None:
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="background-task-close",
    )
    ctx = await BootstrapPipeline(config).run()

    assert ctx.background_tasks

    await ctx.close()

    assert all(task.done() for task in ctx.background_tasks)
    assert ctx.workspace_index._scan_task is None


@pytest.mark.asyncio
async def test_bootstrap_identity_persists(tmp_path: Path) -> None:
    config = BootstrapConfig(state_dir=tmp_path, session_id="s1")
    ctx1 = await BootstrapPipeline(config).run()
    await ctx1.memory.close()
    await ctx1.embeddings.cache.close()

    ctx2 = await BootstrapPipeline(config).run()
    assert ctx2.identity.continuity.boot_count == ctx1.identity.continuity.boot_count + 1
    await ctx2.memory.close()
    await ctx2.embeddings.cache.close()


@pytest.mark.asyncio
async def test_first_boot_seeds_identity(tmp_path: Path) -> None:
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="first-session",
        persona_name="Aurora",
        user_name="Alex",
        user_bio="A software engineer who values brevity and correctness.",
    )
    ctx = await BootstrapPipeline(config).run()

    # Self-model should have baseline personality
    assert ctx.identity.self_model.name == "Aurora"
    assert "clarity" in ctx.identity.self_model.values
    assert "action-oriented" in ctx.identity.self_model.traits
    assert len(ctx.identity.self_model.current_goals) >= 1

    # User-model should have baseline profile
    assert ctx.identity.user_model.explicit_preferences.get("name") == "Alex"
    assert ctx.identity.user_model.explicit_preferences.get("bio") == config.user_bio
    assert len(ctx.identity.user_model.inferred_goals) >= 1
    assert len(ctx.identity.user_model.known_boundaries) >= 1

    # Moral warning should have been logged
    events = ctx.tracer.store.query()
    warnings = [e for e in events if "moral_warning" in str(e.message)]
    assert len(warnings) == 1

    await ctx.memory.close()
    await ctx.embeddings.cache.close()


@pytest.mark.asyncio
async def test_subsequent_boot_does_not_reseed(tmp_path: Path) -> None:
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="s1",
        persona_name="Aurora",
        user_name="Alex",
    )
    ctx1 = await BootstrapPipeline(config).run()
    # Mutate a value to detect re-seeding
    ctx1.identity.self_model.name = "Modified"
    ctx1.identity.save()
    await ctx1.memory.close()
    await ctx1.embeddings.cache.close()

    ctx2 = await BootstrapPipeline(config).run()
    assert ctx2.identity.self_model.name == "Modified"
    await ctx2.memory.close()
    await ctx2.embeddings.cache.close()


@pytest.mark.asyncio
async def test_clean_boot_reseeds_identity(tmp_path: Path) -> None:
    config = BootstrapConfig(
        state_dir=tmp_path,
        session_id="s1",
        persona_name="Aurora",
        user_name="Alex",
    )
    ctx1 = await BootstrapPipeline(config).run()
    ctx1.identity.self_model.name = "Modified"
    ctx1.identity.save()
    await ctx1.memory.close()
    await ctx1.embeddings.cache.close()

    config2 = BootstrapConfig(
        state_dir=tmp_path,
        session_id="s2",
        clean_boot=True,
        persona_name="Nova",
        user_name="Jordan",
    )
    ctx2 = await BootstrapPipeline(config2).run()
    assert ctx2.identity.self_model.name == "Nova"
    assert ctx2.identity.user_model.explicit_preferences.get("name") == "Jordan"
    await ctx2.memory.close()
    await ctx2.embeddings.cache.close()


@pytest.mark.asyncio
async def test_bootstrap_restores_executive_snapshot(tmp_path: Path) -> None:
    config = BootstrapConfig(state_dir=tmp_path, session_id="s1")
    ctx1 = await BootstrapPipeline(config).run()
    ctx1.executive.set_intention("restore me")
    ctx1.executive.add_goal("bootstrap goal")
    snapshot_path = tmp_path / "executive.json"
    ctx1.executive.save_snapshot(snapshot_path)
    await ctx1.memory.close()
    await ctx1.embeddings.cache.close()

    ctx2 = await BootstrapPipeline(config).run()
    assert ctx2.executive.intention == "restore me"
    assert "bootstrap goal" in ctx2.executive.active_goals
    await ctx2.memory.close()
    await ctx2.embeddings.cache.close()


@pytest.mark.asyncio
async def test_bootstrap_loads_skills(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir(parents=True)
    skill_file = plugins_dir / "hello_skill.py"
    skill_file.write_text(
        "from opencas.plugins import SkillEntry\n"
        "SKILL_ENTRY = SkillEntry(skill_id='hello', name='Hello', description='Says hello.')\n"
    )
    config = BootstrapConfig(state_dir=state_dir, session_id="s1")
    ctx = await BootstrapPipeline(config).run()

    assert ctx.skill_registry is not None
    skill = ctx.skill_registry.get("hello")
    assert skill is not None
    assert skill.name == "Hello"

    await ctx.memory.close()
    await ctx.embeddings.cache.close()


@pytest.mark.asyncio
async def test_bootstrap_prefers_explicit_workspace_root(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    workspace_root = tmp_path / "repo"
    workspace_root.mkdir()
    config = BootstrapConfig(
        state_dir=state_dir,
        session_id="workspace-test",
        workspace_root=workspace_root,
    )
    ctx = await BootstrapPipeline(config).run()

    assert ctx.config.workspace_root == workspace_root.resolve()
    assert ctx.sandbox.allowed_roots == [workspace_root.resolve()]

    await ctx.memory.close()
    await ctx.embeddings.cache.close()


def test_bootstrap_config_dedupes_workspace_roots(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    primary = tmp_path / "primary"
    secondary = tmp_path / "secondary"
    config = BootstrapConfig(
        state_dir=state_dir,
        workspace_root=primary,
        workspace_roots=[secondary, primary],
    ).resolve_paths()

    roots = config.all_workspace_roots()
    assert roots == [primary.resolve(), secondary.resolve()]
    assert config.primary_workspace_root() == primary.resolve()
    assert config.agent_workspace_root() == (primary.resolve() / "workspace").resolve()


def test_bootstrap_config_supports_explicit_managed_workspace_root(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    primary = tmp_path / "primary"
    managed = tmp_path / "managed"
    config = BootstrapConfig(
        state_dir=state_dir,
        workspace_root=primary,
        managed_workspace_root=managed,
    ).resolve_paths()

    assert config.agent_workspace_root() == managed.resolve()
    assert config.all_workspace_roots() == [primary.resolve(), managed.resolve()]


def test_build_bootstrap_config_loads_persisted_model_routing(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    save_persisted_model_routing_state(
        state_dir,
        PersistedModelRoutingState(
            default_llm_model="openai/gpt-5.3-codex",
            model_routing=ModelRoutingConfig(
                mode="tiered",
                light_model="google/gemini-2.5-flash",
                standard_model="openai/gpt-5.3-codex",
                high_model="anthropic/claude-sonnet-4-6",
                extra_high_model="codex-cli/gpt-5.3-codex",
                auto_escalation=True,
            ),
        ),
    )
    args = SimpleNamespace(
        state_dir=str(state_dir),
        session_id="routing-test",
        agent_profile_id="general_technical_operator",
        workspace_root=str(tmp_path),
        workspace_extra_root=[],
        cycle_interval=600,
        daydream_interval=720,
        baa_heartbeat_interval=120,
        credential_profile_id=[],
        credential_env_key=[],
        telegram_enabled=None,
        telegram_bot_token=None,
        telegram_dm_policy=None,
        telegram_allow_from=[],
        telegram_poll_interval=None,
        telegram_pairing_ttl=None,
        default_llm_model=None,
        embedding_model_id=None,
        provider_config_path=None,
        provider_env_path=None,
        credential_source_config_path=None,
        credential_source_env_path=None,
    )
    persisted_telegram = SimpleNamespace(
        enabled=False,
        bot_token=None,
        dm_policy="pairing",
        allow_from=[],
        poll_interval_seconds=1.0,
        pairing_ttl_seconds=3600,
        api_base_url="https://api.telegram.org",
    )

    config = _build_bootstrap_config(args, persisted_telegram)

    assert config.default_llm_model == "openai/gpt-5.3-codex"
    assert config.model_routing.mode.value == "tiered"
    assert config.model_routing.light_model == "google/gemini-2.5-flash"
    assert config.model_routing.extra_high_model == "codex-cli/gpt-5.3-codex"
    assert config.cycle_interval == 600
    assert config.daydream_interval == 720
    assert config.baa_heartbeat_interval == 120


def test_build_bootstrap_config_heals_stale_persisted_model_routing(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    provider_material = state_dir / "provider_material"
    provider_material.mkdir(parents=True)
    (provider_material / "config.json").write_text(
        json.dumps(
            {
                "defaultModel": "zai-coding/glm-5.1",
                "providers": {
                    "zai-coding": {
                        "baseUrl": "https://api.z.ai/api/coding/paas/v4",
                        "auth": "api-key",
                        "api": "openai-completions",
                        "models": [
                            {
                                "id": "zai-coding/glm-5.1",
                                "name": "zai-coding/glm-5.1",
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    save_persisted_model_routing_state(
        state_dir,
        PersistedModelRoutingState(
            default_llm_model="zai-anthropic/glm-5.1",
            model_routing=ModelRoutingConfig(
                mode="single",
                single_model="zai-anthropic/glm-5.1",
                standard_model="zai-anthropic/glm-5.1",
                high_model="zai-anthropic/glm-5.1",
                extra_high_model="zai-anthropic/glm-5.1",
            ),
        ),
    )
    args = SimpleNamespace(
        state_dir=str(state_dir),
        session_id="routing-heal-test",
        agent_profile_id="general_technical_operator",
        workspace_root=str(tmp_path),
        workspace_extra_root=[],
        cycle_interval=600,
        daydream_interval=720,
        baa_heartbeat_interval=120,
        credential_profile_id=[],
        credential_env_key=[],
        telegram_enabled=None,
        telegram_bot_token=None,
        telegram_dm_policy=None,
        telegram_allow_from=[],
        telegram_poll_interval=None,
        telegram_pairing_ttl=None,
        default_llm_model=None,
        embedding_model_id=None,
        provider_config_path=None,
        provider_env_path=None,
        credential_source_config_path=None,
        credential_source_env_path=None,
    )
    persisted_telegram = SimpleNamespace(
        enabled=False,
        bot_token=None,
        dm_policy="pairing",
        allow_from=[],
        poll_interval_seconds=1.0,
        pairing_ttl_seconds=3600,
        api_base_url="https://api.telegram.org",
    )

    config = _build_bootstrap_config(args, persisted_telegram)

    assert config.default_llm_model == "zai-coding/glm-5.1"
    assert config.model_routing.standard_model == "zai-coding/glm-5.1"
    assert config.model_routing.high_model == "zai-coding/glm-5.1"
    assert config.cycle_interval == 600
    assert config.daydream_interval == 720
    assert config.baa_heartbeat_interval == 120
    repaired_payload = json.loads((provider_material / "config.json").read_text(encoding="utf-8"))
    assert repaired_payload["defaultModel"] == "zai-coding/glm-5.1"


@pytest.mark.asyncio
async def test_bootstrap_copies_provider_material_per_app(tmp_path: Path) -> None:
    source_dir = tmp_path / "source-auth"
    source_dir.mkdir()
    source_config = source_dir / "config.json"
    source_env = source_dir / ".env"
    source_env.write_text('GEMINI_API_KEY="gemini-test-key"\n', encoding="utf-8")
    source_config.write_text(
        json.dumps(
            {
                "authProfiles": {
                    "kimi-coding:default": {
                        "provider": "kimi-coding",
                        "type": "api_key",
                        "key": "kimi-test-key",
                    },
                    "google:default": {
                        "provider": "google",
                        "type": "api_key",
                        "key": "${GEMINI_API_KEY}",
                    },
                },
                "authOrder": {
                    "kimi-coding": ["kimi-coding:default"],
                    "google": ["google:default"],
                },
            }
        ),
        encoding="utf-8",
    )

    config = BootstrapConfig(
        state_dir=tmp_path / "state",
        session_id="provider-copy-test",
        default_llm_model="kimi-coding/k2p5",
        credential_source_config_path=source_config,
        credential_source_env_path=source_env,
        credential_profile_ids=["kimi-coding:default", "google:default"],
        credential_env_keys=["GEMINI_API_KEY"],
    )
    ctx = await BootstrapPipeline(config).run()

    copied_config = ctx.config.state_dir / "provider_material" / "config.json"
    copied_env = ctx.config.state_dir / "provider_material" / ".env"
    assert copied_config.exists()
    assert copied_env.exists()

    copied_payload = json.loads(copied_config.read_text(encoding="utf-8"))
    assert sorted(copied_payload["authProfiles"].keys()) == [
        "google:default",
        "kimi-coding:default",
    ]
    assert (
        copied_payload["authProfiles"]["kimi-coding:default"]["key"] == "kimi-test-key"
    )
    assert copied_env.read_text(encoding="utf-8").strip() == 'GEMINI_API_KEY="gemini-test-key"'

    await ctx.memory.close()
    await ctx.embeddings.cache.close()


@pytest.mark.asyncio
async def test_bootstrap_disables_hnsw_on_python_314_plus(tmp_path: Path) -> None:
    config = BootstrapConfig(
        state_dir=tmp_path / "state",
        session_id="hnsw-guard-test",
    )
    ctx = await BootstrapPipeline(config).run()

    if sys.version_info >= (3, 14):
        assert ctx.embeddings.cache.hnsw_backend is None

    await ctx.memory.close()
    await ctx.embeddings.cache.close()


def test_cli_bootstrap_config_preserves_default_embedding_model_when_flag_omitted(tmp_path: Path) -> None:
    args = SimpleNamespace(
        state_dir=str(tmp_path / "state"),
        session_id="cli-test",
        agent_profile_id="general_technical_operator",
        workspace_root=str(tmp_path),
        workspace_extra_root=[],
        default_llm_model=None,
        embedding_model_id=None,
        provider_config_path=None,
        provider_env_path=None,
        credential_source_config_path=None,
        credential_source_env_path=None,
        credential_profile_id=[],
        credential_env_key=[],
        telegram_enabled=None,
        telegram_bot_token=None,
        telegram_dm_policy=None,
        telegram_allow_from=[],
        telegram_poll_interval=None,
        telegram_pairing_ttl=None,
    )
    persisted_telegram = SimpleNamespace(
        enabled=False,
        bot_token=None,
        dm_policy="pairing",
        allow_from=[],
        poll_interval_seconds=1.0,
        pairing_ttl_seconds=3600,
        api_base_url="https://api.telegram.org",
    )

    config = _build_bootstrap_config(args, persisted_telegram)

    assert config.default_llm_model is None
    assert config.embedding_model_id == "google/embeddinggemma-300m"


def test_cli_bootstrap_config_accepts_qdrant_backend_flags(tmp_path: Path) -> None:
    args = SimpleNamespace(
        state_dir=str(tmp_path / "state"),
        session_id="cli-test",
        agent_profile_id="general_technical_operator",
        workspace_root=str(tmp_path),
        workspace_extra_root=[],
        default_llm_model=None,
        embedding_model_id=None,
        qdrant_url="http://127.0.0.1:6333",
        qdrant_api_key=None,
        qdrant_collection="episodes_embed_v1",
        provider_config_path=None,
        provider_env_path=None,
        credential_source_config_path=None,
        credential_source_env_path=None,
        credential_profile_id=[],
        credential_env_key=[],
        telegram_enabled=None,
        telegram_bot_token=None,
        telegram_dm_policy=None,
        telegram_allow_from=[],
        telegram_poll_interval=None,
        telegram_pairing_ttl=None,
    )
    persisted_telegram = SimpleNamespace(
        enabled=False,
        bot_token=None,
        dm_policy="pairing",
        allow_from=[],
        poll_interval_seconds=1.0,
        pairing_ttl_seconds=3600,
        api_base_url="https://api.telegram.org",
    )

    config = _build_bootstrap_config(args, persisted_telegram)

    assert config.qdrant_url == "http://127.0.0.1:6333"
    assert config.qdrant_collection == "episodes_embed_v1"
    assert config.qdrant_auto_start is True
    assert config.qdrant_required is True


def test_cli_bootstrap_config_prefers_materialized_bundle_default_model(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    provider_material = state_dir / "provider_material"
    provider_material.mkdir(parents=True)
    (provider_material / "config.json").write_text(
        json.dumps({"defaultModel": "kimi-coding/k2p5"}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        state_dir=str(state_dir),
        session_id="cli-test",
        agent_profile_id="general_technical_operator",
        workspace_root=str(tmp_path),
        workspace_extra_root=[],
        default_llm_model=None,
        embedding_model_id=None,
        provider_config_path=None,
        provider_env_path=None,
        credential_source_config_path=None,
        credential_source_env_path=None,
        credential_profile_id=[],
        credential_env_key=[],
        telegram_enabled=None,
        telegram_bot_token=None,
        telegram_dm_policy=None,
        telegram_allow_from=[],
        telegram_poll_interval=None,
        telegram_pairing_ttl=None,
    )
    persisted_telegram = SimpleNamespace(
        enabled=False,
        bot_token=None,
        dm_policy="pairing",
        allow_from=[],
        poll_interval_seconds=1.0,
        pairing_ttl_seconds=3600,
        api_base_url="https://api.telegram.org",
    )

    config = _build_bootstrap_config(args, persisted_telegram)

    assert config.default_llm_model == "kimi-coding/k2p5"
    assert config.embedding_model_id == "google/embeddinggemma-300m"


def test_pipeline_defaults_to_local_gemma_embedding_model(tmp_path: Path) -> None:
    pipeline = BootstrapPipeline(BootstrapConfig(state_dir=tmp_path, embedding_model_id=None))
    assert pipeline._resolve_embedding_model() == "google/embeddinggemma-300m"
