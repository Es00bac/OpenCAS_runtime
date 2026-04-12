"""CLI entry point for OpenCAS autonomous agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime
from opencas.telegram_config import load_telegram_runtime_config


def _read_materialized_default_model(state_dir: Path) -> str | None:
    config_path = state_dir / "provider_material" / "config.json"
    if not config_path.exists():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("defaultModel")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _build_bootstrap_config(args, persisted_telegram) -> BootstrapConfig:
    """Build BootstrapConfig without clobbering model defaults with None."""
    state_dir = Path(args.state_dir).expanduser().resolve()
    config_kwargs = {
        "state_dir": state_dir,
        "session_id": args.session_id,
        "agent_profile_id": args.agent_profile_id,
        "workspace_root": Path(args.workspace_root).expanduser().resolve(),
        "workspace_roots": [
            Path(root).expanduser().resolve() for root in args.workspace_extra_root
        ],
        "credential_profile_ids": list(args.credential_profile_id),
        "credential_env_keys": list(args.credential_env_key),
        "telegram_enabled": (
            persisted_telegram.enabled
            if args.telegram_enabled is None
            else args.telegram_enabled
        ),
        "telegram_bot_token": (
            args.telegram_bot_token
            if args.telegram_bot_token is not None
            else persisted_telegram.bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        ),
        "telegram_dm_policy": args.telegram_dm_policy or persisted_telegram.dm_policy,
        "telegram_allow_from": (
            list(args.telegram_allow_from)
            if args.telegram_allow_from
            else list(persisted_telegram.allow_from)
        ),
        "telegram_poll_interval_seconds": (
            args.telegram_poll_interval
            if args.telegram_poll_interval is not None
            else persisted_telegram.poll_interval_seconds
        ),
        "telegram_pairing_ttl_seconds": (
            args.telegram_pairing_ttl
            if args.telegram_pairing_ttl is not None
            else persisted_telegram.pairing_ttl_seconds
        ),
        "telegram_api_base_url": persisted_telegram.api_base_url,
    }
    if args.default_llm_model is not None:
        config_kwargs["default_llm_model"] = args.default_llm_model
    else:
        materialized_default = _read_materialized_default_model(state_dir)
        if materialized_default:
            config_kwargs["default_llm_model"] = materialized_default
    if args.embedding_model_id is not None:
        config_kwargs["embedding_model_id"] = args.embedding_model_id
    if args.provider_config_path:
        config_kwargs["provider_config_path"] = (
            Path(args.provider_config_path).expanduser().resolve()
        )
    if args.provider_env_path:
        config_kwargs["provider_env_path"] = (
            Path(args.provider_env_path).expanduser().resolve()
        )
    if args.credential_source_config_path:
        config_kwargs["credential_source_config_path"] = (
            Path(args.credential_source_config_path).expanduser().resolve()
        )
    if args.credential_source_env_path:
        config_kwargs["credential_source_env_path"] = (
            Path(args.credential_source_env_path).expanduser().resolve()
        )
    return BootstrapConfig(**config_kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenCAS autonomous agent")
    parser.add_argument(
        "--state-dir",
        default="./.opencas",
        help="Directory for agent state (default: ./.opencas)",
    )
    parser.add_argument("--session-id", default=None, help="Session identifier")
    parser.add_argument(
        "--agent-profile-id",
        default="general_technical_operator",
        help="Built-in agent runtime profile id (default: general_technical_operator)",
    )
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Primary workspace root the agent is allowed to operate in (default: current directory)",
    )
    parser.add_argument(
        "--workspace-extra-root",
        action="append",
        default=[],
        help="Additional workspace roots the agent is allowed to operate in. May be passed multiple times.",
    )
    parser.add_argument(
        "--cycle-interval",
        type=int,
        default=300,
        help="Creative cycle interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--consolidation-interval",
        type=int,
        default=86400,
        help="Consolidation interval in seconds (default: 86400)",
    )
    parser.add_argument(
        "--default-llm-model",
        default=None,
        help="Default model reference for conversation/tool use (e.g. kimi-coding/k2p5)",
    )
    parser.add_argument(
        "--embedding-model-id",
        default=None,
        help="Embedding model reference (e.g. google/gemini-embedding-2-preview)",
    )
    parser.add_argument(
        "--provider-config-path",
        default=None,
        help="Explicit OpenLLMAuth config.json path for this app instance.",
    )
    parser.add_argument(
        "--provider-env-path",
        default=None,
        help="Explicit OpenLLMAuth .env path for this app instance.",
    )
    parser.add_argument(
        "--credential-source-config-path",
        default=None,
        help="Source OpenLLMAuth config to copy selected credentials from into app-local state.",
    )
    parser.add_argument(
        "--credential-source-env-path",
        default=None,
        help="Source .env file to copy selected provider env vars from into app-local state.",
    )
    parser.add_argument(
        "--credential-profile-id",
        action="append",
        default=[],
        help="Auth profile id to copy into app-local provider material. May be passed multiple times.",
    )
    parser.add_argument(
        "--credential-env-key",
        action="append",
        default=[],
        help="Environment variable name to copy into app-local provider material. May be passed multiple times.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    parser.add_argument(
        "--with-server",
        action="store_true",
        help="Run autonomous mode with the FastAPI server (default: server disabled)",
    )
    parser.add_argument(
        "--telegram-enabled",
        dest="telegram_enabled",
        action="store_true",
        default=None,
        help="Enable Telegram integration.",
    )
    parser.add_argument(
        "--telegram-disabled",
        dest="telegram_enabled",
        action="store_false",
        help="Disable Telegram integration even if persisted settings exist.",
    )
    parser.add_argument(
        "--telegram-bot-token",
        default=None,
        help="Telegram bot token from BotFather.",
    )
    parser.add_argument(
        "--telegram-dm-policy",
        choices=["disabled", "pairing", "allowlist", "open"],
        default=None,
        help="Telegram direct-message access policy.",
    )
    parser.add_argument(
        "--telegram-allow-from",
        action="append",
        default=[],
        help="Telegram user id to allow without pairing. May be passed multiple times.",
    )
    parser.add_argument(
        "--telegram-poll-interval",
        type=float,
        default=None,
        help="Telegram long-poll retry interval in seconds.",
    )
    parser.add_argument(
        "--telegram-pairing-ttl",
        type=int,
        default=None,
        help="Telegram pairing-code lifetime in seconds.",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch the interactive TUI bootstrap wizard instead of using CLI arguments",
    )
    args = parser.parse_args()

    if args.tui:
        from opencas.bootstrap.tui import main as tui_main

        return tui_main()

    async def _run() -> None:
        state_dir = Path(args.state_dir).expanduser().resolve()
        persisted_telegram = load_telegram_runtime_config(state_dir)
        config = _build_bootstrap_config(args, persisted_telegram)
        ctx = await BootstrapPipeline(config).run()
        runtime = AgentRuntime(ctx)
        await runtime.tom.load()

        if args.with_server:
            await runtime.run_autonomous_with_server(
                host=args.host,
                port=args.port,
                cycle_interval=args.cycle_interval,
                consolidation_interval=args.consolidation_interval,
            )
        else:
            await runtime.run_autonomous(
                cycle_interval=args.cycle_interval,
                consolidation_interval=args.consolidation_interval,
            )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nShutdown requested by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
