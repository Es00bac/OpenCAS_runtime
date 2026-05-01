"""Setup and configuration screens for the OpenCAS bootstrap TUI."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    Markdown,
    RadioButton,
    RadioSet,
    Select,
    Static,
    Switch,
    TextArea,
)

from opencas.bootstrap.tui_components import HelpText, NavButtons, StepHeader
from opencas.bootstrap.tui_state import STATE, discover_model_choices, scan_openllmauth_profiles
from opencas.runtime.agent_profile import BUILTIN_AGENT_PROFILES


class WorkspaceScreen(Screen):
    def compose(self) -> ComposeResult:
        yield StepHeader(10, 16, "Agent Home & Workspace")
        yield HelpText(
            "The agent needs a place to keep its memories, and boundaries around where it can work."
        )

        yield Label("Where should the agent store its state?", classes="field-label")
        yield Input(placeholder="./.opencas", value=STATE.state_dir, id="input-state-dir")
        yield HelpText("This folder will contain databases, snapshots, telemetry, and identity files.")

        yield Static()
        yield Label("Primary workspace root:", classes="field-label")
        yield Input(placeholder=".", value=STATE.workspace_root, id="input-workspace")
        yield HelpText(
            "The main directory the agent is allowed to read, write, and run commands in."
        )

        yield Static()
        yield Label("Additional workspace roots (optional, comma-separated):", classes="field-label")
        yield Input(
            placeholder="/home/you/projects, /mnt/data",
            value=STATE.workspace_extra,
            id="input-workspace-extra",
        )
        yield HelpText(
            "Extra directories the agent can access. Useful if you want it to work across multiple repos."
        )

        yield Static()
        yield Label("Managed workspace root (optional):", classes="field-label")
        yield Input(
            placeholder="Defaults to <primary workspace>/workspace",
            value=STATE.managed_workspace_root,
            id="input-managed-workspace",
        )
        yield HelpText(
            "Where agent-created work should live. Leave blank to use a managed `workspace/` "
            "folder under the primary workspace root."
        )

        yield NavButtons(show_back=True, next_label="Continue →")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            STATE.state_dir = self.query_one("#input-state-dir", Input).value or "./.opencas"
            STATE.workspace_root = self.query_one("#input-workspace", Input).value or "."
            STATE.workspace_extra = self.query_one("#input-workspace-extra", Input).value or ""
            STATE.managed_workspace_root = self.query_one(
                "#input-managed-workspace", Input
            ).value or ""
            self.app.push_screen("credentials")


# -----------------------------------------------------------------------------
# Screen 11: Credentials
# -----------------------------------------------------------------------------

class CredentialsScreen(Screen):
    """Provider credential setup with explanations for new users."""

    def compose(self) -> ComposeResult:
        yield StepHeader(11, 16, "Brains & Credentials")
        yield HelpText(
            "OpenCAS needs access to Large Language Models (LLMs) to think, plan, and talk with you. "
            "We use a gateway called open_llm_auth to manage credentials safely."
        )

        yield Label("How would you like to provide LLM access?", classes="field-label")
        with RadioSet(id="provider-mode"):
            yield RadioButton(
                "Use my existing open_llm_auth config (simplest)",
                value=(STATE.provider_mode == "auto"),
                id="mode-auto",
            )
            yield RadioButton(
                "Copy specific profiles into the agent's local state",
                value=(STATE.provider_mode == "copy"),
                id="mode-copy",
            )
            yield RadioButton(
                "Point to custom config files directly",
                value=(STATE.provider_mode == "custom"),
                id="mode-custom",
            )

        yield Static()
        self._copy_container = Container(id="copy-options")
        with self._copy_container:
            profiles = scan_openllmauth_profiles()
            if profiles:
                yield Label("Profiles found in ~/.open_llm_auth/config.json:", classes="field-label")
                for pid, label in profiles:
                    yield Checkbox(label, value=(pid in STATE.selected_profiles), id=f"profile-{pid.replace(":", "_")}")
            else:
                yield Label(
                    "No profiles found in ~/.open_llm_auth/config.json. "
                    "You may need to configure open_llm_auth first.",
                    classes="text-warning",
                )
            yield Label("Environment variables to copy (optional, comma-separated):", classes="field-label")
            yield Input(
                placeholder="GOOGLE_API_KEY, OPENAI_API_KEY",
                value=", ".join(STATE.credential_env_keys),
                id="input-env-keys",
            )
            yield Label("Source config.json path:", classes="field-label")
            yield Input(
                placeholder=str(Path.home() / ".open_llm_auth" / "config.json"),
                value=STATE.credential_source_config,
                id="input-copy-config",
            )
            yield Label("Source .env path (optional):", classes="field-label")
            yield Input(
                placeholder="/path/to/.env",
                value=STATE.credential_source_env_path,
                id="input-copy-env",
            )

        yield Static()
        self._custom_container = Container(id="custom-options")
        with self._custom_container:
            yield Label("Path to custom config.json:", classes="field-label")
            yield Input(
                placeholder="/path/to/config.json",
                value=STATE.provider_config_path,
                id="input-custom-config",
            )
            yield Label("Path to custom .env:", classes="field-label")
            yield Input(
                placeholder="/path/to/.env",
                value=STATE.provider_env_path,
                id="input-custom-env",
            )

        yield HelpText(
            "If you're unsure, choose 'Use my existing config'. The agent will look at "
            "~/.open_llm_auth/config.json automatically."
        )
        yield NavButtons(show_back=True, next_label="Continue →")

    def on_mount(self) -> None:
        self._update_visibility()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.pressed.id == "mode-auto":
            STATE.provider_mode = "auto"
        elif event.pressed.id == "mode-copy":
            STATE.provider_mode = "copy"
        elif event.pressed.id == "mode-custom":
            STATE.provider_mode = "custom"
        self._update_visibility()

    def _update_visibility(self) -> None:
        copy_opts = self.query_one("#copy-options", Container)
        custom_opts = self.query_one("#custom-options", Container)
        copy_opts.styles.display = "block" if STATE.provider_mode == "copy" else "none"
        custom_opts.styles.display = "block" if STATE.provider_mode == "custom" else "none"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            if STATE.provider_mode == "copy":
                profiles = scan_openllmauth_profiles()
                STATE.selected_profiles = [
                    pid for pid, _ in profiles
                    if self.query_one(f"#profile-{pid.replace(":", "_")}", Checkbox).value
                ]
                env_input = self.query_one("#input-env-keys", Input).value
                STATE.credential_env_keys = [
                    token.strip()
                    for token in env_input.split(",")
                    if token.strip()
                ]
                STATE.credential_source_config = (
                    self.query_one("#input-copy-config", Input).value
                    or str(Path.home() / ".open_llm_auth" / "config.json")
                )
                STATE.credential_source_env_path = (
                    self.query_one("#input-copy-env", Input).value or ""
                )
                STATE.provider_config_path = ""
                STATE.provider_env_path = ""
            elif STATE.provider_mode == "custom":
                STATE.selected_profiles = []
                STATE.credential_env_keys = []
                STATE.credential_source_config = ""
                STATE.credential_source_env_path = ""
                STATE.provider_config_path = self.query_one("#input-custom-config", Input).value
                STATE.provider_env_path = self.query_one("#input-custom-env", Input).value
            else:
                STATE.selected_profiles = []
                STATE.credential_env_keys = []
                STATE.credential_source_config = ""
                STATE.credential_source_env_path = ""
                STATE.provider_config_path = ""
                STATE.provider_env_path = ""
            self.app.push_screen("models")


# -----------------------------------------------------------------------------
# Screen 12: Models
# -----------------------------------------------------------------------------

class ModelsScreen(Screen):
    def compose(self) -> ComposeResult:
        model_choices = discover_model_choices(STATE)
        yield StepHeader(12, 16, "Model Preferences")
        yield HelpText(
            "Which AI models should the agent use for thinking and remembering? "
            "If you don't know, the defaults are solid starting points."
        )

        yield Label("Default chat / reasoning model:", classes="field-label")
        yield Select(
            model_choices["chat"],
            prompt="Configured chat models",
            allow_blank=False,
            value=STATE.default_llm_model,
            id="select-llm",
        )
        yield HelpText(
            "This dropdown is built from the provider material you selected in the previous step. "
            "Use a custom override only if the model exists but is not listed yet."
        )
        yield Input(
            placeholder="Optional custom model override",
            value="",
            id="input-llm-custom",
        )

        yield Static()
        yield Label("Embedding model:", classes="field-label")
        yield Select(
            model_choices["embedding"],
            prompt="Configured embedding models",
            allow_blank=False,
            value=STATE.embedding_model_id,
            id="select-embedding",
        )
        yield HelpText(
            "Embeddings turn text into vectors so the agent can search its memory semantically. "
            "If no provider-backed embedding model is configured, `local-fallback` keeps the system usable offline."
        )

        yield Static()
        yield Label("Routing mode:", classes="field-label")
        yield Select(
            [
                ("Single model", "single"),
                ("Tiered by complexity", "tiered"),
            ],
            allow_blank=False,
            value=STATE.model_routing_mode,
            id="select-routing-mode",
        )
        yield HelpText(
            "Single mode uses the default chat model for everything. Tiered mode lets you pin "
            "different models for light, standard, high, and extra-high reasoning."
        )

        yield Label("Tiered light model (optional):", classes="field-label")
        yield Input(
            placeholder="Falls back to the default/standard model",
            value=STATE.routing_light_model,
            id="input-routing-light",
        )
        yield Label("Tiered standard model (optional):", classes="field-label")
        yield Input(
            placeholder="Falls back to the default model",
            value=STATE.routing_standard_model,
            id="input-routing-standard",
        )
        yield Label("Tiered high model (optional):", classes="field-label")
        yield Input(
            placeholder="Falls back to the standard/default model",
            value=STATE.routing_high_model,
            id="input-routing-high",
        )
        yield Label("Tiered extra-high model (optional):", classes="field-label")
        yield Input(
            placeholder="Falls back to the high/standard/default model",
            value=STATE.routing_extra_high_model,
            id="input-routing-extra-high",
        )
        yield Horizontal(
            Label("Allow automatic escalation between tiers  ", classes="field-label"),
            Switch(value=STATE.routing_auto_escalation, id="sw-routing-auto-escalation"),
        )

        yield NavButtons(show_back=True, next_label="Continue →")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            custom_llm = self.query_one("#input-llm-custom", Input).value.strip()
            if custom_llm:
                STATE.default_llm_model = custom_llm
            else:
                STATE.default_llm_model = str(
                    self.query_one("#select-llm", Select).value or "anthropic/claude-sonnet-4-6"
                )
            STATE.embedding_model_id = str(
                self.query_one("#select-embedding", Select).value or "local-fallback"
            )
            STATE.model_routing_mode = str(
                self.query_one("#select-routing-mode", Select).value or "single"
            )
            STATE.routing_light_model = self.query_one("#input-routing-light", Input).value
            STATE.routing_standard_model = self.query_one("#input-routing-standard", Input).value
            STATE.routing_high_model = self.query_one("#input-routing-high", Input).value
            STATE.routing_extra_high_model = self.query_one(
                "#input-routing-extra-high", Input
            ).value
            STATE.routing_auto_escalation = self.query_one(
                "#sw-routing-auto-escalation", Switch
            ).value
            self.app.push_screen("advanced")


# -----------------------------------------------------------------------------
# Screen 13: Advanced
# -----------------------------------------------------------------------------

class AdvancedScreen(Screen):
    def compose(self) -> ComposeResult:
        yield StepHeader(13, 16, "Runtime & Retrieval Settings")
        yield HelpText(
            "These settings control the runtime loop, the web dashboard, and how semantic retrieval is backed."
        )

        yield Horizontal(
            Label("Run the web dashboard and API server  ", classes="field-label"),
            Switch(value=STATE.use_server, id="sw-server"),
        )
        yield HelpText(
            "Strongly recommended. The dashboard lets you chat, inspect state, and manage sessions visually."
        )

        yield Static()
        yield Label("Server host:", classes="field-label")
        yield Input(value=STATE.host, id="input-host")

        yield Label("Server port:", classes="field-label")
        yield Input(value=STATE.port, id="input-port")

        yield Static()
        yield Label("Creative cycle interval (seconds):", classes="field-label")
        yield Input(value=STATE.cycle_interval, id="input-cycle")
        yield HelpText(
            "How often the agent evaluates its creative ladder, daydreams, and background work. "
            "Default is 600s (10 minutes)."
        )

        yield Static()
        yield Label("Daydream interval (seconds):", classes="field-label")
        yield Input(value=STATE.daydream_interval, id="input-daydream")
        yield HelpText(
            "How often the agent runs background imaginative reflection when idle. "
            "Default is 720s (12 minutes)."
        )

        yield Static()
        yield Label("Heartbeat interval (seconds):", classes="field-label")
        yield Input(value=STATE.baa_heartbeat_interval, id="input-heartbeat")
        yield HelpText(
            "How often the background heartbeat decays somatic state and samples queue health. "
            "Default is 120s (2 minutes)."
        )

        yield Static()
        yield Label("Consolidation interval (seconds):", classes="field-label")
        yield Input(value=STATE.consolidation_interval, id="input-consolidation")
        yield HelpText(
            "How often the agent performs deep-memory consolidation. "
            "Default is 86400s (24 hours)."
        )

        yield Static()
        yield Label("Qdrant URL (optional):", classes="field-label")
        yield Input(
            placeholder="http://localhost:6333",
            value=STATE.qdrant_url,
            id="input-qdrant-url",
        )
        yield Label("Qdrant API key (optional):", classes="field-label")
        yield Input(
            placeholder="Qdrant API key",
            value=STATE.qdrant_api_key,
            id="input-qdrant-api-key",
        )
        yield Label("Qdrant collection:", classes="field-label")
        yield Input(
            placeholder="opencas_embeddings",
            value=STATE.qdrant_collection,
            id="input-qdrant-collection",
        )
        yield HelpText(
            "Leave Qdrant blank to rely on the local vector backends. If configured, OpenCAS can "
            "use the remote collection for embeddings."
        )

        yield Horizontal(
            Label("Enable local HNSW index  ", classes="field-label"),
            Switch(value=STATE.hnsw_enabled, id="sw-hnsw-enabled"),
        )
        yield Label("HNSW M:", classes="field-label")
        yield Input(value=STATE.hnsw_m, id="input-hnsw-m")
        yield Label("HNSW ef_construction:", classes="field-label")
        yield Input(value=STATE.hnsw_ef_construction, id="input-hnsw-ef")
        yield HelpText(
            "These tune the local approximate-nearest-neighbor index. Defaults are sensible; "
            "only change them if you know you need different trade-offs."
        )

        yield NavButtons(show_back=True, next_label="Integrations →")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            STATE.use_server = self.query_one("#sw-server", Switch).value
            STATE.host = self.query_one("#input-host", Input).value or "127.0.0.1"
            STATE.port = self.query_one("#input-port", Input).value or "8080"
            STATE.cycle_interval = self.query_one("#input-cycle", Input).value or "600"
            STATE.daydream_interval = self.query_one("#input-daydream", Input).value or "720"
            STATE.baa_heartbeat_interval = (
                self.query_one("#input-heartbeat", Input).value or "120"
            )
            STATE.consolidation_interval = (
                self.query_one("#input-consolidation", Input).value or "86400"
            )
            STATE.qdrant_url = self.query_one("#input-qdrant-url", Input).value
            STATE.qdrant_api_key = self.query_one("#input-qdrant-api-key", Input).value
            STATE.qdrant_collection = (
                self.query_one("#input-qdrant-collection", Input).value or "opencas_embeddings"
            )
            STATE.hnsw_enabled = self.query_one("#sw-hnsw-enabled", Switch).value
            STATE.hnsw_m = self.query_one("#input-hnsw-m", Input).value or "16"
            STATE.hnsw_ef_construction = self.query_one("#input-hnsw-ef", Input).value or "200"
            self.app.push_screen("integrations")


# -----------------------------------------------------------------------------
# Screen 14: Integrations
# -----------------------------------------------------------------------------

class IntegrationsScreen(Screen):
    def compose(self) -> ComposeResult:
        yield StepHeader(14, 16, "Integrations & Safety")
        yield HelpText(
            "These options cover on-demand MCP servers, Telegram access, and execution sandboxing."
        )

        yield Horizontal(
            Label("Auto-register configured MCP servers  ", classes="field-label"),
            Switch(value=STATE.mcp_auto_register, id="sw-mcp-auto-register"),
        )
        yield Label("MCP servers JSON (optional):", classes="field-label")
        yield TextArea(STATE.mcp_servers_json, id="input-mcp-servers")
        yield HelpText(
            "Provide a JSON array of MCP server objects. Leave blank if you do not want any "
            "preconfigured MCP servers at bootstrap."
        )

        yield Static()
        yield Horizontal(
            Label("Enable Telegram integration  ", classes="field-label"),
            Switch(value=STATE.telegram_enabled, id="sw-telegram-enabled"),
        )
        yield Label("Telegram bot token (optional):", classes="field-label")
        yield Input(
            placeholder="123456:ABCDEF",
            value=STATE.telegram_bot_token,
            id="input-telegram-bot-token",
        )
        yield Label("Telegram DM policy:", classes="field-label")
        yield Select(
            [
                ("Disabled", "disabled"),
                ("Pairing", "pairing"),
                ("Allowlist", "allowlist"),
                ("Open", "open"),
            ],
            allow_blank=False,
            value=STATE.telegram_dm_policy,
            id="select-telegram-dm-policy",
        )
        yield Label("Telegram allow-from IDs (optional):", classes="field-label")
        yield Input(
            placeholder="12345, 67890",
            value=STATE.telegram_allow_from,
            id="input-telegram-allow-from",
        )
        yield Label("Telegram poll interval (seconds):", classes="field-label")
        yield Input(
            value=STATE.telegram_poll_interval_seconds,
            id="input-telegram-poll-interval",
        )
        yield Label("Telegram pairing TTL (seconds):", classes="field-label")
        yield Input(
            value=STATE.telegram_pairing_ttl_seconds,
            id="input-telegram-pairing-ttl",
        )
        yield Label("Telegram API base URL:", classes="field-label")
        yield Input(
            value=STATE.telegram_api_base_url,
            id="input-telegram-api-base-url",
        )

        yield Static()
        yield Label("Sandbox mode:", classes="field-label")
        yield Select(
            [
                ("Off", "off"),
                ("Workspace only", "workspace-only"),
                ("Allow list", "allow-list"),
                ("Docker", "docker"),
            ],
            allow_blank=False,
            value=STATE.sandbox_mode,
            id="select-sandbox-mode",
        )
        yield Label("Sandbox allowed roots (optional):", classes="field-label")
        yield Input(
            placeholder="/home/you/project, /mnt/shared",
            value=STATE.sandbox_allowed_roots,
            id="input-sandbox-allowed-roots",
        )
        yield HelpText(
            "Allow-list mode uses the roots you specify here. Workspace-only mode relies on the "
            "workspace roots you configured earlier."
        )

        yield NavButtons(show_back=True, next_label="Review →")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            STATE.mcp_auto_register = self.query_one("#sw-mcp-auto-register", Switch).value
            STATE.mcp_servers_json = self.query_one("#input-mcp-servers", TextArea).text
            STATE.telegram_enabled = self.query_one("#sw-telegram-enabled", Switch).value
            STATE.telegram_bot_token = self.query_one("#input-telegram-bot-token", Input).value
            STATE.telegram_dm_policy = str(
                self.query_one("#select-telegram-dm-policy", Select).value or "pairing"
            )
            STATE.telegram_allow_from = self.query_one("#input-telegram-allow-from", Input).value
            STATE.telegram_poll_interval_seconds = (
                self.query_one("#input-telegram-poll-interval", Input).value or "1.0"
            )
            STATE.telegram_pairing_ttl_seconds = (
                self.query_one("#input-telegram-pairing-ttl", Input).value or "3600"
            )
            STATE.telegram_api_base_url = (
                self.query_one("#input-telegram-api-base-url", Input).value
                or "https://api.telegram.org"
            )
            STATE.sandbox_mode = str(
                self.query_one("#select-sandbox-mode", Select).value or "workspace-only"
            )
            STATE.sandbox_allowed_roots = self.query_one(
                "#input-sandbox-allowed-roots", Input
            ).value
            self.app.push_screen("review")


# -----------------------------------------------------------------------------
# Screen 15: Review
# -----------------------------------------------------------------------------

class ReviewScreen(Screen):
    def compose(self) -> ComposeResult:
        yield StepHeader(15, 16, "Review Your Configuration")

        extra_roots = [r.strip() for r in STATE.workspace_extra.split(",") if r.strip()]
        profiles_txt = ", ".join(STATE.selected_profiles) if STATE.selected_profiles else "(auto-detect)"
        env_keys_txt = ", ".join(STATE.credential_env_keys) if STATE.credential_env_keys else "(none)"
        managed_workspace_txt = STATE.managed_workspace_root or "(default: <primary>/workspace)"
        mcp_servers_txt = "configured" if STATE.mcp_servers_json.strip() else "(none)"
        telegram_allow_txt = STATE.telegram_allow_from or "(none)"
        sandbox_roots_txt = STATE.sandbox_allowed_roots or "(none)"

        collab_prefs = []
        if STATE.collab_pair:
            collab_prefs.append("pair-programming")
        if STATE.collab_async:
            collab_prefs.append("async notes")
        if STATE.collab_backforth:
            collab_prefs.append("discussion-heavy")
        if STATE.collab_minimal:
            collab_prefs.append("minimal interruptions")

        goals_md = ""
        if STATE.goal_1 or STATE.goal_2 or STATE.goal_3:
            goals_md = "\n### Initial Goals\n"
            if STATE.goal_1:
                goals_md += f"- **Goal 1:** {STATE.goal_1}\n"
            if STATE.goal_2:
                goals_md += f"- **Goal 2:** {STATE.goal_2}\n"
            if STATE.goal_3:
                goals_md += f"- **Goal 3:** {STATE.goal_3}\n"
            goals_md += f"- **Timeframe:** {STATE.goal_timeframe}\n"

        md = f"""
### Partnership Profile
- **Agent archetype:** {BUILTIN_AGENT_PROFILES[STATE.agent_profile_id].display_name}
- **Agent name:** {STATE.persona_name}
- **Persona accent:** {STATE.persona_accent}
- **Your name:** {STATE.user_name or '(not set)'}

### About You
- **Job/role:** {STATE.user_job or '(not set)'}
- **Interests:** {STATE.user_interests or '(not set)'}
- **Comm style:** {STATE.user_comm_style}
- **Learning style:** {STATE.learning_preference.replace('_', ' ')}
- **Feedback style:** {STATE.feedback_style.replace('_', ' ')}
- **Help style:** {STATE.help_style.replace('_', ' ')}
- **Collaboration prefs:** {', '.join(collab_prefs) or '(none)'}

### Vision
- **Main help:** {STATE.vision_main_help or '(not set)'}
- **Engagement:** {STATE.vision_engagement_style.replace('_', ' ')}
- **6-month success:** {STATE.vision_success_six_months or '(not set)'}
- **Working notes:** {STATE.vision_working_notes or '(not set)'}
{goals_md}
### Environment
- **State directory:** `{STATE.state_dir}`
- **Primary workspace:** `{STATE.workspace_root}`
- **Extra workspaces:** {', '.join(extra_roots) or '(none)'}
- **Managed workspace:** `{managed_workspace_txt}`

### Intelligence
- **Chat model:** `{STATE.default_llm_model}`
- **Embedding model:** `{STATE.embedding_model_id}`
- **Routing mode:** {STATE.model_routing_mode}
- **Light model:** `{STATE.routing_light_model or '(default)'}` 
- **Standard model:** `{STATE.routing_standard_model or '(default)'}` 
- **High model:** `{STATE.routing_high_model or '(default)'}` 
- **Extra-high model:** `{STATE.routing_extra_high_model or '(default)'}` 
- **Auto escalation:** {'enabled' if STATE.routing_auto_escalation else 'disabled'}
- **Credential mode:** {STATE.provider_mode}
- **Profiles:** {profiles_txt}
- **Copied env keys:** {env_keys_txt}

### Retrieval & Vectors
- **Qdrant URL:** `{STATE.qdrant_url or '(not set)'}`
- **Qdrant collection:** `{STATE.qdrant_collection or 'opencas_embeddings'}`
- **Local HNSW:** {'enabled' if STATE.hnsw_enabled else 'disabled'}
- **HNSW M / ef_construction:** {STATE.hnsw_m} / {STATE.hnsw_ef_construction}

### Integrations & Safety
- **MCP auto-register:** {'enabled' if STATE.mcp_auto_register else 'disabled'}
- **MCP servers:** {mcp_servers_txt}
- **Telegram:** {'enabled' if STATE.telegram_enabled else 'disabled'}
- **Telegram policy:** {STATE.telegram_dm_policy}
- **Telegram allow-from:** {telegram_allow_txt}
- **Sandbox mode:** {STATE.sandbox_mode}
- **Sandbox allowed roots:** {sandbox_roots_txt}

### Runtime
- **Dashboard server:** {'enabled' if STATE.use_server else 'disabled'}
- **Host:** `{STATE.host}:{STATE.port}`
- **Creative cycle:** {STATE.cycle_interval}s
- **Daydream cadence:** {STATE.daydream_interval}s
- **Heartbeat cadence:** {STATE.baa_heartbeat_interval}s
- **Consolidation:** {STATE.consolidation_interval}s
        """
        yield Markdown(md)
        yield HelpText(
            "If everything looks good, click Bootstrap Agent to bring your CAS to life."
        )
        yield NavButtons(show_back=True, next_label="Bootstrap Agent →")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-next":
            self.app.push_screen("bootstrap")
