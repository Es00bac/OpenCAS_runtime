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
        yield StepHeader(10, 15, "Agent Home & Workspace")
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

        yield NavButtons(show_back=True, next_label="Continue →")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            STATE.state_dir = self.query_one("#input-state-dir", Input).value or "./.opencas"
            STATE.workspace_root = self.query_one("#input-workspace", Input).value or "."
            STATE.workspace_extra = self.query_one("#input-workspace-extra", Input).value or ""
            self.app.push_screen("credentials")


# -----------------------------------------------------------------------------
# Screen 11: Credentials
# -----------------------------------------------------------------------------

class CredentialsScreen(Screen):
    """Provider credential setup with explanations for new users."""

    def compose(self) -> ComposeResult:
        yield StepHeader(11, 15, "Brains & Credentials")
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

        yield Static()
        self._custom_container = Container(id="custom-options")
        with self._custom_container:
            yield Label("Path to custom config.json:", classes="field-label")
            yield Input(placeholder="/path/to/config.json", id="input-custom-config")
            yield Label("Path to custom .env:", classes="field-label")
            yield Input(placeholder="/path/to/.env", id="input-custom-env")

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
                STATE.credential_source_config = str(Path.home() / ".open_llm_auth" / "config.json")
            elif STATE.provider_mode == "custom":
                STATE.selected_profiles = []
                STATE.credential_env_keys = []
                STATE.credential_source_config = ""
                STATE.provider_config_path = self.query_one("#input-custom-config", Input).value
                STATE.provider_env_path = self.query_one("#input-custom-env", Input).value
            else:
                STATE.selected_profiles = []
                STATE.credential_env_keys = []
                STATE.credential_source_config = ""
            self.app.push_screen("models")


# -----------------------------------------------------------------------------
# Screen 12: Models
# -----------------------------------------------------------------------------

class ModelsScreen(Screen):
    def compose(self) -> ComposeResult:
        model_choices = discover_model_choices(STATE)
        yield StepHeader(12, 15, "Model Preferences")
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
            self.app.push_screen("advanced")


# -----------------------------------------------------------------------------
# Screen 13: Advanced
# -----------------------------------------------------------------------------

class AdvancedScreen(Screen):
    def compose(self) -> ComposeResult:
        yield StepHeader(13, 15, "Autonomy & Server Settings")
        yield HelpText(
            "These control how often the agent thinks autonomously and whether it runs a web dashboard."
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
            "Default is 300s (5 minutes)."
        )

        yield Static()
        yield Label("Consolidation interval (seconds):", classes="field-label")
        yield Input(value=STATE.consolidation_interval, id="input-consolidation")
        yield HelpText(
            "How often the agent performs deep-memory consolidation. "
            "Default is 86400s (24 hours)."
        )

        yield NavButtons(show_back=True, next_label="Review →")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            STATE.use_server = self.query_one("#sw-server", Switch).value
            STATE.host = self.query_one("#input-host", Input).value or "127.0.0.1"
            STATE.port = self.query_one("#input-port", Input).value or "8080"
            STATE.cycle_interval = self.query_one("#input-cycle", Input).value or "300"
            STATE.consolidation_interval = (
                self.query_one("#input-consolidation", Input).value or "86400"
            )
            self.app.push_screen("review")


# -----------------------------------------------------------------------------
# Screen 14: Review
# -----------------------------------------------------------------------------

class ReviewScreen(Screen):
    def compose(self) -> ComposeResult:
        yield StepHeader(14, 15, "Review Your Configuration")

        extra_roots = [r.strip() for r in STATE.workspace_extra.split(",") if r.strip()]
        profiles_txt = ", ".join(STATE.selected_profiles) if STATE.selected_profiles else "(auto-detect)"
        env_keys_txt = ", ".join(STATE.credential_env_keys) if STATE.credential_env_keys else "(none)"

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

### Intelligence
- **Chat model:** `{STATE.default_llm_model}`
- **Embedding model:** `{STATE.embedding_model_id}`
- **Credential mode:** {STATE.provider_mode}
- **Profiles:** {profiles_txt}
- **Copied env keys:** {env_keys_txt}

### Runtime
- **Dashboard server:** {'enabled' if STATE.use_server else 'disabled'}
- **Host:** `{STATE.host}:{STATE.port}`
- **Creative cycle:** {STATE.cycle_interval}s
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


