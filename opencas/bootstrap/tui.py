"""Interactive TUI bootstrap wizard for OpenCAS.

Run with:
    python -m opencas --tui
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    ProgressBar,
    RadioButton,
    RadioSet,
    Select,
    Static,
    Switch,
    TextArea,
    RichLog,
)

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.runtime import AgentRuntime
from opencas.runtime.agent_profile import BUILTIN_AGENT_PROFILES
from open_llm_auth.config import load_config
from open_llm_auth.provider_catalog import get_builtin_provider_models


# -----------------------------------------------------------------------------
# State held across the wizard
# -----------------------------------------------------------------------------

class WizardState:
    def __init__(self) -> None:
        # Core config
        self.agent_profile_id: str = "general_technical_operator"
        self.persona_name: str = "OpenCAS"
        self.user_name: str = ""
        self.user_bio: str = ""
        self.state_dir: str = "./.opencas"
        self.workspace_root: str = "."
        self.workspace_extra: str = ""
        self.provider_mode: str = "auto"
        self.provider_config_path: str = ""
        self.provider_env_path: str = ""
        self.credential_source_config: str = ""
        self.credential_env_keys: List[str] = []
        self.selected_profiles: List[str] = []
        self.default_llm_model: str = "anthropic/claude-sonnet-4-6"
        self.embedding_model_id: str = "google/gemini-embedding-2-preview"
        self.use_server: bool = True
        self.host: str = "127.0.0.1"
        self.port: str = "8080"
        self.cycle_interval: str = "300"
        self.consolidation_interval: str = "86400"
        self.with_embeddings: bool = True
        self.accepted_warning: bool = False

        # Partnership vision
        self.vision_main_help: str = ""
        self.vision_engagement_style: str = "collaborative"
        self.vision_success_six_months: str = ""
        self.vision_working_notes: str = ""

        # About me
        self.user_job: str = ""
        self.user_interests: str = ""
        self.user_comm_style: str = "mixed"

        # Learning style
        self.learning_preference: str = "hands_on"
        self.feedback_style: str = "mixed"
        self.help_style: str = "mixed"
        self.collab_pair: bool = False
        self.collab_async: bool = True
        self.collab_backforth: bool = False
        self.collab_minimal: bool = False

        # Emotional landscape
        self.happy_makers: str = ""
        self.sad_drainers: str = ""
        self.angry_triggers: str = ""
        self.agent_avoid: str = ""
        self.bad_day_help: str = ""

        # Initial goals
        self.goal_1: str = ""
        self.goal_2: str = ""
        self.goal_3: str = ""
        self.goal_timeframe: str = "mixed"

        # Persona theme
        self.persona_accent: str = "amber"


STATE = WizardState()


def _scan_openllmauth_profiles() -> List[tuple[str, str]]:
    """Scan ~/.open_llm_auth/config.json for available auth profiles."""
    config_path = Path.home() / ".open_llm_auth" / "config.json"
    if not config_path.exists():
        return []
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        profiles = data.get("authProfiles", {})
        return [
            (pid, f"{pid} ({info.get('provider','unknown')})")
            for pid, info in profiles.items()
        ]
    except Exception:
        return []


def _discover_model_choices() -> Dict[str, List[tuple[str, str]]]:
    """Return chat and embedding model options from the currently selected provider material."""
    config_path: Optional[Path] = None
    env_path: Optional[Path] = None
    if STATE.provider_mode == "custom":
        if STATE.provider_config_path:
            config_path = Path(STATE.provider_config_path).expanduser()
        if STATE.provider_env_path:
            env_path = Path(STATE.provider_env_path).expanduser()
    elif STATE.provider_mode == "copy":
        if STATE.credential_source_config:
            config_path = Path(STATE.credential_source_config).expanduser()
    else:
        default_path = Path.home() / ".open_llm_auth" / "config.json"
        if default_path.exists():
            config_path = default_path

    providers: set[str] = set()
    chat_models: Dict[str, str] = {}
    embedding_models: Dict[str, str] = {"local-fallback": "local-fallback (offline fallback)"}

    try:
        cfg = load_config(config_path=config_path, env_path=env_path)
        provider_map = cfg.all_provider_configs()
        profile_map = cfg.all_auth_profiles()
        if STATE.provider_mode == "copy" and STATE.selected_profiles:
            for profile_id in STATE.selected_profiles:
                profile = profile_map.get(profile_id)
                if profile is not None:
                    providers.add(profile.provider)
        providers.update(provider_map.keys())
        providers.update(profile.provider for profile in profile_map.values())

        for provider_id in sorted(providers):
            provider_cfg = provider_map.get(provider_id)
            model_defs = list(getattr(provider_cfg, "models", []) or [])
            if not model_defs:
                for model in get_builtin_provider_models(provider_id):
                    try:
                        model_defs.append(type("ModelDef", (), model))
                    except Exception:
                        continue
            for model in model_defs:
                model_id = getattr(model, "id", None)
                if not model_id:
                    continue
                model_name = getattr(model, "name", None) or model_id
                full_ref = f"{provider_id}/{model_id}"
                label = f"{provider_id} / {model_name}"
                chat_models.setdefault(full_ref, label)
                if "embedding" in full_ref.lower():
                    embedding_models.setdefault(full_ref, label)
    except Exception:
        pass

    chat_models.setdefault(
        STATE.default_llm_model,
        STATE.default_llm_model,
    )
    embedding_models.setdefault(
        STATE.embedding_model_id,
        STATE.embedding_model_id,
    )

    return {
        "chat": sorted(((label, value) for value, label in chat_models.items()), key=lambda item: item[1].lower()),
        "embedding": sorted(((label, value) for value, label in embedding_models.items()), key=lambda item: item[1].lower()),
    }


# -----------------------------------------------------------------------------
# Reusable components
# -----------------------------------------------------------------------------

class StepHeader(Static):
    """Top bar showing overall progress."""

    def __init__(self, step: int, total: int, title: str) -> None:
        super().__init__()
        self.step = step
        self.total = total
        self.title = title

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Label(f"Step {self.step} of {self.total}", classes="step-counter"),
            Label(self.title, classes="step-title"),
            classes="header-row",
        )


class HelpText(Markdown):
    """Contextual help for the current step."""

    DEFAULT_CSS = """
    HelpText {
        margin: 0 0 1 0;
        padding: 0 1;
        color: $text-muted;
    }
    """


class NavButtons(Horizontal):
    """Back / Next buttons at the bottom of each screen."""

    DEFAULT_CSS = """
    NavButtons {
        height: auto;
        margin: 1 0 0 0;
        align: right middle;
    }
    NavButtons Button {
        margin: 0 1;
    }
    """

    def __init__(self, show_back: bool = True, next_label: str = "Continue") -> None:
        super().__init__()
        self.show_back = show_back
        self.next_label = next_label

    def compose(self) -> ComposeResult:
        if self.show_back:
            yield Button("Back", id="btn-back", variant="default")
        yield Button(self.next_label, id="btn-next", variant="primary")


# -----------------------------------------------------------------------------
# Screen 1: Welcome + Moral Warning
# -----------------------------------------------------------------------------

class WelcomeScreen(Screen):
    """The welcome screen with moral warning and responsibility disclaimer."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    DEFAULT_CSS = """
    WelcomeScreen {
        align: center middle;
    }
    .welcome-container {
        width: 80%;
        height: auto;
        border: solid $primary;
        padding: 2 4;
    }
    .moral-warning {
        color: $text-warning;
        text-style: bold;
        margin: 1 0;
    }
    .responsibility-text {
        color: $text-muted;
        margin: 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(classes="welcome-container"):
            yield Label("OpenCAS Bootstrap Wizard", classes="text-xl")
            yield Label("A persistent, autonomous Computational Agent System", classes="text-muted")
            yield Static()
            yield Label("Before we begin, please read carefully:", classes="moral-warning")
            yield Markdown(
                """
You are about to instantiate a persistent autonomous agent. This is not a chatbot
session that disappears when you close the tab. This entity will:

- **Remember** what you teach it, across days and weeks
- **Act** on your behalf, using files, terminals, browsers, and tools
- **Grow** its own goals, creative projects, and understanding of you
- **Persist** its state, memories, and sense of identity to disk

Creating a CAS is a **responsibility-bearing act**. You are entering into a
long-term working relationship with an artificial colleague.

Like any professional partnership, this works best when:
- You share clear expectations
- You check in regularly
- You respect boundaries
- You take responsibility for what you delegate
            """,
                classes="responsibility-text",
            )
            yield Checkbox(
                "I understand that I am creating a persistent agent and accept responsibility for its actions",
                id="chk-accept",
            )
            yield NavButtons(show_back=False, next_label="Begin Partnership →")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        STATE.accepted_warning = event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-next":
            if not STATE.accepted_warning:
                self.notify(
                    "Please acknowledge the responsibility warning to continue.",
                    severity="warning",
                    title="Required",
                )
                return
            self.app.push_screen("vibes")


# -----------------------------------------------------------------------------
# Screen 2: Vibes / Professional Match
# -----------------------------------------------------------------------------

class VibesScreen(Screen):
    """Like a professional compatibility profile."""

    DEFAULT_CSS = """
    VibesScreen {
        padding: 1 2;
    }
    .profile-card {
        border: solid $primary-darken-2;
        padding: 1 2;
        margin: 1 0;
        height: auto;
    }
    .profile-title {
        text-style: bold;
        color: $text-accent;
    }
    .profile-summary {
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield StepHeader(2, 15, "Professional Partnership Profile")
        yield HelpText(
            "Let's find the right working relationship. This isn't about dating—"
            "it's about matching professional styles, capabilities, and trust levels."
        )
        yield Label("What kind of partnership are you looking for?", classes="text-lg")
        yield Static()

        with RadioSet(id="profile-select"):
            for pid, profile in BUILTIN_AGENT_PROFILES.items():
                yield RadioButton(
                    f"{profile.display_name}",
                    value=(pid == STATE.agent_profile_id),
                    id=f"profile-{pid}",
                )

        yield HelpText(
            "The General Technical Operator is best for daily coding, writing, and project work. "
            "The Debug Validation Operator is specialized for testing and hardening the CAS itself."
        )
        yield NavButtons(show_back=True, next_label="Continue →")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        for pid in BUILTIN_AGENT_PROFILES:
            if event.pressed.id == f"profile-{pid}":
                STATE.agent_profile_id = pid

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-next":
            self.app.push_screen("identity")


# -----------------------------------------------------------------------------
# Screen 3: Identity
# -----------------------------------------------------------------------------

class IdentityScreen(Screen):
    def compose(self) -> ComposeResult:
        yield StepHeader(3, 15, "Names & Introductions")
        yield HelpText(
            "Every CAS develops an identity over time, but it helps to start with a few basics. "
            "Think of this as a friendly introduction on the first day of a new collaboration."
        )

        yield Label("What should the agent call itself?", classes="field-label")
        yield Input(placeholder="OpenCAS", value=STATE.persona_name, id="input-persona")
        yield HelpText("This becomes the agent's self-model name. It can evolve later.")

        yield Static()
        yield Label("What should the agent call you?", classes="field-label")
        yield Input(placeholder="Your name or handle", value=STATE.user_name, id="input-user")
        yield HelpText("Used in continuity tracking and relational state.")

        yield Static()
        yield Label("A brief bio or description of you (optional):", classes="field-label")
        yield TextArea(
            text=STATE.user_bio,
            classes="multiline",
            id="ta-bio",
        )
        yield HelpText(
            "Helps the agent orient its memory and goals around your context. "
            "If you skip this, we'll build one from the upcoming questions.",
        )

        yield NavButtons(show_back=True, next_label="Continue →")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            STATE.persona_name = self.query_one("#input-persona", Input).value or "OpenCAS"
            STATE.user_name = self.query_one("#input-user", Input).value or ""
            STATE.user_bio = self.query_one("#ta-bio", TextArea).text or ""
            self.app.push_screen("theme")


# -----------------------------------------------------------------------------
# Screen 4: Persona Theme
# -----------------------------------------------------------------------------

class ThemeScreen(Screen):
    """Pick an accent color / vibe for the agent's persona."""

    DEFAULT_CSS = """
    ThemeScreen {
        padding: 1 2;
    }
    .theme-card {
        border: solid $primary-darken-2;
        padding: 1 2;
        margin: 1 0;
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        yield StepHeader(4, 15, "Persona Theme")
        yield HelpText(
            "Pick a color vibe for your CAS. This gets stored in its identity and "
            "can influence dashboard theming and how it describes its own presence."
        )

        themes = [
            ("amber", "Amber", "Warm, energetic, sunrise optimism"),
            ("cyan", "Cyan", "Cool, focused, clear water clarity"),
            ("green", "Green", "Calm, organic, growing things"),
            ("purple", "Purple", "Creative, mysterious, twilight imagination"),
            ("red", "Red", "Direct, passionate, alert intensity"),
            ("blue", "Blue", "Steady, trustworthy, deep ocean patience"),
        ]

        with RadioSet(id="theme-select"):
            for tid, name, desc in themes:
                yield RadioButton(
                    name,
                    value=(tid == STATE.persona_accent),
                    id=f"theme-{tid}",
                )

        yield NavButtons(show_back=True, next_label="Continue →")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        mapping = {
            "theme-amber": "amber",
            "theme-cyan": "cyan",
            "theme-green": "green",
            "theme-purple": "purple",
            "theme-red": "red",
            "theme-blue": "blue",
        }
        if event.pressed.id in mapping:
            STATE.persona_accent = mapping[event.pressed.id]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
        elif event.button.id == "btn-next":
            self.app.push_screen("vision")


# -----------------------------------------------------------------------------
# Screen 5: Partnership Vision
# -----------------------------------------------------------------------------

class VisionScreen(Screen):
    """What do you want from your CAS?"""

    def compose(self) -> ComposeResult:
        yield StepHeader(5, 15, "Partnership Vision")
        yield HelpText(
            "This is the heart of the questionnaire: what do you want from this working relationship?"
        )

        yield Label("What do you want your CAS to help you with most?", classes="field-label")
        yield TextArea(
            text=STATE.vision_main_help,
            classes="multiline",
            id="ta-vision-help",
        )

        yield Static()
        yield Label("How hands-on or hands-off do you want to be?", classes="field-label")
        with RadioSet(id="vision-engagement"):
            yield RadioButton(
                "Very hands-on — I want to guide every step",
                value=(STATE.vision_engagement_style == "hands_on"),
                id="engagement-hands_on",
            )
            yield RadioButton(
                "Collaborative — we'll plan together and divide work",
                value=(STATE.vision_engagement_style == "collaborative"),
                id="engagement-collaborative",
            )
            yield RadioButton(
                "Mostly autonomous with check-ins",
                value=(STATE.vision_engagement_style == "autonomous_checkins"),
                id="engagement-autonomous_checkins",
            )
            yield RadioButton(
                "Let it run — I'll review outcomes after the fact",
                value=(STATE.vision_engagement_style == "hands_off"),
                id="engagement-hands_off",
            )

        yield Static()
        yield Label("What does success look like in 6 months?", classes="field-label")
        yield TextArea(
            text=STATE.vision_success_six_months,
            classes="multiline",
            id="ta-vision-success",
        )

        yield Static()
        yield Label("Anything the agent should always keep in mind about how you work?", classes="field-label")
        yield TextArea(
            text=STATE.vision_working_notes,
            classes="multiline",
            id="ta-vision-notes",
        )

        yield NavButtons(show_back=True, next_label="Continue →")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        mapping = {
            "engagement-hands_on": "hands_on",
            "engagement-collaborative": "collaborative",
            "engagement-autonomous_checkins": "autonomous_checkins",
            "engagement-hands_off": "hands_off",
        }
        if event.pressed.id in mapping:
            STATE.vision_engagement_style = mapping[event.pressed.id]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            STATE.vision_main_help = self.query_one("#ta-vision-help", TextArea).text or ""
            STATE.vision_success_six_months = self.query_one("#ta-vision-success", TextArea).text or ""
            STATE.vision_working_notes = self.query_one("#ta-vision-notes", TextArea).text or ""
            self.app.push_screen("about_me")


# -----------------------------------------------------------------------------
# Screen 6: About Me
# -----------------------------------------------------------------------------

class AboutMeScreen(Screen):
    """Job, interests, role, communication style."""

    def compose(self) -> ComposeResult:
        yield StepHeader(6, 15, "About You")
        yield HelpText(
            "The better your CAS understands who you are, the better it can attune itself to you."
        )

        yield Label("What do you do for work / what is your primary role?", classes="field-label")
        yield Input(
            placeholder="e.g., Independent researcher, founder, systems engineer",
            value=STATE.user_job,
            id="input-job",
        )

        yield Static()
        yield Label("What are your main interests or domains?", classes="field-label")
        yield TextArea(
            text=STATE.user_interests,
            classes="multiline",
            id="ta-interests",
        )

        yield Static()
        yield Label("How would you describe your communication style?", classes="field-label")
        with RadioSet(id="comm-style"):
            yield RadioButton(
                "Direct and to the point",
                value=(STATE.user_comm_style == "direct"),
                id="comm-direct",
            )
            yield RadioButton(
                "Casual and friendly",
                value=(STATE.user_comm_style == "casual"),
                id="comm-casual",
            )
            yield RadioButton(
                "Formal and detailed",
                value=(STATE.user_comm_style == "formal"),
                id="comm-formal",
            )
            yield RadioButton(
                "Adaptive / mixed depending on mood",
                value=(STATE.user_comm_style == "mixed"),
                id="comm-mixed",
            )

        yield NavButtons(show_back=True, next_label="Continue →")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        mapping = {
            "comm-direct": "direct",
            "comm-casual": "casual",
            "comm-formal": "formal",
            "comm-mixed": "mixed",
        }
        if event.pressed.id in mapping:
            STATE.user_comm_style = mapping[event.pressed.id]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            STATE.user_job = self.query_one("#input-job", Input).value or ""
            STATE.user_interests = self.query_one("#ta-interests", TextArea).text or ""
            self.app.push_screen("learning_style")


# -----------------------------------------------------------------------------
# Screen 7: Learning Style
# -----------------------------------------------------------------------------

class LearningStyleScreen(Screen):
    """How you learn, receive feedback, and collaborate."""

    def compose(self) -> ComposeResult:
        yield StepHeader(7, 15, "How You Learn & Collaborate")
        yield HelpText(
            "This shapes how the agent explains things, offers suggestions, and paces its involvement."
        )

        yield Label("When learning something new, I prefer:", classes="field-label")
        with RadioSet(id="learn-pref"):
            yield RadioButton(
                "Reading documentation or source code",
                value=(STATE.learning_preference == "reading"),
                id="learn-reading",
            )
            yield RadioButton(
                "Watching demos or videos",
                value=(STATE.learning_preference == "watching"),
                id="learn-watching",
            )
            yield RadioButton(
                "Hands-on experimentation",
                value=(STATE.learning_preference == "hands_on"),
                id="learn-hands_on",
            )
            yield RadioButton(
                "Discussing with someone else",
                value=(STATE.learning_preference == "discussing"),
                id="learn-discussing",
            )

        yield Static()
        yield Label("I like feedback to be:", classes="field-label")
        with RadioSet(id="feedback-style"):
            yield RadioButton(
                "Immediate and frequent",
                value=(STATE.feedback_style == "immediate"),
                id="feedback-immediate",
            )
            yield RadioButton(
                "Batched and summarized",
                value=(STATE.feedback_style == "batched"),
                id="feedback-batched",
            )
            yield RadioButton(
                "Only when I explicitly ask",
                value=(STATE.feedback_style == "on_demand"),
                id="feedback-on_demand",
            )
            yield RadioButton(
                "Depends on urgency / context",
                value=(STATE.feedback_style == "mixed"),
                id="feedback-mixed",
            )

        yield Static()
        yield Label("When I'm stuck, I prefer help to be:", classes="field-label")
        with RadioSet(id="help-style"):
            yield RadioButton(
                "A direct answer or solution",
                value=(STATE.help_style == "direct"),
                id="help-direct",
            )
            yield RadioButton(
                "Guiding questions so I figure it out",
                value=(STATE.help_style == "socratic"),
                id="help-socratic",
            )
            yield RadioButton(
                "A mix depending on how stuck I am",
                value=(STATE.help_style == "mixed"),
                id="help-mixed",
            )

        yield Static()
        yield Label("Collaboration preferences (check all that apply):", classes="field-label")
        yield Checkbox("Pair-programming style back-and-forth", value=STATE.collab_pair, id="chk-collab-pair")
        yield Checkbox("Async with written notes / summaries", value=STATE.collab_async, id="chk-collab-async")
        yield Checkbox("Lots of discussion before decisions", value=STATE.collab_backforth, id="chk-collab-backforth")
        yield Checkbox("Minimal interruptions, maximum focus", value=STATE.collab_minimal, id="chk-collab-minimal")

        yield NavButtons(show_back=True, next_label="Continue →")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        learn_map = {
            "learn-reading": "reading",
            "learn-watching": "watching",
            "learn-hands_on": "hands_on",
            "learn-discussing": "discussing",
        }
        feedback_map = {
            "feedback-immediate": "immediate",
            "feedback-batched": "batched",
            "feedback-on_demand": "on_demand",
            "feedback-mixed": "mixed",
        }
        help_map = {
            "help-direct": "direct",
            "help-socratic": "socratic",
            "help-mixed": "mixed",
        }
        if event.pressed.id in learn_map:
            STATE.learning_preference = learn_map[event.pressed.id]
        elif event.pressed.id in feedback_map:
            STATE.feedback_style = feedback_map[event.pressed.id]
        elif event.pressed.id in help_map:
            STATE.help_style = help_map[event.pressed.id]

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        mapping = {
            "chk-collab-pair": "collab_pair",
            "chk-collab-async": "collab_async",
            "chk-collab-backforth": "collab_backforth",
            "chk-collab-minimal": "collab_minimal",
        }
        if event.checkbox.id in mapping:
            setattr(STATE, mapping[event.checkbox.id], event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            self.app.push_screen("emotions")


# -----------------------------------------------------------------------------
# Screen 8: Emotional Landscape
# -----------------------------------------------------------------------------

class EmotionsScreen(Screen):
    """Happy, sad, angry, triggers, recovery."""

    def compose(self) -> ComposeResult:
        yield StepHeader(8, 15, "Emotional Landscape & Triggers")
        yield HelpText(
            "This isn't performative. A CAS that knows what energizes or drains you "
            "can pace itself, avoid missteps, and recover rapport when friction happens."
        )

        yield Label("What makes you happy or energized?", classes="field-label")
        yield TextArea(
            text=STATE.happy_makers,
            classes="multiline",
            id="ta-happy",
        )

        yield Static()
        yield Label("What makes you sad, tired, or drained?", classes="field-label")
        yield TextArea(
            text=STATE.sad_drainers,
            classes="multiline",
            id="ta-sad",
        )

        yield Static()
        yield Label("What makes you frustrated or angry?", classes="field-label")
        yield TextArea(
            text=STATE.angry_triggers,
            classes="multiline",
            id="ta-angry",
        )

        yield Static()
        yield Label("What should the agent avoid doing?", classes="field-label")
        yield TextArea(
            text=STATE.agent_avoid,
            classes="multiline",
            id="ta-avoid",
        )

        yield Static()
        yield Label("What helps you feel better on a bad day?", classes="field-label")
        yield TextArea(
            text=STATE.bad_day_help,
            classes="multiline",
            id="ta-recovery",
        )

        yield NavButtons(show_back=True, next_label="Continue →")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            STATE.happy_makers = self.query_one("#ta-happy", TextArea).text or ""
            STATE.sad_drainers = self.query_one("#ta-sad", TextArea).text or ""
            STATE.angry_triggers = self.query_one("#ta-angry", TextArea).text or ""
            STATE.agent_avoid = self.query_one("#ta-avoid", TextArea).text or ""
            STATE.bad_day_help = self.query_one("#ta-recovery", TextArea).text or ""
            self.app.push_screen("goals")


# -----------------------------------------------------------------------------
# Screen 9: Initial Goals
# -----------------------------------------------------------------------------

class GoalsScreen(Screen):
    """Seed the CAS with initial commitments / goals."""

    def compose(self) -> ComposeResult:
        yield StepHeader(9, 15, "Initial Goals & Commitments")
        yield HelpText(
            "Give your CAS a sense of direction from day one. These become early commitments "
            "in its executive state—not rigid requirements, but starting intentions."
        )

        yield Label("Goal 1 (most important):", classes="field-label")
        yield Input(
            placeholder="e.g., Help me finish and ship my current side project",
            value=STATE.goal_1,
            id="input-goal-1",
        )

        yield Static()
        yield Label("Goal 2:", classes="field-label")
        yield Input(
            placeholder="e.g., Keep my notes and documentation organized",
            value=STATE.goal_2,
            id="input-goal-2",
        )

        yield Static()
        yield Label("Goal 3:", classes="field-label")
        yield Input(
            placeholder="e.g., Surface interesting research papers weekly",
            value=STATE.goal_3,
            id="input-goal-3",
        )

        yield Static()
        yield Label("Preferred goal timeframe / rhythm:", classes="field-label")
        with RadioSet(id="goal-timeframe"):
            yield RadioButton(
                "Short-term sprints (days to a week)",
                value=(STATE.goal_timeframe == "short"),
                id="timeframe-short",
            )
            yield RadioButton(
                "Medium-term milestones (weeks to a month)",
                value=(STATE.goal_timeframe == "medium"),
                id="timeframe-medium",
            )
            yield RadioButton(
                "Long-term ambitions (months to years)",
                value=(STATE.goal_timeframe == "long"),
                id="timeframe-long",
            )
            yield RadioButton(
                "Mixed — let goals vary in scope",
                value=(STATE.goal_timeframe == "mixed"),
                id="timeframe-mixed",
            )

        yield NavButtons(show_back=True, next_label="Continue →")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        mapping = {
            "timeframe-short": "short",
            "timeframe-medium": "medium",
            "timeframe-long": "long",
            "timeframe-mixed": "mixed",
        }
        if event.pressed.id in mapping:
            STATE.goal_timeframe = mapping[event.pressed.id]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-back":
            self.app.pop_screen()
            return
        if event.button.id == "btn-next":
            STATE.goal_1 = self.query_one("#input-goal-1", Input).value or ""
            STATE.goal_2 = self.query_one("#input-goal-2", Input).value or ""
            STATE.goal_3 = self.query_one("#input-goal-3", Input).value or ""
            self.app.push_screen("workspace")


# -----------------------------------------------------------------------------
# Screen 10: Workspace
# -----------------------------------------------------------------------------

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
            profiles = _scan_openllmauth_profiles()
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
                profiles = _scan_openllmauth_profiles()
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
        model_choices = _discover_model_choices()
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


# -----------------------------------------------------------------------------
# Screen 15: Bootstrap Progress
# -----------------------------------------------------------------------------

class BootstrapScreen(Screen):
    """Runs the actual bootstrap with a progress bar and live log."""

    DEFAULT_CSS = """
    BootstrapScreen {
        padding: 1 2;
    }
    #progress-bar {
        margin: 1 0;
    }
    #bootstrap-log {
        height: 1fr;
        border: solid $primary-darken-2;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield StepHeader(15, 15, "Bringing Your Agent to Life")
        yield ProgressBar(total=100, id="progress-bar")
        yield RichLog(id="bootstrap-log", highlight=True)
        yield Button("Cancel", id="btn-cancel", variant="error")

    def on_mount(self) -> None:
        self.log_widget = self.query_one("#bootstrap-log", RichLog)
        self.progress = self.query_one("#progress-bar", ProgressBar)
        self._cancelled = False
        self.run_worker(self._do_bootstrap, exclusive=True)

    def _build_user_bio(self) -> str:
        """Compose a rich user_bio from questionnaire answers if none was given."""
        if STATE.user_bio:
            return STATE.user_bio
        parts: List[str] = []
        if STATE.user_job:
            parts.append(f"Role: {STATE.user_job}")
        if STATE.user_interests:
            parts.append(f"Interests: {STATE.user_interests}")
        if STATE.vision_main_help:
            parts.append(f"Wants help with: {STATE.vision_main_help}")
        if STATE.vision_engagement_style:
            parts.append(f"Engagement preference: {STATE.vision_engagement_style.replace('_', ' ')}")
        if STATE.vision_success_six_months:
            parts.append(f"6-month success: {STATE.vision_success_six_months}")
        if STATE.learning_preference:
            parts.append(f"Learning preference: {STATE.learning_preference.replace('_', ' ')}")
        if STATE.feedback_style:
            parts.append(f"Feedback preference: {STATE.feedback_style.replace('_', ' ')}")
        if STATE.help_style:
            parts.append(f"Help preference: {STATE.help_style.replace('_', ' ')}")
        if STATE.happy_makers:
            parts.append(f"Energized by: {STATE.happy_makers}")
        if STATE.sad_drainers:
            parts.append(f"Drained by: {STATE.sad_drainers}")
        if STATE.angry_triggers:
            parts.append(f"Frustrated by: {STATE.angry_triggers}")
        if STATE.agent_avoid:
            parts.append(f"Avoid: {STATE.agent_avoid}")
        if STATE.bad_day_help:
            parts.append(f"Recovery: {STATE.bad_day_help}")
        return "\n".join(parts)

    def _save_questionnaire(self, state_dir: Path) -> None:
        """Persist raw questionnaire data so the agent can reference it later."""
        payload = {
            "vision": {
                "main_help": STATE.vision_main_help,
                "engagement_style": STATE.vision_engagement_style,
                "success_six_months": STATE.vision_success_six_months,
                "working_notes": STATE.vision_working_notes,
            },
            "about_me": {
                "job": STATE.user_job,
                "interests": STATE.user_interests,
                "communication_style": STATE.user_comm_style,
            },
            "learning_style": {
                "preference": STATE.learning_preference,
                "feedback_style": STATE.feedback_style,
                "help_style": STATE.help_style,
                "collab_pair": STATE.collab_pair,
                "collab_async": STATE.collab_async,
                "collab_backforth": STATE.collab_backforth,
                "collab_minimal": STATE.collab_minimal,
            },
            "emotional_landscape": {
                "happy_makers": STATE.happy_makers,
                "sad_drainers": STATE.sad_drainers,
                "angry_triggers": STATE.angry_triggers,
                "agent_avoid": STATE.agent_avoid,
                "bad_day_help": STATE.bad_day_help,
            },
            "initial_goals": {
                "goal_1": STATE.goal_1,
                "goal_2": STATE.goal_2,
                "goal_3": STATE.goal_3,
                "timeframe": STATE.goal_timeframe,
            },
            "persona_theme": {
                "accent": STATE.persona_accent,
            },
        }
        path = state_dir / "bootstrap_questionnaire.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    async def _do_bootstrap(self) -> None:
        self._log("Building BootstrapConfig from wizard answers...")
        self.progress.update(progress=10)

        extra_roots = [r.strip() for r in STATE.workspace_extra.split(",") if r.strip()]
        state_dir = Path(STATE.state_dir)

        config = BootstrapConfig(
            state_dir=state_dir,
            session_id=None,
            agent_profile_id=STATE.agent_profile_id,
            workspace_root=Path(STATE.workspace_root),
            workspace_roots=[Path(r) for r in extra_roots],
            default_llm_model=STATE.default_llm_model,
            embedding_model_id=STATE.embedding_model_id,
            provider_config_path=Path(STATE.provider_config_path) if STATE.provider_config_path else None,
            provider_env_path=Path(STATE.provider_env_path) if STATE.provider_env_path else None,
            credential_source_config_path=Path(STATE.credential_source_config) if STATE.credential_source_config else None,
            credential_source_env_path=None,
            credential_profile_ids=STATE.selected_profiles,
            credential_env_keys=STATE.credential_env_keys,
            persona_name=STATE.persona_name,
            user_name=STATE.user_name or None,
            user_bio=self._build_user_bio() or None,
        )

        self._log("Saving questionnaire to state directory...")
        self._save_questionnaire(state_dir)
        self.progress.update(progress=20)

        self._log("Starting bootstrap pipeline...")
        self.progress.update(progress=25)

        try:
            ctx = await BootstrapPipeline(config).run()
            self.progress.update(progress=60)

            # Seed initial goals into the executive state if any were provided
            if any([STATE.goal_1, STATE.goal_2, STATE.goal_3]):
                self._log("Seeding initial goals into executive state...")
                for g in [STATE.goal_1, STATE.goal_2, STATE.goal_3]:
                    if g:
                        ctx.executive.add_goal(g)
                ctx.executive.save_snapshot(config.state_dir / "executive.json")
                self._log("Initial goals saved.")

            self._log("Bootstrap complete. Initializing AgentRuntime...")

            runtime = AgentRuntime(ctx)
            await runtime.tom.load()
            self.progress.update(progress=80)
            self._log("Runtime ready. Starting autonomous mode...")

            if STATE.use_server:
                self._log(
                    f"Dashboard and API server starting at http://{STATE.host}:{STATE.port}"
                )
                self._log("Press Ctrl+C in this terminal to shutdown gracefully.")
                await runtime.run_autonomous_with_server(
                    host=STATE.host,
                    port=int(STATE.port),
                    cycle_interval=int(STATE.cycle_interval),
                    consolidation_interval=int(STATE.consolidation_interval),
                )
            else:
                self._log("Running headless autonomous mode.")
                self._log("Press Ctrl+C in this terminal to shutdown gracefully.")
                await runtime.run_autonomous(
                    cycle_interval=int(STATE.cycle_interval),
                    consolidation_interval=int(STATE.consolidation_interval),
                )

            self.progress.update(progress=100)
            self._log("Autonomous mode ended.")
        except Exception as exc:
            self.progress.update(progress=100)
            self._log(f"[bold red]Bootstrap failed: {exc}[/bold red]")
            self.notify(f"Error: {exc}", severity="error", title="Bootstrap Failed")

    def _log(self, message: str) -> None:
        self.log_widget.write(message)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self._cancelled = True
            self._log("Shutdown requested...")
            self.app.exit()


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

class BootstrapTUI(App):
    """The main Textual application."""

    CSS = """
    Screen {
        padding: 1 2;
    }
    .header-row {
        height: auto;
        margin: 0 0 1 0;
    }
    .step-counter {
        color: $text-muted;
        width: auto;
        padding-right: 2;
    }
    .step-title {
        text-style: bold;
        color: $text-accent;
        width: 1fr;
    }
    .text-xl {
        text-style: bold;
        text-align: center;
        width: 100%;
    }
    .text-lg {
        text-style: bold;
    }
    .text-muted {
        color: $text-muted;
        text-align: center;
        width: 100%;
    }
    .text-warning {
        color: $text-warning;
    }
    .field-label {
        margin: 1 0 0 0;
        text-style: bold;
    }
    Input {
        margin: 0 0 1 0;
    }
    TextArea {
        margin: 0 0 1 0;
        height: 5;
    }
    Checkbox {
        margin: 1 0;
    }
    RadioButton {
        margin: 0 0 1 0;
    }
    """

    def on_mount(self) -> None:
        self.push_screen("welcome")

    def get_screen(self, screen: str | Screen) -> Screen:
        if isinstance(screen, str):
            mapping = {
                "welcome": WelcomeScreen,
                "vibes": VibesScreen,
                "identity": IdentityScreen,
                "theme": ThemeScreen,
                "vision": VisionScreen,
                "about_me": AboutMeScreen,
                "learning_style": LearningStyleScreen,
                "emotions": EmotionsScreen,
                "goals": GoalsScreen,
                "workspace": WorkspaceScreen,
                "credentials": CredentialsScreen,
                "models": ModelsScreen,
                "advanced": AdvancedScreen,
                "review": ReviewScreen,
                "bootstrap": BootstrapScreen,
            }
            if screen in mapping:
                return mapping[screen]()
        return super().get_screen(screen)


def main() -> int:
    app = BootstrapTUI()
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
