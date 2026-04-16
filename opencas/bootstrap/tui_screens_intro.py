"""Intro and persona bootstrap screens for the OpenCAS TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Input, Label, Markdown, RadioButton, RadioSet, Static, TextArea

from opencas.bootstrap.tui_components import HelpText, NavButtons, StepHeader
from opencas.bootstrap.tui_state import STATE
from opencas.runtime.agent_profile import BUILTIN_AGENT_PROFILES


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

