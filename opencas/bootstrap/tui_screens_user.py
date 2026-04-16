"""User-context and goal bootstrap screens for the OpenCAS TUI."""

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
        yield StepHeader(2, 16, "Professional Partnership Profile")
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
        yield StepHeader(3, 16, "Names & Introductions")
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
        yield StepHeader(4, 16, "Persona Theme")
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
        yield StepHeader(5, 16, "Partnership Vision")
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
        yield StepHeader(6, 16, "About You")
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
        yield StepHeader(7, 16, "How You Learn & Collaborate")
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
        yield StepHeader(8, 16, "Emotional Landscape & Triggers")
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
        yield StepHeader(9, 16, "Initial Goals & Commitments")
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

