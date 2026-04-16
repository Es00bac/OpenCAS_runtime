"""Interactive TUI bootstrap wizard for OpenCAS.

Run with:
    python -m opencas --tui
"""

from __future__ import annotations

import sys
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Footer, Header

from opencas.bootstrap.tui_state import STATE

from opencas.bootstrap.tui_screens_intro import (
    IdentityScreen,
    ThemeScreen,
    VibesScreen,
    VisionScreen,
    WelcomeScreen,
)
from opencas.bootstrap.tui_screens_user import (
    AboutMeScreen,
    EmotionsScreen,
    GoalsScreen,
    LearningStyleScreen,
)
from opencas.bootstrap.tui_runtime import BootstrapScreen
from opencas.bootstrap.tui_screens_setup import (
    AdvancedScreen,
    CredentialsScreen,
    ModelsScreen,
    ReviewScreen,
    WorkspaceScreen,
)


# -----------------------------------------------------------------------------
# State held across the wizard
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Reusable components
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Screen 1: Welcome + Moral Warning
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Screen 10: Workspace
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Screen 15: Bootstrap Progress
# -----------------------------------------------------------------------------

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
