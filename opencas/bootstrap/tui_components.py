"""Reusable Textual widgets shared across the bootstrap wizard."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Label, Markdown, Static


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
