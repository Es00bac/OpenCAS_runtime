from textual.containers import Horizontal
from textual.widgets import Button

from opencas.bootstrap.tui_components import HelpText, NavButtons, StepHeader


def test_step_header_compose_renders_header_row():
    header = StepHeader(2, 5, "Identity")

    rows = list(header.compose())

    assert header.step == 2
    assert header.total == 5
    assert header.title == "Identity"
    assert len(rows) == 1
    assert isinstance(rows[0], Horizontal)
    assert "header-row" in rows[0].classes


def test_nav_buttons_hides_back_button_when_requested():
    buttons = NavButtons(show_back=False, next_label="Next")

    children = list(buttons.compose())

    assert [child.id for child in children if isinstance(child, Button)] == ["btn-next"]


def test_help_text_has_muted_css_contract():
    assert 'color: $text-muted;' in HelpText.DEFAULT_CSS
