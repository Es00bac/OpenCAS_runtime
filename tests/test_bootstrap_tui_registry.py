from opencas.bootstrap.tui import BootstrapTUI
from opencas.bootstrap.tui_screens_intro import WelcomeScreen
from opencas.bootstrap.tui_screens_user import GoalsScreen
from opencas.bootstrap.tui_runtime import BootstrapScreen
from opencas.bootstrap.tui_screens_setup import (
    CredentialsScreen,
    IntegrationsScreen,
    ReviewScreen,
    WorkspaceScreen,
)


def test_bootstrap_tui_resolves_profile_screens() -> None:
    app = BootstrapTUI()
    assert isinstance(app.get_screen("welcome"), WelcomeScreen)
    assert isinstance(app.get_screen("goals"), GoalsScreen)


def test_bootstrap_tui_resolves_setup_screens() -> None:
    app = BootstrapTUI()
    assert isinstance(app.get_screen("workspace"), WorkspaceScreen)
    assert isinstance(app.get_screen("credentials"), CredentialsScreen)
    assert isinstance(app.get_screen("integrations"), IntegrationsScreen)
    assert isinstance(app.get_screen("review"), ReviewScreen)


def test_bootstrap_tui_resolves_runtime_screen() -> None:
    app = BootstrapTUI()
    assert isinstance(app.get_screen("bootstrap"), BootstrapScreen)
