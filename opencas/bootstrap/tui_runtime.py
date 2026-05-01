"""Bootstrap runtime screen for the OpenCAS TUI wizard."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Button, ProgressBar, RichLog

from opencas.bootstrap import BootstrapPipeline
from opencas.bootstrap.tui_bootstrap import build_bootstrap_config, save_questionnaire
from opencas.bootstrap.tui_components import StepHeader
from opencas.bootstrap.tui_state import STATE
from opencas.runtime import AgentRuntime
from opencas.runtime.tom_intention_mirror import reconcile_completed_runtime_intentions


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
        yield StepHeader(16, 16, "Bringing Your Agent to Life")
        yield ProgressBar(total=100, id="progress-bar")
        yield RichLog(id="bootstrap-log", highlight=True)
        yield Button("Cancel", id="btn-cancel", variant="error")

    def on_mount(self) -> None:
        self.log_widget = self.query_one("#bootstrap-log", RichLog)
        self.progress = self.query_one("#progress-bar", ProgressBar)
        self._cancelled = False
        self.run_worker(self._do_bootstrap, exclusive=True)

    async def _do_bootstrap(self) -> None:
        self._log("Building BootstrapConfig from wizard answers...")
        self.progress.update(progress=10)

        config = build_bootstrap_config(STATE)
        state_dir = config.state_dir

        self._log("Saving questionnaire to state directory...")
        save_questionnaire(STATE, state_dir)
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
            await reconcile_completed_runtime_intentions(runtime)
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
                    daydream_interval=int(STATE.daydream_interval),
                    baa_heartbeat_interval=int(STATE.baa_heartbeat_interval),
                    consolidation_interval=int(STATE.consolidation_interval),
                )
            else:
                self._log("Running headless autonomous mode.")
                self._log("Press Ctrl+C in this terminal to shutdown gracefully.")
                await runtime.run_autonomous(
                    cycle_interval=int(STATE.cycle_interval),
                    daydream_interval=int(STATE.daydream_interval),
                    baa_heartbeat_interval=int(STATE.baa_heartbeat_interval),
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
