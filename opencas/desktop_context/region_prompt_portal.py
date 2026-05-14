"""XDG desktop-portal global shortcut daemon for OpenCAS region prompts."""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


PORTAL_SERVICE = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
PORTAL_INTERFACE = "org.freedesktop.portal.GlobalShortcuts"
REQUEST_INTERFACE = "org.freedesktop.portal.Request"
SHORTCUT_ID = "opencas_region_prompt"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_python() -> str:
    return os.environ.get("OPENCAS_REGION_PROMPT_PYTHON") or shutil.which("python3") or "/usr/bin/python3"


def default_region_prompt_script() -> Path:
    return repo_root() / "scripts" / "opencas_region_prompt.py"


def build_region_prompt_command(
    *,
    python_executable: Optional[str] = None,
    script_path: Optional[Path] = None,
) -> list[str]:
    return [
        python_executable or default_python(),
        str(script_path or default_region_prompt_script()),
    ]


class PortalRegionPromptBinding:
    """Register one desktop-wide portal shortcut and launch the region prompt on activation."""

    def __init__(
        self,
        *,
        shortcut_text: str = "F23",
        description: str = "OpenCAS desktop region prompt",
        command: Optional[list[str]] = None,
    ) -> None:
        self.shortcut_text = shortcut_text
        self.description = description
        self.command = command or build_region_prompt_command()
        self._registered = False
        self._bus: Any = None
        self._loop: Any = None
        self._session_handle: Optional[str] = None
        self._active_process: Optional[subprocess.Popen[bytes]] = None

    def run(self) -> None:
        import dbus
        from dbus.mainloop.glib import DBusGMainLoop
        from gi.repository import GLib

        DBusGMainLoop(set_as_default=True)
        self._bus = dbus.SessionBus()
        self._loop = GLib.MainLoop()
        self._bus.add_signal_receiver(
            self._on_portal_activated,
            signal_name="Activated",
            dbus_interface=PORTAL_INTERFACE,
            bus_name=PORTAL_SERVICE,
            path=PORTAL_PATH,
        )
        self._bus.add_signal_receiver(
            self._on_portal_deactivated,
            signal_name="Deactivated",
            dbus_interface=PORTAL_INTERFACE,
            bus_name=PORTAL_SERVICE,
            path=PORTAL_PATH,
        )
        GLib.idle_add(self._request_session)
        print(f"OpenCAS region prompt shortcut daemon requesting {self.shortcut_text}.", flush=True)
        self._loop.run()

    def _portal(self) -> Any:
        import dbus

        if self._bus is None:
            raise RuntimeError("Session bus is unavailable")
        proxy = self._bus.get_object(PORTAL_SERVICE, PORTAL_PATH)
        return dbus.Interface(proxy, PORTAL_INTERFACE)

    def _request_path(self, token: str) -> str:
        if self._bus is None:
            raise RuntimeError("Session bus is unavailable")
        unique_name = self._bus.get_unique_name()
        sender = unique_name[1:].replace(".", "_") if unique_name.startswith(":") else unique_name
        return f"/org/freedesktop/portal/desktop/request/{sender}/{token}"

    def _request_session(self) -> bool:
        try:
            handle_token = self._token("create")
            request_path = self._request_path(handle_token)
            self._bus.add_signal_receiver(
                self._on_create_session_response,
                signal_name="Response",
                dbus_interface=REQUEST_INTERFACE,
                bus_name=PORTAL_SERVICE,
                path=request_path,
            )
            self._portal().CreateSession(
                {
                    "handle_token": handle_token,
                    "session_handle_token": self._token("session"),
                }
            )
        except Exception as exc:
            print(f"OpenCAS region prompt shortcut portal error: {exc}", file=sys.stderr, flush=True)
        return False

    def _request_binding(self) -> None:
        import dbus

        if self._bus is None or not self._session_handle:
            print("OpenCAS region prompt shortcut session failed.", file=sys.stderr, flush=True)
            return
        handle_token = self._token("bind")
        request_path = self._request_path(handle_token)
        self._bus.add_signal_receiver(
            self._on_bind_shortcuts_response,
            signal_name="Response",
            dbus_interface=REQUEST_INTERFACE,
            bus_name=PORTAL_SERVICE,
            path=request_path,
        )
        shortcuts = [
            dbus.Struct(
                [
                    SHORTCUT_ID,
                    dbus.Dictionary(
                        {
                            "description": self.description,
                            "preferred_trigger": self.shortcut_text,
                        },
                        signature="sv",
                    ),
                ],
                signature=None,
            )
        ]
        self._portal().BindShortcuts(
            dbus.ObjectPath(self._session_handle),
            shortcuts,
            "",
            {"handle_token": handle_token},
        )

    def _token(self, prefix: str) -> str:
        return f"opencas_region_{prefix}_{secrets.token_hex(4)}"

    def _on_create_session_response(self, response: int, results: dict[str, object]) -> None:
        import dbus

        if response != 0:
            print("OpenCAS region prompt shortcut session was not approved.", file=sys.stderr, flush=True)
            return
        session_handle = results.get("session_handle")
        if not isinstance(session_handle, (str, dbus.ObjectPath)) or not session_handle:
            print("OpenCAS region prompt shortcut session returned no handle.", file=sys.stderr, flush=True)
            return
        self._session_handle = str(session_handle)
        self._request_binding()

    def _on_bind_shortcuts_response(self, response: int, results: dict[str, object]) -> None:
        shortcuts = results.get("shortcuts")
        if response != 0 or not shortcuts:
            print(f"OpenCAS region prompt shortcut {self.shortcut_text} was not approved.", file=sys.stderr, flush=True)
            return
        self._registered = True
        print(f"OpenCAS region prompt shortcut ready on {self.shortcut_text}.", flush=True)

    def _on_portal_activated(
        self,
        session_handle: object,
        shortcut_id: str,
        _timestamp: object,
        _options: object,
    ) -> None:
        if not self._registered or str(session_handle) != self._session_handle or shortcut_id != SHORTCUT_ID:
            return
        if self._active_process is not None and self._active_process.poll() is None:
            return
        self._active_process = subprocess.Popen(
            self.command,
            cwd=str(repo_root()),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _on_portal_deactivated(
        self,
        session_handle: object,
        shortcut_id: str,
        _timestamp: object,
        _options: object,
    ) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the OpenCAS region prompt global-shortcut portal daemon.")
    parser.add_argument("--shortcut", default=os.environ.get("OPENCAS_REGION_PROMPT_SHORTCUT", "F23"))
    parser.add_argument("--python", dest="python_executable", default=None)
    parser.add_argument("--script", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    command = build_region_prompt_command(
        python_executable=args.python_executable,
        script_path=Path(args.script).expanduser() if args.script else None,
    )
    try:
        PortalRegionPromptBinding(shortcut_text=args.shortcut, command=command).run()
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"OpenCAS region prompt shortcut daemon failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
