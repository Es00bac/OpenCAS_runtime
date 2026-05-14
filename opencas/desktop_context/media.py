"""MPRIS media pause/resume helpers for spoken desktop-context updates."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from typing import Any, Callable


CommandRunner = Callable[..., subprocess.CompletedProcess[Any]]


class MprisMediaController:
    """Pause currently playing MPRIS players and resume only those players."""

    def __init__(
        self,
        *,
        runner: CommandRunner | None = None,
        busctl_path: str | None = None,
        ydotool_path: str | None = None,
        ydotoold_path: str | None = None,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._runner = runner or subprocess.run
        self._busctl_path = busctl_path
        self._ydotool_path = ydotool_path
        self._ydotoold_path = ydotoold_path
        self._timeout_seconds = max(0.1, float(timeout_seconds))

    def pause_playing(self) -> dict[str, Any]:
        """Pause every currently-playing MPRIS player and return the paused names."""

        paused_players: list[str] = []
        errors: list[dict[str, str]] = []
        for player in self.playing_players():
            result = self._call_player(player, "Pause")
            if result.get("ok"):
                paused_players.append(player)
            else:
                errors.append({"player": player, "action": "Pause", "error": result.get("error", "")})
        return {"paused_players": paused_players, "errors": errors}

    def resume_players(self, players: Iterable[str]) -> dict[str, Any]:
        """Resume exactly the player names returned by pause_playing()."""

        resumed_players: list[str] = []
        errors: list[dict[str, str]] = []
        for raw_player in players:
            player = str(raw_player or "").strip()
            if not player:
                continue
            result = self._call_player(player, "Play")
            if result.get("ok"):
                resumed_players.append(player)
            else:
                errors.append({"player": player, "action": "Play", "error": result.get("error", "")})
        return {"resumed_players": resumed_players, "errors": errors}

    def set_players_rate(self, players: Iterable[str], rate: float) -> dict[str, Any]:
        """Set MPRIS playback rate for the provided player names when supported."""

        target_rate = _normalize_rate(rate)
        rate_set_players: list[str] = []
        errors: list[dict[str, str]] = []
        for raw_player in players:
            player = str(raw_player or "").strip()
            if not player:
                continue
            result = self._set_player_rate(player, target_rate)
            if result.get("ok"):
                rate_set_players.append(player)
            else:
                errors.append({"player": player, "action": "SetRate", "error": result.get("error", "")})
        return {"rate_set_players": rate_set_players, "rate": target_rate, "errors": errors}

    def set_youtube_browser_rate(self, rate: float) -> dict[str, Any]:
        """Use YouTube browser shortcuts to set playback speed when MPRIS Rate is read-only.

        YouTube exposes speed control through Shift+, and Shift+. even when the
        browser's MPRIS adapter refuses to set the Player.Rate property. This is
        intended for the foreground YouTube player that was just paused/resumed
        for spoken commentary.
        """

        target_rate = _normalize_rate(rate)
        ydotool = self._ydotool()
        if not ydotool:
            return {"ok": False, "rate": target_rate, "method": "ydotool_youtube_shortcuts", "error": "ydotool_unavailable"}
        events = self.youtube_rate_key_events(target_rate)
        daemon = None
        temp_dir = None
        env = os.environ.copy()
        socket_path = env.get("YDOTOOL_SOCKET") or f"/run/user/{os.getuid()}/.ydotool_socket"
        if not os.path.exists(socket_path):
            ydotoold = self._ydotoold()
            if not ydotoold:
                return {
                    "ok": False,
                    "rate": target_rate,
                    "method": "ydotool_youtube_shortcuts",
                    "error": "ydotoold_unavailable",
                }
            temp_dir = tempfile.TemporaryDirectory(prefix="opencas-ydotool-")
            socket_path = os.path.join(temp_dir.name, "ydotool.sock")
            daemon = subprocess.Popen(
                [
                    ydotoold,
                    "--socket-path",
                    socket_path,
                    "--socket-perm",
                    "0600",
                    "--socket-own",
                    f"{os.getuid()}:{os.getgid()}",
                    "--mouse-off",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not os.path.exists(socket_path):
                if daemon.poll() is not None:
                    break
                time.sleep(0.03)
            if not os.path.exists(socket_path):
                if daemon.poll() is None:
                    daemon.terminate()
                if temp_dir is not None:
                    temp_dir.cleanup()
                return {
                    "ok": False,
                    "rate": target_rate,
                    "method": "ydotool_youtube_shortcuts",
                    "error": "ydotoold_socket_unavailable",
                }
        env["YDOTOOL_SOCKET"] = socket_path
        try:
            completed = self._runner(
                [ydotool, "key", "-d", "25", *events],
                capture_output=True,
                text=True,
                timeout=max(2.0, self._timeout_seconds),
                env=env,
            )
        except Exception as exc:
            completed = subprocess.CompletedProcess(
                [ydotool, "key", "-d", "25", *events],
                1,
                "",
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            if daemon is not None and daemon.poll() is None:
                daemon.terminate()
                try:
                    daemon.wait(timeout=1.0)
                except Exception:
                    daemon.kill()
            if temp_dir is not None:
                temp_dir.cleanup()
        if completed.returncode == 0:
            return {
                "ok": True,
                "rate": target_rate,
                "method": "ydotool_youtube_shortcuts",
                "shortcut_events": len(events),
            }
        error = str(completed.stderr or completed.stdout or "").strip()
        return {
            "ok": False,
            "rate": target_rate,
            "method": "ydotool_youtube_shortcuts",
            "error": error or f"ydotool_exit_{completed.returncode}",
        }

    @staticmethod
    def youtube_rate_key_events(rate: float) -> list[str]:
        """Return ydotool key events that reset YouTube speed then set target rate."""

        target_rate = _normalize_rate(rate)
        minimum_rate = 0.25
        step = 0.25
        # YouTube supports 0.25x through 2.0x. Decrease enough times to hit the
        # floor from any normal YouTube speed, then increase to the target.
        decrease_steps = 8
        increase_steps = int(round((target_rate - minimum_rate) / step))
        decrease = ["42:1", "51:1", "51:0", "42:0"]  # Shift+Comma, YouTube speed down
        increase = ["42:1", "52:1", "52:0", "42:0"]  # Shift+Period, YouTube speed up
        events: list[str] = []
        for _ in range(decrease_steps):
            events.extend(decrease)
        for _ in range(max(0, increase_steps)):
            events.extend(increase)
        return events

    def playing_players(self) -> list[str]:
        """Return MPRIS player bus names whose PlaybackStatus is Playing."""

        players: list[str] = []
        for player in self._list_mpris_players():
            status = self._playback_status(player)
            if status == "Playing":
                players.append(player)
        return players

    def current_media(self) -> list[dict[str, Any]]:
        """Return metadata for active MPRIS media, including paused players."""

        items: list[dict[str, Any]] = []
        for player in self._list_mpris_players():
            status = self._playback_status(player)
            if status not in {"Playing", "Paused", "Stopped"}:
                continue
            metadata = self._metadata(player)
            if status == "Stopped" and not (metadata.get("xesam:title") or metadata.get("xesam:url")):
                continue
            items.append(
                {
                    "player": player,
                    "status": status,
                    "title": metadata.get("xesam:title") or "",
                    "artist": metadata.get("xesam:artist") or "",
                    "album": metadata.get("xesam:album") or "",
                    "url": metadata.get("xesam:url") or "",
                    "length_us": metadata.get("mpris:length"),
                    "position_us": self._position(player),
                    "raw": metadata,
                }
            )
        return items

    def _busctl(self) -> str | None:
        return self._busctl_path or shutil.which("busctl")

    def _ydotool(self) -> str | None:
        return self._ydotool_path or shutil.which("ydotool")

    def _ydotoold(self) -> str | None:
        return self._ydotoold_path or shutil.which("ydotoold")

    def _list_mpris_players(self) -> list[str]:
        busctl = self._busctl()
        if not busctl:
            return []
        completed = self._run([busctl, "--user", "list", "--no-legend"])
        if completed.returncode != 0:
            return []
        players: list[str] = []
        for line in str(completed.stdout or "").splitlines():
            parts = line.split()
            if not parts:
                continue
            name = parts[0].strip()
            if name.startswith("org.mpris.MediaPlayer2.") and name not in players:
                players.append(name)
        return players

    def _playback_status(self, player: str) -> str | None:
        busctl = self._busctl()
        if not busctl:
            return None
        completed = self._run(
            [
                busctl,
                "--user",
                "get-property",
                player,
                "/org/mpris/MediaPlayer2",
                "org.mpris.MediaPlayer2.Player",
                "PlaybackStatus",
            ]
        )
        if completed.returncode != 0:
            return None
        return _parse_busctl_string_value(str(completed.stdout or ""))

    def _call_player(self, player: str, method: str) -> dict[str, Any]:
        busctl = self._busctl()
        if not busctl:
            return {"ok": False, "error": "busctl_unavailable"}
        completed = self._run(
            [
                busctl,
                "--user",
                "call",
                player,
                "/org/mpris/MediaPlayer2",
                "org.mpris.MediaPlayer2.Player",
                method,
            ]
        )
        if completed.returncode == 0:
            return {"ok": True}
        error = str(completed.stderr or completed.stdout or "").strip()
        return {"ok": False, "error": error or f"busctl_exit_{completed.returncode}"}

    def _set_player_rate(self, player: str, rate: float) -> dict[str, Any]:
        busctl = self._busctl()
        if not busctl:
            return {"ok": False, "error": "busctl_unavailable"}
        completed = self._run(
            [
                busctl,
                "--user",
                "set-property",
                player,
                "/org/mpris/MediaPlayer2",
                "org.mpris.MediaPlayer2.Player",
                "Rate",
                "d",
                _format_rate(rate),
            ]
        )
        if completed.returncode == 0:
            return {"ok": True}
        error = str(completed.stderr or completed.stdout or "").strip()
        return {"ok": False, "error": error or f"busctl_exit_{completed.returncode}"}

    def _metadata(self, player: str) -> dict[str, Any]:
        busctl = self._busctl()
        if not busctl:
            return {}
        completed = self._run(
            [
                busctl,
                "--user",
                "get-property",
                player,
                "/org/mpris/MediaPlayer2",
                "org.mpris.MediaPlayer2.Player",
                "Metadata",
            ]
        )
        if completed.returncode != 0:
            return {}
        return _parse_busctl_metadata(str(completed.stdout or ""))

    def _position(self, player: str) -> int | None:
        busctl = self._busctl()
        if not busctl:
            return None
        completed = self._run(
            [
                busctl,
                "--user",
                "get-property",
                player,
                "/org/mpris/MediaPlayer2",
                "org.mpris.MediaPlayer2.Player",
                "Position",
            ]
        )
        if completed.returncode != 0:
            return None
        return _parse_busctl_int_value(str(completed.stdout or ""))

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[Any]:
        try:
            return self._runner(
                args,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            return subprocess.CompletedProcess(args, 1, "", f"{type(exc).__name__}: {exc}")


def _parse_busctl_string_value(value: str) -> str | None:
    """Parse busctl string output such as ``s "Playing"``."""

    text = value.strip()
    if not text:
        return None
    if text.startswith("s "):
        text = text[2:].strip()
    try:
        parts = shlex.split(text)
    except ValueError:
        parts = []
    if parts:
        return parts[0]
    return text.strip('"') or None


def _normalize_rate(rate: float) -> float:
    try:
        value = float(rate)
    except (TypeError, ValueError):
        value = 1.0
    return round(max(0.25, min(2.0, value)), 2)


def _format_rate(rate: float) -> str:
    text = f"{_normalize_rate(rate):.2f}".rstrip("0").rstrip(".")
    return text or "1"


def _parse_busctl_int_value(value: str) -> int | None:
    """Parse busctl integer output such as ``x 123456``."""

    text = value.strip()
    if not text:
        return None
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    for token in reversed(tokens):
        try:
            return int(token)
        except ValueError:
            continue
    return None


def _parse_busctl_metadata(value: str) -> dict[str, Any]:
    """Best-effort parser for compact busctl MPRIS metadata output."""

    metadata: dict[str, Any] = {}
    text = " ".join(str(value or "").split())
    if not text:
        return metadata
    tokens = shlex.split(text)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith(("xesam:", "mpris:", "vlc:")):
            index += 1
            continue
        key = token
        type_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        if type_token == "s" and index + 2 < len(tokens):
            metadata[key] = tokens[index + 2]
            index += 3
            continue
        if type_token in {"x", "u", "i"} and index + 2 < len(tokens):
            try:
                metadata[key] = int(tokens[index + 2])
            except ValueError:
                metadata[key] = tokens[index + 2]
            index += 3
            continue
        if type_token == "as" and index + 3 < len(tokens):
            try:
                count = int(tokens[index + 2])
            except ValueError:
                count = 0
            values = tokens[index + 3 : index + 3 + max(0, count)]
            metadata[key] = ", ".join(values)
            index += 3 + max(0, count)
            continue
        index += 1
    return metadata
