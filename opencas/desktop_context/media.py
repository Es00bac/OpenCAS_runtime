"""MPRIS media pause/resume helpers for spoken desktop-context updates."""

from __future__ import annotations

import shlex
import shutil
import subprocess
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
        timeout_seconds: float = 2.0,
    ) -> None:
        self._runner = runner or subprocess.run
        self._busctl_path = busctl_path
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
