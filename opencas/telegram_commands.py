"""Slash-command router for the Telegram integration.

Keeps command handling out of `telegram_integration.py` so the bot service stays
a thin transport shell. Each command reads state from `runtime.ctx` and returns
a text reply ready to send back to the user.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


CommandHandler = Callable[["TelegramCommandRouter", int, str, List[str]], Awaitable[Optional[str]]]


# (command, description) pairs shown in Telegram's command menu via setMyCommands.
BOT_COMMAND_MENU: List[Tuple[str, str]] = [
    ("new", "Start a new conversation session"),
    ("session", "Show the current session id"),
    ("help", "List available commands"),
    ("status", "System overview (somatic, musubi, identity)"),
    ("somatic", "Somatic state breakdown"),
    ("musubi", "Relational resonance (trust/presence/attunement)"),
    ("identity", "Self-model summary"),
    ("user", "How OpenCAS models you"),
    ("continuity", "Boot count and last session"),
    ("whoami", "Your Telegram pairing info"),
]


class TelegramCommandRouter:
    """Parse and dispatch `/command` messages for a `TelegramBotService`."""

    def __init__(self, runtime: Any, *, state_path: Path | str) -> None:
        self.runtime = runtime
        self._state_path = Path(state_path)
        self._lock = asyncio.Lock()
        self._state = self._load()
        self._handlers: Dict[str, CommandHandler] = {
            "new": _cmd_new,
            "session": _cmd_session,
            "help": _cmd_help,
            "start": _cmd_help,
            "status": _cmd_status,
            "somatic": _cmd_somatic,
            "musubi": _cmd_musubi,
            "relational": _cmd_musubi,
            "identity": _cmd_identity,
            "self": _cmd_identity,
            "user": _cmd_user_model,
            "continuity": _cmd_continuity,
            "whoami": _cmd_whoami,
        }

    def is_command(self, text: str) -> bool:
        stripped = (text or "").lstrip()
        return stripped.startswith("/")

    def parse(self, text: str, *, bot_username: Optional[str] = None) -> Tuple[str, List[str]]:
        parts = (text or "").strip().split()
        if not parts or not parts[0].startswith("/"):
            return "", []
        head = parts[0][1:]
        if "@" in head:
            head, mention = head.split("@", 1)
            if bot_username and mention.lower() != bot_username.lower():
                return "", []
        return head.lower(), parts[1:]

    async def dispatch(
        self,
        *,
        chat_id: int,
        user_id: int,
        text: str,
        bot_username: Optional[str] = None,
    ) -> Optional[str]:
        name, args = self.parse(text, bot_username=bot_username)
        if not name:
            return None
        handler = self._handlers.get(name)
        if handler is None:
            return f"Unknown command: /{name}\nSend /help for the list."
        try:
            return await handler(self, chat_id, text, args)
        except Exception as exc:
            return f"[Command error: {exc}]"

    def session_id_for(self, chat_type: str, chat_id: int) -> str:
        """Return the current session id for a chat, respecting /new resets."""
        counter = int(self._state.get("session_counters", {}).get(str(chat_id), 0) or 0)
        base = f"telegram:{chat_type}:{chat_id}"
        return f"{base}:{counter}" if counter else base

    async def begin_new_session(self, chat_id: int) -> str:
        async with self._lock:
            counters = self._state.setdefault("session_counters", {})
            counters[str(chat_id)] = int(counters.get(str(chat_id), 0) or 0) + 1
            self._state["updated_at"] = time.time()
            self._save(self._state)
            return str(counters[str(chat_id)])

    def _load(self) -> Dict[str, Any]:
        if not self._state_path.exists():
            return {"session_counters": {}}
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw.setdefault("session_counters", {})
                return raw
        except Exception:
            pass
        return {"session_counters": {}}

    def _save(self, state: Dict[str, Any]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8"
        )


def _fmt_float(value: Any, precision: int = 2) -> str:
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "—"


def _bar(value: float, width: int = 10, *, bipolar: bool = False) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "[" + "·" * width + "]"
    if bipolar:
        v = max(-1.0, min(1.0, v))
        filled = int(round((v + 1.0) / 2.0 * width))
    else:
        v = max(0.0, min(1.0, v))
        filled = int(round(v * width))
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def _primary_emotion(state: Any) -> Optional[str]:
    try:
        from opencas.somatic.modulators import SomaticModulators

        return SomaticModulators(state)._infer_primary_emotion().value
    except Exception:
        return None


async def _cmd_new(
    router: "TelegramCommandRouter", chat_id: int, text: str, args: List[str]
) -> str:
    counter = await router.begin_new_session(chat_id)
    return (
        "Started a new conversation.\n"
        f"Session counter: {counter}. Prior context will not be carried over."
    )


async def _cmd_session(
    router: "TelegramCommandRouter", chat_id: int, text: str, args: List[str]
) -> str:
    sid = router.session_id_for("private", chat_id)
    return f"Current session id: {sid}"


async def _cmd_help(
    router: "TelegramCommandRouter", chat_id: int, text: str, args: List[str]
) -> str:
    lines = ["OpenCAS Telegram commands:"]
    for name, description in BOT_COMMAND_MENU:
        lines.append(f"/{name} — {description}")
    lines.append("")
    lines.append("Anything without a leading slash is sent to OpenCAS as a normal message.")
    return "\n".join(lines)


async def _cmd_somatic(
    router: "TelegramCommandRouter", chat_id: int, text: str, args: List[str]
) -> str:
    somatic_mgr = getattr(router.runtime.ctx, "somatic", None)
    if somatic_mgr is None:
        return "Somatic subsystem is not available."
    state = somatic_mgr.state
    emotion = _primary_emotion(state) or "—"
    tag = getattr(state, "somatic_tag", None) or "—"
    updated = getattr(state, "updated_at", None)
    updated_s = updated.isoformat() if hasattr(updated, "isoformat") else str(updated or "—")

    rows: List[Tuple[str, float, bool]] = [
        ("arousal  ", state.arousal, False),
        ("fatigue  ", state.fatigue, False),
        ("tension  ", state.tension, False),
        ("valence  ", state.valence, True),
        ("focus    ", state.focus, False),
        ("energy   ", state.energy, False),
        ("certainty", state.certainty, False),
    ]

    lines = [
        "Somatic state",
        f"primary emotion: {emotion}",
        f"somatic tag:     {tag}",
        f"updated at:      {updated_s}",
        "",
    ]
    for label, value, bipolar in rows:
        lines.append(f"{label} {_bar(value, bipolar=bipolar)} {_fmt_float(value)}")

    try:
        salience = float(state.to_memory_salience_modifier())
        lines.append("")
        lines.append(f"memory salience modifier: ×{_fmt_float(salience)}")
    except Exception:
        pass

    return "\n".join(lines)


async def _cmd_musubi(
    router: "TelegramCommandRouter", chat_id: int, text: str, args: List[str]
) -> str:
    rel = getattr(router.runtime.ctx, "relational", None)
    if rel is None:
        return "Relational (musubi) subsystem is not available."
    state = rel.state
    dims = dict(getattr(state, "dimensions", {}) or {})
    updated = getattr(state, "updated_at", None)
    updated_s = updated.isoformat() if hasattr(updated, "isoformat") else str(updated or "—")

    lines = [
        "Relational resonance (musubi)",
        f"composite musubi: {_bar(state.musubi, bipolar=True)} {_fmt_float(state.musubi)}",
        f"source tag:       {getattr(state, 'source_tag', None) or '—'}",
        f"updated at:       {updated_s}",
        "",
    ]
    for key in ("trust", "resonance", "presence", "attunement"):
        value = dims.get(key, 0.0)
        lines.append(f"{key:<10} {_bar(value, bipolar=True)} {_fmt_float(value)}")

    breadcrumb = getattr(state, "continuity_breadcrumb", None)
    if breadcrumb:
        lines.append("")
        lines.append(f"continuity: {breadcrumb}")

    return "\n".join(lines)


async def _cmd_identity(
    router: "TelegramCommandRouter", chat_id: int, text: str, args: List[str]
) -> str:
    identity = getattr(router.runtime.ctx, "identity", None)
    if identity is None:
        return "Identity subsystem is not available."
    sm = identity.self_model
    lines = [
        f"Self-model: {sm.name} v{sm.version}",
        f"intention:  {sm.current_intention or '—'}",
    ]
    if sm.values:
        lines.append("values:     " + ", ".join(sm.values[:6]))
    if sm.traits:
        lines.append("traits:     " + ", ".join(sm.traits[:6]))
    if sm.current_goals:
        lines.append("")
        lines.append("current goals:")
        for goal in sm.current_goals[:5]:
            lines.append(f"  • {goal}")
    if sm.narrative:
        narrative = sm.narrative.strip()
        if len(narrative) > 400:
            narrative = narrative[:400].rstrip() + "…"
        lines.append("")
        lines.append("narrative:")
        lines.append(narrative)
    return "\n".join(lines)


async def _cmd_user_model(
    router: "TelegramCommandRouter", chat_id: int, text: str, args: List[str]
) -> str:
    identity = getattr(router.runtime.ctx, "identity", None)
    if identity is None:
        return "Identity subsystem is not available."
    um = identity.user_model
    lines = [
        "User model",
        f"trust level: {_fmt_float(um.trust_level)}",
    ]
    prefs = dict(getattr(um, "explicit_preferences", {}) or {})
    if prefs:
        lines.append("")
        lines.append("explicit preferences:")
        for key, value in list(prefs.items())[:6]:
            lines.append(f"  • {key}: {value}")
    if um.inferred_goals:
        lines.append("")
        lines.append("inferred goals:")
        for goal in um.inferred_goals[:5]:
            lines.append(f"  • {goal}")
    if um.known_boundaries:
        lines.append("")
        lines.append("known boundaries:")
        for boundary in um.known_boundaries[:5]:
            lines.append(f"  • {boundary}")
    if um.uncertainty_areas:
        lines.append("")
        lines.append("uncertainty areas:")
        for area in um.uncertainty_areas[:5]:
            lines.append(f"  • {area}")
    return "\n".join(lines)


async def _cmd_continuity(
    router: "TelegramCommandRouter", chat_id: int, text: str, args: List[str]
) -> str:
    identity = getattr(router.runtime.ctx, "identity", None)
    if identity is None:
        return "Identity subsystem is not available."
    cs = identity.continuity
    last_shutdown = (
        cs.last_shutdown_time.isoformat() if cs.last_shutdown_time else "—"
    )
    updated = cs.updated_at.isoformat() if hasattr(cs.updated_at, "isoformat") else str(cs.updated_at)
    return "\n".join(
        [
            "Continuity",
            f"version:         {cs.version}",
            f"boot count:      {cs.boot_count}",
            f"last session:    {cs.last_session_id or '—'}",
            f"last shutdown:   {last_shutdown}",
            f"updated at:      {updated}",
        ]
    )


async def _cmd_status(
    router: "TelegramCommandRouter", chat_id: int, text: str, args: List[str]
) -> str:
    ctx = router.runtime.ctx
    lines: List[str] = ["OpenCAS status"]

    identity = getattr(ctx, "identity", None)
    if identity is not None:
        sm = identity.self_model
        cs = identity.continuity
        lines.append(f"self:        {sm.name} v{sm.version}  (boots={cs.boot_count})")
        if sm.current_intention:
            lines.append(f"intention:   {sm.current_intention}")

    somatic_mgr = getattr(ctx, "somatic", None)
    if somatic_mgr is not None:
        ss = somatic_mgr.state
        emotion = _primary_emotion(ss) or "—"
        lines.append(
            "somatic:     "
            f"emotion={emotion}  arousal={_fmt_float(ss.arousal)}  "
            f"fatigue={_fmt_float(ss.fatigue)}  focus={_fmt_float(ss.focus)}"
        )

    rel = getattr(ctx, "relational", None)
    if rel is not None:
        ms = rel.state
        dims = dict(getattr(ms, "dimensions", {}) or {})
        lines.append(
            "musubi:      "
            f"composite={_fmt_float(ms.musubi)}  trust={_fmt_float(dims.get('trust'))}  "
            f"presence={_fmt_float(dims.get('presence'))}  attunement={_fmt_float(dims.get('attunement'))}"
        )

    config = getattr(ctx, "config", None)
    if config is not None:
        model = getattr(config, "default_llm_model", None)
        if model:
            lines.append(f"model:       {model}")

    lines.append("")
    lines.append(f"session:     {router.session_id_for('private', chat_id)}")
    return "\n".join(lines)


async def _cmd_whoami(
    router: "TelegramCommandRouter", chat_id: int, text: str, args: List[str]
) -> str:
    return (
        "This chat is paired with OpenCAS.\n"
        f"chat id: {chat_id}\n"
        f"session: {router.session_id_for('private', chat_id)}"
    )
