"""Speech-safe adaptation of assistant responses for TTS."""

from __future__ import annotations

import re

DEFAULT_MAX_SPOKEN_RESPONSE_CHARS = 5_000

_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_ABS_PATH_RE = re.compile(r"(?<!\w)(?:~|/[\w .@+-]+(?:/[\w .@+-]+)+)")
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s]+")
_URL_RE = re.compile(r"https?://\S+")
_MARKDOWN_TABLE_RE = re.compile(r"(?m)^\s*\|.*\|\s*$")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_ISO_TIMESTAMP_RE = re.compile(
    r"\b20\d\d-\d\d-\d\d[T ][0-2]\d:[0-5]\d(?::[0-5]\d(?:\.\d+)?)?(?:Z|[+-]\d\d:?\d\d)?\b"
)
_TECHNICAL_META_RE = re.compile(
    r"^\s*(?:files_changed|timestamp|summary_stats|traceback|stack trace|raw analysis|"
    r"system message|system|developer|tool|debug log|logs?):\s*",
    re.IGNORECASE,
)


def prepare_spoken_response_text(
    response_text: str,
    *,
    max_chars: int | None = DEFAULT_MAX_SPOKEN_RESPONSE_CHARS,
) -> str:
    """Adapt a written assistant response for natural speech.

    The goal is to preserve the user-facing English answer while keeping
    visual-only details in the chat: code blocks, paths, URLs, tables,
    terminal output, logs, timestamps, and system/debug metadata.
    """

    text = str(response_text or "").strip()
    if not text:
        return ""

    visual_notes: list[str] = []

    if _FENCED_CODE_RE.search(text):
        text = _FENCED_CODE_RE.sub(" ", text)
        visual_notes.append("I put code in the chat for you to review visually.")

    if _MARKDOWN_TABLE_RE.search(text):
        text = _MARKDOWN_TABLE_RE.sub(" ", text)
        visual_notes.append("I included a table in the chat, so look there for the details.")

    if _URL_RE.search(text):
        text = _URL_RE.sub(" ", text)
        visual_notes.append("I included a link in the chat.")

    if _ABS_PATH_RE.search(text) or _WINDOWS_PATH_RE.search(text):
        text = _ABS_PATH_RE.sub(" ", text)
        text = _WINDOWS_PATH_RE.sub(" ", text)
        visual_notes.append("I included file paths in the chat instead of reading them out loud.")

    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(_inline_replacement(visual_notes), text)

    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^\s{0,3}#{1,6}\s+", "", stripped)
        stripped = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", stripped)
        if _line_is_visual_only(stripped):
            visual_notes.append("I left technical details in the chat for visual review.")
            continue
        lines.append(stripped)

    text = " ".join(lines)
    text = _ISO_TIMESTAMP_RE.sub(" ", text)
    text = re.sub(r"[*_>#]+", " ", text)
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"\s+", " ", text).strip()

    deduped_notes = list(dict.fromkeys(visual_notes))
    if deduped_notes:
        text = " ".join([text, *deduped_notes]).strip()
    if not text:
        text = "I put the details in the chat for you to review visually."
    return _limit_spoken_text(text, max_chars=max_chars)


def _inline_replacement(visual_notes: list[str]):
    def replace(match: re.Match[str]) -> str:
        value = match.group(1).strip()
        if not value:
            return " "
        if any(marker in value for marker in ("/", "\\", "{", "}", "=", "(", ")", ";", "$")):
            visual_notes.append("I included inline technical text in the chat for visual review.")
            return " "
        return value

    return replace


def _line_is_visual_only(line: str) -> bool:
    lowered = line.lower()
    if _TECHNICAL_META_RE.search(line):
        return True
    if line.startswith(("$ ", "Traceback ", "File \"", "+", "-", "@@", "{", "}", "[", "]")):
        return True
    if _ABS_PATH_RE.search(line) or _WINDOWS_PATH_RE.search(line) or _URL_RE.search(line):
        return True
    if _ISO_TIMESTAMP_RE.search(line):
        return True
    if "```" in line:
        return True
    if lowered.startswith(("system message", "developer message", "tool call", "raw json", "json:")):
        return True
    return False


def _limit_spoken_text(text: str, *, max_chars: int | None) -> str:
    normalized = " ".join(str(text or "").split()).strip()
    if max_chars is None or max_chars <= 0 or len(normalized) <= max_chars:
        return normalized
    trimmed = normalized[: max(80, max_chars)].rstrip()
    sentence_break = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"))
    if sentence_break > max_chars * 0.6:
        trimmed = trimmed[: sentence_break + 1]
    else:
        trimmed = trimmed.rsplit(" ", 1)[0].rstrip().rstrip(",;:")
        if trimmed and not trimmed.endswith((".", "!", "?")):
            trimmed += "."
    return f"{trimmed} There is more detail in the chat."
