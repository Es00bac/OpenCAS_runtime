"""Codec plugin for OpenCAS — hashing and encoding utilities."""

from __future__ import annotations

import base64
import hashlib
import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.models import ToolResult

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ALGORITHMS = {"md5", "sha1", "sha256", "sha512", "blake2b", "blake2s"}


def _hash_text(args: Dict[str, Any]) -> ToolResult:
    algo = str(args.get("algorithm", "sha256")).lower()
    if algo not in _ALGORITHMS:
        return ToolResult(
            success=False,
            output=f"algorithm must be one of {sorted(_ALGORITHMS)}",
            metadata={"algorithm": algo},
        )
    if "path" in args and args["path"]:
        path = Path(str(args["path"])).expanduser()
        if not path.is_file():
            return ToolResult(success=False, output=f"file not found: {path}", metadata={})
        h = hashlib.new(algo)
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return ToolResult(
            success=True,
            output=h.hexdigest(),
            metadata={"algorithm": algo, "path": str(path), "size": path.stat().st_size},
        )
    text = args.get("text")
    if text is None:
        return ToolResult(success=False, output="provide 'text' or 'path'", metadata={})
    h = hashlib.new(algo)
    h.update(str(text).encode("utf-8"))
    return ToolResult(success=True, output=h.hexdigest(), metadata={"algorithm": algo})


def _base64_encode(args: Dict[str, Any]) -> ToolResult:
    text = args.get("text")
    if text is None:
        return ToolResult(success=False, output="text is required", metadata={})
    url_safe = bool(args.get("url_safe", False))
    raw = str(text).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw) if url_safe else base64.b64encode(raw)
    return ToolResult(success=True, output=encoded.decode("ascii"), metadata={"url_safe": url_safe})


def _base64_decode(args: Dict[str, Any]) -> ToolResult:
    encoded = args.get("text")
    if encoded is None:
        return ToolResult(success=False, output="text is required", metadata={})
    url_safe = bool(args.get("url_safe", False))
    try:
        raw = (
            base64.urlsafe_b64decode(str(encoded))
            if url_safe
            else base64.b64decode(str(encoded), validate=True)
        )
    except (ValueError, base64.binascii.Error) as exc:
        return ToolResult(success=False, output=f"decode error: {exc}", metadata={})
    try:
        decoded = raw.decode("utf-8")
        text_safe = True
    except UnicodeDecodeError:
        decoded = raw.decode("latin-1", errors="replace")
        text_safe = False
    return ToolResult(success=True, output=decoded, metadata={"bytes": len(raw), "utf8": text_safe})


def _url_encode(args: Dict[str, Any]) -> ToolResult:
    text = args.get("text")
    if text is None:
        return ToolResult(success=False, output="text is required", metadata={})
    plus = bool(args.get("plus", False))
    fn = urllib.parse.quote_plus if plus else urllib.parse.quote
    return ToolResult(success=True, output=fn(str(text)), metadata={"plus": plus})


def _url_decode(args: Dict[str, Any]) -> ToolResult:
    text = args.get("text")
    if text is None:
        return ToolResult(success=False, output="text is required", metadata={})
    plus = bool(args.get("plus", False))
    fn = urllib.parse.unquote_plus if plus else urllib.parse.unquote
    return ToolResult(success=True, output=fn(str(text)), metadata={"plus": plus})


def _slugify(args: Dict[str, Any]) -> ToolResult:
    text = args.get("text")
    if text is None:
        return ToolResult(success=False, output="text is required", metadata={})
    max_length = int(args.get("max_length", 80))
    slug = _SLUG_RE.sub("-", str(text).lower()).strip("-")[:max_length]
    return ToolResult(success=True, output=slug or "n-a", metadata={"length": len(slug)})


def register_skills(skill_registry, tools) -> None:
    tools.register(
        "hash_text",
        "Hash a string ('text') or file ('path') with md5/sha1/sha256/sha512/blake2b/blake2s.",
        lambda name, args: _hash_text(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "path": {"type": "string"},
                "algorithm": {"type": "string", "enum": sorted(_ALGORITHMS)},
            },
        },
    )
    tools.register(
        "base64_encode",
        "Base64-encode a UTF-8 string. Set url_safe=true for URL-safe alphabet.",
        lambda name, args: _base64_encode(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {"text": {"type": "string"}, "url_safe": {"type": "boolean"}},
            "required": ["text"],
        },
    )
    tools.register(
        "base64_decode",
        "Base64-decode a string back to UTF-8 (with latin-1 fallback).",
        lambda name, args: _base64_decode(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {"text": {"type": "string"}, "url_safe": {"type": "boolean"}},
            "required": ["text"],
        },
    )
    tools.register(
        "url_encode",
        "Percent-encode a string for use in URLs. Set plus=true to encode spaces as '+'.",
        lambda name, args: _url_encode(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {"text": {"type": "string"}, "plus": {"type": "boolean"}},
            "required": ["text"],
        },
    )
    tools.register(
        "url_decode",
        "Decode percent-encoded text. Set plus=true to also decode '+' as space.",
        lambda name, args: _url_decode(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {"text": {"type": "string"}, "plus": {"type": "boolean"}},
            "required": ["text"],
        },
    )
    tools.register(
        "slugify",
        "Convert text to a URL-safe slug (lowercase, dashes for non-alphanumeric).",
        lambda name, args: _slugify(args),
        ActionRiskTier.READONLY,
        {
            "type": "object",
            "properties": {"text": {"type": "string"}, "max_length": {"type": "integer"}},
            "required": ["text"],
        },
    )
