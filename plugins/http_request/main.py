"""HTTP request plugin for OpenCAS — full method support with headers/body/auth."""

from __future__ import annotations

import base64
import json as _json
from typing import Any, Dict

from opencas.autonomy.models import ActionRiskTier
from opencas.tools.models import ToolResult

ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_BYTES = 524_288


def _http_request(args: Dict[str, Any]) -> ToolResult:
    url = str(args.get("url", "")).strip()
    if not url:
        return ToolResult(success=False, output="url is required", metadata={})
    if not (url.startswith("http://") or url.startswith("https://")):
        return ToolResult(
            success=False,
            output="url must start with http:// or https://",
            metadata={"url": url},
        )

    method = str(args.get("method", "GET")).upper().strip()
    if method not in ALLOWED_METHODS:
        return ToolResult(
            success=False,
            output=f"method must be one of {sorted(ALLOWED_METHODS)}",
            metadata={"method": method},
        )

    headers = dict(args.get("headers") or {})
    if not isinstance(headers, dict):
        return ToolResult(success=False, output="headers must be an object", metadata={})

    auth = args.get("auth")
    if isinstance(auth, dict) and auth.get("user") is not None and auth.get("password") is not None:
        token = base64.b64encode(f"{auth['user']}:{auth['password']}".encode("utf-8")).decode("ascii")
        headers.setdefault("Authorization", f"Basic {token}")
    bearer = args.get("bearer_token")
    if isinstance(bearer, str) and bearer:
        headers.setdefault("Authorization", f"Bearer {bearer}")
    headers.setdefault("User-Agent", "OpenCAS-http_request/1.0")

    timeout = float(args.get("timeout_seconds", DEFAULT_TIMEOUT))
    max_bytes = int(args.get("max_bytes", DEFAULT_MAX_BYTES))

    body_arg = args.get("body")
    json_arg = args.get("json")
    form_arg = args.get("form")
    follow_redirects = bool(args.get("follow_redirects", True))

    request_kwargs: Dict[str, Any] = {"headers": headers}
    if json_arg is not None:
        request_kwargs["json"] = json_arg
    elif form_arg is not None:
        if not isinstance(form_arg, dict):
            return ToolResult(success=False, output="form must be an object", metadata={})
        request_kwargs["data"] = form_arg
    elif body_arg is not None:
        request_kwargs["content"] = str(body_arg).encode("utf-8") if not isinstance(body_arg, (bytes, bytearray)) else bytes(body_arg)

    try:
        import httpx
    except ImportError:
        return ToolResult(success=False, output="httpx is required for http_request", metadata={})

    try:
        with httpx.Client(timeout=timeout, follow_redirects=follow_redirects) as client:
            response = client.request(method, url, **request_kwargs)
    except httpx.HTTPError as exc:
        return ToolResult(
            success=False,
            output=f"http error: {exc}",
            metadata={"url": url, "method": method, "error_type": type(exc).__name__},
        )
    except Exception as exc:
        return ToolResult(
            success=False,
            output=str(exc),
            metadata={"url": url, "method": method, "error_type": type(exc).__name__},
        )

    raw = response.content[: max_bytes + 1]
    truncated = len(raw) > max_bytes
    body_bytes = raw[:max_bytes]
    try:
        body_text = body_bytes.decode("utf-8")
    except UnicodeDecodeError:
        body_text = body_bytes.decode("latin-1", errors="replace")

    parse_json = bool(args.get("parse_json", False))
    output = body_text
    metadata: Dict[str, Any] = {
        "url": url,
        "method": method,
        "status": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "bytes": len(body_bytes),
        "truncated": truncated,
    }
    if parse_json and body_text:
        try:
            output = _json.dumps(_json.loads(body_text), indent=2)
            metadata["parsed_json"] = True
        except _json.JSONDecodeError as exc:
            return ToolResult(
                success=False,
                output=f"JSON parse failed: {exc}",
                metadata=metadata,
            )

    return ToolResult(
        success=200 <= response.status_code < 400,
        output=output if output else f"({method} {url} → {response.status_code})",
        metadata=metadata,
    )


def register_skills(skill_registry, tools) -> None:
    tools.register(
        "http_request",
        "Make an HTTP request with method, headers, body (json/form/raw), basic auth or bearer token. Supports GET/POST/PUT/PATCH/DELETE/HEAD/OPTIONS.",
        lambda name, args: _http_request(args),
        ActionRiskTier.NETWORK,
        {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "method": {"type": "string", "enum": sorted(ALLOWED_METHODS)},
                "headers": {"type": "object"},
                "json": {"description": "JSON body (object/array)."},
                "form": {"type": "object", "description": "URL-encoded form fields."},
                "body": {"type": "string", "description": "Raw body string."},
                "auth": {
                    "type": "object",
                    "properties": {"user": {"type": "string"}, "password": {"type": "string"}},
                },
                "bearer_token": {"type": "string"},
                "timeout_seconds": {"type": "number"},
                "max_bytes": {"type": "integer"},
                "parse_json": {"type": "boolean"},
                "follow_redirects": {"type": "boolean"},
            },
            "required": ["url"],
        },
    )
