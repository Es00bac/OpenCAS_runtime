"""Web tool adapters for search and fetch."""

from __future__ import annotations

import html.parser
import json
from typing import Any, Dict, List, Optional

from ..models import ToolResult


class _TextExtractor(html.parser.HTMLParser):
    """Simple HTML-to-text extractor."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, _attrs: Any) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "nav", "footer", "header"):
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip == 0:
            stripped = data.strip()
            if stripped:
                self._chunks.append(stripped)

    def get_text(self) -> str:
        text = " ".join(self._chunks)
        # Collapse whitespace
        return " ".join(text.split())


class WebToolAdapter:
    """Adapter for web search and fetch operations."""

    def __init__(self) -> None:
        pass

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        try:
            if name == "web_fetch":
                return await self._web_fetch(args)
            if name == "web_search":
                return await self._web_search(args)
            return ToolResult(success=False, output=f"Unknown web tool: {name}", metadata={})
        except Exception as exc:
            return ToolResult(success=False, output=str(exc), metadata={"error_type": type(exc).__name__})

    async def _web_fetch(self, args: Dict[str, Any]) -> ToolResult:
        import asyncio

        url = str(args.get("url", ""))
        max_length = int(args.get("max_length", 8000))
        if not url:
            return ToolResult(success=False, output="url is required", metadata={})

        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, output="httpx is required for web_fetch", metadata={})

        def _fetch() -> str:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "OpenCAS/1.0"})
                resp.raise_for_status()
                return resp.text

        html = await asyncio.to_thread(_fetch)
        extractor = _TextExtractor()
        await asyncio.to_thread(lambda: extractor.feed(html))
        text = extractor.get_text()
        if len(text) > max_length:
            text = text[:max_length] + "\n[truncated]"
        return ToolResult(
            success=True,
            output=text,
            metadata={"url": url, "length": len(text)},
        )

    async def _web_search(self, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", ""))
        if not query:
            return ToolResult(success=False, output="query is required", metadata={})

        try:
            import httpx
        except ImportError:
            return ToolResult(success=False, output="httpx is required for web_search", metadata={})

        import asyncio

        def _search() -> List[Dict[str, str]]:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                # Lightweight DuckDuckGo HTML search
                resp = client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                resp.raise_for_status()
                html = resp.text
                results: List[Dict[str, str]] = []
                import re
                for match in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html):
                    href = match.group(1)
                    title = re.sub(r'<[^>]+>', '', match.group(2))
                    if href.startswith("//"):
                        href = "https:" + href
                    results.append({"title": title.strip(), "url": href})
                return results[:10]

        results = await asyncio.to_thread(_search)
        return ToolResult(
            success=True,
            output=json.dumps({"ok": True, "results": results}, indent=2),
            metadata={"query": query, "result_count": len(results)},
        )
