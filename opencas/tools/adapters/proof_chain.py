"""Tool adapter for proof-chain lookup surfaces."""

from __future__ import annotations

import json
from typing import Any, Dict

from opencas.proof_chain import lookup_promise_chains

from ..models import ToolResult


class ProofChainToolAdapter:
    """Expose bounded proof-chain joins as readonly tools."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def __call__(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name != "proof_chain_promise_lookup":
            return ToolResult(False, f"Unknown proof-chain tool: {name}", {})
        payload = await lookup_promise_chains(
            self.runtime,
            since=args.get("since"),
            limit=int(args.get("limit") or 50),
        )
        return ToolResult(
            bool(payload.get("available")),
            json.dumps(payload),
            payload,
        )
