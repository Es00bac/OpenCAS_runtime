"""Read-only proof-chain diagnostics routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from opencas.proof_chain import lookup_promise_chains, summarize_claims


def build_proof_router(runtime: Any) -> APIRouter:
    """Expose recent proof-chain claims for operator diagnostics."""

    router = APIRouter(tags=["proof"])

    @router.get("/api/proof/claims")
    async def list_claims(
        claim_type: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        store = getattr(runtime, "proof_store", None)
        if store is None:
            return {"available": False, "claims": []}
        claims = await store.list_claims(
            claim_type=claim_type,
            status=status,
            limit=limit,
        )
        return {
            "available": True,
            "claims": [claim.model_dump(mode="json") for claim in claims],
        }

    @router.get("/api/proof/summary")
    async def summary() -> dict[str, Any]:
        store = getattr(runtime, "proof_store", None)
        if store is None:
            return {"available": False}
        recent = await store.list_claims(limit=100)
        return {
            "available": True,
            **summarize_claims(recent),
        }

    @router.get("/api/proof-chain/promise-lookup")
    async def promise_lookup(
        since: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        return await lookup_promise_chains(runtime, since=since, limit=limit)

    return router
