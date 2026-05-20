"""SQLite store for proof-chain claims."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from .models import ProofClaim, ProofClaimType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS proof_claims (
    claim_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_proof_claims_created_at
ON proof_claims(created_at);

CREATE INDEX IF NOT EXISTS idx_proof_claims_type_status_created_at
ON proof_claims(claim_type, status, created_at);
"""


class ProofStore:
    """Async SQLite persistence for proof claims."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> "ProofStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        return self

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def save_claim(self, claim: ProofClaim) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO proof_claims (
                claim_id, created_at, claim_type, subject, status, payload
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                created_at = excluded.created_at,
                claim_type = excluded.claim_type,
                subject = excluded.subject,
                status = excluded.status,
                payload = excluded.payload
            """,
            (
                str(claim.claim_id),
                claim.created_at.isoformat(),
                claim.claim_type.value,
                claim.subject,
                claim.status.value,
                claim.model_dump_json(),
            ),
        )
        await self._db.commit()

    async def get_claim(self, claim_id: str) -> ProofClaim | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT payload FROM proof_claims WHERE claim_id = ? LIMIT 1",
            (str(claim_id),),
        )
        row = await cursor.fetchone()
        return ProofClaim.model_validate_json(row["payload"]) if row else None

    async def list_claims(
        self,
        *,
        claim_type: ProofClaimType | str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[ProofClaim]:
        assert self._db is not None
        clauses: list[str] = []
        params: list[str | int] = []
        if claim_type is not None:
            clauses.append("claim_type = ?")
            params.append(claim_type.value if isinstance(claim_type, ProofClaimType) else str(claim_type))
        if status is not None:
            clauses.append("status = ?")
            params.append(str(status))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(200, int(limit))))
        cursor = await self._db.execute(
            f"""
            SELECT payload FROM proof_claims
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        rows = await cursor.fetchall()
        return [ProofClaim.model_validate_json(row["payload"]) for row in rows]

