from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opencas.api import LLMClient
from opencas.embeddings import EmbeddingService
from opencas.embeddings.models import EmbeddingRecord
from .store import WorkspaceStore, WorkspaceGistRecord

PROMPT_VERSION = "workspace_gist_v1"

@dataclass
class GistCandidate:
    attempt_no: int
    gist_text: str
    gist_json: dict[str, Any]
    cosine_similarity: float
    drift_score: float
    accepted: bool

def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector size mismatch: {len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))

def weighted_centroid(vectors: list[list[float]], weights: list[float]) -> list[float]:
    if not vectors:
        raise ValueError("No vectors provided")
    dim = len(vectors[0])
    acc = [0.0] * dim
    total = 0.0
    for vec, w in zip(vectors, weights):
        total += w
        for i in range(dim):
            acc[i] += vec[i] * w
    if total == 0:
        total = 1.0
    return l2_normalize([x / total for x in acc])

def chunk_text(text: str, chunk_chars: int = 6000, overlap: int = 400) -> list[str]:
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_chars)
        out.append(text[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out

def render_gist_text(payload: dict[str, Any]) -> str:
    summary = str(payload.get("summary", "")).strip()
    purpose = str(payload.get("purpose", "")).strip()
    why_exists = str(payload.get("why_exists", "")).strip()
    notes = str(payload.get("notes", "")).strip()

    parts = [p for p in [summary, purpose, why_exists] if p]
    gist = " ".join(parts)
    if notes:
        gist = f"{gist} Note: {notes}"
    return gist.strip()

def build_gist_prompt(
    *,
    path: Path,
    checksum: str,
    file_kind: str,
    content_excerpt: str,
    previous_failures: list[str],
) -> str:
    failure_block = ""
    if previous_failures:
        joined = "\n".join(f"- {x}" for x in previous_failures)
        failure_block = f"\nPrevious weak attempts:\n{joined}\n"

    return f"""
You are generating a compressed workspace gist for a file in an autonomous agent system.

Return strict JSON with keys:
- summary
- purpose
- why_exists
- notes

Rules:
- Be concrete and compact.
- Focus on what the file IS, what it DOES, and WHY it exists.
- Do not include fluff.
- Do not speculate beyond evidence.
- If the content seems partial, generated, binary, or ambiguous, say so in notes.
- Keep the full rendered gist under ~110 words.

Path: {path}
Checksum: {checksum}
File kind: {file_kind}
{failure_block}
Content excerpt:
\"\"\"
{content_excerpt}
\"\"\"
""".strip()

async def embed_file_semantics(
    *,
    extracted_text: str,
    embeddings_client: EmbeddingService,
    namespace: str,
    max_chunks: int = 24,
) -> tuple[list[float], str]:
    text = extracted_text.strip()
    if not text:
        raise ValueError("No extracted text available for semantic embedding")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("Unable to chunk extracted text")

    chunks = chunks[:max_chunks]
    vectors = [
        list(
            (
                await embeddings_client.embed(
                    chunk,
                    meta={
                        "workspace_namespace": namespace,
                        "chunk_index": index,
                        "chunk_count": len(chunks),
                    },
                    task_type="workspace_file_chunk",
                )
            ).vector
        )
        for index, chunk in enumerate(chunks)
    ]
    weights = [max(1.0, len(chunk)) for chunk in chunks]
    centroid = weighted_centroid(vectors, weights)
    status = "full" if len(chunks) == 1 else "chunked"
    return centroid, status


async def store_workspace_vector(
    embeddings_client: EmbeddingService,
    *,
    key: str,
    vector: list[float],
    metadata: dict[str, Any],
) -> str:
    record = EmbeddingRecord(
        source_hash=key,
        model_id=embeddings_client.model_id,
        dimension=len(vector),
        vector=list(vector),
        meta={**metadata, "source_hash_key": key},
    )
    await embeddings_client.cache.put(record)
    return str(record.embedding_id)

async def generate_validated_gist(
    *,
    path: Path,
    checksum: str,
    file_kind: str,
    content_excerpt: str,
    extracted_text: str,
    llm_client: LLMClient,
    embeddings_client: EmbeddingService,
    store: WorkspaceStore,
    llm_model: str,
    embedding_model: str,
    similarity_threshold: float = 0.78,
    max_attempts: int = 3,
) -> WorkspaceGistRecord:
    """
    Generates a gist, validates it semantically against file content,
    retries up to max_attempts, and stores the best accepted candidate
    or the best fallback candidate with a warning note.
    """
    file_vector: list[float] | None
    try:
        file_vector, content_text_status = await embed_file_semantics(
            extracted_text=extracted_text,
            embeddings_client=embeddings_client,
            namespace=f"workspace:file:{checksum}",
        )
        content_embedding_ref = await store_workspace_vector(
            embeddings_client,
            key=f"workspace:file:{checksum}",
            vector=file_vector,
            metadata={
                "checksum": checksum,
                "path": str(path),
                "kind": file_kind,
                "content_text_status": content_text_status,
            },
        )
        content_embedding_model = embedding_model
        content_embedding_dim = len(file_vector)
    except ValueError:
        file_vector = None
        content_text_status = "unreadable"
        content_embedding_ref = None
        content_embedding_model = None
        content_embedding_dim = None

    await store.upsert_checksum(
        checksum=checksum,
        size_bytes=len(extracted_text.encode("utf-8", errors="ignore")),
        file_kind=file_kind,
        mime_type=None,
        content_text_status=content_text_status,
        content_preview=content_excerpt[:1000],
        content_embedding_ref=content_embedding_ref,
        content_embedding_model=content_embedding_model,
        content_embedding_dim=content_embedding_dim,
    )

    previous_failures: list[str] = []
    best: GistCandidate | None = None

    for attempt_no in range(1, max_attempts + 1):
        prompt = build_gist_prompt(
            path=path,
            checksum=checksum,
            file_kind=file_kind,
            content_excerpt=content_excerpt,
            previous_failures=previous_failures,
        )

        # Assuming llm_client can parse JSON. If not, we might need to prompt for plain text or handle JSON manually.
        response = await llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=llm_model,
            complexity="light",
            payload={"system": "You are an expert file indexer returning structured JSON.", "max_tokens": 1024}
        )
        response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        try:
            # simple extract json
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start != -1 and end != -1:
                response_json = json.loads(response_text[start:end+1])
            else:
                response_json = {"summary": response_text.strip()}
        except Exception:
            response_json = {"summary": response_text.strip()}

        gist_text = render_gist_text(response_json).strip()
        if not gist_text:
            continue
            
        gist_embedding = await embeddings_client.embed(
            gist_text,
            meta={
                "checksum": checksum,
                "path": str(path),
                "kind": file_kind,
                "candidate_attempt_no": attempt_no,
            },
            task_type="workspace_gist_candidate",
        )
        gist_vector = l2_normalize(list(gist_embedding.vector))

        if file_vector is None:
            # Can't compare meaningfully, accept whatever the LLM guessed (often "This is a binary file")
            sim = 1.0
        else:
            sim = cosine_similarity(file_vector, gist_vector)
            
        drift = 1.0 - sim
        accepted = sim >= similarity_threshold

        candidate = GistCandidate(
            attempt_no=attempt_no,
            gist_text=gist_text,
            gist_json=response_json,
            cosine_similarity=sim,
            drift_score=drift,
            accepted=accepted,
        )

        await store.insert_gist_attempt(
            checksum=checksum,
            attempt_no=attempt_no,
            gist_text=gist_text,
            gist_json=json.dumps(response_json, ensure_ascii=False),
            cosine_similarity=sim,
            drift_score=drift,
            accepted_flag=accepted,
            failure_reason=None if accepted else f"similarity {sim:.4f} < threshold {similarity_threshold:.4f}",
        )

        if best is None or candidate.cosine_similarity > best.cosine_similarity:
            best = candidate

        if accepted:
            gist_embedding_ref = await store_workspace_vector(
                embeddings_client,
                key=f"workspace:gist:{checksum}",
                vector=gist_vector,
                metadata={
                    "checksum": checksum,
                    "path": str(path),
                    "kind": file_kind,
                    "accepted": True,
                    "attempt_no": attempt_no,
                },
            )

            record = WorkspaceGistRecord(
                checksum=checksum,
                gist_text=gist_text,
                gist_json=json.dumps(response_json, ensure_ascii=False),
                llm_model=llm_model,
                prompt_version=PROMPT_VERSION,
                gist_embedding_ref=gist_embedding_ref,
                gist_embedding_model=embedding_model,
                gist_embedding_dim=len(gist_vector),
                cosine_similarity=sim,
                drift_score=drift,
                accepted_flag=True,
                needs_further_reading=False,
                attempt_count=attempt_no,
                updated_at=store.utcnow(),
            )
            await store.upsert_gist(record)
            return record

        previous_failures.append(gist_text)

    # If we get here, best must be set (unless all attempts generated empty text, which is unlikely)
    if best is None:
        best = GistCandidate(1, "File could not be parsed or summarized.", {}, 0.0, 1.0, False)

    fallback_text = best.gist_text
    fallback_note = " Note: This gist may not fully capture the file's semantics and might need to be read further."
    if fallback_note.strip() not in fallback_text:
        fallback_text = f"{fallback_text}{fallback_note}"

    fallback_embedding = await embeddings_client.embed(
        fallback_text,
        meta={
            "checksum": checksum,
            "path": str(path),
            "kind": file_kind,
            "fallback": True,
        },
        task_type="workspace_gist_candidate",
    )
    fallback_vector = l2_normalize(list(fallback_embedding.vector))

    gist_embedding_ref = await store_workspace_vector(
        embeddings_client,
        key=f"workspace:gist:{checksum}",
        vector=fallback_vector,
        metadata={
            "checksum": checksum,
            "path": str(path),
            "kind": file_kind,
            "accepted": False,
            "attempt_no": best.attempt_no,
            "fallback": True,
        },
    )

    record = WorkspaceGistRecord(
        checksum=checksum,
        gist_text=fallback_text,
        gist_json=json.dumps(best.gist_json, ensure_ascii=False),
        llm_model=llm_model,
        prompt_version=PROMPT_VERSION,
        gist_embedding_ref=gist_embedding_ref,
        gist_embedding_model=embedding_model,
        gist_embedding_dim=len(fallback_vector),
        cosine_similarity=best.cosine_similarity,
        drift_score=best.drift_score,
        accepted_flag=False,
        needs_further_reading=True,
        attempt_count=max_attempts,
        updated_at=store.utcnow(),
    )
    await store.upsert_gist(record)
    return record
