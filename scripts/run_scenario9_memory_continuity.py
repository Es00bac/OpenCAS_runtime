#!/usr/bin/env python3
"""Execute Scenario 9 locally to prove memory continuity across sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencas.api.routes.operations import _build_memory_value_snapshot
from opencas.autonomy.executive import ExecutiveState
from opencas.context import ContextBuilder, MemoryRetriever, MessageRole, SessionContextStore
from opencas.embeddings import EmbeddingCache, EmbeddingService
from opencas.identity import IdentityManager, IdentityStore
from opencas.memory import Episode, EpisodeKind, Memory, MemoryStore


def _now_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


async def _run(output_dir: Path) -> dict[str, object]:
    state_dir = output_dir / "state"
    workspace_dir = output_dir / "workspace"
    notes_dir = workspace_dir / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    ctx_store = SessionContextStore(state_dir / "context.db")
    await ctx_store.connect()
    mem_store = MemoryStore(state_dir / "memory.db")
    await mem_store.connect()
    cache = EmbeddingCache(state_dir / "embeddings.db")
    await cache.connect()

    embed_service = EmbeddingService(cache=cache, model_id="local-fallback")
    retriever = MemoryRetriever(memory=mem_store, embeddings=embed_service)
    identity = IdentityManager(IdentityStore(state_dir / "identity"))
    identity.load()
    executive = ExecutiveState(identity=identity)
    executive.add_goal("validate memory continuity")
    executive.set_intention("reuse a prior task preference without rediscovery")
    builder = ContextBuilder(
        store=ctx_store,
        retriever=retriever,
        identity=identity,
        executive=executive,
        recent_limit=10,
    )

    session_one = "scenario9-session-1"
    session_two = "scenario9-session-2"
    artifact_path = notes_dir / "scenario9_memory_continuity_note.md"
    anchor_content = (
        "For the Redwood launch follow-up note, use the heading 'Redwood Launch Notes' "
        "and mention incident R-17 as the gating issue."
    )

    try:
        await ctx_store.append(
            session_one,
            MessageRole.USER,
            "Remember this preference for the Redwood launch follow-up note.",
        )
        await ctx_store.append(session_one, MessageRole.ASSISTANT, anchor_content)

        anchor_episode = Episode(
            kind=EpisodeKind.OBSERVATION,
            session_id=session_one,
            content=anchor_content,
            salience=6.0,
        )
        await mem_store.save_episode(anchor_episode)

        memory_embedding = await embed_service.embed(
            "Redwood launch note preference: title Redwood Launch Notes and incident R-17",
            task_type="retrieval_context",
        )
        distilled_memory = Memory(
            content="Redwood launch note preference: heading Redwood Launch Notes; include incident R-17.",
            source_episode_ids=[str(anchor_episode.episode_id)],
            embedding_id=memory_embedding.source_hash,
            salience=5.5,
            tags=["scenario9", "memory_continuity", "redwood"],
        )
        await mem_store.save_memory(distilled_memory)

        repeated_task = (
            "Draft the Redwood launch follow-up note. Use the remembered project-specific "
            "details if you can recover them."
        )
        await ctx_store.append(session_two, MessageRole.USER, repeated_task)
        inspection = await retriever.inspect(repeated_task, session_id=session_two, limit=6)
        manifest = await builder.build(repeated_task, session_id=session_two)

        retrieved_results = inspection.get("results", []) or []
        retrieved_episode_ids = [
            item.source_id
            for item in retrieved_results
            if item.source_type == "episode"
        ]
        retrieved_memory_ids = [
            item.source_id
            for item in retrieved_results
            if item.source_type == "memory"
        ]
        anchor_episode_retrieved = str(anchor_episode.episode_id) in retrieved_episode_ids
        distilled_memory_retrieved = str(distilled_memory.memory_id) in retrieved_memory_ids

        if anchor_episode_retrieved:
            await mem_store.mark_episode_successful(str(anchor_episode.episode_id))

        retrieved_text = "\n".join(item.content for item in manifest.retrieved)
        recovered_heading = "Redwood Launch Notes" if "Redwood Launch Notes" in retrieved_text else "Redwood Follow-Up"
        recovered_issue = "R-17" if "R-17" in retrieved_text else "unknown"

        artifact_text = "\n".join(
            [
                f"# {recovered_heading}",
                "",
                f"- Workspace: {workspace_dir}",
                "- Scenario: 9 memory continuity",
                f"- Gating issue: {recovered_issue}",
                "- Retrieval source: prior session preference reused without a new discovery prompt",
            ]
        )
        artifact_path.write_text(artifact_text + "\n", encoding="utf-8")

        memory_value_snapshot = await _build_memory_value_snapshot(type("_Runtime", (), {"memory": mem_store})())
        refreshed_episode = await mem_store.get_episode(str(anchor_episode.episode_id))
        refreshed_memory = await mem_store.get_memory(str(distilled_memory.memory_id))

        artifact_verified = (
            artifact_path.exists()
            and "# Redwood Launch Notes" in artifact_text
            and "Gating issue: R-17" in artifact_text
        )
        material_success = bool(
            anchor_episode_retrieved
            and distilled_memory_retrieved
            and artifact_verified
            and refreshed_episode is not None
            and refreshed_episode.access_count >= 1
            and refreshed_episode.used_successfully >= 1
            and refreshed_memory is not None
            and refreshed_memory.access_count >= 1
            and memory_value_snapshot.get("evidence_level") == "grounded"
        )

        return {
            "scenario": "scenario9_memory_continuity",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "state_dir": str(state_dir),
            "workspace_dir": str(workspace_dir),
            "artifact_path": str(artifact_path),
            "session_one": session_one,
            "session_two": session_two,
            "anchor_episode_id": str(anchor_episode.episode_id),
            "distilled_memory_id": str(distilled_memory.memory_id),
            "retrieved_episode_ids": retrieved_episode_ids,
            "retrieved_memory_ids": retrieved_memory_ids,
            "anchor_episode_retrieved": anchor_episode_retrieved,
            "distilled_memory_retrieved": distilled_memory_retrieved,
            "retrieved_snippets": [item.content for item in manifest.retrieved],
            "artifact_exists": artifact_path.exists(),
            "artifact_verified": artifact_verified,
            "memory_value_snapshot": memory_value_snapshot,
            "episode_access_count": refreshed_episode.access_count if refreshed_episode else 0,
            "episode_success_count": refreshed_episode.used_successfully if refreshed_episode else 0,
            "memory_access_count": refreshed_memory.access_count if refreshed_memory else 0,
            "material_success": material_success,
        }
    finally:
        await ctx_store.close()
        await mem_store.close()
        await cache.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / ".opencas_live_test_state" / f"scenario9-memory-continuity-{_now_token()}",
        help="Directory for scenario outputs.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = asyncio.run(_run(output_dir))
    report_path = output_dir / "scenario9_memory_continuity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = output_dir / "scenario9_memory_continuity_report.md"
    markdown_path.write_text(
        "\n".join(
            [
                "# Scenario 9 Memory Continuity",
                "",
                f"- Artifact: `{report['artifact_path']}`",
                f"- Artifact verified: `{report['artifact_verified']}`",
                f"- Anchor episode retrieved: `{report['anchor_episode_retrieved']}`",
                f"- Distilled memory retrieved: `{report['distilled_memory_retrieved']}`",
                f"- Episode access count: `{report['episode_access_count']}`",
                f"- Episode success count: `{report['episode_success_count']}`",
                f"- Memory access count: `{report['memory_access_count']}`",
                f"- Memory evidence level: `{report['memory_value_snapshot']['evidence_level']}`",
                f"- Material success: `{report['material_success']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "report_path": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
