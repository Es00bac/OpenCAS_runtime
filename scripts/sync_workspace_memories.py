"""Sync workspace text files into episodic memory."""

import asyncio
from pathlib import Path

from opencas.bootstrap import BootstrapPipeline
from opencas.maintenance import build_repo_local_bootstrap_config
from opencas.memory import ArtifactMemoryBridge


REPO_ROOT = Path(__file__).resolve().parent.parent


async def main():
    config = build_repo_local_bootstrap_config(
        REPO_ROOT,
        session_id="workspace-sync",
        clean_boot=False,
    )
    ctx = await BootstrapPipeline(config).run()

    bridge = ArtifactMemoryBridge(
        state_dir=config.state_dir,
        memory=ctx.memory,
        embeddings=ctx.embeddings,
        chunk_chars=2000,
        overlap_chars=200,
        max_bytes=500_000,
    )

    workspace_root = config.agent_workspace_root()

    # Reflective context (self notes, prototypes, research, daydream digests, refusal
    # reflections) only becomes searchable when ingested through the bridge.
    # Without these sweeps, retrieval cannot surface Bulma's own reflections during
    # later work — which was the regression that motivated this script's expansion.
    sync_dirs = [
        workspace_root / "Chronicles",
        workspace_root / "self",
        workspace_root / "reflections",
        workspace_root / "daydream-lab",
    ]
    for target in sync_dirs:
        if not target.exists():
            print(f"Missing directory (skipped): {target}")
            continue
        print(f"Syncing {target} ...")
        result = await bridge.sync_directory(target)
        print(f"  result: {result}")

    # Also sync top-level markdown files in the managed workspace.
    md_files = [p for p in workspace_root.glob("*.md") if p.is_file()]
    if md_files:
        print(f"Syncing {len(md_files)} top-level markdown files ...")
        for path in md_files:
            result = await bridge.sync_directory(path)
            print(f"  {path.name}: {result}")

    await ctx.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
