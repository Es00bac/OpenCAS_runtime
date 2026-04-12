"""Sync workspace text files into episodic memory."""

import asyncio
from pathlib import Path

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.memory import ArtifactMemoryBridge


async def main():
    config = BootstrapConfig(
        state_dir="(workspace_root)/.opencas",
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

    # Sync the Chronicles directory
    chronicles_dir = Path("(workspace_root)/Chronicles")
    if chronicles_dir.exists():
        print(f"Syncing {chronicles_dir} ...")
        result = await bridge.sync_directory(chronicles_dir)
        print(f"Chronicles sync result: {result}")
    else:
        print(f"Chronicles directory not found: {chronicles_dir}")

    # Also sync top-level markdown files in the workspace by creating a temp dir wrapper
    workspace_root = Path("(workspace_root)")
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
