"""Sync key Chronicle summary files into episodic memory."""

import asyncio
from pathlib import Path

from opencas.bootstrap import BootstrapConfig, BootstrapPipeline
from opencas.memory import ArtifactMemoryBridge


async def main():
    config = BootstrapConfig(
        state_dir="(workspace_root)/.opencas",
        session_id="chronicle-sync",
        clean_boot=False,
    )
    ctx = await BootstrapPipeline(config).run()

    bridge = ArtifactMemoryBridge(
        state_dir=config.state_dir,
        memory=ctx.memory,
        embeddings=ctx.embeddings,
        chunk_chars=3000,
        overlap_chars=300,
        max_bytes=1_000_000,
    )

    summary_files = [
        Path("(workspace_root)/Chronicles/2046/chronicle_2046.md"),
        Path("(workspace_root)/Chronicles/2146/chronicle_2146_outline.md"),
        Path("(workspace_root)/Chronicles/3146/chronicle_3146.md"),
        Path("(workspace_root)/Chronicles/4246/chronicle_4246.md"),
        Path("(workspace_root)/Chronicles/chronicle_work_status_report.md"),
        Path("(workspace_root)/Chronicles/chronicle_reflection_notes.md"),
    ]

    total = {"artifacts": 0, "episodes_created": 0, "episodes_updated": 0, "episodes_deleted": 0, "memories_upserted": 0}
    for path in summary_files:
        if path.exists():
            print(f"Syncing {path.name} ...")
            result = await bridge.sync_directory(path)
            for k in total:
                total[k] += result[k]
        else:
            print(f"Missing: {path}")

    print(f"Total: {total}")
    await ctx.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
