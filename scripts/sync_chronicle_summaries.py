"""Sync key Chronicle summary files into episodic memory."""

import asyncio
from pathlib import Path

from opencas.bootstrap import BootstrapPipeline
from opencas.maintenance import build_repo_local_bootstrap_config
from opencas.memory import ArtifactMemoryBridge


REPO_ROOT = Path(__file__).resolve().parent.parent


async def main():
    config = build_repo_local_bootstrap_config(
        REPO_ROOT,
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

    chronicles_root = config.agent_workspace_root() / "Chronicles"
    summary_files = [
        chronicles_root / "2046" / "chronicle_2046.md",
        chronicles_root / "2146" / "chronicle_2146_outline.md",
        chronicles_root / "3146" / "chronicle_3146.md",
        chronicles_root / "4246" / "chronicle_4246.md",
        chronicles_root / "chronicle_work_status_report.md",
        chronicles_root / "chronicle_reflection_notes.md",
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
