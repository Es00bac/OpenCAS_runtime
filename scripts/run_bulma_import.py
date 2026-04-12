"""One-way cutover import: OpenBulma v4 → OpenCAS.

Usage:
    source .venv/bin/activate
    python scripts/run_bulma_import.py [--dry-run] [--resume]

Clears the existing OpenCAS agent state, then imports Bulma's memories,
identity, daydreams, skills, governance history, sessions, and all other
state from OpenBulma v4.  Uses ~/work as the curated workspace directory.
"""

import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from opencas.bootstrap.config import BootstrapConfig
from opencas.bootstrap.pipeline import BootstrapPipeline
from opencas.runtime.agent_loop import AgentRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("bulma_import")

# ── Paths ────────────────────────────────────────────────────────────────────
BULMA_STATE_DIR   = Path("/mnt/xtra/openbulma-v4/state")
OPENCAS_STATE_DIR = Path("/mnt/xtra/OpenCAS/.opencas")
TEST_STATE_DIR    = Path("/mnt/xtra/OpenCAS/.opencas_live_test_state")
WORKSPACE_DIR     = Path.home() / "work"
CHECKPOINT_PATH   = Path("/mnt/xtra/OpenCAS/.opencas_import_checkpoint.json")


def clear_agent_state(resume: bool, yes: bool = False) -> None:
    """Delete existing OpenCAS agent state directories."""
    if resume:
        logger.info("--resume: keeping existing state, will pick up from checkpoint.")
        return

    existing = [d for d in (OPENCAS_STATE_DIR, TEST_STATE_DIR) if d.exists()]
    if existing and not yes:
        print("\nWARNING: The following state directories will be permanently deleted:")
        for d in existing:
            print(f"  {d}")
        print("\nThis will wipe an existing Bulma/OpenCAS identity and all imported state.")
        answer = input("Type 'yes' to confirm, anything else to abort: ").strip().lower()
        if answer != "yes":
            logger.info("Aborted by user.")
            sys.exit(0)

    for state_dir in (OPENCAS_STATE_DIR, TEST_STATE_DIR):
        if state_dir.exists():
            logger.info("Clearing %s", state_dir)
            shutil.rmtree(state_dir)

    OPENCAS_STATE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Fresh state directory created at %s", OPENCAS_STATE_DIR)


async def run(dry_run: bool = False, resume: bool = False, yes: bool = False) -> None:
    # Validate source
    if not BULMA_STATE_DIR.exists():
        logger.error("Bulma state directory not found: %s", BULMA_STATE_DIR)
        sys.exit(1)

    identity_check = BULMA_STATE_DIR / "identity" / "profile.json"
    if not identity_check.exists():
        logger.error("Missing identity/profile.json in Bulma state — aborting.")
        sys.exit(1)

    if not WORKSPACE_DIR.exists():
        logger.error("Workspace dir not found: %s", WORKSPACE_DIR)
        sys.exit(1)

    logger.info("Source  : %s", BULMA_STATE_DIR)
    logger.info("State   : %s", OPENCAS_STATE_DIR)
    logger.info("Workspace: %s", WORKSPACE_DIR)

    if dry_run:
        logger.info("DRY RUN — validating source counts only, no state will be written.")

    # Clear existing agent state (skip in resume or dry-run)
    if not dry_run:
        clear_agent_state(resume, yes=yes)

    logger.info("Bootstrapping OpenCAS runtime...")
    config = BootstrapConfig(
        state_dir=OPENCAS_STATE_DIR,
        session_id="bulma-import",
        clean_boot=not resume,
    )
    ctx = await BootstrapPipeline(config).run()
    runtime = AgentRuntime(ctx)
    logger.info("Runtime ready.")

    if dry_run:
        from opencas.legacy.importer import BulmaImportTask
        task = BulmaImportTask(
            BULMA_STATE_DIR,
            runtime=runtime,
            curated_workspace_dir=WORKSPACE_DIR,
        )
        report = await task.validate()
        logger.info("Validation report:\n%s", report.model_dump_json(indent=2))
        await runtime._close_stores()
        return

    logger.info("Starting import — this may take several minutes...")
    report = await runtime.import_bulma(
        bulma_state_dir=BULMA_STATE_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        curated_workspace_dir=WORKSPACE_DIR,
    )

    logger.info("Import complete.")
    report_path = OPENCAS_STATE_DIR / "import_report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Report written to %s", report_path)

    # Print summary
    r = report.model_dump()
    logger.info("Episodes imported      : %s", r.get("episodes_imported", 0))
    logger.info("Edges imported         : %s", r.get("edges_imported", 0))
    logger.info("Daydream sparks        : %s", r.get("daydream_sparks_imported", 0))
    logger.info("Daydream initiatives   : %s", r.get("daydream_initiatives_imported", 0))
    logger.info("Sessions imported      : %s", r.get("sessions_imported", 0))
    logger.info("Skills imported        : %s", r.get("skills_imported", 0))
    logger.info("Governance entries     : %s", r.get("governance_entries_imported", 0))
    logger.info("Workspaces imported    : %s", r.get("workspaces_imported", 0))
    logger.info("Errors                 : %s", r.get("errors", []))

    if r.get("errors"):
        logger.warning("Import finished with errors — check report at %s", report_path)
        sys.exit(1)

    # Clean up checkpoint on success
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint file removed.")

    await runtime._close_stores()
    logger.info("Done.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    resume  = "--resume"  in sys.argv
    yes     = "--yes"     in sys.argv
    asyncio.run(run(dry_run=dry_run, resume=resume, yes=yes))
