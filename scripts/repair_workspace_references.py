#!/usr/bin/env python3
"""Normalize stale workspace path references in OpenCAS SQLite state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opencas.maintenance import (
    build_repo_local_bootstrap_config,
    repair_workspace_references_in_sqlite,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair stale Chronicle/workspace references inside SQLite state files.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path.cwd()),
        help="Repository root used to derive the managed workspace root.",
    )
    parser.add_argument(
        "--state-dir",
        action="append",
        default=[],
        help="State directory containing SQLite files. May be repeated. Defaults to <repo>/.opencas.",
    )
    parser.add_argument(
        "--managed-root",
        default=None,
        help="Override the managed workspace root. Defaults to <repo>/workspace.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report candidate rewrites without modifying the databases.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    config = build_repo_local_bootstrap_config(
        repo_root,
        session_id="workspace-reference-repair",
        managed_workspace_root=Path(args.managed_root).expanduser().resolve()
        if args.managed_root
        else None,
    )
    managed_root = config.agent_workspace_root()
    state_dirs = [Path(p).expanduser().resolve() for p in args.state_dir] or [repo_root / ".opencas"]

    summaries = []
    for state_dir in state_dirs:
        for db_path in sorted(state_dir.glob("*.db")):
            summaries.append(
                repair_workspace_references_in_sqlite(
                    db_path,
                    repo_root=repo_root,
                    managed_root=managed_root,
                    dry_run=args.dry_run,
                ).to_dict()
            )

    print(
        json.dumps(
            {
                "repo_root": str(repo_root),
                "managed_root": str(managed_root),
                "dry_run": args.dry_run,
                "state_dirs": [str(path) for path in state_dirs],
                "summaries": summaries,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
