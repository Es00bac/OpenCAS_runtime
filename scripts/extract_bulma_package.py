"""
Extract a clean, operator-reviewable data package from an OpenBulma v4 state directory.

This script is READ-ONLY on the Bulma state. It writes only to the output package
directory. No OpenCAS runtime or embedding provider is required.

Usage:
    python scripts/extract_bulma_package.py [--bulma-state PATH] [--out PATH] [--dry-run]

Defaults:
    --bulma-state   /mnt/xtra/openbulma-v4/state
    --out           ./bulma-package-YYYYMMDD-HHMMSS

Output structure:
    <out>/
      REVIEW.md                 # operator pruning guide
      preflight.json            # preflight report (episodes, edges, secrets, clutter)
      manifest.json             # per-file manifest with sizes and sha256s
      memory/                   # episodes, edges, emotion history, consolidation, goal threads
      identity/                 # profile, rebuild audit
      self-knowledge/           # index.json
      daydream/                 # sparks, initiatives, outcomes, history, conflicts
      somatic/                  # current state, musubi
      executive/                # goals, commitments, workspace snapshot (no events.jsonl — too large)
      workspaces/               # creative workspace files (chronicle chapters, etc.)
      work-products/            # document draft JSONs
      skills/                   # registry.json
      relationship.json
      SKIPPED.md                # what was omitted and why
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Inline helpers (no opencas.legacy dependency so script is portable)
# ---------------------------------------------------------------------------

SECRET_KEY_PARTS = ("token", "api_key", "apikey", "secret", "password", "credential")

CLUTTER_DIRS = (
    "backups",
    "logs",
    "migration_runs",
    "root_owned_quarantine",
    "tool-result-spill",
    "foreground-artifacts",
    "foreground-workbench",
    "document-drafts",
    "deliverable-schemas",
    "heartbeat",
    "runtime-hooks",
    "webhooks",
    "qdrant_storage",
    "pids",
    "run",
    "dashboard",
    "reports",
    "audit",
    "audit-2026",
)

ESSENTIAL_FILES: List[Tuple[str, str]] = [
    # (source relative to bulma state, dest relative to package)
    ("memory/episodes.jsonl",                "memory/episodes.jsonl"),
    ("memory/edges.jsonl",                   "memory/edges.jsonl"),
    ("memory/emotion_history.jsonl",         "memory/emotion_history.jsonl"),
    ("memory/consolidation_reports.jsonl",   "memory/consolidation_reports.jsonl"),
    ("memory/goal_threads.jsonl",            "memory/goal_threads.jsonl"),
    ("memory/quality_snapshot.json",         "memory/quality_snapshot.json"),
    ("identity/profile.json",               "identity/profile.json"),
    ("identity/rebuild-audit.json",          "identity/rebuild-audit.json"),
    ("self-knowledge/index.json",            "self-knowledge/index.json"),
    ("daydream/sparks.jsonl",               "daydream/sparks.jsonl"),
    ("daydream/initiatives.jsonl",           "daydream/initiatives.jsonl"),
    ("daydream/spark_outcomes.jsonl",        "daydream/spark_outcomes.jsonl"),
    ("daydream/history.jsonl",               "daydream/history.jsonl"),
    ("daydream/conflicts.json",              "daydream/conflicts.json"),
    ("daydream/status.json",                 "daydream/status.json"),
    ("somatic/current.json",                "somatic/current.json"),
    ("somatic/musubi.json",                  "somatic/musubi.json"),
    ("somatic/history.jsonl",               "somatic/history.jsonl"),
    ("executive/goals.json",                "executive/goals.json"),
    ("executive/commitments.json",           "executive/commitments.json"),
    ("executive/workspace.json",             "executive/workspace.json"),
    # executive/events.jsonl intentionally skipped — can be 100k+ lines
    ("skills/registry.json",               "skills/registry.json"),
    ("relationship.json",                   "relationship.json"),
]

# Directories copied recursively (not individual files)
ESSENTIAL_DIRS: List[Tuple[str, str]] = [
    ("workspaces",    "workspaces"),
    ("work-products", "work-products"),
]

# Files that may contain secrets and should be redacted before copying
REDACT_PATHS = {
    "identity/profile.json",
    "self-knowledge/index.json",
    "daydream/config.json",
    "somatic/musubi.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            k: ("***" if any(p in str(k).lower() for p in SECRET_KEY_PARTS) and v else _redact(v))
            for k, v in payload.items()
        }
    if isinstance(payload, list):
        return [_redact(i) for i in payload]
    return payload


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def _count_dir_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def _walk_files(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _preflight(bulma: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "source": str(bulma),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "episodes": 0,
        "episodes_parse_errors": 0,
        "edges": 0,
        "duplicate_episode_ids": 0,
        "sparks": 0,
        "initiatives": 0,
        "somatic_history_entries": 0,
        "sessions": 0,
        "work_products": 0,
        "workspace_files": 0,
        "skills": 0,
        "secret_bearing_files": [],
        "clutter_present": {},
        "missing_essential": [],
        "notes": [],
    }

    ep_ids: set[str] = set()
    ep_path = bulma / "memory" / "episodes.jsonl"
    if ep_path.exists():
        for line in ep_path.open(encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                eid = obj.get("id", "")
                if eid in ep_ids:
                    report["duplicate_episode_ids"] += 1
                ep_ids.add(eid)
                report["episodes"] += 1
            except Exception:
                report["episodes_parse_errors"] += 1

    edges_path = bulma / "memory" / "edges.jsonl"
    if edges_path.exists():
        report["edges"] = _count_lines(edges_path)

    sparks_path = bulma / "daydream" / "sparks.jsonl"
    if sparks_path.exists():
        report["sparks"] = _count_lines(sparks_path)

    init_path = bulma / "daydream" / "initiatives.jsonl"
    if init_path.exists():
        report["initiatives"] = _count_lines(init_path)

    somatic_hist = bulma / "somatic" / "history.jsonl"
    if somatic_hist.exists():
        report["somatic_history_entries"] = _count_lines(somatic_hist)

    sessions_dir = bulma / "sessions"
    report["sessions"] = len(list(sessions_dir.glob("*.json"))) if sessions_dir.exists() else 0

    wp_dir = bulma / "work-products"
    report["work_products"] = _count_dir_files(wp_dir)

    ws_dir = bulma / "workspaces"
    report["workspace_files"] = _count_dir_files(ws_dir)

    skills_path = bulma / "skills" / "registry.json"
    if skills_path.exists():
        try:
            registry = json.loads(skills_path.read_text(encoding="utf-8"))
            entries = registry if isinstance(registry, list) else registry.get("skills", [])
            report["skills"] = len(entries)
        except Exception:
            pass

    for path in _walk_files(bulma):
        rel = str(path.relative_to(bulma))
        if any(part in path.name.lower() for part in SECRET_KEY_PARTS):
            report["secret_bearing_files"].append(rel)

    for name in CLUTTER_DIRS:
        d = bulma / name
        if d.exists():
            report["clutter_present"][name] = _count_dir_files(d)

    for src_rel, _ in ESSENTIAL_FILES:
        if not (bulma / src_rel).exists():
            report["missing_essential"].append(src_rel)

    if report["duplicate_episode_ids"]:
        report["notes"].append(
            f"{report['duplicate_episode_ids']} duplicate episode IDs found — importer deduplicates by Bulma ID."
        )
    if report["episodes_parse_errors"]:
        report["notes"].append(
            f"{report['episodes_parse_errors']} episode lines failed to parse."
        )
    if report["edges"] > 50_000:
        report["notes"].append(
            f"{report['edges']:,} edges present — this is large. "
            "Consider pruning low-confidence edges before import: "
            "`jq 'select(.confidence > 0.3)' memory/edges.jsonl > memory/edges.filtered.jsonl`"
        )

    return report


def _copy_file(src: Path, dst: Path, redact: bool = False) -> Dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if redact:
        try:
            raw = json.loads(src.read_text(encoding="utf-8"))
            cleaned = _redact(raw)
            dst.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)
    return {
        "path": str(dst),
        "size_bytes": dst.stat().st_size,
        "sha256": _sha256(dst),
    }


def _copy_dir(src: Path, dst: Path) -> Tuple[int, int]:
    """Return (files_copied, files_skipped)."""
    copied = skipped = 0
    if not src.exists():
        return 0, 0
    for source in _walk_files(src):
        rel = source.relative_to(src)
        target = dst / rel
        if source.is_symlink():
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied, skipped


def _write_review_md(out: Path, preflight: Dict[str, Any], skipped: List[str]) -> None:
    lines = [
        "# Bulma Package — Operator Review Guide",
        "",
        f"Extracted: {preflight['checked_at']}  ",
        f"Source: `{preflight['source']}`",
        "",
        "---",
        "",
        "## What Is Here",
        "",
        "| Category | Count |",
        "| --- | --- |",
        f"| Episodes | {preflight['episodes']:,} |",
        f"| Memory edges | {preflight['edges']:,} |",
        f"| Daydream sparks | {preflight['sparks']:,} |",
        f"| Daydream initiatives | {preflight['initiatives']:,} |",
        f"| Sessions | {preflight['sessions']:,} |",
        f"| Work-product drafts | {preflight['work_products']:,} |",
        f"| Workspace files | {preflight['workspace_files']:,} |",
        f"| Skills | {preflight['skills']:,} |",
        "",
        "---",
        "",
        "## What To Review Before Import",
        "",
        "### 1. Memory edges (`memory/edges.jsonl`)",
        "",
        f"There are **{preflight['edges']:,} edges**. Most agents import all of them, but you can thin",
        "low-confidence ones now to reduce import time:",
        "",
        "```bash",
        "# Preview confidence distribution",
        "jq -r '.confidence' memory/edges.jsonl | sort -n | uniq -c | tail -20",
        "",
        "# Keep only edges with confidence > 0.3 (adjust threshold as needed)",
        "jq -c 'select(.confidence > 0.3)' memory/edges.jsonl > memory/edges.filtered.jsonl",
        "mv memory/edges.filtered.jsonl memory/edges.jsonl",
        "```",
        "",
        "### 2. Work-product drafts (`work-products/`)",
        "",
        f"**{preflight['work_products']:,} draft JSON files.** These are Bulma-era document artifacts.",
        "Review and delete any that are obsolete or redundant before import.",
        "Each file is a self-contained JSON object with `objective`, `summary`, and `content`.",
        "",
        "```bash",
        "# Quick summary of all objectives",
        "for f in work-products/*.json; do jq -r '.objective // .summary // \"(no label)\"' \"$f\"; done | sort",
        "```",
        "",
        "### 3. Workspaces (`workspaces/`)",
        "",
        "Creative workspace files (chronicle chapters, scripts, etc.).",
        "These will be copied into OpenCAS workspace storage on import.",
        "Delete any sub-directories you do not want migrated.",
        "",
        "### 4. Identity (`identity/profile.json`)",
        "",
        "This becomes the OpenCAS self-model. Review the `coreNarrative`, `values`, `traits`,",
        "and `ongoingGoals` fields. Edit directly if anything is stale.",
        "",
        "### 5. Skills (`skills/registry.json`)",
        "",
        f"**{preflight['skills']} skills** in the registry. OpenCAS will import enabled skills only.",
        "Set `\"enabled\": false` on any you want to exclude.",
        "",
        "---",
        "",
        "## Notes From Preflight",
        "",
    ]

    for note in preflight.get("notes", []):
        lines.append(f"- {note}")
    if not preflight.get("notes"):
        lines.append("- No issues detected.")

    if preflight.get("missing_essential"):
        lines.append("")
        lines.append("### Missing source files (will be skipped on import)")
        for m in preflight["missing_essential"]:
            lines.append(f"- `{m}`")

    if preflight.get("secret_bearing_files"):
        lines.append("")
        lines.append("### Files with secret-like names (already redacted in this package)")
        for s in preflight["secret_bearing_files"]:
            lines.append(f"- `{s}`")

    lines += [
        "",
        "---",
        "",
        "## When You Are Ready To Import",
        "",
        "Point the importer at this directory:",
        "",
        "```python",
        "# In a Python session with the OpenCAS venv active:",
        "from opencas.legacy.importer import BulmaImportTask",
        "from pathlib import Path",
        "",
        "# The runtime must be a fully-started AgentRuntime instance.",
        "# This is handled by the import API route or a dedicated script.",
        "task = BulmaImportTask(",
        f"    bulma_state_dir=Path('{out}'),",
        "    runtime=runtime,",
        ")",
        "report = await task.run()",
        "print(report.model_dump_json(indent=2))",
        "```",
        "",
        "The import is checkpointed — if it fails partway through, re-running resumes",
        "from the last completed phase.",
        "",
        "---",
        "",
        "## What Was Skipped",
        "",
    ]

    for s in skipped:
        lines.append(f"- {s}")

    (out / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def _write_skipped_md(out: Path, skipped: List[str]) -> None:
    lines = ["# Skipped Categories", "", "The following Bulma state directories were not copied:", ""]
    for s in skipped:
        lines.append(f"- {s}")
    (out / "SKIPPED.md").write_text("\n".join(lines), encoding="utf-8")


def run(bulma_state: Path, out: Path, dry_run: bool) -> None:
    print(f"Source:  {bulma_state}")
    print(f"Output:  {out}")
    if dry_run:
        print("DRY RUN — no files will be written.")

    if not bulma_state.exists():
        print(f"ERROR: Bulma state directory not found: {bulma_state}", file=sys.stderr)
        sys.exit(1)

    print("\nRunning preflight...")
    preflight = _preflight(bulma_state)

    print(f"  Episodes:       {preflight['episodes']:,}")
    print(f"  Edges:          {preflight['edges']:,}")
    print(f"  Sparks:         {preflight['sparks']:,}")
    print(f"  Work products:  {preflight['work_products']:,}")
    print(f"  Workspace files:{preflight['workspace_files']:,}")
    print(f"  Sessions:       {preflight['sessions']:,}")
    if preflight["notes"]:
        for note in preflight["notes"]:
            print(f"  NOTE: {note}")

    if dry_run:
        print("\nDry run complete. No output written.")
        print(json.dumps(preflight, indent=2))
        return

    out.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[Dict[str, Any]] = []
    skipped_reasons: List[str] = []

    # Essential individual files
    print("\nCopying essential files...")
    for src_rel, dst_rel in ESSENTIAL_FILES:
        src = bulma_state / src_rel
        if not src.exists():
            skipped_reasons.append(f"`{src_rel}` — not present in source")
            continue
        should_redact = src_rel in REDACT_PATHS
        entry = _copy_file(src, out / dst_rel, redact=should_redact)
        entry["source"] = src_rel
        entry["redacted"] = should_redact
        manifest_entries.append(entry)
        tag = " (redacted)" if should_redact else ""
        print(f"  {dst_rel}{tag}")

    # Essential directories
    print("\nCopying workspace directories...")
    for src_rel, dst_rel in ESSENTIAL_DIRS:
        src = bulma_state / src_rel
        if not src.exists():
            skipped_reasons.append(f"`{src_rel}/` — not present in source")
            continue
        copied, sym_skipped = _copy_dir(src, out / dst_rel)
        print(f"  {dst_rel}/  ({copied} files)")
        if sym_skipped:
            skipped_reasons.append(f"`{src_rel}/` — {sym_skipped} symlinks skipped")
        manifest_entries.append({"source": src_rel + "/", "files_copied": copied})

    # Record clutter as skipped
    for name in CLUTTER_DIRS:
        d = bulma_state / name
        if d.exists():
            count = preflight["clutter_present"].get(name, 0)
            skipped_reasons.append(f"`{name}/` — {count} files, clutter category (not needed for import)")

    # Write metadata
    (out / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps({"created_at": preflight["checked_at"], "entries": manifest_entries}, indent=2),
        encoding="utf-8",
    )
    _write_skipped_md(out, skipped_reasons)
    _write_review_md(out, preflight, skipped_reasons)

    total_size = sum(
        e.get("size_bytes", 0) for e in manifest_entries if isinstance(e.get("size_bytes"), int)
    )
    print(f"\nPackage written to: {out}")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Files in manifest: {len(manifest_entries)}")
    print(f"\nNext step: read {out}/REVIEW.md and prune before import.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a clean Bulma data package for operator review.")
    parser.add_argument(
        "--bulma-state",
        type=Path,
        default=Path("/mnt/xtra/openbulma-v4/state"),
        help="Path to the OpenBulma v4 state directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output package directory (default: ./bulma-package-YYYYMMDD-HHMMSS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight only, print report, write nothing",
    )
    args = parser.parse_args()

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out = Path(f"bulma-package-{stamp}")

    run(args.bulma_state, args.out, args.dry_run)


if __name__ == "__main__":
    main()
