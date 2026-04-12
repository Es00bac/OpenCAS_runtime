# OpenCAS Delta State (Changes Since Last Checkup)
**Date:** 2026-04-09
**Previous Checkpoint:** `docs/opencas-deep-system-audit-2026-04-09.md` and initial baseline.

## 1. Scope of Delta
This document tracks explicit changes made to the OpenCAS repository since the previous deep system audit.

## 2. New Subsystems & Files
*   **`opencas/embeddings/hnsw_backend.py`**: Added to the codebase to support `hnswlib`. This directly solves the `O(N)` SQLite brute-force search limitation identified in early audits. The requirements list was updated to include `hnswlib>=0.7.0`.
*   **`opencas/tools/mcp_client.py` & `mcp_adapter.py`**: Implemented standard Model Context Protocol (MCP) integrations. This closes the feature-parity gap with LegacyPrototype-v4 regarding external tool usage.
*   **`scripts/sweep_operator_processes.py`**: A new script added to ensure process hygiene and cleanup after intensive background executions (BAA runs).

## 3. Metrics & Scenarios Executed
*   **Test Count:** Increased from **644** to **838** collected `pytest` functions.
*   **Completed Qualification Scenarios:**
    *   `Scenario 1`: Integrated operator workflow (browser inspection, PTY editing).
    *   `Scenario 2`: Repo triage to working note scaffolding.
    *   `Scenario 3`: Operator intervention and recovery validation.
    *   `Scenario 4`: Recovery from PTY/editor tool friction (vim write error classification).
    *   `Scenario 5`: Browser drift recovery and durable screenshot evidence.
    *   `Scenario 6`: Provider-backed timeout cleanup and harness exits.

## 4. Documentation & Process
*   **TaskList.md** was heavily updated to reflect new Workstreams (PR-001 through PR-010).
*   **New Workflow Focus:** The active task priority officially shifted from "building new architecture" to "Inner-life operationalization" (`PR-009`) and "Operator substrate hardening" (`PR-010`).

## 5. Items Checked But Not Yet Changed
*   `BaaPauseEvent` is still unhandled by the background execution engine.
*   `plan_mode` persistence has not yet been addressed.
