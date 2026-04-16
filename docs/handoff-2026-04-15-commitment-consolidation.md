# Handoff Document: Commitment Consolidation & Auto-Resume
**Date:** 2026-04-15  
**Author:** Claude Code (Kimi)  
**Status:** Historical mid-flight snapshot from before `PR-019` through `PR-021` were completed

> Historical note: this handoff captured the mid-flight state of the 2026-04-14 through 2026-04-15 commitment work. Parts of the implementation status here are now superseded by later fixes in `PR-019` through `PR-021`. Use [TaskList.md](../TaskList.md) and [opencas-continuation-program-2026-04-15.md](opencas-continuation-program-2026-04-15.md) for the live frontier.

---

## 1. Context & Problem Statement

The user (Jarrod) is running a live OpenCAS instance ("Balma") at `./`. A core issue was identified: Balma frequently makes verbal self-commitments in chat (e.g., *"I'll carry this thread forward when I'm ready"*), but these commitments evaporate because nothing persists them as actionable work. The user wants:

1. **Self-commitments captured from assistant responses** during chat.
2. **Auto-resume** when the agent recovers from fatigue/overload.
3. **Nightly consolidation** that deduplicates commitments using embeddings + LLM reasoning, extracts missed commitments from chat logs, and ensures every surviving commitment is linked to executable work.

---

## 2. Completed Work

### A. Dashboard Chat 400 Fix (Kimi API)
**Files:**
- `../open_llm_auth/src/open_llm_auth/providers/anthropic_compatible.py`

**What changed:**
- Added default `tool_choice: {"type": "auto"}` when tools are present but no explicit choice given.
- Added orphan filtering in `_convert_messages()`: drops `tool_result` blocks whose `tool_use_id` has no matching assistant `tool_calls`.
- Added debug print on 400 errors.

**Status:** Live.

### B. Fatigue Decay Fix
**Files:**
- `./opencas/runtime/scheduler.py`

**What changed:**
- Added `somatic.decay()` call inside `_baa_heartbeat_loop()` (ticks every 60s).
- This was previously dead code — `decay()` existed in `somatic/manager.py` but was never invoked.

**Status:** Live.

### C. Auto-Resume for Deferred Work
**Files:**
- `./opencas/autonomy/executive.py` — added `resume_deferred_work()`
- `./opencas/runtime/scheduler.py` — added `_last_cycle_allowed` tracking and `_on_cycle_resume()`

**What changed:**
- When `_cycle_loop` transitions from paused → resumed (fatigue dropped below threshold), it triggers `resume_deferred_work()`.
- That method unblocks all `BLOCKED` commitments to `ACTIVE`, re-enqueues their linked work, and restores the task queue.

**Status:** Live.

### D. Self-Commitment Capture from Assistant Responses
**Files:**
- `./opencas/runtime/agent_loop.py` — added `_extract_self_commitments()` and inline capture logic in `converse()`

**What changed:**
- After every assistant response, regex patterns scan for deferral language.
- On match, creates a `Commitment` in `commitments.db` (BLOCKED if fatigued, else ACTIVE) and records a ToM `Intention` (actor: `self`).
- Regex covers phrases like: *"I'll carry this forward when I'm ready"*, *"We can get back to this later"*, *"back to the Chronicles later"*, etc.

**Status:** Live.

---

## 3. Incomplete Work: Hybrid LLM + Embedding Commitment Consolidation

### What exists
**Files modified but incomplete:**
- `./opencas/consolidation/models.py`
- `./opencas/consolidation/engine.py`
- `./opencas/runtime/agent_loop.py` (needs `commitment_store` and `work_store` wired into `NightlyConsolidationEngine`)

### What was added
1. **`ConsolidationResult`** (`models.py`) got new fields:
   - `commitments_consolidated: int = 0`
   - `commitment_clusters_formed: int = 0`
   - `commitment_work_objects_created: int = 0`
   - `commitments_extracted_from_chat: int = 0`

2. **`NightlyConsolidationEngine.__init__`** (`engine.py`) now accepts:
   - `commitment_store: Optional[CommitmentStore] = None`
   - `work_store: Optional[WorkStore] = None`

3. **Call sites inserted** in `engine.py` `run()` method (lines ~140-149):
   ```python
   if self.commitment_store:
       commitment_result = await self._consolidate_commitments(
           similarity_threshold=similarity_threshold
       )
       result.commitments_consolidated = commitment_result.get("commitments_merged", 0)
       ...
       chat_extracted = await self._extract_commitments_from_chat_logs()
       result.commitments_extracted_from_chat = chat_extracted
   ```

### What is MISSING
**Two methods in `consolidation/engine.py` are called but NOT YET DEFINED:**
- `_consolidate_commitments(self, similarity_threshold: float) -> Dict[str, Any]`
- `_extract_commitments_from_chat_logs(self) -> int`

**Also missing:** `AgentRuntime.__init__` does NOT yet pass `commitment_store` or `work_store` into `NightlyConsolidationEngine`.

---

## 4. Implementation Spec for Missing Methods

### `_consolidate_commitments(similarity_threshold)`

**Purpose:** Deduplicate active/blocked commitments via embedding clustering + LLM merge decisions, then ensure survivors have executable work objects.

**Steps:**
1. Fetch all `ACTIVE` and `BLOCKED` commitments via `self.commitment_store.list_by_status()`.
2. Embed each commitment content (`task_type="retrieval_query"`).
3. Cluster by cosine similarity (reuse greedy clusterer pattern from `_cluster_episodes()`).
4. For each cluster with size > 1:
   - **LLM call:** Send the cluster contents to the LLM with a merge prompt. Ask it to pick a survivor, explain why, and note if any are already satisfied/subsumed.
   - **Fallback heuristic:** If LLM fails, pick survivor by: most recent `updated_at` > highest `priority` > most `linked_work_ids` > `ACTIVE` over `BLOCKED`.
   - Merge `linked_work_ids` and `linked_task_ids` from all cluster members into survivor (deduplicate).
   - Set survivor status to `ACTIVE`.
   - Update all other members to `CommitmentStatus.ABANDONED`.
   - Save survivor via `self.commitment_store.save()`.
5. For every survivor that has **zero** `linked_work_ids`:
   - Create a `WorkObject(content=commitment.content, stage=WorkStage.MICRO_TASK, commitment_id=str(commitment.commitment_id))`.
   - Save it via `self.work_store.save(work)`.
   - Link it via `self.commitment_store.link_work(str(commitment.commitment_id), str(work.work_id))`.
6. Return stats dict: `{"clusters_formed": int, "commitments_merged": int, "work_objects_created": int}`.

### `_extract_commitments_from_chat_logs()`

**Purpose:** Use LLM to scan recent conversation episodes for self-commitments that the real-time regex missed.

**Steps:**
1. Query `self.memory.list_non_compacted_episodes(limit=200)` or similar, filtering for `kind == "turn"` and recent timestamps (last 48-72h).
2. Extract assistant turns and the 1-2 preceding user turns for context.
3. Build an LLM prompt:
   > "Review these recent conversation excerpts. Extract any self-commitments or promises the assistant made to the user that are not yet tracked. Return a JSON list of objects with fields: content, inferred_status (active/blocked/completed), reason."
4. Parse the response.
5. For each extracted commitment:
   - Skip if an existing commitment already covers it (simple substring or embedding similarity check against existing commitments).
   - Create a new `Commitment` with status from LLM or default `ACTIVE`.
6. Return the count of newly extracted commitments.

### Wire the stores into `AgentRuntime`

**File:** `opencas/runtime/agent_loop.py`, around line 233

```python
self.consolidation = NightlyConsolidationEngine(
    memory=self.memory,
    embeddings=context.embeddings,
    llm=self.llm,
    identity=context.identity,
    tracer=self.tracer,
    curation_store=getattr(context, "curation_store", None),
    tom_store=getattr(context, "tom_store", None),
    commitment_store=self.commitment_store,
    work_store=getattr(context, "work_store", None),
)
```

---

## 5. Key Files

| File | State | Notes |
|------|-------|-------|
| `opencas/consolidation/engine.py` | **Incomplete** | Missing `_consolidate_commitments` and `_extract_commitments_from_chat_logs` |
| `opencas/consolidation/models.py` | Modified | New result fields added |
| `opencas/runtime/agent_loop.py` | Modified (partial) | Self-commitment capture live; consolidation engine wiring incomplete |
| `opencas/autonomy/executive.py` | Modified | `resume_deferred_work()` live |
| `opencas/runtime/scheduler.py` | Modified | Auto-resume on cycle recovery live |
| `opencas/autonomy/commitment_store.py` | Unchanged | Reusable: `list_by_status`, `save`, `link_work`, `update_status` |
| `opencas/autonomy/work_store.py` | Unchanged | Reusable: `save`, `get` |
| `opencas/autonomy/models.py` | Unchanged | `WorkObject`, `WorkStage`, `Commitment`, `CommitmentStatus` |

---

## 6. Live Process

The OpenCAS server is currently running. Before testing any code changes, **restart the server** so the new Python module code is loaded. Use:

```bash
# Find and kill existing server
kill -9 $(lsof -t -i :8080) 2>/dev/null
rm -f .opencas/instance.lock .opencas/run.lock

# Restart
source .venv/bin/activate
python -m opencas --with-server --port 8080
```

---

## 7. User Intent Summary

> "Most important thing is that this agent actually takes action and doesn't just talk about taking action."

The user's priority is **executable outcomes**, not just storage. Every commitment that survives consolidation must have a path to execution:
- `Commitment` → `WorkObject` → creative ladder → BAA queue → tool use.
- Auto-resume ensures paused work unblocks when the agent recovers.
- LLM extraction ensures nuanced chat commitments aren't lost to regex blindness.
