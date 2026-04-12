# OpenCAS Claude/Codex Handoff

Date: 2026-04-08

Purpose:
- give Claude enough context to continue work immediately
- preserve the intent of the earlier Claude Code + kimi build phase
- preserve the implementation and validation state from the recent Codex hardening phase
- make alternating handoff between Claude and Codex low-friction

Related:
- [TaskList.md](/mnt/xtra/OpenCAS/TaskList.md)
- [documentation-map.md](/mnt/xtra/OpenCAS/docs/documentation-map.md)
- [production-readiness-status-2026-04-09.md](/mnt/xtra/OpenCAS/docs/production-readiness-status-2026-04-09.md)
- [first-regular-use-deployment-checklist.md](/mnt/xtra/OpenCAS/docs/first-regular-use-deployment-checklist.md)
- [AGENTS.md](/mnt/xtra/OpenCAS/AGENTS.md)
- [CLAUDE.md](/mnt/xtra/OpenCAS/CLAUDE.md)
- [OPENCAS_PRODUCT_SPEC.md](/mnt/xtra/OpenCAS/OPENCAS_PRODUCT_SPEC.md)
- [opencas-production-readiness-audit-2026-04-08.md](/mnt/xtra/OpenCAS/docs/opencas-production-readiness-audit-2026-04-08.md)
- [opencas-production-program-plan-2026-04-08.md](/mnt/xtra/OpenCAS/docs/opencas-production-program-plan-2026-04-08.md)
- [opencas-comprehensive-audit.md](/mnt/xtra/OpenCAS/docs/opencas-comprehensive-audit.md)
- [opencas-architecture-and-comparison.md](/mnt/xtra/OpenCAS/docs/opencas-architecture-and-comparison.md)

## 1. Project Intent

The earlier Claude Code + kimi phase was not trying to build a narrow coding copilot.

The consistent intent, visible in the code and in the repo audit artifacts, is:

- build a high-trust local-first CAS runtime
- keep long-horizon state, not just session-local chat memory
- integrate identity, somatic state, relational state, ToM, memory, execution, planning, governance, and dashboard surfaces into one runtime
- support real work across:
  - coding
  - writing
  - project management
  - browser use
  - terminal-native/TUI workflows
  - persistent project continuity
- preserve the ambitious architecture rather than collapsing it into a simpler but less differentiated assistant

The correct continuation strategy is:

- do not delete the unusual subsystems just because they are unusual
- force them to justify themselves through real operator performance, control, inspectability, and evaluation
- treat browser/PTTY/editor use as first-class operator substrate, not side features

## 2. Prior Planning Artifacts

Two planning documents were created to stabilize direction before hardening work:

- [opencas-production-readiness-audit-2026-04-08.md](/mnt/xtra/OpenCAS/docs/opencas-production-readiness-audit-2026-04-08.md)
- [opencas-production-program-plan-2026-04-08.md](/mnt/xtra/OpenCAS/docs/opencas-production-program-plan-2026-04-08.md)

Those documents already encode the main judgment:

- OpenCAS is a serious substrate, not a toy
- the missing work is operational hardening, control-plane maturity, evaluation, and higher-level operator workflows
- the next phase should not be more abstract cognition work by default
- the next phase should be:
  - correctness
  - control
  - operator substrate
  - production qualification

## 3. Recent Commit Trajectory

Recent commits, newest first:

- `8bdfd8f` `Wire operations control plane into dashboard`
- `bcbb780` `Document operations API routes in CLAUDE.md`
- `96f5df3` `Add operations API routes for dashboard control plane`
- `4bd13d4` `Document workflow tools in CLAUDE.md`
- `b37f8c5` `Add higher-level operator workflow tools`
- `ebb84e5` `Document PTY normalized screen text in CLAUDE.md`
- `2604277` `Add ANSI-stripped cleaned output to PTY supervisor`
- `a672936` `Keep full harness outputs for internal PTY parsing`
- `22a8580` `Make PTY operator flow viable for autonomous agents`
- `f3bd2c7` `Add per-app auth bootstrap and live validation harness`
- `46c537f` `Add agent-visible runtime status tool`
- `105fcb2` `Harden execution control plane and add operator sessions`
- `2be9abe` `Add AGENTS.md with full project context for multi-agent handoff`
- `c37b0a2` `Add OpenCAS comprehensive dashboard (Phase C)`
- `fa327af` `Implement emotionally-aware memory retrieval with multi-signal fusion`

Interpretation:

- earlier work built broad substrate and dashboard reach
- the most recent work focused on production-readiness hardening
- the system is now materially better at:
  - workspace-root attachment
  - browser/PTTY operator sessions
  - per-app auth bootstrap
  - live validation
  - tool curation
  - PTY tool usability under autonomous agent control
  - workflow-level operator tooling
  - dashboard control-plane visibility

## 4. What Was Implemented In The Recent Codex Phase

### 4.1 Embeddings and retrieval hardening

Implemented:

- model/task-specific embedding identity instead of text-only identity
- ANN retrieval now preserves real similarity scores
- backward-compatible cache lookup for older records
- backfill now persists the identifier the runtime expects
- memory projection endpoint no longer embeds on GET
- diagnostics/monitor routes no longer trigger mutating/provider-costly embedding probes
- remote embedding failures or rate limits degrade gracefully through local fallback
- HNSW local ANN is disabled on Python 3.14 because native stability was causing segfaults

Important:

- embeddings are enabled and working
- Gemini embeddings are live
- intermittent `429` happens under load, but the system continues
- only HNSW acceleration is disabled on this interpreter for stability

Primary files:

- [service.py](/mnt/xtra/OpenCAS/opencas/embeddings/service.py)
- [hnsw_backend.py](/mnt/xtra/OpenCAS/opencas/embeddings/hnsw_backend.py)
- [qdrant_backend.py](/mnt/xtra/OpenCAS/opencas/embeddings/qdrant_backend.py)
- [backfill.py](/mnt/xtra/OpenCAS/opencas/embeddings/backfill.py)
- [memory.py](/mnt/xtra/OpenCAS/opencas/api/routes/memory.py)
- [monitor.py](/mnt/xtra/OpenCAS/opencas/api/routes/monitor.py)
- [doctor.py](/mnt/xtra/OpenCAS/opencas/diagnostics/doctor.py)

### 4.2 Workspace and execution control-plane hardening

Implemented:

- explicit workspace-root model with primary root helpers
- CLI support for multiple workspace roots
- process and PTY sessions default to attached workspace roots
- browser operator supervisor and tools
- PTY supervisor and tools
- runtime/dashboard visibility into process/PTTY/browser sessions
- runtime inspection tools for the agent itself

Primary files:

- [config.py](/mnt/xtra/OpenCAS/opencas/bootstrap/config.py)
- [pipeline.py](/mnt/xtra/OpenCAS/opencas/bootstrap/pipeline.py)
- [__main__.py](/mnt/xtra/OpenCAS/opencas/__main__.py)
- [process.py](/mnt/xtra/OpenCAS/opencas/tools/adapters/process.py)
- [pty.py](/mnt/xtra/OpenCAS/opencas/tools/adapters/pty.py)
- [browser.py](/mnt/xtra/OpenCAS/opencas/tools/adapters/browser.py)
- [browser_supervisor.py](/mnt/xtra/OpenCAS/opencas/execution/browser_supervisor.py)
- [pty_supervisor.py](/mnt/xtra/OpenCAS/opencas/execution/pty_supervisor.py)
- [runtime_state.py](/mnt/xtra/OpenCAS/opencas/tools/adapters/runtime_state.py)
- [workflow_state.py](/mnt/xtra/OpenCAS/opencas/tools/adapters/workflow_state.py)
- [agent_loop.py](/mnt/xtra/OpenCAS/opencas/runtime/agent_loop.py)

### 4.3 Agent profiles and validation harness

Implemented:

- builtin agent profiles:
  - `general_technical_operator`
  - `debug_validation_operator`
- profile material is injected into system context
- live validation harness that boots a temporary high-trust debug agent and tests:
  - runtime introspection
  - workflow introspection
  - filesystem access
  - browser tools
  - PTY/TUI tools
  - agent-mediated tool use

Primary files:

- [agent_profile.py](/mnt/xtra/OpenCAS/opencas/runtime/agent_profile.py)
- [builder.py](/mnt/xtra/OpenCAS/opencas/context/builder.py)
- [run_live_debug_validation.py](/mnt/xtra/OpenCAS/scripts/run_live_debug_validation.py)

### 4.4 Per-app auth bootstrap

Implemented:

- OpenCAS can copy provider config and env material into its own state dir
- it does not need to share one global provider pipeline
- copied auth profiles and copied env keys are used to create app-local provider material

This was done because the user explicitly wanted app-local credential handling and asked that OpenLLMAuth per-app use be respected.

Primary files:

- [provider_material.py](/mnt/xtra/OpenCAS/opencas/bootstrap/provider_material.py)
- [config.py](/mnt/xtra/OpenCAS/opencas/bootstrap/config.py)
- [pipeline.py](/mnt/xtra/OpenCAS/opencas/bootstrap/pipeline.py)

## 5. External Credential Context

The user explicitly instructed:

- if credentials are needed, use `openbulma-v4`’s `.env`
- use OpenLLMAuth per-app support rather than one shared provider pipeline
- copy credentials, do not just point at the same live pipeline
- use openbulma’s embedding credentials too

Relevant paths:

- `/mnt/xtra/openbulma-v4/.env`
- `~/.open_llm_auth/config.json`
- `/mnt/xtra/open_llm_auth`

Observed auth profiles in `~/.open_llm_auth/config.json`:

- `anthropic:claude-code-pro`
- `google:default`
- `kimi-coding:default`
- `openai-codex:chatgpt-plus`
- `zaicoding:personal`

The current live validation harness uses copied material for:

- `kimi-coding:default`
- `google:default`

and env keys:

- `GEMINI_API_KEY`
- `QDRANT_API_KEY`
- `MEMORY_EMBED_MODEL_PROFILE`
- `MEMORY_EMBED_AUTH_PROFILE`
- `MEMORY_EMBED_COLLECTION`
- `MEMORY_EMBED_DIMENSIONS`
- `MEMORY_EMBED_READY_MIN_RATIO`

## 6. Current Validated State

### Tests

Latest full suite result:

- `837 passed in 183.95s`

### Live validation

Use:

- `source .venv/bin/activate && python scripts/run_live_debug_validation.py`

What that harness does:

- boots a temporary `debug_validation_operator`
- uses copied per-app provider material
- uses live Kimi conversation + Gemini embeddings
- runs direct checks and agent-mediated checks

Important reports:

- successful agent-mediated PTY/browser run:
  - [debug-validation-20260408-201426](/mnt/xtra/OpenCAS/.opencas_live_test_state/debug-validation-20260408-201426/live_debug_validation_report.md)
- latest hygiene run with artifacts redirected into state dir:
  - [debug-validation-20260408-201845](/mnt/xtra/OpenCAS/.opencas_live_test_state/debug-validation-20260408-201845/live_debug_validation_report.md)
- successful supervised Codex real-work eval:
  - [debug-validation-20260408-214117](/mnt/xtra/OpenCAS/.opencas_live_test_state/debug-validation-20260408-214117/live_debug_validation_report.md)
- current Kilo Code validation baseline:
  - [debug-validation-20260408-220155](/mnt/xtra/OpenCAS/.opencas_live_test_state/debug-validation-20260408-220155/live_debug_validation_report.md)
- latest Kilo supervised-work diagnosis:
  - [debug-validation-20260408-223339](/mnt/xtra/OpenCAS/.opencas_live_test_state/debug-validation-20260408-223339/live_debug_validation_report.md)
- latest Kilo supervised-work success:
  - [debug-validation-20260408-231052](/mnt/xtra/OpenCAS/.opencas_live_test_state/debug-validation-20260408-231052/live_debug_validation_report.md)
- focused standard-vim TUI success:
  - [debug-validation-20260408-232901](/mnt/xtra/OpenCAS/.opencas_live_test_state/debug-validation-20260408-232901/live_debug_validation_report.md)
- focused writing-workflow success:
  - [debug-validation-20260408-234917](/mnt/xtra/OpenCAS/.opencas_live_test_state/debug-validation-20260408-234917/live_debug_validation_report.md)
- focused project-management-workflow success:
  - [debug-validation-20260408-235606](/mnt/xtra/OpenCAS/.opencas_live_test_state/debug-validation-20260408-235606/live_debug_validation_report.md)
- focused writing-revision-workflow success:
  - [debug-validation-20260409-000125](/mnt/xtra/OpenCAS/.opencas_live_test_state/debug-validation-20260409-000125/live_debug_validation_report.md)
- focused integrated-operator-workflow success:
  - [debug-validation-20260409-003117](/mnt/xtra/OpenCAS/.opencas_live_test_state/debug-validation-20260409-003117/live_debug_validation_report.md)
- local operator-intervention recovery success:
  - [scenario3_operator_recovery_report.md](/mnt/xtra/OpenCAS/.opencas_live_test_state/scenario3-operator-recovery-20260409-170416/scenario3_operator_recovery_report.md)
- local repo-triage scenario success:
  - [scenario2_repo_triage_report.md](/mnt/xtra/OpenCAS/.opencas_live_test_state/scenario2-repo-triage-20260409-171519/scenario2_repo_triage_report.md)
- local PTY/tool-friction recovery success:
  - [scenario4_tool_friction_report.md](/mnt/xtra/OpenCAS/.opencas_live_test_state/scenario4-tool-friction-20260409-172334/scenario4_tool_friction_report.md)
- local browser-drift recovery success:
  - [scenario5_browser_drift_report.md](/mnt/xtra/OpenCAS/.opencas_live_test_state/scenario5-browser-drift-20260409-174329/scenario5_browser_drift_report.md)
- local provider-backed cleanup success:
  - [scenario6_provider_cleanup_report.md](/mnt/xtra/OpenCAS/.opencas_live_test_state/scenario6-provider-cleanup-20260409-174942/scenario6_provider_cleanup_report.md)

Interpretation:

- agent-mediated `claude` TUI probing works
- agent-mediated `codex` TUI probing works
- supervised `codex` PTY work can produce a real artifact
- agent-mediated browser probing works
- direct `kilocode` TUI probing works and reaches the ready chat UI without login friction
- `kilo run --auto` works once repo-local bootstrap config is present
- agent-mediated PTY editing of standard `vim` now produces and verifies a real artifact
- the higher-level writing workflow now creates a commitment, an active plan, and the requested artifact in live validation
- the higher-level project-management workflow now creates a durable commitment, an active plan, and a verified report artifact in live validation
- the higher-level writing workflow now also supports a successful draft->revision->verification loop in live validation
- one bounded integrated task can now combine `workflow_create_plan`, browser inspection, and PTY `vim` editing in the same live run and still produce a verified artifact
- explicit PTY/editor tool friction now classifies correctly as `vim_write_error`, recovers through operator follow-up, and leaves a verified artifact plus clean cleanup
- explicit browser drift now recovers through the operations browser control path, leaves durable screenshot evidence, and closes cleanly
- minimal provider-backed prompt timeouts now leave a clean harness exit and a zero-count stale-process sweep
- harness artifacts are now stored under:
  - `.opencas_live_test_state/<run-id>/workspace_artifacts/`

Nuance:

- `a672936` fixed a harness-only issue where large PTY outputs were truncated too early for internal JSON/session parsing
- later work fixed two runtime self-approval gaps:
  - `pty_kill` and `pty_remove` were being treated as generic shell actions under high-trust stress states
  - `workflow_supervise_session` was not inheriting the same bounded interactive risk metadata as direct PTY tools
- the harness now supports bounded per-prompt timeouts so long-running supervised-work evals fail cleanly instead of hanging indefinitely
- the harness now also has a bounded total run timeout and sweeps leaked PTY/process/browser sessions after each prompt
- the harness now force-exits after report write so completed runs cannot linger as idle Python processes
- PTY poll/observe/interact payloads now include a coarse `screen_state` heuristic so workflows can reason about shell prompts, `vim` insert mode, auth gates, and generic interactive readiness without re-parsing raw terminal output every time
- `workflow_supervise_session` now returns a `supervision_advisory` and uses adaptive observe timings so idle/blocked TUI states stop or shorten cleanly instead of burning unnecessary supervision rounds
- `PtySupervisor` now retains last observed PTY state/output, and `/api/operations/sessions` plus the dashboard Operations tab surface that summary for live operator inspection
- `/api/operations/sessions/pty/{session_id}` now supports PTY detail inspection with optional live refresh, and the dashboard uses it for non-destructive session inspection
- `/api/operations/sessions/pty/{session_id}/input` now supports operator follow-up input with immediate refresh, and the dashboard exposes that as a PTY `Send Input` action
- `/api/operations/sessions/browser/{session_id}` now supports browser detail inspection with optional live snapshot refresh, and the dashboard exposes that as a browser `View` action
- `/api/operations/sessions/browser/{session_id}/navigate` now supports redirecting a live browser session with immediate refresh, and the dashboard exposes that as a browser `Navigate` action
- `/api/operations/sessions/browser/{session_id}/click`, `/type`, `/press`, and `/wait` now support richer operator intervention against a live browser session with immediate refresh, and the dashboard exposes `Click`, `Type`, `Press`, and `Wait` actions from browser session detail
- `BrowserSupervisor` now persists last browser snapshot metadata per session and cleans up superseded screenshot temp files on replacement/close
- `/api/operations/sessions/browser/{session_id}/capture` and `/screenshot` now support explicit screenshot capture plus file serving, and the dashboard can capture and display the latest browser screenshot inline
- `/api/operations/sessions/browser/{session_id}` now also supports explicit browser-session deletion, and the dashboard exposes that as a browser `Close` action
- `/api/operations/sessions/pty` and `/api/operations/sessions/browser` now also support scoped bulk clear operations, and the dashboard exposes those as `Clear PTY Scope` and `Clear Browser Scope`
- `/api/operations/work/{work_id}` now supports work-item detail and patch updates, and the dashboard exposes operator actions for work stage/content/blocker edits
- `/api/operations/commitments/{commitment_id}` now supports commitment detail and patch updates, and the dashboard exposes operator actions for status/content edits
- `/api/operations/plans/{plan_id}` now supports patch updates for plan status/content, and the dashboard exposes operator actions for direct plan edits from the detail pane
- those edit routes now validate stage/status inputs at the request boundary, so malformed operator edits return `422` instead of throwing handler-time parsing errors
- the dashboard now renders in-pane form controls for those detail views instead of relying on prompt dialogs for the main edit paths
- `workflow_create_writing_task` and `workflow_create_plan` were fixed to supply explicit `plan_id` values and mark created plans `active`, after a focused live validation exposed the broken call shape against the real `PlanStore`
- the user asked that live external-tool evals favor `kilocode` over `codex` to avoid burning paid tokens
- the first integrated operator live eval failed for a valid reason: `ToolLoopGuard.MAX_ROUNDS=16` was too tight for legitimate planning + browser + PTY work. That breaker is now widened to `24` while the identical-call breaker remains intact.

Concrete Kilo state:

- credentials are present locally via `~/.local/share/kilo/auth.json`
- models are available locally, including:
  - `kilo/kilo-auto/free`
  - `kimi-for-coding/k2p5`
- repo-local bootstrap config now exists at:
  - [opencode.json](/mnt/xtra/OpenCAS/.opencode/opencode.json)
- `kilo run --auto "Create /tmp/kilo_test_note.txt ..."` succeeds under that config and writes the file correctly
- `workflow_supervise_session` now stages new TUI sessions into `start -> observe readiness -> submit`
- with that staged start path, `kilocode_supervised_work` now succeeds in live validation

## 7. What Is Still Wrong Or Incomplete

### 7.1 PTY/TUI operator path is viable, but needs generalization

The PTY substrate is viable and cleaned output is now exposed. `kilocode` supervised work now succeeds in the live harness after staging startup and task submission separately. Standard non-AI PTY editing through `vim` also succeeds in the live harness.

This means:

- the agent can launch and read Kilo Code
- the agent can supervise bounded external work through Kilo Code and produce a verified artifact
- the agent can operate a standard full-screen Linux TUI (`vim`) and verify the resulting filesystem artifact
- the key fix was avoiding prompt injection during terminal startup

Latest direct diagnosis:

- when driven through PTY in a single `start+submit` interaction, the Kilo TUI receives the prompt text into the composer
- the prompt then remains in the composer instead of submitting/executing
- separating startup from task submission resolves the issue
- therefore the broader lesson is that full-screen TUIs may need readiness staging before input, not just raw send-key behavior

Next improvement:

- generalize staged readiness handling for other TUIs where startup races can strand input
- deepen `screen_state` heuristics and use them to drive explicit follow-up actions instead of only advisory output

This remains the highest-value operator-substrate gap, but it is now a generalization problem rather than a basic viability problem.

### 7.2 Dashboard/control plane is improved but not finished

The dashboard is no longer read-only observability. It now has a real Operations tab backed by `/api/operations/*`.

It now also exposes the latest qualification aggregate through `/api/operations/qualification`, plus an overview card and an operations-panel view that surface:

- total live validation runs
- direct and agent success rates
- average run duration
- weakest current agent-check labels
- the exact JSON snapshot path backing the view

The operations sessions view is also now scope-aware in the UI, not just in the API:

- `/api/operations/sessions` returns per-scope PTY/browser counts
- the dashboard lets operators filter to a specific scope and applies scoped cleanup actions against that chosen scope
- this closes a real control-plane gap where multi-scope execution existed in the runtime but not in the visible operator surface
- browser session detail is now also in-pane form-driven for navigate/click/type/press/wait, instead of relying on chained `window.prompt(...)` dialogs

Still needed:

- better multi-scope visibility and filtering
- richer browser screenshot/download ergonomics beyond the current inline capture/display path
- deeper receipt/history drill-down from the same control plane

### 7.3 Live operator evaluation still needs deeper tasks

Current live validation now proves:

- startup viability
- state inspection
- basic tool operation
- TUI startup probing for Claude, Codex, and Kilo Code
- real PTY editing work through standard `vim`
- one bounded supervised Codex artifact-producing task
- one bounded supervised Kilo artifact-producing task
- one bounded higher-level writing-workflow artifact-producing task
- one bounded higher-level project-management-workflow artifact-producing task
- one bounded higher-level writing revision-loop artifact-producing task
- one bounded integrated planning + browser + PTY artifact-producing task

Qualification aggregate snapshot:

- generator: [summarize_live_validations.py](/mnt/xtra/OpenCAS/scripts/summarize_live_validations.py)
- repeated-run orchestrator: [run_qualification_cycle.py](/mnt/xtra/OpenCAS/scripts/run_qualification_cycle.py)
- markdown snapshot: [live_validation_summary.md](/mnt/xtra/OpenCAS/docs/qualification/live_validation_summary.md)
- JSON snapshot: [live_validation_summary.json](/mnt/xtra/OpenCAS/docs/qualification/live_validation_summary.json)
- current aggregate: `23` runs, direct success `0.948`, agent success `0.966`, average duration `132.93s`
- current weakest labels in the aggregate reflect historical rather than current-state failures:
  - `kilocode_supervised_work` at `0.4` success due to earlier submission-race/timeout failures before staged TUI readiness landed
  - `integrated_operator_workflow` at `0.5` success due to the old `ToolLoopGuard.MAX_ROUNDS=16` breaker before it was widened to `24`
- repeated qualification now has a bounded runner that defaults to focused agent-only cycles, so requalification does not automatically re-run the full direct-probe matrix unless requested
- the operations dashboard also now exposes recent validation runs directly from `.opencas_live_test_state`, so operators can inspect the latest run-level evidence without opening report files by hand
- those recent validation runs now also have per-run detail drill-down in the Operations pane, including direct-check outcomes and failed agent-check responses
- the qualification aggregate now also emits explicit requalification suggestions, each with a ready-to-run `scripts/run_qualification_cycle.py` command for the weak label
- those requalification suggestions now include an operator note and the latest captured failure snippet, so the dashboard qualification panel acts as a lightweight remediation playbook rather than just a command list
- operators can now explicitly start a bounded qualification rerun from the dashboard, and that rerun is tracked as a background process under the `qualification` scope instead of being an invisible shell job

It does not yet prove:

- sustained real work through `kilocode`
- long multi-step project management work
- longer writing workflows with review feedback from separate sessions or tools
- cost/latency performance under long sessions

### 7.4 Higher-level operator workflows still need deeper evals

The runtime now has explicit workflow tools for commitments, plans, writing, repo triage, and session supervision.

Still needed:

- validate them in longer autonomous runs, not just tool-level tests
- confirm the agent consistently chooses them instead of falling back to low-level choreography
- add more durable project-management and writing evals on top of them

## 8. Exact Next Steps For Claude

Work in this order.

### Step 1: treat qualification as the main frontier

Reason:

- the operator substrate is now real and live-validated across PTY, browser, writing, planning, and integrated workflows
- the remaining risk is operational trustworthiness under repeated and longer runs, not primitive capability absence
- the qualification summary now gives a durable scoreboard to judge whether new work is actually improving the system

Success criteria:

- any new eval work updates the qualification snapshot
- do not regress the current aggregate without a deliberate reason captured in docs
- do not leave provider-backed validation runs alive after completion or failure

### Step 2: keep control-plane progress visible

Goal:

- avoid burying operational state in raw run directories
- keep exposing useful operator state through `/api/operations/*` and the dashboard

Acceptance criteria:

- operators can see the current qualification snapshot without reading markdown files directly
- further dashboard additions should prefer concrete control or evaluation value over cosmetic polish

### Step 3: deepen qualification with repeated integrated evals

Recommended order:

- rerun bounded integrated evals after local reproduction only
- add longer writing/project-management flows with revision/intervention
- measure failures, timeouts, and tool-message cost in the aggregate summary
- keep using `kilocode` over provider-costly alternatives where it gives comparable signal

### Step 4: continue control-plane hardening where it supports qualification

Priority order:

1. multi-scope visibility and filtering
2. project management workflow
3. repo triage / coding workflow
4. external tool supervision workflow

Target:

- verify the agent consistently chooses workflow tools when appropriate
- prove longer bounded work loops over the higher-level abstractions, not just low-level tool success

## 9. Working Conventions For Claude

### Git

Use git conservatively and intentionally.

- do not touch unrelated untracked files
- do not clean/reset the worktree
- commit coherent slices only
- keep validation tied to each commit

Current tracked worktree should be clean after the latest committed slice. If it is not, inspect `opencas/api/routes/operations.py`, `opencas/dashboard/static/index.html`, and `opencas/execution/process_supervisor.py` first; those are the active qualification/control-plane files.

Current unrelated untracked artifacts include:

- `.omc/`
- `.opencas_live_test_state/`
- `.research/`
- `2026-04-06-134754-this-session-is-being-continued-from-a-previous-c.txt`
- `AUDIT_OPENCAS_VS_BULMA.md`
- `test_script.sh`

Leave them alone unless the user explicitly says otherwise.

### Validation expectations

For infrastructure changes:

- run targeted tests first
- then run the full suite if the change is broad

For operator-substrate changes:

- run the live harness
- inspect the written report, not just process exit
- preserve the report path in docs/handoff context
- do not leave live validation processes running in background shells
- prefer one bounded live run after local reproduction instead of repeated overlapping runs

### Philosophy

Do not simplify the project into a narrower assistant unless there is a measured reason.

The correct continuation is:

- preserve architecture
- improve operational truthfulness
- improve real work capability
- improve inspectability
- improve cost/control behavior

## 10. How To Hand Back To Codex Later

If Claude does substantial work and the user later hands the repo back to Codex, the return handoff should include:

- commit list with intent per commit
- exact tests run and outcomes
- latest live validation report path
- whether PTY normalized screen text was implemented
- whether a real `claude` or `codex` work session was successfully supervised
- which docs were updated
- any remaining blockers

The easiest way to preserve continuity is:

1. update this file or create a new dated handoff in `docs/`
2. include exact report paths under `.opencas_live_test_state/`
3. include exact commit ids
4. include unresolved risks, not just completed work

## 11. Immediate Resume Command Set

For Claude, this is the minimal resume sequence:

```bash
cd /mnt/xtra/OpenCAS
git status --short
git log --oneline -12
source .venv/bin/activate && pytest -q
```

Then:

- inspect `/api/operations/qualification`, `/api/operations/validation-runs`, and `/api/operations/sessions` behavior first
- prefer bounded qualification reruns through `scripts/run_qualification_cycle.py` or the dashboard rerun control rather than ad hoc live harness shells
- verify no stale `run_live_debug_validation.py`, `kilocode`, or `kilo run` processes remain after any live test
- continue qualification/control-plane work before adding new primitive tools

## 12. Latest Qualification Control-Plane Slice

Recent completed slices extended the operator surface around qualification:

- qualification aggregate in the dashboard
- recent validation run summaries
- per-run detail drill-down
- remediation playbook hints and rerun recommendations
- explicit bounded rerun launch via `ProcessSupervisor`
- background process session visibility and kill support
- process output previews and scoped process cleanup
- direct navigation from process detail back to qualification and recent runs
- active qualification reruns rendered directly inside the qualification panel
- recent validation runs rendered directly inside the qualification panel
- process detail now reflects refreshed process state and supports removing completed jobs directly
- qualification panels now auto-refresh on a bounded 5-second cadence while reruns are active
- qualification recommendations can now jump directly to recent runs filtered by the weak label
- recent validation run lists now surface compact failure signals like `aborted` and failed labels
- filtered validation run detail now preserves the focus label and highlights matching agent checks
- qualification reruns now carry provenance metadata such as source label and operator note
- qualification recommendations now show a simple latest-vs-previous trend for the weak label
- recommendation rows now also summarize latest/previous outcomes and whether a rerun for that label is already active
- recommendation rows now also show the latest completed matching run when no rerun is active, so completion is visible inline instead of only through recent-run drill-down
- recommendation rows now expose a direct `Latest Result` jump into focused run detail when a completed matching run exists
- qualification rerun launches are now appended to `.opencas_live_test_state/qualification_rerun_history.jsonl`, and recommendation rows surface the latest request time/process per label so operator intent survives process completion
- qualification reruns now also carry a generated request ID into `scripts/run_qualification_cycle.py`, which appends a matching completion event with return code and generated run IDs back into the same JSONL history; recommendation rows can therefore show which request finished, not just a loose recent-run match
- the qualification summary now also exposes a bounded recent-rerun history list, and the dashboard renders it directly so operators can inspect recent `requested`/`completed` events with request IDs, process IDs, return codes, and generated run IDs without leaving the qualification panel
- recent rerun history rows now expose direct actions into the latest result or the label-filtered run list when that context exists
- process detail now resolves rerun request/completion provenance by `request_id`, and validation-run detail resolves matching rerun provenance by generated run ID plus focused label, so rerun lineage is visible in the detail views as well as in the qualification panel
- recent rerun history completion rows now also carry weak-label comparison/trend data when they map cleanly to one label, so the history panel itself can show whether the rerun improved or regressed that check
- validation-run detail and process detail now also expose direct drill-down actions into the latest result and label-filtered run list when rerun provenance is present
- qualification recommendations and rerun-history completion rows now also carry a small rolling per-label success-rate window, including recent rate, previous rate when available, and delta in points, so aggregate movement is visible alongside symbolic trend text
- the operations API now also exposes a per-label qualification detail view, and the dashboard can drill into it from weakest checks, recommendations, and rerun-history rows to inspect one weak label’s stats, delta cues, recent runs, rerun history, active reruns, and recommendation in one place
- the operations API now also exposes a rerun-detail view keyed by `request_id`, and the dashboard can drill into it from rerun-history rows and process detail to inspect one rerun request’s launch metadata, completion metadata, active process state, and latest produced validation run in one place
- that rerun-detail view is now request-centric rather than latest-run-centric: it also shows all produced validation runs for the request and per-label outcome summaries with trend/rate-window context, so multi-iteration reruns can be reviewed from one place
- request-centric rerun detail now also computes first-vs-latest per-label progress inside the request itself, so operators can tell whether a rerun actually improved over its own iterations rather than only looking at global trend text
- there is now a repo-local stale-process sweep tool at `scripts/sweep_operator_processes.py`, plus a dedicated qualification execution plan in `docs/qualification/testing-execution-plan-2026-04-09.md`, to keep long-running testing bounded and prevent leftover provider-backed jobs from burning usage
- direct CLI launches of `scripts/run_qualification_cycle.py` now also auto-generate rerun provenance (`request_id` plus `requested` event), so testing-plan reruns from the shell appear in the same request-detail/provenance flow as API-launched reruns
- `scripts/run_live_debug_validation.py` now accepts forwarded provenance flags from the qualification runner, so the manual CLI rerun path no longer breaks on `--request-id` / `--rerun-history-path`
- there is now a remediation rollup generator at `scripts/summarize_qualification_remediation.py`, producing `docs/qualification/qualification_remediation_rollup.{json,md}` and surfacing recent reruns as `continue_testing`, `investigate_runner`, or `code_change_justified`
- the operations API now records durable operator-action history for PTY/browser/process intervention paths under each runtime `state_dir`, and PTY/browser/process detail views surface recent operator actions directly
- there is now a local scenario runner at `scripts/run_scenario3_operator_recovery.py` that validates operator intervention and recovery through the operations control plane without burning provider calls; latest report: `.opencas_live_test_state/scenario3-operator-recovery-20260409-170416/scenario3_operator_recovery_report.md`
- there is now a local scenario runner at `scripts/run_scenario2_repo_triage.py` that validates the repo-triage-to-note path through the real workflow adapter against the current OpenCAS repo; latest report: `.opencas_live_test_state/scenario2-repo-triage-20260409-171519/scenario2_repo_triage_report.md`

If resuming after this point, the highest-value next work is:

1. qualification-job completion/refresh ergonomics in the dashboard
2. broader repeated bounded qualification cycles using the existing runner
3. deeper operator history/reason tracing, not more raw tool adapters
