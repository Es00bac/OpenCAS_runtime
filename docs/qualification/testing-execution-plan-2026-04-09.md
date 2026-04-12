# OpenCAS Qualification Testing Execution Plan

This plan is designed to be followed by a lighter model such as GPT 5.4-mini without needing broad project context reconstruction on every run.

## Goals

- keep qualification work bounded and repeatable
- prevent stale background processes from burning provider usage
- distinguish testing/evaluation work from coding/remediation work
- produce enough evidence to decide the next coding change with confidence

## Operating Split

- `GPT 5.4-mini`: testing, qualification reruns, triage, dashboard inspection, report updates
- `GPT 5.4 High`: code changes, design changes, nontrivial debugging, cross-cutting refactors

## Important Constraint

Codex model selection is controlled by the chat/runtime, not by OpenCAS repository code. The repo can standardize commands and testing flow, but it cannot switch this Codex session between 5.4-mini and 5.4 High automatically from inside the workspace.

What can be automated safely:

- bounded test commands
- qualification rerun commands
- process sweeps before and after runs
- result summarization

What still requires the human/operator:

- deciding which Codex model to use for the current session

## Pre-Run Safety Sweep

Run this before any live or repeated qualification work:

```bash
cd (workspace_root)
source .venv/bin/activate
python scripts/sweep_operator_processes.py --json
```

If stale provider-backed jobs are present, terminate them:

```bash
python scripts/sweep_operator_processes.py --kill
```

If local-only leftovers also need cleanup, use:

```bash
python scripts/sweep_operator_processes.py --include-local-test-tools --kill
```

Default sweep scope:

- `run_live_debug_validation.py`
- `run_qualification_cycle.py`
- `kilocode`
- `kilo run`

Optional local-only scope:

- `pytest`
- `mpv`
- `edge-tts`

## Phase 1: Baseline Health Check

Run these before changing code:

```bash
timeout 180s .venv/bin/python -m pytest tests/test_operations_routes.py -q
timeout 120s .venv/bin/python -m pytest tests/test_dashboard_api.py -q
timeout 120s .venv/bin/python -m pytest tests/test_sweep_operator_processes.py -q
```

If any of these fail, stop qualification work and hand off to GPT 5.4 High for code repair.

## Phase 2: Qualification Inspection

Inspect current state without launching new provider-backed runs:

1. open the dashboard and inspect:
   - `/api/operations/qualification`
   - `/api/operations/validation-runs?limit=10`
   - weak-label detail views
   - rerun request detail views
2. identify one weak label at a time
3. inspect:
   - latest matching run
   - rerun history for that label
   - request-centric rerun detail if a recent rerun exists

## Phase 3: Bounded Rerun Execution

For a single weak label:

```bash
python scripts/run_qualification_cycle.py \
  --agent-check-label <label> \
  --iterations 2 \
  --prompt-timeout-seconds 180 \
  --run-timeout-seconds 420
```

Rules:

- do not launch multiple provider-backed reruns in parallel
- do not run broad qualification when one weak label is enough to reproduce the problem
- prefer `--agent-check-label` focused reruns over wide reruns

After the rerun:

```bash
python scripts/summarize_live_validations.py \
  --runs-dir .opencas_live_test_state \
  --output-dir docs/qualification
```

Direct CLI runs of `scripts/run_qualification_cycle.py` now auto-record rerun provenance into `.opencas_live_test_state/qualification_rerun_history.jsonl`, even when no API request launched them.

## Phase 4: Post-Run Cleanup

Always sweep immediately after a rerun or live validation:

```bash
python scripts/sweep_operator_processes.py --json
```

If anything stale remains:

```bash
python scripts/sweep_operator_processes.py --kill
```

Use local-tool cleanup only when the run is fully complete and you do not need the current local process anymore:

```bash
python scripts/sweep_operator_processes.py --include-local-test-tools --kill
```

## Phase 5: Decision Rule

Hand off to GPT 5.4 High for coding only if at least one of these is true:

- the same weak label fails in repeated bounded reruns
- the rerun request detail shows no improvement across generated runs
- the operator surface is missing the inspection/control affordance needed to understand the failure
- the failure is reproducible and local, not a one-off provider issue

Stay on GPT 5.4-mini when:

- inspecting dashboards
- reviewing qualification summaries
- running focused reruns
- updating test notes or reports
- verifying process cleanup

## Token / Usage Discipline

- never leave provider-backed runs unattended
- no overlapping `run_live_debug_validation.py` or `run_qualification_cycle.py` jobs
- one focused rerun is better than one broad rerun
- prefer inspection first, rerun second, code change third
- if Kimi usage rises unexpectedly, run the sweep script before any further action

## Suggested Loop

1. pre-run sweep
2. focused local tests
3. inspect weak label
4. run one bounded rerun
5. sweep again
6. inspect request detail and label detail
7. decide:
   - continue testing on mini
   - or hand off to High for code changes

## Deliverables After Each Testing Session

- current weak label being investigated
- whether the latest rerun improved, regressed, or stayed flat
- whether stale processes were found and killed
- whether a code change is justified
- exact next recommended action
