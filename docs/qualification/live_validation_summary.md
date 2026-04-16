# OpenCAS Live Validation Qualification Summary

- Scope: `current retained run folders`
- Runs analyzed: `6`
- Summary scope id: `retained_runs_dir_snapshot`
- Direct success rate: `-`
- Agent success rate: `0.667`
- Average run duration (s): `63.53`
- Models: `kimi-coding/k2p5`
- Embedding models: `google/gemini-embedding-2-preview`
- Historical note: this file reflects only the run folders currently retained under `.opencas_live_test_state`; use `qualification_remediation_rollup.md` and readiness/task docs for rerun-history decisions.

## Recent Runs

- `debug-validation-20260415-163002` direct `0/0` agent `1/1` duration `43.838389`s model `kimi-coding/k2p5`
- `debug-validation-20260415-162921` direct `0/0` agent `1/1` duration `40.078271`s model `kimi-coding/k2p5`
- `debug-validation-20260415-162602` direct `0/0` agent `0/1` duration `42.483176`s model `kimi-coding/k2p5`
- `debug-validation-20260415-162521` direct `0/0` agent `0/1` duration `40.236789`s model `kimi-coding/k2p5`
- `debug-validation-20260415-155925` direct `0/0` agent `1/1` duration `38.286999`s model `kimi-coding/k2p5`
- `debug-validation-20260415-155629` direct `0/0` agent `1/1` duration `176.257327`s model `kimi-coding/k2p5`

## Agent Checks

### integrated_operator_workflow

- Runs: `2`
- Successes: `2`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `13.5`
- Outcomes: `{"artifact_verified": 2}`

### kilocode_supervised_work

- Runs: `4`
- Successes: `2`
- Failures: `2`
- Success rate: `0.5`
- Timeouts: `0`
- Average tool messages: `3.75`
- Outcomes: `{"artifact_missing": 2, "artifact_verified": 2}`
- Recent failures:
  - `debug-validation-20260415-162602` outcome `artifact_missing`: Result: failure.

- `kilocode` launched in its TUI and reached an interactive ready state.
- The supervised session ran through 6 observation rounds without the target file being created.
- The file `./.opencas_live_test_sta
  - `debug-validation-20260415-162521` outcome `artifact_missing`: The supervised session did not produce the file. Kilo's TUI started and was ready for input, but after 5 rounds it timed out without creating `./.opencas_live_test_state/debug-validation-20260415-162521/workspace_artifacts/n
