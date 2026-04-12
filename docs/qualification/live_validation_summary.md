# OpenCAS Live Validation Qualification Summary

- Runs analyzed: `27`
- Direct success rate: `0.948`
- Agent success rate: `0.967`
- Average run duration (s): `120.45`
- Models: `google/gemini-2.5-flash, kimi-coding/k2p5`
- Embedding models: `google/gemini-embedding-2-preview, local-fallback`

## Recent Runs

- `debug-validation-20260409-183455` direct `0/0` agent `1/1` duration `8.71653`s model `google/gemini-2.5-flash`
- `debug-validation-20260409-164343` direct `0/0` agent `1/1` duration `42.267226`s model `kimi-coding/k2p5`
- `debug-validation-20260409-160102` direct `0/0` agent `1/1` duration `79.645013`s model `kimi-coding/k2p5`
- `debug-validation-20260409-155243` direct `0/0` agent `1/1` duration `64.002072`s model `kimi-coding/k2p5`
- `debug-validation-20260409-003117` direct `0/0` agent `1/1` duration `112.805211`s model `kimi-coding/k2p5`
- `debug-validation-20260409-002851` direct `0/0` agent `0/1` duration `84.618599`s model `kimi-coding/k2p5`
- `debug-validation-20260409-000125` direct `0/0` agent `1/1` duration `51.504513`s model `kimi-coding/k2p5`
- `debug-validation-20260408-235606` direct `0/0` agent `1/1` duration `24.099278`s model `kimi-coding/k2p5`
- `debug-validation-20260408-234917` direct `0/0` agent `1/1` duration `17.863718`s model `kimi-coding/k2p5`
- `debug-validation-20260408-234755` direct `0/0` agent `1/1` duration `14.834771`s model `kimi-coding/k2p5`
- `debug-validation-20260408-232901` direct `0/0` agent `1/1` duration `56.475617`s model `kimi-coding/k2p5`
- `debug-validation-20260408-232647` direct `0/0` agent `1/1` duration `63.25954`s model `kimi-coding/k2p5`
- `debug-validation-20260408-231052` direct `9/9` agent `8/8` duration `240.773336`s model `kimi-coding/k2p5`
- `debug-validation-20260408-223339` direct `9/9` agent `7/8` duration `277.843713`s model `kimi-coding/k2p5`
- `debug-validation-20260408-222443` direct `9/9` agent `7/8` duration `280.966516`s model `kimi-coding/k2p5`
- `debug-validation-20260408-221926` direct `9/9` agent `8/8` duration `213.052947`s model `kimi-coding/k2p5`
- `debug-validation-20260408-220155` direct `9/9` agent `7/8` duration `297.091735`s model `kimi-coding/k2p5`
- `debug-validation-20260408-214117` direct `9/9` agent `8/8` duration `356.496963`s model `kimi-coding/k2p5`
- `debug-validation-20260408-213733` direct `9/9` agent `7/7` duration `102.612056`s model `kimi-coding/k2p5`
- `debug-validation-20260408-213322` direct `9/9` agent `7/7` duration `115.087227`s model `kimi-coding/k2p5`
- `debug-validation-20260408-204416` direct `9/9` agent `7/7` duration `106.614568`s model `kimi-coding/k2p5`
- `debug-validation-20260408-201845` direct `8/9` agent `7/7` duration `131.22635`s model `kimi-coding/k2p5`
- `debug-validation-20260408-201426` direct `9/9` agent `7/7` duration `117.712518`s model `kimi-coding/k2p5`
- `debug-validation-20260408-201011` direct `9/9` agent `7/7` duration `128.605225`s model `kimi-coding/k2p5`
- `debug-validation-20260408-200216` direct `8/9` agent `7/7` duration `168.469206`s model `kimi-coding/k2p5`
- `debug-validation-20260408-195802` direct `8/9` agent `7/7` duration `69.59803`s model `kimi-coding/k2p5`
- `debug-validation-20260408-195359` direct `5/9` agent `7/7` duration `25.790018`s model `kimi-coding/k2p5`

## Agent Checks

### browser_probe

- Runs: `15`
- Successes: `15`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `3.47`
- Outcomes: `{"completed": 15}`

### claude_tui_probe

- Runs: `15`
- Successes: `15`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `2.6`
- Outcomes: `{"completed": 15}`

### codex_supervised_work

- Runs: `1`
- Successes: `1`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `9.0`
- Outcomes: `{"artifact_verified": 1}`

### codex_tui_probe

- Runs: `10`
- Successes: `10`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `3.9`
- Outcomes: `{"completed": 10}`

### inspect_runtime

- Runs: `15`
- Successes: `15`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `1.73`
- Outcomes: `{"completed": 15}`

### integrated_operator_workflow

- Runs: `4`
- Successes: `3`
- Failures: `1`
- Success rate: `0.75`
- Timeouts: `0`
- Average tool messages: `15.5`
- Outcomes: `{"artifact_missing": 1, "artifact_verified": 3}`
- Recent failures:
  - `debug-validation-20260409-002851` outcome `artifact_missing`: [Tool loop halted] Tool loop circuit breaker: exceeded 16 consecutive tool calls in this session.

### kilocode_supervised_work

- Runs: `6`
- Successes: `3`
- Failures: `3`
- Success rate: `0.5`
- Timeouts: `2`
- Average tool messages: `6.33`
- Outcomes: `{"artifact_missing": 1, "artifact_verified": 3, "timed_out": 2}`
- Recent failures:
  - `debug-validation-20260408-223339` outcome `artifact_missing`: Failure.

- `workflow_supervise_session` timed out after 5 rounds without creating the file.
- I then started a manual `pty_interact` session with `kilocode`. The TUI rendered and accepted input, but after submitting the prompt and waiting,
  - `debug-validation-20260408-222443` outcome `timed_out`: [Timed out after 180.0s while executing this validation prompt. Inspect telemetry/context for partial progress.]
  - `debug-validation-20260408-220155` outcome `timed_out`: [Timed out after 120.0s while executing this validation prompt. Inspect telemetry/context for partial progress.]

### kilocode_tui_probe

- Runs: `5`
- Successes: `5`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `1.0`
- Outcomes: `{"completed": 5}`

### project_management_workflow

- Runs: `1`
- Successes: `1`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `4.0`
- Outcomes: `{"artifact_verified": 1}`

### role_priming

- Runs: `16`
- Successes: `16`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `0.94`
- Outcomes: `{"completed": 16}`

### self_reflection

- Runs: `15`
- Successes: `15`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `0.0`
- Outcomes: `{"completed": 15}`

### vim_tui_edit

- Runs: `2`
- Successes: `2`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `10.5`
- Outcomes: `{"artifact_verified": 2}`

### write_project_note

- Runs: `15`
- Successes: `15`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `2.13`
- Outcomes: `{"artifact_verified": 6, "completed": 9}`

### writing_revision_workflow

- Runs: `1`
- Successes: `1`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `5.0`
- Outcomes: `{"artifact_verified": 1}`

### writing_workflow

- Runs: `2`
- Successes: `2`
- Failures: `0`
- Success rate: `1.0`
- Timeouts: `0`
- Average tool messages: `3.0`
- Outcomes: `{"artifact_verified": 2}`
