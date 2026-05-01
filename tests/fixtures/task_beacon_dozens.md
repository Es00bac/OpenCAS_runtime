# OpenCAS Task List

## In Progress

- `TASK-101` Build/test now active alpha
  - owner: Codex
  - status: in progress
  - result:
    - active alpha should stay in now

- `TASK-101` Build/test now active alpha
  - owner: Codex
  - status: planning
  - result:
    - duplicate planning note should not change the bucket

- `TASK-102` Build/test now active beta
  - owner: Codex
  - status: verifying
  - result:
    - verifying beta should stay in now

- `TASK-102` Build/test now active beta
  - owner: Codex
  - status: in progress
  - result:
    - later duplicate should not outrank the active one

- `TASK-103` Build/test now active gamma
  - owner: Codex
  - status: running
  - result:
    - running gamma should stay in now

- `TASK-103` Build/test now active gamma
  - owner: Codex
  - status: in progress
  - result:
    - duplicate active gamma should not change the bucket

## Next Up / Backlog

- `TASK-201` Build/test next blocked alpha
  - owner: Codex
  - status: blocked
  - result:
    - blocked by the flaky pytest shard

- `TASK-201` Build/test next blocked alpha
  - owner: Codex
  - status: pending
  - result:
    - newer duplicate should not outrank the blocked one

- `TASK-202` Build/test next pending beta
  - owner: Codex
  - status: pending
  - result:
    - the last build result is still open

- `TASK-202` Build/test next pending beta
  - owner: Codex
  - status: in progress
  - result:
    - live duplicate should still collapse into next

- `TASK-203` Build/test next waiting gamma
  - owner: Codex
  - status: pending
  - result:
    - the last integration check is still open

- `TASK-203` Build/test next waiting gamma
  - owner: Codex
  - status: pending
  - result:
    - second duplicate should stay in next

## Next Up / Backlog

- `TASK-204` Build/test next pending delta
  - owner: Codex
  - status: pending
  - result:
    - queued follow-up should stay in next

- `TASK-204` Build/test next pending delta
  - owner: Codex
  - status: pending
  - result:
    - pending follow-up should still stay in next

- `TASK-205` Build/test next blocked epsilon
  - owner: Codex
  - status: pending
  - result:
    - pending epsilon should not outrank the older blocked alpha

- `TASK-205` Build/test next blocked epsilon
  - owner: Codex
  - status: pending
  - result:
    - newer pending duplicate should not outrank the older blocked one

## Recently Completed

- `TASK-301` Build/test later completed alpha
  - owner: Codex
  - status: completed
  - result:
    - completed fragment should stay in later

- `TASK-301` Build/test later completed alpha
  - owner: Codex
  - result:
    - historical duplicate should not change the bucket

- `TASK-302` Build/test later completed beta
  - owner: Codex
  - status: queued
  - result:
    - queued historical follow-up should stay later

- `TASK-302` Build/test later completed beta
  - owner: Codex
  - status: completed
  - result:
    - duplicate completion should remain later

## Archived Completions

- `TASK-303` Build/test later archived gamma
  - owner: Codex
  - status: completed
  - result:
    - archived completion should remain later

- `TASK-303` Build/test later archived gamma
  - owner: Codex
  - result:
    - archived duplicate should stay quiet
