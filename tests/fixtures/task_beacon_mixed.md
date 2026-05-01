# OpenCAS Task List

## In Progress

- `TASK-100` Build gate repair
  - owner: Codex
  - status: in progress
  - result:
    - keep the build gate green and avoid noise

- `TASK-101` Build/test gate with blocker
  - owner: Codex
  - status: in progress
  - result:
    - blocked by the flaky pytest shard

- `TASK-107` Build pipeline restore
  - owner: Codex
  - status: in progress
  - result:
    - keep the build pipeline moving

- `TASK-109` Build/test follow-through
  - owner: Codex
  - status: in progress
  - result:
    - first pass on the regression harness

- `TASK-113` Build/test reducer cleanup
  - owner: Codex
  - status: in progress
  - result:
    - latest revision should surface first in the now bucket

## Next Up / Backlog

- `TASK-102` Build/test cleanup
  - owner: Codex
  - status: pending
  - result:
    - earlier note that should be superseded by a later fragment

- `TASK-103` Build/test staging pass
  - owner: Codex
  - status: queued
  - result:
    - waiting on the nightly build to finish

- `TASK-108` Build/test blocker follow-up
  - owner: Codex
  - status: blocked
  - result:
    - blocked on the final validation report

- `TASK-109` Build/test follow-through
  - owner: Codex
  - status: in progress
  - result:
    - second pass should win because it is more recent

- `TASK-110` Build/test summary cleanup
  - owner: Codex
  - status: pending
  - result:
    - first pass through the compact summary reducer

- `TASK-110` Build/test summary cleanup
  - owner: Codex
  - status: pending
  - result:
    - second pass should win because it is more recent

## Next Up / Backlog

- `TASK-102` Build/test cleanup
  - owner: Codex
  - status: pending
  - result:
    - later fragment should win because it is more recent

- `TASK-104` Build/test validation sweep
  - owner: Codex
  - status: pending
  - result:
    - validation sweep for the reducer output

- `TASK-112` Build/test summary polish
  - owner: Codex
  - status: pending
  - result:
    - keep the beacon summary compact and readable

## Recently Completed

- `TASK-105` Build/test result archive
  - owner: Codex
  - status: completed
  - result:
    - verified the quiet output contract

- `TASK-106` Build/test archive sweep
  - owner: Codex
  - result:
    - archived build/test fragment should stay in later

- `TASK-111` Build/test failed run
  - owner: Codex
  - status: failed
  - result:
    - failed validation should land in later

- `TASK-114` Build/test completed follow-up
  - owner: Codex
  - status: completed
  - result:
    - newer completed fragment should stay in later
