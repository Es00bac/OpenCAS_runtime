# OpenCAS Task List

## In Progress

- `TASK-301` Build/test active fragment
  - owner: Codex
  - status: in progress
  - result:
    - active fragments land in now

- `TASK-302` Build/test empty-state duplicate
  - owner: Codex
  - result:
    - empty-state duplicate should still collapse deterministically

## Next Up / Backlog

- `TASK-302` Build/test empty-state duplicate
  - owner: Codex
  - status: blocked
  - result:
    - blocked duplicate should win the merge

- `TASK-303` Build/test blocked fragment
  - owner: Codex
  - status: blocked
  - result:
    - blocked fragment should land in next

## Next Up / Backlog

- `TASK-304` Build/test completed fragment
  - owner: Codex
  - status: completed
  - result:
    - completed fragment should land in later

## Recently Completed

- `TASK-304` Build/test completed fragment
  - owner: Codex
  - result:
    - historical empty-state duplicate should not change the bucket

- `TASK-305` Build/test historical empty-state fragment
  - owner: Codex
  - result:
    - empty-state historical fragment should stay in later
