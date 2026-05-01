# OpenCAS Task List

## In Progress

- `TASK-901` Build/test unknown fragment
  - owner: Codex
  - result:
    - missing status should stay quiet when the reducer cannot classify it

- `TASK-903` Build/test conflicting fragment
  - owner: Codex
  - status: in progress
  - result:
    - active fragment should lose to the completed duplicate

- `TASK-905` Build/test active fragment
  - owner: Codex
  - status: in progress
  - result:
    - explicit active fragment stays in now

## Background Context

- `TASK-902` Build/test stale fragment
  - owner: Codex
  - status: in progress
  - result:
    - stale from a previous run should not stay active

- `TASK-903` Build/test conflicting fragment
  - owner: Codex
  - status: completed
  - result:
    - completed duplicate should make the conflict conservative

## Recently Completed

- `TASK-904` Build/test archived fragment
  - owner: Codex
  - status: completed
  - result:
    - archived fragment should stay in later
