# OpenCAS Production Program Plan

Date: 2026-04-08

Related:
- [Production Readiness Audit](opencas-production-readiness-audit-2026-04-08.md)
- [Deep System Audit](opencas-deep-system-audit-2026-04-09.md)
- [Product Spec](../OPENCAS_PRODUCT_SPEC.md)

## Purpose

This document turns the audit into a single integrated program plan.

It is meant to cover:

- hardening the current system
- filling major capability gaps
- integrating all subsystems coherently
- bootstrapping the first production-capable agent profile
- defining milestones, dependencies, acceptance criteria, and rollout order

This is a planning document, not an implementation spec for a single sprint.

## Current Status Update

As of 2026-04-09, this program plan remains the long-horizon structure, but the active execution state is now tracked in:

- [TaskList.md](../TaskList.md)
- [production-readiness-status-2026-04-09.md](production-readiness-status-2026-04-09.md)
- [testing-execution-plan-2026-04-09.md](qualification/testing-execution-plan-2026-04-09.md)

Current milestone reality:

- Milestone 1 through Milestone 7 are materially implemented at substrate level
- Milestone 8 is partially implemented through workflow tools, operator substrate, and control-plane work
- Milestone 9, qualification and deployment readiness, is now the dominant frontier

The 2026-04-09 deep audit refined this further:

- the remaining problem is not mainly missing primitive tools
- the remaining problem is incomplete fusion between:
  - Bulma-style inner-life systems
  - claw-code / OpenClaw-style operator systems
- the plan therefore needs one additional explicit emphasis:
  - inner-life operationalization driven by qualification evidence

The current execution priority is not broad feature expansion. It is:

1. repeated bounded qualification on weak labels
2. longer integrated day-to-day scenarios
3. inner-life operationalization where the audit shows under-coupling
4. remediation-guided code changes only when justified by evidence
5. deployment-readiness checklist definition for first regular-use testing

## Program Outcome

The target outcome is not "a nicer prototype." The target outcome is:

OpenCAS becomes a production-capable, high-trust, local-first general technical operator that can do real work across coding, writing, project management, shell and CLI operations, browser use, and terminal-native/TUI workflows, while preserving continuity, memory, judgment, and economic operation over time.

## Non-Goals

This plan does not aim to:

- reduce OpenCAS to a basic coding assistant
- optimize for demo aesthetics over operational correctness
- expand cognition-themed subsystems without proving value
- support every imaginable app before core operator workflows are reliable

## Program Principles

### 1. Hardening before expansion

The current system is broad enough. The immediate program priority is correctness, safety, control, and evaluation.

### 2. Keep the ambitious architecture

Memory, relational continuity, somatic modulation, project growth, and long-horizon work are part of the product thesis. They should be retained, but forced to justify themselves in measured outcomes.

### 3. Build the operator substrate explicitly

Browser-use and PTY/TUI interaction are not small add-ons. They are first-class capability layers and must be designed as such.

### 4. Modes over undifferentiated behavior

The same substrate should support multiple work modes, but not every subsystem should have equal weight in every task.

### 5. Every subsystem must have observable value

A subsystem that cannot improve task completion, continuity, cost, safety, or operator trust should be simplified or demoted.

## Program Structure

The work should be executed as nine milestones:

1. Foundation and governance
2. Correctness and safety hardening
3. Workspace and coding-agent bootstrap
4. Memory and retrieval operationalization
5. Control plane and observability
6. Browser operator capability
7. PTY/TUI/editor operator capability
8. General work modes and first production agent profile
9. Qualification, soak testing, and production rollout

Each milestone below includes:

- purpose
- primary work
- dependencies
- key integration points
- acceptance criteria

## Milestone 0: Foundation And Program Governance

### Purpose

Create the execution model for the production program so the later work does not fragment.

### Primary work

- define the canonical production target:
  - "general technical operator"
- define canonical environments:
  - local repo workspace
  - shell environment
  - browser session
  - PTY terminal session
  - dashboard/operator session
- define success metrics:
  - task completion rate
  - intervention rate
  - wall-clock latency
  - token cost
  - external API cost
  - retry count
  - unsafe-action rate
  - operator trust score
- define release tracks:
  - `research`
  - `staging`
  - `production`
- define subsystem status labels:
  - `experimental`
  - `operational`
  - `qualified`

### Dependencies

- none

### Key integration points

- all milestones must use the same success metrics and environment model

### Acceptance criteria

- a written program charter exists
- every major subsystem has an owner/status
- production qualification metrics are defined before implementation work accelerates

## Milestone 1: Correctness And Safety Hardening

### Purpose

Remove correctness bugs and close the largest trust gaps in the current system.

### Primary work

#### Retrieval and embedding correctness

- fix embedding cache keying to include:
  - embedding model id
  - embedding task type
  - source hash
- preserve actual backend similarity scores from ANN backends
- make retrieval fusion consume real scores rather than synthetic rank-based approximations
- add tests covering:
  - same text embedded under multiple models
  - same text embedded under multiple task types
  - ANN score consistency

#### Bootstrap/runtime hardening

- fix async bootstrap/runtime guard behavior for ANN backends
- verify all startup paths operate correctly under the actual event-loop model

#### Policy and safety hardening

- move self-approval boundaries from exact-string matching to structured policy evaluation
- normalize action risk policy around:
  - tool family
  - command permission class
  - resource impact
  - destructive potential
  - ambiguity
- replace brittle shell blocking with structured command parsing and auditable safety decisions
- reduce or eliminate reliance on `shell=True` where possible

#### API/dashboard hardening

- remove state mutation from GET endpoints
- remove provider-cost side effects from observability endpoints
- fix route/runtime object mismatches hidden by tests

### Dependencies

- Milestone 0

### Key integration points

- memory/retrieval
- self-approval
- tool registry
- dashboard/API
- bootstrap pipeline

### Acceptance criteria

- no GET endpoint performs embedding or mutating work
- mixed-model embedding retrieval tests pass
- safety decisions are explainable and structured
- bootstrap succeeds cleanly under production async runtime
- dashboard routes operate correctly against live objects, not only fakes

## Milestone 2: Workspace And Coding-Agent Bootstrap

### Purpose

Make “coding in a real repo” a first-class operating mode instead of an incidental behavior.

### Primary work

#### Workspace attachment model

- define a canonical workspace object with:
  - repo root
  - working branch
  - build command
  - test command
  - lint/format commands
  - package manager/toolchain hints
  - allowed roots
  - secret handling policy
  - network policy
  - mutation policy

#### Coding bootstrap flow

- add a coding-mode bootstrap sequence that verifies:
  - workspace is attached
  - VCS is readable
  - toolchain is available
  - commands are known or discoverable
  - write roots are correct
  - receipts/checkpoints are enabled

#### Coding task model

- define a coding task lifecycle with:
  - objective
  - constraints
  - repository context
  - success criteria
  - evidence
  - patch set
  - verification result
  - recovery path

### Dependencies

- Milestone 1

### Key integration points

- bootstrap pipeline
- task/work stores
- execution receipts
- shell/process tools
- dashboard workspace controls

### Acceptance criteria

- a workspace can be attached explicitly and inspected
- a coding task can run end-to-end with receipts and verifiable outputs
- the runtime never ambiguously operates outside the intended repo roots

## Milestone 3: Memory And Retrieval Operationalization

### Purpose

Turn the memory system from an interesting architecture into a measured production subsystem.

### Primary work

#### Retrieval architecture

- adopt ANN-first retrieval for real workloads
- keep brute-force paths only for constrained fallback or testing
- define retrieval tiers:
  - immediate conversational memory
  - project memory
  - durable autobiographical memory
  - archive

#### Embedding strategy

- introduce task-type-aware embeddings for:
  - conversation
  - coding/workspace search
  - project/task summaries
  - writing/documents
  - identity/relational traces
- define model and cost policy per embedding type

#### Memory budgets and promotion rules

- define budget and promotion policies for:
  - what gets embedded
  - what gets summarized
  - what gets consolidated
  - what gets demoted to archive

#### Evaluation

- build retrieval evaluation sets:
  - coding tasks
  - user continuity tasks
  - writing/project tasks
- measure:
  - relevance
  - latency
  - token impact
  - cost impact

### Dependencies

- Milestone 1
- partial dependency on Milestone 2 for coding-task evaluation sets

### Key integration points

- context builder
- memory retriever
- consolidation
- somatic/relational/identity memory weighting

### Acceptance criteria

- retrieval quality is benchmarked and tracked
- latency/cost budgets are explicit
- brute-force fallback is no longer the default production path
- task-type embeddings are operational and measurable

## Milestone 4: Control Plane And Operator UX

### Purpose

Turn the dashboard from an observability surface into an operator control plane.

### Primary work

#### Dashboard architecture

- define the dashboard as an operator console, not a static monitor
- prefer websocket/event-driven updates for live state where appropriate

#### Required control surfaces

- live chat/control session
- task tree explorer
- execution receipt viewer
- plan viewer/editor
- workspace attachment and readiness panel
- process/session inspector
- approval queue and policy decision view
- memory/debug view for retrieval inspection

#### Operator trust surfaces

- expose:
  - why the agent took an action
  - what policy allowed it
  - what evidence it used
  - what changed in the workspace
  - what is currently running

### Dependencies

- Milestones 1 to 3

### Key integration points

- API server
- websocket bridge
- execution/task stores
- memory/retrieval inspection
- self-approval and governance

### Acceptance criteria

- a supervisor can operate the system without reading source code
- the dashboard shows live work, decisions, plans, and receipts coherently
- task and tool supervision is possible in real time

## Milestone 5: Browser Operator Capability

### Purpose

Give OpenCAS real browser-use capability rather than static web text retrieval.

### Primary work

#### Browser operator substrate

- add a browser session manager using a real browser engine such as Playwright
- model browser state explicitly:
  - session id
  - tabs
  - current page
  - auth/session persistence
  - action history
  - artifacts

#### Browser action family

- navigation
- click
- type
- keypress
- select
- upload/download
- wait-for conditions
- screenshot
- accessibility tree / DOM extraction
- structured page snapshot

#### Browser receipts

- log every browser action with:
  - timestamp
  - target
  - reason
  - resulting page state/artifact

#### Browser safety policy

- define approval and policy rules for:
  - login/auth use
  - external site navigation
  - downloads/uploads
  - form submission
  - purchase/destructive flows

### Dependencies

- Milestones 1 and 4

### Key integration points

- tool registry
- task execution
- dashboard live supervision
- identity/user trust policy

### Acceptance criteria

- the agent can complete multi-step browser workflows with receipts
- a supervisor can inspect or replay the browser action chain
- browser sessions are stable enough for repeated tasks

## Milestone 6: PTY/TUI/Editor Operator Capability

### Purpose

Give OpenCAS the ability to operate terminal-native interactive tools like a human user.

### Why this is separate from shell/process support

Current shell and process tools provide command execution and stdin/stdout handling. They do not provide terminal-state interaction. Human-like use of terminal apps requires a PTY-backed interaction substrate.

### Primary work

#### PTY session manager

- create real pseudo-terminal sessions
- track:
  - session id
  - command
  - cwd
  - environment
  - terminal size
  - lifecycle state

#### Terminal state model

- support:
  - ANSI-aware screen rendering
  - cursor location tracking
  - scrollback capture
  - resize handling
  - focused region extraction

#### Terminal action family

- send text
- send key combos
- send navigation keys
- send control sequences
- resize terminal
- capture visible screen
- capture scrollback
- wait for screen pattern/state

#### TUI interaction loop

- build a stepwise closed-loop model:
  - observe screen state
  - decide action
  - act
  - wait
  - verify new state

#### Editor interaction strategy

OpenCAS should support three editor paths:

1. direct file-operation path
2. structured editor integration path
3. PTY/TUI-driven editor path for selected terminal editors

Recommendation:

- keep direct file tools as the default coding path
- add PTY-driven support for one or two target terminal tools first
- do not attempt arbitrary visual-editor control before the PTY substrate is stable

#### Target app set for first support

Initial PTY/TUI operator targets should include:

- terminal REPLs
- simple interactive setup/login flows
- one terminal editor workflow
- one git/TUI workflow
- one chat/IRC/TUI workflow

The point is to prove the substrate on real classes of tools, not to claim universal support immediately.

### Dependencies

- Milestones 1, 4, and 5

### Key integration points

- tool registry
- process/session management
- dashboard/session viewer
- task orchestration
- policy/approval surfaces

### Acceptance criteria

- OpenCAS can drive an interactive terminal application through a PTY session
- screen-state and action-state are inspectable by the operator
- at least one editor-like and one TUI-like workflow complete reliably

## Milestone 7: General Work Modes And First Production Agent Profile

### Purpose

Turn the substrate into a coherent first production agent rather than a collection of capabilities.

### First production profile

The first production profile should be:

`general_technical_operator`

This agent should be able to:

- code in a repo
- write and revise documents
- manage tasks and plans
- operate shell and CLI workflows
- use a browser
- operate selected terminal-native interactive tools

### Work mode framework

Define explicit work modes:

- `coding`
- `writing`
- `project_management`
- `research`
- `operator`

Each mode should configure:

- tool access defaults
- planning granularity
- retrieval weighting
- memory salience policy
- approval thresholds
- response style

### Writing and project management support

Writing and project management should not be treated as incidental.

Required support:

- document/task templates
- outline and revision workflow
- task decomposition and dependency mapping
- milestone and issue tracking memory
- meeting/note/decision memory extraction

### Human relationship, emotion, somatic, and temporal systems

These should be retained, but constrained by operational purpose.

#### Relationship model should support

- preference learning
- trust calibration
- tone alignment
- long-horizon continuity

#### Somatic model should support

- urgency calibration
- pacing
- prompt style modulation
- interruption/recovery behavior

#### Temporal distance should support

- recency weighting
- project continuity
- distinction between urgent now and durable later

These systems should be reviewed against explicit metrics, not intuition.

### Dependencies

- Milestones 1 to 6

### Key integration points

- memory
- identity
- relational engine
- somatic engine
- task/planning system
- browser and PTY operator layers

### Acceptance criteria

- one named production profile exists with explicit operating rules
- that profile can complete benchmark tasks across coding, writing, project management, browser use, and TUI use
- mode switching is explicit and inspectable

## Milestone 8: Evaluation, Soak Testing, And Production Qualification

### Purpose

Prove the system performs sustained useful work safely and economically.

### Primary work

#### Benchmark suite

Create benchmark families for:

- coding tasks
- writing/editing tasks
- planning/project tasks
- browser workflows
- PTY/TUI workflows
- mixed continuity tasks spanning multiple sessions

#### Long-run soak testing

Run sustained tests that cover:

- memory growth
- background loop stability
- recovery from interrupted tasks
- dashboard supervision
- process/session cleanup
- browser/session cleanup
- PTY/session cleanup

#### Qualification gates

Define production gates for:

- completion rate
- unsafe action rate
- operator intervention rate
- average cost per completed task
- latency budgets
- crash/recovery rate
- retrieval quality

### Dependencies

- Milestones 1 to 7

### Key integration points

- all major subsystems

### Acceptance criteria

- qualification thresholds are documented
- benchmark runs are repeatable
- production promotion requires passing those gates

## Cross-Cutting Workstreams

These run across multiple milestones and must not be left implicit.

## Workstream A: Policy And Trust

### Required decisions

- what actions are self-approvable by default
- what actions require operator confirmation
- how trust evolves by evidence
- what constitutes a boundary violation

### Deliverables

- structured policy model
- auditable approval decisions
- operator-visible trust state

## Workstream B: Receipts And Provenance

### Required decisions

- what every meaningful action must log
- how evidence is tied to decisions
- how replay/debugging works

### Deliverables

- consistent receipt schema across shell, browser, PTY, edits, and tasks
- dashboard receipt inspection

## Workstream C: Cost And Model Routing

### Required decisions

- which models are used for:
  - control logic
  - planning
  - synthesis
  - embeddings
  - consolidation
- when to use cheap vs expensive inference

### Deliverables

- model routing policy
- embedding strategy by task type
- token and external cost dashboards

## Workstream D: Plugin, MCP, And External Capability Integration

### Required decisions

- what core capabilities are native
- what capabilities should be MCP/plugin mediated
- how external tools are normalized into policy and receipt layers

### Deliverables

- plugin capability contract
- MCP tool normalization policy
- stable registration/lifecycle rules

## Workstream E: Writing And Project Management As First-Class Work

### Required decisions

- what the writing workflow looks like
- how projects, milestones, notes, and decisions persist in memory
- how task planning and execution connect to authored artifacts

### Deliverables

- writing task schema
- project/task planning schema
- templates and quality checks for non-code work

## Integration Guardrails

The following guardrails should be enforced during implementation.

### 1. One receipt model across action families

Shell, browser, PTY, editing, and planning work should all produce compatible evidence records.

### 2. One policy model across action families

Do not create separate safety universes for shell, browser, and PTY actions.

### 3. One session model across interaction surfaces

Conversation sessions, browser sessions, PTY sessions, and workspace sessions should be linkable.

### 4. Memory must remain budgeted

Every new capability must define what it stores, what it summarizes, and what it discards.

### 5. Operator visibility is mandatory

If the agent can do something, the operator must be able to inspect what happened and why.

## Priority Order Inside The Program

The practical priority order should be:

1. Milestone 0
2. Milestone 1
3. Milestone 2
4. Milestone 3
5. Milestone 4
6. Milestone 5
7. Milestone 6
8. Milestone 7
9. Milestone 8

This order is strict for production readiness.

Reason:

- browser and PTY layers should not be built on top of weak policy, weak receipts, or weak control-plane visibility
- the first production agent profile should not be defined until the substrate is stable enough to support it

## Suggested Execution Batches

To keep the program tractable, the work should be batched like this:

### Batch A

- Milestone 0
- Milestone 1

Outcome:

- trustworthy foundation

### Batch B

- Milestone 2
- Milestone 3
- core parts of Milestone 4

Outcome:

- stable coding/bootstrap and measurable memory/runtime behavior

### Batch C

- complete Milestone 4
- Milestone 5

Outcome:

- usable operator console and browser workflows

### Batch D

- Milestone 6
- Milestone 7

Outcome:

- PTY/TUI/editor support and first real production agent profile

### Batch E

- Milestone 8

Outcome:

- qualification and rollout decision

## Readiness Checklist For The First Production Agent

Before declaring the first production-capable agent, all of the following should be true:

- retrieval correctness issues are resolved
- self-approval policy is structured and auditable
- dashboard is operational as a control plane
- workspace attachment is explicit and reliable
- coding tasks are reproducible and receipt-backed
- browser workflows are real, not HTML fetches
- PTY/TUI workflows are real, not blind stdin writes
- writing and project-management workflows are first-class
- cost budgets are defined and monitored
- long-running soak tests pass
- qualification benchmarks pass

## Immediate Planning Next Steps

The next planning artifacts should be created in this order:

1. milestone ownership map and subsystem owners
2. acceptance-test matrix by milestone
3. canonical receipt schema across shell/browser/PTY/task actions
4. structured policy schema for approvals and trust
5. workspace attachment and agent-profile spec

## Final Program Judgment

OpenCAS does not need a conceptual reset. It needs disciplined productionization.

The right program is:

- harden the current core
- define explicit operator substrates for browser and PTY interaction
- make writing and project management first-class alongside coding
- unify policy, receipts, sessions, and control surfaces
- qualify the result against real work and real budgets

If executed in this order, OpenCAS can become the kind of agent platform the product spec is aiming at. If the order is ignored, the project is likely to accumulate more surface area without gaining the reliability needed for real deployment.
