# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Added
- **Living spec (#6)**: `docs/spec.md`, a verified orientation document covering what the project is, the two
  entry points and why only one is maintained, the construction workflow as it actually runs, what retrieval
  grounding does and does not guard against, and the current shipped/next state.
- **Explicit construction handoff (#8)**: the post-construction question window now states which agent
  is answering, keeps the continue-or-hand-off choice on screen, and ends only on the user's explicit
  confirmation — recorded by a tool call and enforced by a session-state gate, not inferred from tone.
  Confirming transfers straight to the retrieval agent rather than stalling at the coordinator.
- **Explicit retrieval handoff (#9)**: the retrieval agent now stays until the user says they are
  finished, inviting the next question after each answer instead of ending the phase on its own
  judgment after a single one. Leaving is recorded by a tool call and enforced by a session-state
  gate, the same mechanism the construction handoff uses.

### Fixed
- **Neo4j connection recovery (#PR)**: a brief database outage no longer disables every graph tool for the
  life of the process. The shared client now reopens its own connection on next use instead of being
  discarded while five modules still held it, so once the database is healthy the next tool call succeeds
  without a restart. Reconnections are logged, and a recovery is reported only after a query has actually
  succeeded.

## [0.4.0] - 2026-08-03

### Added
- **Contributor workflow (#5)**: `CONTRIBUTING.md` (PR/branch conventions, testing requirements, design-decision
  and CHANGELOG policy) and `CHANGELOG.md` (this file, with backfilled release history).

### Changed
- `.gitignore` narrowed from ignoring all of `docs/` to just `docs/superpowers/` (design specs/plans) and
  `docs/backlog/` (defect/follow-up notes) (#5) — a prior blanket ignore had silently untracked two design
  specs during the 0.3.0 merge, recovered from git history rather than lost.
- Conversational model bumped to DeepSeek V4-Flash's official `0731` release, superseding the preview build.

## [0.3.0] - 2026-08-02

### Added
- **Grounded graphrag retrieval (#4)**: schema-profile-aware querying with completeness and partition
  annotations, replacing reliance on conversational recall. Adds context filtering for ADK's cross-agent
  event history, a two-layer schema-profile cache, and bounded/summarized query results.

## [0.2.1] - 2026-07-30

### Fixed
- **`schema_refinement_loop` feedback-clobbering bug (#3)**: the loop's `before_agent_callback` reset
  feedback on every iteration instead of once per invocation, discarding critique mid-loop.

### Added
- Per-turn invocation cap on `schema_refinement_loop` (#3), preventing the coordinator from silently
  chaining multiple full propose/critique rounds within a single user turn.

## [0.2.0] - 2026-07-29

### Added
- **Foundation (#2)**: file sources via `fsspec`, driver-side CSV loading, and OpenRouter with per-job
  model selection.

### Security
- **Path traversal in file source resolution (#2)**: an unchecked relative path (e.g. `"../.env"`) could
  escape the configured source root and read arbitrary files, including the OpenRouter key and Neo4j
  credentials in `.env`.

### Fixed
- Driver-side CSV loading replaces `LOAD CSV FROM file:///`, which Neo4j Aura rejects (#2).
- `finished()`'s private-API parent-agent lookup, replaced with a documented public-API path (#2).
- A ragged CSV row could silently erase properties an earlier row had already loaded onto the same
  entity (#2).
- `schema_refinement_loop` could silently revert a user-requested schema fix with no error surfaced (#2).
- File reads defaulted to the platform's locale encoding instead of `utf-8`, and OpenRouter calls with no
  `max_tokens` cap could pre-authorize against the full token ceiling and fail with an opaque 402 (#2).

## [0.1.0] - 2026-07-26

### Fixed
- `tool_result` API mismatch between unit tests and the real `(key, result)` signature (#1).
- Integration tests' Neo4j test-container password default, and a result-key mismatch reading
  `send_query()`'s output (#1).

### Added
- Documentation of the colima/Testcontainers `DOCKER_HOST` + Ryuk workaround needed to run integration
  tests locally (#1).

[Unreleased]: https://github.com/volkovyy-00/agentic-kg/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/volkovyy-00/agentic-kg/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/volkovyy-00/agentic-kg/releases/tag/v0.1.0
