# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Fixed
- **Over-matching relationship joins (#23)**: relationship loading now warns when a construction rule
  matches both endpoints more times than the rows it read, not only when it matches too few. A join key
  coarser than the fact each row describes — joining on a node's stored property rather than the
  column identifying it — previously reported a clean build; where `MERGE` collapsed the duplicates
  back, the graph looked correct too. The warning names the relationship type and both endpoints, and
  nothing about what a build writes has changed. Warnings raised by rules that loaded successfully now
  also survive a partial failure as a structured list rather than only as text inside the error
  message.

## [0.5.3] - 2026-08-16

### Fixed
- **Invented construction warnings (#22)**: the construction agent no longer presents a "Construction
  warnings" section when the loader reported none. Its instruction previously said what to do when the
  build result carried warnings and nothing about when it did not, and a schema-critic reply from
  earlier in the session — which uses the same word — could be relabelled as construction output. The
  agent now reports only the warnings carried by the most recent build, whether those arrive as a
  warnings list on success or inside the error message of a partial failure, and writes no section
  when that build reported none. Warnings from another agent, another tool, or an earlier build of the
  same graph are no longer repeated as though they described the current one. The context filter that
  keeps other agents' turns out of this agent's requests, wired on in 0.5.0, is now covered by tests
  against exactly that scenario.

## [0.5.2] - 2026-08-14

### Fixed
- **Orphaned reference columns (#21)**: approval now refuses a construction plan that leaves an
  approved file's reference column unreachable — a column identifying rows in one file and named
  identically in another, which the plan neither keys a node by nor preserves through collapsing.
  Such a plan could previously only be made approvable by dropping the affected relationship
  entirely, which happened silently and produced a graph missing every edge of that type. The
  check reads the approved files and fails open: a source it cannot read is reported as unverified
  rather than blocking approval.

## [0.5.1] - 2026-08-14

### Fixed
- **Withheld schema approval (#20)**: the schema agent no longer tells the user a construction plan is
  "not ready for approval" when nothing is actually blocking it. When the critic's remaining objections
  describe the source data rather than a fault in the plan — or when the refinement budget for the turn
  is spent — the agent now reads whether approval would succeed and presents the plan with the choice to
  approve it as it stands or ask for a change, instead of ending the turn on a verdict of its own. A plan
  that genuinely cannot be built is still refused, and now says which joins are the reason.

## [0.5.0] - 2026-08-09

### Added
- **Typed graph properties (#13)**: a construction plan can now declare a property as `integer`,
  `float` or `boolean`, and the CSV loaders write real Neo4j types instead of strings — so
  filtering, range comparison, sorting and aggregation over quantities, lead times, prices and
  costs return correct answers without the query casting or cleaning the value first. Currency
  formatting and thousands separators are stripped on the way in, including negatives written
  either way round (`-$42.00`, `$-42.00`) and in accounting parentheses (`($42.00)`). A new `column_type_hint` tool
  gives the schema agents the evidence to propose and challenge a type. Values that cannot be read
  as their declared type are reported and cleared rather than silently kept as text, and a column
  failing on most of a batch stops that rule outright. Identifiers and join columns stay text by
  rule, since they are matched as raw CSV values.
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
- **Ruff for linting and formatting (#15)**: `ruff check` / `ruff format --check` now cover `src/` and
  `tests/`; config lives in `pyproject.toml`'s `[tool.ruff]`. Dev-only — no user-facing behavior change.

### Changed
- **README rewritten as an independent project doc (#7)**: drops the deeplearning.ai course framing
  (course link, "not a production tool" disclaimer, "Special Thanks" section) for a single-line fork
  acknowledgment, and fixes content that had gone stale since the fork — the missing `graphrag_agent`
  pipeline stage, an outdated test count, a broken Google ADK link, and a roadmap checklist now
  superseded by `docs/spec.md` and this file. The same stale "companion project" framing is fixed in
  `pyproject.toml`, `LICENSE.txt`, and `.github/copilot-instructions.md`.
- **PRs now merge with a real merge commit, not squash (#16)**: enforced at the GitHub-settings level;
  every commit on a branch now lands in `main`'s history as-is. See `CONTRIBUTING.md`.

### Removed
- **Local `upstream` git remote (#14)**: this clone is local-only now. GitHub still shows the repo as
  forked from `neo4j-contrib/agentic-kg`; there's just no local remote to sync from anymore.

### Fixed
- **The user's goal approval is now recorded (#12)**: `user_intent_agent_v2` could ask its clarifying
  questions and transfer away in the same reply, so the user's agreement was heard by the coordinator,
  which holds no approval tool — `approved_user_goal` was never written, and the workflow continued on a
  goal the system had never recorded as approved. The intent phase can no longer end without a recorded
  approval, and can no longer be walked out of via ADK's injected `transfer_to_agent` tool. An approval
  that the user then revises no longer counts: the gate compares the approved goal against the current
  one, so a revision must be approved again. The coordinator's own delegating call is also filtered out
  of this agent's context, so the tool taken away is not left behind as a worked example.
- **Handoff gates are no longer bypassable (#11)**: both the construction and retrieval handoff gates
  guarded only their own `finished` tool, while ADK independently injected a `transfer_to_agent` tool —
  and an instruction advertising it — into every sub-agent with a parent or peers. The model could leave
  either phase through that tool with the confirmation flag still unset, which is the exact defect both
  gates were built to prevent. Both gated agents now strip that tool out of every request before the
  model sees it, so `finished` is the only exit the model can choose. The construction gate's refusal
  message was also corrected: it now tells the model a retry will succeed when the confirmation was
  recorded later in the same reply, instead of sending it back to re-ask the user. The construction
  agent additionally drops other agents' turns from its context, the same filtering the retrieval agent
  already did — otherwise the coordinator's own delegating call stayed in history as a worked example of
  the tool that was just taken away.
- **Neo4j connection recovery (#10)**: a brief database outage no longer disables every graph tool for the
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

[Unreleased]: https://github.com/volkovyy-00/agentic-kg/compare/v0.5.3...HEAD
[0.5.3]: https://github.com/volkovyy-00/agentic-kg/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/volkovyy-00/agentic-kg/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/volkovyy-00/agentic-kg/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/volkovyy-00/agentic-kg/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/volkovyy-00/agentic-kg/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/volkovyy-00/agentic-kg/releases/tag/v0.1.0
