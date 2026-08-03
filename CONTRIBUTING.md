# Contributing to agentic-kg

This is being developed into a real program, not a teaching artifact (see `CLAUDE.md` for the full
positioning note) — treat changes accordingly: prefer the choice that makes a working program over one
that mirrors the deeplearning.ai course structure it was forked from.

## Before you start

- Read `CLAUDE.md` — it holds the architecture map (two coordinators, the `variants` pattern, state-passing
  via ADK session state, tool result conventions) and the current work-in-progress section. Don't re-derive
  decisions that are already recorded there.
- Set up the project per `README.md` (`uv venv && uv sync`, `.env` from `.env.example`).
- `docs/superpowers/specs/` and `docs/superpowers/plans/` may hold design notes for past and current
  sub-projects, but they're gitignored local scratch (see "Documenting significant design decisions" below)
  — don't expect them in a fresh clone, and don't rely on them being there.

## Branches and PRs

- Branch names are descriptive, not ticket-prefixed: `graphrag-grounding`, `foundation-file-sources-and-models`.
- Open a PR against `main`. This repo has two remotes (`origin` = your fork, `upstream` =
  `neo4j-contrib/agentic-kg`) — `gh` commands need `--repo <owner>/agentic-kg` or they'll resolve against
  the wrong one.
- PRs are squash-merged. Write the PR title as the single line that should stand as the commit's summary.
- No CI is currently configured — run the test suite yourself before opening a PR (see below).

## Commits and PR descriptions

Explain *why*, not just *what* — the diff already shows what changed. A good commit/PR body names the
failure or gap that motivated the change and what specifically was tried, the way `63e6d99`'s and `2d9fa8a`'s
messages do. Avoid narrating file-by-file changes; that's what `git diff --stat` is for.

## Testing

```bash
uv run pytest -q                 # unit tests — fast, no external deps, run before every PR
uv run pytest -q -m integration  # integration tests — need Docker (Testcontainers); skip cleanly without it
```

A PR that changes retrieval, construction, or Neo4j access code should include or update unit tests; skip
only with a stated reason (e.g. "covered by the existing fake in `tests/unit/fakes.py`").

## Documenting significant design decisions

This project doesn't use a separate ADR directory. `docs/superpowers/` (specs and plans) and `docs/backlog/`
(defect/follow-up notes) are gitignored by deliberate policy (see `.gitignore`: "keep design docs and plans
local, out of the published repo") — nothing under either is shared via git. The rest of `docs/` is tracked normally;
that's where a durable, shared project doc (e.g. the living spec) belongs. If you write a design spec before
implementing a decision with long-term impact (new sub-agent architecture, a change to state-passing
conventions, a new external dependency, a retrieval/construction behavior change with user-visible
consequences), treat it as your own local working notes, not a shared project record: a fresh clone will not
have it, and nobody else's spec will be in yours either. If a decision needs to be discoverable by future
contributors, the durable place for its rationale is the PR description — that's what survives the squash
merge into `main`'s history. Skip writing a spec at all for tactical bug fixes, refactors, or anything whose
rationale fits directly in the PR description.

The same applies to defect write-ups: if you hit something real but out of scope for the PR you're in, a
handoff note under `docs/backlog/` is fine as your own scratch reference or as a draft to paste into an
external tracker, but it will not be visible to anyone else from the repo. Anything another contributor needs
to see belongs in an issue, not a local file.

## CHANGELOG

Every PR that adds, changes, or fixes user-visible behavior updates `CHANGELOG.md`:

- Semantic versioning, three segments only (`MAJOR.MINOR.PATCH`) — no `0.1.3.1`.
  - Patch: bug fixes, small internal changes.
  - Minor: a completed sub-project or a significant new capability.
  - Major: breaking changes to the agent/tool contract or graph schema conventions.
- Entries describe impact, in past tense, referencing the PR number for traceability — not file paths, line
  counts, or review-comment history (those belong in the git log and PR discussion).
- Typos, formatting, and comment-only changes don't need an entry.

Example:

```markdown
## [0.3.0] - 2026-08-02

### Added
- **Grounded graphrag retrieval (#4)**: schema-profile-aware querying with completeness and
  partition annotations, replacing reliance on conversational recall.
```

## Getting help

Open an issue, or check `CLAUDE.md`'s architecture section first — most "why does this work this way"
questions are already answered there.
