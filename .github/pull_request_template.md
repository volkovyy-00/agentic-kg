<!-- Title: "KG-NN: <what changes, in plain words>". No ticket? Add the `no-ticket` label. -->

**Ticket:** KG-NN — https://stormdoc.atlassian.net/browse/KG-NN
**Release:** X.Y.Z (patch / minor / major) — or `no-release` label if nothing user-visible changes

## Why

<!-- The failure or gap that motivated this, and what was tried. The diff shows *what*. -->

## What changes for a user

<!-- Mirrors the CHANGELOG entry. Name any behaviour that is deliberately NOT changed. -->

## Decisions a future session needs

<!-- Anything a later change must not undo, and why. Link the spec/plan/intent in the private
     notes repo by path if one exists (e.g. docs/superpowers/specs/2026-09-20-foo-design.md). -->

## Checklist

- [ ] `pyproject.toml` bumped, `uv lock` run, and `CHANGELOG.md` has `## [X.Y.Z] - YYYY-MM-DD` on top with entries citing `(#PR, KG-NN)` — or `no-release`
- [ ] `uv run pytest -q`, `ruff check .`, `ruff format --check .`, `pyright` pass
- [ ] Integration tests run locally if Neo4j access, loading or retrieval changed
- [ ] `CLAUDE.md` updated if an architectural invariant or command changed (status goes to Jira, not here)
