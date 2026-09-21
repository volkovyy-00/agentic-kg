# Contributing to agentic-kg

This is being developed into a real program, not a teaching artifact (see `CLAUDE.md` for the full
positioning note) — treat changes accordingly: prefer the choice that makes a working program over one
that mirrors the deeplearning.ai course structure it was forked from.

## Before you start

- Read `CLAUDE.md` — the architecture map (two coordinators, the `variants` pattern, state-passing via
  ADK session state, tool result conventions) and the invariants a change must not break.
- Read the Jira ticket you are working on (project `KG`, <https://stormdoc.atlassian.net/browse/KG>),
  including its comments: that is where the current state of the work is.
- Set up the project per `README.md` (`uv venv && uv sync`, `.env` from `.env.example`), then run
  `gh repo set-default volkovyy-00/agentic-kg` once — see *Branches and PRs* for why.

## Where project knowledge lives

Each kind of knowledge has exactly one home. Write it there, and point to it from anywhere else rather
than copying it — a copy is what goes stale.

| Kind of knowledge | Home | Shared via git? |
|---|---|---|
| What is planned, in progress, blocked; why a piece of work was asked for; a handoff between sessions | **Jira ticket** (description + comments) | no — Jira |
| What changed, in which release | **`CHANGELOG.md`** | yes |
| Why the code was changed this way | **Commit messages and the PR description** | yes (merge commits keep them in `main`) |
| How the system works; invariants a change must not break; commands | **`CLAUDE.md`** | yes |
| What the program is and why it exists | **`docs/spec.md`** | yes |
| How to contribute | **this file** | yes |
| Specs, plans and approved intents for a piece of work | **`docs/superpowers/`** — a separate, private git repo nested in the working copy | no — link it from the ticket and the PR by path |

Rules that follow from the table:

- **No status in `CLAUDE.md` or `docs/spec.md`.** "Current work", "latest release", "next up" — those
  belong to Jira and `CHANGELOG.md`, and go stale anywhere else. `CLAUDE.md` holds what stays true until
  the code changes.
- **A session that stops mid-ticket leaves a comment on the ticket**: what is done, what is next, and
  anything learned that isn't yet in a commit. The next session starts from the ticket, not from a
  local handoff file.
- **A decision a later change must not undo** goes in the PR description (*Decisions a future session
  needs*) and, if it constrains the architecture, in `CLAUDE.md`. A spec under `docs/superpowers/` is the
  long form, but no one else's clone has it.
- **Something real but out of scope** for the PR you're in becomes a Jira ticket, not a local note.

## Tickets

Work is tracked in Jira project `KG`. Reference the ticket key wherever the work shows up:

- **Branch:** `KG-21-streaming-column-checks`.
- **PR title:** `KG-21: Stream the distinct-value column checks`.
- **Commit subject**, where a commit is specific to the ticket: `fix(file_tools): … (KG-21)`.
- **`CHANGELOG.md` entry:** `(#62, KG-21)`.

If the Jira GitHub integration is installed, a key in the branch, PR title or commit links them to the
ticket automatically. Work with no ticket — a typo fix, a dependency bump — takes the `no-ticket` label
instead; if the work is bigger than that, open a ticket first.

## Branches and PRs

- Open a PR against `main` of `volkovyy-00/agentic-kg`, using the PR template.
- **The fork trap.** This repo is a GitHub fork of `neo4j-contrib/agentic-kg`, and a clone may also carry
  an `upstream` remote pointing there. Without a default set, `gh` resolves a fork's commands against the
  *parent* repo — `gh pr view` fails and `gh pr create` targets `neo4j-contrib`. Run
  `gh repo set-default volkovyy-00/agentic-kg` once per clone (check with `gh repo set-default --view`),
  or pass `--repo volkovyy-00/agentic-kg` on every `gh` command.
- PRs are merged with a real merge commit (`gh pr merge --merge`), not squashed or rebased — every commit
  on the branch lands in `main`'s history as-is, so write meaningful individual commit messages, not just
  a summary-worthy PR title.
- CI runs unit tests (`pytest -q`), Ruff, pyright, SonarCloud and the *PR conventions* check (ticket and
  release, below) on every PR against `main`. Integration tests need Docker and aren't run in CI — run
  them yourself before opening a PR (see below).
- Dependabot (`.github/dependabot.yml`) opens weekly PRs for `uv` dependencies and pinned GitHub Actions.
  They are exempt from the ticket and release rules; review and merge them like any other PR.

## Releases: one per PR

Every PR that adds, changes, or fixes user-visible behaviour is its own release. In that PR:

1. Bump `version` in `pyproject.toml` to the next version after `main`'s:
   - **Patch** — bug fixes, small internal changes.
   - **Minor** — a significant new capability, or a completed sub-project.
   - **Major** — breaking changes to the agent/tool contract or graph schema conventions.

   Three segments only (`MAJOR.MINOR.PATCH`) — no `0.1.3.1`.
2. Update `uv.lock`'s own `agentic-kg` version to match (`uv lock`; if your `uv` reformats the rest of
   the lock, revert that and change only the version line).
3. Add a `## [X.Y.Z] - YYYY-MM-DD` section at the top of `CHANGELOG.md` holding this PR's entries.
4. Put the version in the PR description.

After the merge, the *Tag release* workflow tags the merge commit `vX.Y.Z`. No separate release commit.

PRs that change nothing user-visible — docs, CI, refactors, test-only changes — take the `no-release`
label and leave the version alone. The *PR conventions* check enforces all of the above.

**Two open PRs claiming the same version:** whichever merges second conflicts on `CHANGELOG.md` (the
identical `pyproject.toml` and `uv.lock` bumps merge cleanly, so don't rely on them to flag it); rebase it
onto `main`, take the next version, and retitle its `CHANGELOG.md` section.

## CHANGELOG entries

- Entries describe impact, in past tense, and end their bold lead with the PR number and ticket key:
  `(#62, KG-21)`, or `(#48)` when there is no ticket. No file paths, line counts, or review-comment
  history — those belong in the git log and PR discussion.
- Typos, formatting, and comment-only changes don't need an entry (and take `no-release`).

Example:

```markdown
## [0.5.4] - 2026-08-17

### Fixed
- **Over-matching relationship joins (#23, KG-2)**: relationship loading now warns when a construction
  rule matches both endpoints more times than the rows it read, not only when it matches too few.
```

## Commits and PR descriptions

Explain *why*, not just *what* — the diff already shows what changed. A good commit/PR body names the
failure or gap that motivated the change and what specifically was tried, the way `63e6d99`'s and
`2d9fa8a`'s messages do. Avoid narrating file-by-file changes; that's what `git diff --stat` is for.
The PR template's *Decisions a future session needs* section is the one a later change reads first —
fill it in whenever the PR settles something that isn't obvious from the code.

## Testing

```bash
uv run pytest -q                 # unit tests — fast, no external deps; also runs in CI on every push/PR
uv run pytest -q -m integration  # integration tests — need Docker (Testcontainers); skip cleanly without it
uv run ruff check .               # lint — run before every PR
uv run ruff format --check .      # formatting check — run before every PR; drop --check to fix locally
uv run pyright                    # static type check — run before every PR; must report 0 errors, CI fails otherwise
```

A PR that changes retrieval, construction, or Neo4j access code should include or update unit tests; skip
only with a stated reason (e.g. "covered by the existing fake in `tests/unit/fakes.py`").

## Design documents

There is no ADR directory. For a decision with long-term impact (a new sub-agent architecture, a change to
state-passing conventions, a new external dependency, a retrieval/construction behaviour change with
user-visible consequences), the design spec, plan and approved intent live in `docs/superpowers/`
(`specs/`, `plans/`, `intents/`). That directory is gitignored by this repo and is its own private git
repo — commit there with `git -C docs/superpowers …`. A git worktree of this repo does not contain it.

Because no fresh clone has those files, anything another contributor or a later session must know goes
where it is shared: the Jira ticket, the PR description, and `CLAUDE.md` if it is an architectural
invariant. Link the spec from the ticket and the PR by path. Skip a spec entirely for tactical bug fixes,
refactors, or anything whose rationale fits in the PR description.

## Getting help

Check `CLAUDE.md`'s architecture section first — most "why does this work this way" questions are
answered there — then the Jira ticket, then `git log` for the change in question.
