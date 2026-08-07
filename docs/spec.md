# agentic-kg — living spec

**Status:** living document. Every claim below was verified against the code and git state at the 0.4.0
release (2026-08-03); for anything that landed since, check `CHANGELOG.md`'s `[Unreleased]` section and
treat this document as the thing to update. This is the "what is this and why does it exist" document.
For the architecture map you need while editing code, read `CLAUDE.md`. For the PR/branch/CHANGELOG
workflow, read `CONTRIBUTING.md`. This document deliberately does not duplicate either.

---

## 1. What this is

A multi-agent system that turns a folder of source files into a Neo4j knowledge graph, then answers
questions over it. It is built on Google ADK (`google-adk>=1.10,<2`) with LiteLLM routing every model
call through OpenRouter, and it talks to Neo4j (`neo4j>=5.28.2,<6`, plus `neo4j-graphrag`) over Bolt —
local or Aura. Python 3.12, dependencies via `uv`.

The user does not write Cypher. They state a goal in conversation; a chain of agents interviews them,
picks input files, proposes a graph schema, builds the graph, and then answers questions about it —
pausing for explicit human approval between each of those stages.

It began as a fork of the companion project to the deeplearning.ai short course *Agentic Knowledge Graph
Construction* (`upstream` remote is `neo4j-contrib/agentic-kg`), but it is now developed as a real
program rather than a teaching artifact. What makes that true, concretely, rather than as a claim:

- Five tagged releases (`v0.1.0` … `v0.4.0`) and five merged PRs beyond the fork point; `main` is six
  commits ahead of `upstream/main` and zero behind.
- A substantially expanded test suite: 280 passing unit tests (4 skipped without Docker) across 4
  integration modules, against upstream's 2 unit and 2 integration files.
- Capabilities built here that the course does not have: retrieval grounding (§4), client-side CSV
  loading that works against Aura, `fsspec`-based file sources, and per-job model selection.
- A contributor workflow, changelog discipline, and a design-decision record.

Course-shaped structure survives in the code — most visibly the `variants` dicts, where each agent's
prompt/tool wiring lives in a sibling `variants.py` keyed by version-suffixed names. That is inherited
history, not a constraint to preserve. Four of the seven agents now have only a single variant. Of the
three `_v1`/`_v2` pairs that remain, `graphrag_agent` selects v2 and keeps v1 unchanged for an explicit
A/B comparison — the one pair documented as a deliberate retention; `user_intent_agent` selects v2; and
`cypher_agent` selects v1, leaving its v2 unselected.

> Known inconsistency: `README.md:9` still describes the project as "a reference implementation for
> learning and experimentation, not a production tool," which contradicts this section, `CLAUDE.md`, and
> `CONTRIBUTING.md`. `.github/copilot-instructions.md:3` does not make that claim, but still calls the
> project a companion to the course. The README has not been revised since 2026-07-29.

---

## 2. The two entry points

`uv run adk web src/agentic_kg/coordinators/` discovers two top-level agents. They share `common/` and
`tools/` and nothing else. They are **not** two supported ways to do the same job:

| | `multi_agent` | `single_agent` |
|---|---|---|
| What it is | The product | A frozen escape hatch |
| Real agent name | `kg_construction_agent_v1` | `single_agent_agent_v1` |
| Coordinator tools | `get_physical_schema`, `get_source_location`, `neo4j_is_ready` | **none** — its only capability is transferring away |
| Can read source files | Yes — the whole `SOURCE_URI` seam | **No** — has no file tools at all |
| Approval gates | Yes, between every stage | None |
| Session state | Nine keys across five stages | `{}` — uses none |
| Commits since the fork | 3 — all three feature PRs (#2, #3, #4) | 1 — mechanical repairs in #2 |
| Test coverage | Yes | Zero tests reference it |

**Use `multi_agent`.** It is the system this project is about.

**`single_agent`** is the course's early exercise, preserved. It is an LLM in front of a Cypher REPL:
one coordinator with an empty toolset that delegates to `agents/cypher_agent` (`cypher_agent_v1`), which
holds `neo4j_is_ready`, `reset_neo4j_data`, `get_physical_schema`, `read_neo4j_cypher`,
`write_neo4j_cypher`, `create_uniqueness_constraint`, and `finished`. It is genuinely good at ad-hoc
direct Cypher — inspect a schema, run a query, hand-build a toy graph from data typed into the prompt.

It is not merely weaker for the actual product workflow; it is incapable of it, and it fails
*unhelpfully*. Asked in a live trace to load `./data/bom/products.csv`, it had no file tool, improvised
six `LOAD CSV FROM 'file:///…'` attempts that Aura rejected outright, hit `Forbidden` on
`dbms.listConfig()`, and then reported the request satisfied on the strength of pre-existing `Product`
nodes it had not created. Those are exactly the Aura defects the Foundation work eliminated — the fix
was tool-side, and `cypher_agent` was never given those tools.

Two traps when reading this code: `cypher_agent` is used by `single_agent` **only**, and
`coordinators/multi_agent/prompts.py` is dead code — never imported, and its text claims otherwise.

---

## 3. The construction workflow

`multi_agent` is a hierarchical `LlmAgent` that delegates in strict sequence through five sub-agents.
The sequence below is as it actually executes, from a traced six-turn session that built a real graph.

Agents do not pass data by return value. Each stage reads and writes keys on ADK session state, and a
later stage's tools fail fast with `tool_error(...)` when an earlier key is missing — that, not the
coordinator's prose, is what actually enforces the ordering.

1. **`user_intent_agent_v2`** — interviews the user for the kind of graph and its description.
   Writes `perceived_user_goal`, then `approved_user_goal` on confirmation. `finished` refuses
   to leave the phase until `approved_user_goal` exists and still matches `perceived_user_goal`,
   so a goal revised after approval must be approved again. ADK's injected `transfer_to_agent`
   tool is stripped from this agent's requests, and every foreign turn — not just the
   coordinator's own delegating call — is filtered out of its context, so on re-entry after
   later phases this agent sees none of those agents' output, and the only example of the
   stripped call is gone with it. Unlike stages 4 and 5 this gate holds no per-turn flag — the
   approval itself is the durable record.
2. **`file_suggestion_agent_v1`** — reads `approved_user_goal`. Lists and samples what is under
   `SOURCE_URI`, proposes a subset. Writes `all_available_files`, `suggested_file_list`,
   `approved_file_list`.
3. **`schema_proposal_agent_coordinator`** — the most elaborate stage: an `LlmAgent` wrapping a
   `LoopAgent` (`schema_refinement_loop`, `max_iterations=2`) that runs `schema_proposal_agent_v1` →
   `schema_critic_agent_v1` → stop-check. The critic inspects joins
   with `join_preview` / `column_stats` / `collapse_check` and returns `valid` or `retry`. Writes
   `proposed_construction_plan`, `feedback`, `approved_construction_plan`.
4. **`graph_construction_agent_v1`** — reads the approved plan, creates uniqueness constraints, and runs
   `build_graph_from_construction_rules`. Writes one key,
   `construction_handoff_confirmed`, a per-turn flag gating the explicit handoff to stage 5. ADK's
   injected `transfer_to_agent` tool is stripped from this agent's requests, so that gated `finished`
   is the only exit the model is offered.
5. **`graphrag_agent_v2`** — answers questions over the finished graph. Reads and writes one key,
   `graphrag_handoff_confirmed`, a per-turn flag gating the explicit handoff back to the coordinator.
   ADK's injected `transfer_to_agent` tool is stripped from `graphrag_agent_v2`'s requests, so that
   gated `finished` is the only exit the model is offered. `graphrag_agent_v1`, the ungated A/B
   baseline, is unchanged and still receives the injected tool.

In the traced run this produced Supplier 20 / Part 88 / Product 10 / Assembly 64 nodes and SUPPLIES 176
/ PART_OF 88 / ASSEMBLY_OF 64 relationships from the bundled `data/bom` example. Final state held eleven
keys: `perceived_user_goal`, `approved_user_goal`, `all_available_files`, `suggested_file_list`,
`approved_file_list`, `schema_refinement_calls_this_turn`, `feedback`, `proposed_construction_plan`,
`approved_construction_plan`, `construction_handoff_confirmed`, `graphrag_handoff_confirmed`.

Source files are read by the application (via `fsspec`, `common/file_source.py`) and loaded with
parameterised `UNWIND` batches — never by the database. There is no Neo4j import directory to manage,
which is what makes the whole path work unchanged on Aura.

**Behaviour observed live that the code does not obviously imply:**

- The per-turn refinement budget works. `schema_refinement_calls_this_turn` capped the loop at one
  invocation, the second call short-circuited with `stopped: …`, and the coordinator correctly fell back
  to asking the user to decide. Note the key is zeroed on every entry to the stage coordinator, so its
  stored value tells you nothing about earlier turns.
- **Stages 4 and 5 each write one flow-control flag; neither records what it did.** Whether a graph was
  built, and what it contains, is still recoverable only from the event transcript or by querying
  Neo4j. A resumed session cannot tell.
- The coordinator's `get_physical_schema` check is framed as "is the database empty," but **nothing
  gates on the answer**. Construction MERGEs into whatever is already there, and the retrieval stage
  will then profile those foreign labels as part of the graph.
- `graph_construction_agent_v1` was observed emitting a "Construction warnings" section that the
  construction tool did not produce — it re-labelled the schema critic's `feedback` text from two turns
  earlier as construction output. The same class of error §4 exists to prevent, one stage later.
- The construction warning heuristic only catches *under*-matching. An `ASSEMBLY_OF` rule with
  `rows: 64, rows_matched: 426` — 6.6× fan-out — passed silently.

---

## 4. Retrieval grounding

`graphrag_agent_v2` (shipped in 0.3.0, PR #4) is the newest and most carefully built subsystem. Its
premise: a retrieval agent cannot see the graph. Everything it "knows" arrives as text — the schema,
query results, and whatever else is in the conversation — and each of those channels can mislead it in a
different way, silently, because in every case the payload *looks* complete.

The organising rule is that **anything the system does not know is spelled out as a word, never left as
an absence**, because an omitted key reads to a model as "fine." Three mechanisms, three distinct
failure modes:

**Context filtering** (`common/adk_context.py`) — ADK rewrites another agent's output into a *user-role*
message carrying a literal `"For context:"` sentinel before any `before_model_callback` sees it. Role
alone therefore cannot distinguish a colleague agent's stale claim from something the human actually
typed. `drop_foreign_context` keys on the sentinel and strips those turns. Scope matters: it removes one
structurally-invisible contamination channel. It deliberately **keeps the agent's own** prior turns, and
it no-ops rather than emptying a request that is entirely foreign. Self-recall is addressed by prompt
instruction only — "query for it again instead" — not by enforcement.

**Schema profiling** (`common/graph_profile.py`) — the underlying library reports property values from
either an exhaustive scan or, above 10,000 rows, five arbitrary sampled values, and the two are
indistinguishable in its output. The profile refuses to launder that: when the distinct count is absent
the values are **dropped**, not passed through, and completeness is reported as `unknown`. Every
property carries always-present tri-state annotations — completeness, uniqueness, and a four-state
`numeric_like` that distinguishes bare numbers from values castable only after cleaning, because
`toFloat('$42.73')` returns null *silently* and aggregating over nulls yields a confident wrong number.
The profile is itself budgeted to 25 entities and 25 patterns, and what is cut is **marked**
`not_profiled` rather than omitted. It never raises; per-entity failures isolate to `profile_error`.

The subtlest thing it guards is **grain**: a relationship type spanning several `(start, end)` label
pairs has pooled statistics that describe no actual pattern, so degree is keyed per triple and each
`partitioned_by` entry carries a `distribution_covers` qualifier — `this_pattern` or
`all_patterns_of_this_type`.

**Result bounding** (`common/neo4j_for_adk.py`, via the `read_neo4j_cypher` wrapper) — the read path is
server-enforced read-only (`default_access_mode=READ_ACCESS`, never text inspection), 30-second timeout,
streams rather than materialises, retains 50 rows, and keeps *counting* to 100,000 before reporting
`row_count_at_least` rather than inventing a total it did not finish. Long lists — embedding vectors —
are replaced by a shape description. Crucially `truncated` (rows) and `values_summarised` (within a row)
are reported **separately**, so a row-complete result can never imply value-complete.

This bound covers reads only. The write path (`send_query`) is knowingly unbounded and eagerly
materialised, since bulk loads legitimately exceed 30s; `graphrag_agent_v2` holds no write tool and
cannot reach it.

These compose in `graphrag_agent`'s `variants.py`, which is the only place all three meet. Two details
that look like accidents and are not: the profiled schema payload deliberately discards the library's
raw property lists (passing both would let the raw copy assert exactly what the profile exists to deny,
and appear first), and the profile flag is exposed as two separate zero-argument tools rather than one
parameterised tool, because ADK would advertise the parameter as required and a model guessing `true`
would trigger a full scan per label on the latency-sensitive construction path.

`graphrag_agent_v1` is retained unchanged for A/B comparison. The comparison's outcome is not recorded
in this repository.

Guarantees here are about *payload shape*. Whether they change model behaviour is deliberately untested —
the end-to-end test asserts on what reached the model, never on what the model said.

---

## 5. Conventions

See `CONTRIBUTING.md` for the branch/PR workflow, testing expectations, and CHANGELOG conventions, and
`CLAUDE.md` for the architecture map. Only the things most likely to bite you are repeated here:

- **Two remotes.** `origin` is the working fork; `upstream` is `neo4j-contrib/agentic-kg` and has a live
  push URL. Work goes to `origin`. `gh` resolves to the wrong repo without an explicit `--repo`.
- **Design notes are local-only.** `.gitignore:26` and `:29` exclude `docs/superpowers/` and
  `docs/backlog/`. Documents there are absent from a fresh clone — do not cite those paths as if a
  reader can open them. Only those two subdirectories are excluded; the rest of `docs/`, including this
  file, is tracked normally and is where a durable, shared project doc belongs.
- **Tests.** `uv run pytest` defaults to `-m 'not integration'`, so it never touches Docker; integration
  tests are opt-in with `-m integration` and skip cleanly when no Docker daemon is reachable.
- No linter or formatter, and no CI, are configured.

---

## 6. Current state

**Shipped** (all PR numbers verified against the repository's merged PRs):

| Version | Date | What |
|---|---|---|
| 0.4.0 | 2026-08-03 | Contributor workflow: `CONTRIBUTING.md`, `CHANGELOG.md`, narrowed `.gitignore` (#5) |
| 0.3.0 | 2026-08-02 | Retrieval grounding — §4 (#4) |
| 0.2.1 | 2026-07-30 | `schema_refinement_loop` feedback-clobbering fix, per-turn invocation cap (#3) |
| 0.2.0 | 2026-07-29 | Foundation: `fsspec` file sources, driver-side CSV loading, OpenRouter + per-job models (#2) |
| 0.1.0 | 2026-07-26 | Test-suite fixes (#1) |

`## [Unreleased]` in `CHANGELOG.md` accumulates merged-but-unreleased changes; check it for anything
landed since 0.4.0.

**Next**, in order — designs are settled and recorded in local notes; specs are not yet written:

1. **Unstructured ingestion** — entity/fact-type agents, chunking, PDF and Markdown loaders, an
   extraction executor, resumability, and scoped resolution, generic rather than hardcoded to the
   bundled furniture example.
2. **Linking** — `CORRESPONDS_TO` correlation to reference tables, and cross-tier retrieval.

Both target SEC 10-K filings alongside `Company_Filings.csv` / `Asset_Manager_Holdings.csv`. **Those
files are not in this repo and must be sourced**, so this work is blocked on an external dependency. The
bundled furniture example must keep working throughout.

**Known rough edges**, verified in code rather than assumed:

- **Stale driver binding.** Five modules bind the `Neo4jForADK` singleton at import time.
  `close_graphdb()` clears the module global but leaves those references pointing at a closed driver
  with no reconnect path — so `neo4j_is_ready`, the one production caller that closes on a failed
  readiness check, permanently disables every Neo4j tool for the life of the process, and silently,
  since tool bodies return `tool_error`.
- **Everything is a string.** The CSV path writes every property as a Neo4j STRING with no type slot in
  the construction plan, so uncast comparisons silently sort lexicographically (`'9' > '30'`) — wrong
  answers rather than errors. §4's `numeric_like` annotation exists to make the retrieval agent cast.
- **`pyproject.toml:3` still reads `version = "0.1.0"`**, four releases behind, as does its mirror in
  `uv.lock:21`. These are the only machine-readable versions in the repo.
- Model configuration drifts from documentation: the models named in `CLAUDE.md` and `CHANGELOG.md` 0.4.0
  exist only in an untracked `.env`. A fresh clone runs on the `gpt-4o`/`gpt-4o-mini` defaults in
  `.env.example` and `common/config.py`.
