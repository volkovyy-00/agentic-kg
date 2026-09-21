# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

A multi-agent system, built on Google ADK (Agent Development Kit) with LiteLLM, that interviews a user, picks source
files, proposes a graph schema, and builds a knowledge graph in Neo4j.

Forked from the companion project to the deeplearning.ai short course "Agentic Knowledge Graph Construction", but
**this is being developed into a real program, not a teaching artifact.** Course-shaped structure that survives in
the code (the `variants` dicts of successive chapter iterations, the "Differences from the course" section below)
is inherited history, not a constraint to preserve — prefer the choice that makes a working program over the one
that mirrors the course. Reproducibility for students is not a design goal.

See `CONTRIBUTING.md` for the PR workflow, testing expectations, and CHANGELOG conventions to follow when
making changes here.

## Current work: unstructured ingestion (3 sub-projects)

Adding ingestion of unstructured documents (PDF/Markdown) alongside the existing CSV path, generic
rather than hardcoded to the bundled furniture example. Split into three sub-projects, built in order.
Each gets its own spec, plan and implementation cycle.

| | Scope | Status |
|---|---|---|
| **1. Foundation** | File sources via `fsspec`, driver-side CSV loading, OpenRouter + per-job models, `finished()` fix | **implemented** |
| **2. Unstructured ingestion** | Entity/fact-type agents, chunking, PDF+Markdown loaders, extraction executor, resumability, scoped resolution | design settled, **spec not written** |
| **3. Linking** | `CORRESPONDS_TO` correlation to reference tables; cross-tier retrieval | design settled, **spec not written** |

- Foundation spec: `docs/superpowers/specs/2026-07-27-foundation-design.md`
- Foundation plan: `docs/superpowers/plans/2026-07-27-foundation.md` (12 TDD tasks — self-contained, assumes no prior context)
- **Decisions for sub-projects 2 and 3: `docs/superpowers/specs/2026-07-27-unstructured-ingestion-decisions.md`**
- Foundation is merged to `main` (the `foundation-file-sources-and-models` branch is gone — deleted after merge); work continues directly on `main`.

All three documents above live under `docs/superpowers/`, which is **gitignored** — they exist in this working
copy, not in a fresh clone. `docs/spec.md` is the only tracked document under `docs/`.

**All design decisions for sub-projects 2 and 3 are already settled** — chunking, extraction context,
resumability, identity model, approval posture, models, and definition of done — and are recorded with their
reasoning in the decisions document above. Read it before writing spec 2 or 3; do not re-derive them. Technical
constraints found while fact-checking Foundation (PdfLoader's `fs`/path contract, missing
`langchain-text-splitters`, `LongRunningFunctionTool`'s falsy-return behaviour, APOC Core-only on Aura) live in
the Foundation spec's *Follow-on work* section.

Target dataset for 2 and 3 is SEC 10-K filings plus `Company_Filings.csv` / `Asset_Manager_Holdings.csv`.
**Those files are not in this repo and must be sourced.** The bundled furniture example must keep working throughout.

**Write specs 2 and 3 against the post-Foundation codebase** — which now also includes graphrag grounding
(below), merged after Foundation and before either spec was written.

### Interleaved and already shipped: graphrag grounding

Between Foundation and sub-projects 2/3, a separate effort — grounding `graphrag_agent` in the graph instead
of conversational recall — was designed, implemented, and merged as `0.3.0` (2026-08-02, PR #4). It is **not**
part of the 3-sub-project plan above; it jumped the queue. See Architecture's *`graphrag_agent_v2`* subsection
for what shipped, and [PR #4](https://github.com/volkovyy-00/agentic-kg/pull/4) / `CHANGELOG.md`'s `0.3.0`
entry for the design record — the underlying spec/plan are gitignored local notes, not something a fresh
clone has. Sub-projects 2 and 3 remain the actual next work and are still unstarted.

Much else has been interleaved since — handoff gates, Neo4j connection self-heal, schema/construction-plan
hardening, a CI/quality wave, dependency movement — none of it touching sub-projects 2/3. The latest
release is `0.6.1` (2026-09-18); `git log` and `CHANGELOG.md`'s `[Unreleased]` are the current state.
**`CHANGELOG.md` is the entry-by-entry record; the *Architecture* sections below carry the reasoning behind
whatever is still load-bearing.** Two pointers
worth having up front: `docs/spec.md` is the living "what is this and why" document (read it alongside this
file, not instead of it), and reachability is now judged by the values a node actually carries rather than by
which file built it (#52, 2026-09-19) — the change the current work builds on.

## Commands

```bash
# Setup
uv venv
uv sync
cp .env.example .env      # then set OPENROUTER_API_KEY and NEO4J_DSN

# Run the agent system (ADK dev web UI)
uv run adk web src/agentic_kg/coordinators/     # http://localhost:8000, add --port 8001 if busy

# Unit tests (fast, no external deps)
uv run pytest -q
uv run pytest tests/unit/test_pydantic_neo4j.py -v   # single file
uv run pytest tests/unit/test_tool_result.py::test_tool_success -v   # single test
uv run pytest --cov --cov-report=term-missing   # with coverage (CI sends coverage.xml to SonarCloud)

# Integration tests (require Docker; spins up Neo4j via Testcontainers; ~12 min, function-scoped containers)
uv run pytest -q -m integration

# Lint / type check (both gate CI)
uv run ruff check . && uv run ruff format --check .
uv run pyright        # must report 0 errors
```

- Python 3.12, dependency/venv management via `uv` (see `pyproject.toml`, `uv.lock`).
- Pinned to `google-adk>=1.28.1,<2` (`pyproject.toml`; the floor is the CVE-2026-4810 fix) — ADK 2.x is a
  breaking rewrite; check which major version any ADK doc, sample, or blog post is describing before trusting
  it against this code.
- The floor is not what you run: the committed `uv.lock` resolves `google-adk 1.39.1` (and `neo4j 6.3.1`),
  so `uv sync` installs those. Check the lock, not `pyproject.toml`, when a behaviour looks version-dependent.
- `pytest` defaults to `-m 'not integration'` (see `[tool.pytest.ini_options]` in `pyproject.toml`), so plain
  `pytest`/`uv run pytest` never touches Docker.
- `tests/integration/conftest.py` has two fixtures: `neo4j_graph` (plain container) and `neo4j_graph_with_apoc`.
  Use the APOC one for anything touching physical/profiled schema — `neo4j_graphrag.get_structured_schema` is
  APOC-only (`apoc.meta.data`/`apoc.meta.graph`), and a stock `neo4j:5` image doesn't have it.
- If using colima instead of Docker Desktop, integration tests need:
  `export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock` and `export TESTCONTAINERS_RYUK_DISABLED=true`.
- Ruff config is `pyproject.toml`'s `[tool.ruff]`; pyright's is `[tool.pyright]` (`basic` mode, `src` only).
  Drop `--check` from `ruff format` to fix locally.
- Two remotes are configured: `origin` (`volkovyy-00/agentic-kg`) and `upstream`
  (`neo4j-contrib/agentic-kg`, the repo this was forked from). No `gh` default repo is set
  (`gh repo set-default --view` reports none), so an unqualified `gh` command can resolve against
  either — always pass `--repo volkovyy-00/agentic-kg` explicitly to keep it targeting this fork.
- Source files are read by the application itself (via `fsspec`, `common/file_source.py`), not by the database, so
  nothing needs to be copied into a Neo4j import directory — this also works unchanged against Neo4j Aura, which
  has no such directory. Point `SOURCE_URI` in `.env` at a folder of source files; the bundled example works with
  `SOURCE_URI=./data/bom` (ask the running agent "Where are my files?" to confirm what it resolved to).

## Architecture

### Layout

- `src/agentic_kg/` — `coordinators/{single_agent,multi_agent/sub_agents}` (what `adk web` loads) · `agents/`
  (standalone `cypher_agent`, `user_intent_agent`) · `tools/` (ADK tool functions) · `common/` (Neo4j, LLM, file
  sources, ADK callbacks) · `domain/` (typed shapes)
- `tests/unit`, `tests/integration` (`integration` marker, Testcontainers) · `data/bom` (bundled example)
- `prototype/` — notebooks and notes, not part of the package

### Two coordinators, one shared tool/agent library

`adk web src/agentic_kg/coordinators/` discovers two independent top-level agents ("coordinators"):

- **`single_agent`** (`coordinators/single_agent/`) — one agent that talks to Neo4j directly via Cypher, delegating
  to `agents/cypher_agent` as a sub-agent for query execution.
- **`multi_agent`** (`coordinators/multi_agent/`) — a hierarchical `LlmAgent` (`full_workflow_agent`),
  registered as `kg_construction_agent_v1` (`MULTI_AGENT_COORDINATOR` in `common/agent_names.py` — this is the
  name to use when polling the ADK API, per the debugging steps below), that delegates,
  in strict sequence, through five sub-agents defined in `coordinators/multi_agent/sub_agents/`:
  1. `user_intent_agent` — establishes `kind_of_graph` / `graph_description`
  2. `file_suggestion_agent` — requires an approved user goal; suggests input files
  3. `schema_proposal_agent` — requires approved file suggestions; proposes a construction plan
  4. `graph_construction_agent` — requires an approved schema; builds the graph
  5. `graphrag_agent` — only usable once `get_physical_schema` shows the graph exists; answers questions over it

When a turn in the dev UI produces no visible response and no spinner, the UI alone can't tell you why (hung
tool call, routing bug, and swallowed exception all look identical from the browser). Cheapest checks first:
poll `GET /apps/{app}/users/{user}/sessions/{id}` directly (frozen event count = nothing happened), then the
undocumented `GET /debug/trace/session/{id}` (spans have `start_time`/`end_time`, but a call that raises never
gets a span — telemetry only fires on success), then the `adk web` server's own stdout, which is the only
place a swallowed exception actually surfaces. Never reload the tab while a turn is genuinely streaming.

Note there are **two separate implementations of similarly-named agents**: `src/agentic_kg/agents/` (standalone
versions, e.g. `cypher_agent` — the one actually wired into `single_agent` — plus `user_intent_agent`, which is not
used by either coordinator, but is still imported by `src/agentic_kg/agent.py` — an orphaned top-level `root_agent`
left over from the original course backport, not reachable via the documented `adk web` command and not part of
either coordinator) vs. `src/agentic_kg/coordinators/multi_agent/sub_agents/` (versions wired
into the full workflow, with richer instructions/tools). They are not interchangeable — check which coordinator
you're editing before reusing code between them. `agents/file_suggestion_agent/` used to be a third standalone
implementation here; Foundation deleted it (see the *variants* section below).

### The `variants` pattern

Every agent's prompt/tool wiring lives in a sibling `variants.py`, not in `agent.py`. Each `variants.py` defines a
`variants` dict keyed by version-suffixed agent names (e.g. `graphrag_agent_v1/_v2`), each holding an
`instruction` string and a `tools` list — these are successive course-chapter iterations of the same agent, growing
more capable (more tools, more validation) at each version. `agent.py` just picks one:

```python
AGENT_NAME = "graphrag_agent_v2"
Agent(name=AGENT_NAME, instruction=variants[AGENT_NAME]["instruction"], tools=variants[AGENT_NAME]["tools"], ...)
```

When adding a new capability to an agent, prefer adding a new numbered variant (or editing the currently-selected
one) over restructuring this dict shape — it mirrors the course's progressive-exercise structure. Some
`variants.py` files reference tools/names that aren't imported into that file (leftover from course scaffolding) —
if you hit a `NameError` there, check whether the referenced symbol exists elsewhere in `tools/` and add the import
rather than assuming the whole file is broken. That advice does not apply to the standalone
`agents/file_suggestion_agent/`: it was unreachable and referenced eight undefined names, and Foundation removed the
directory outright rather than patching it — if you're looking for it, only the `multi_agent` implementation at
`coordinators/multi_agent/sub_agents/file_suggestion_agent/` exists now.

### State passing: ADK session state, not return values

Agents/tools do not pass data through Python return values between sub-agents — they read/write keys on
`tool_context.state` (Google ADK's session state), e.g. `user_goal`, `approved_user_goal`, `suggested_files`,
`approved_construction_plan`. Tools follow a consistent get/set/approve naming convention per concept (e.g.
`set_perceived_user_goal` → `approve_perceived_user_goal` → `get_approved_user_goal` in
`tools/user_goal_tools.py`), and later agents' tools typically fail fast with a `tool_error(...)` if an earlier
stage's state key is missing — that's how the "requires approved X" sequencing in the coordinator instructions is
actually enforced. When tracing a bug across agents, look at which state keys a tool reads/writes before assuming
control flow is the issue.

`schema_proposal_agent`'s `schema_refinement_calls_this_turn` state key caps `schema_refinement_loop` to one
invocation per user turn (deliberate): `reset_schema_refinement_turn_budget` (coordinator
`before_agent_callback`) zeroes it, `prepare_refinement_loop_invocation` (`refinement_loop`
`before_agent_callback`) increments/checks it and short-circuits a second call with a result beginning `"stopped:"`.

**Plan checks: one rule set, two enforcement points.** `check_construction_plan_consistency` (joins, endpoint
labels, typed columns) and `check_reference_columns_are_reachable` (every approved file's reference columns can
still be reached) run:

- **At approval** — in `approve_proposed_construction_plan`, and in the read tool the current variant uses,
  `get_proposed_construction_plan_with_approval_check` (`tools/construction_plan_tools.py`), through a shared
  `_read_plan_for_approval`, so what the coordinator presents as approvable can't drift from what approval will
  do. The plain `get_proposed_construction_plan` survives only because the older `v1`/`v2` variants still call
  it. The read tool's error branch still returns the plan alongside the problems; its success branch says only
  that those checks passed, not that it's the *right* plan — accepting remaining critic objections is the
  user's call.
- **Inside `schema_refinement_loop`** — its `StopChecker` runs both through `find_plan_problems(state)`, and a
  plan with either problem goes back for another iteration. The stop-check writes a `retry` composite to
  `feedback` via `state_delta`, never by mutating state (`AgentTool` forwards only the delta out of the loop's
  child session). The loop runs at most two iterations, so a problem the second revision introduces still
  surfaces at approval.

Approval is the guarantee: the loop's copy is fail-open behind one guard, approval propagates a crashed check so
it fails closed, and reachability itself fails open on an unreadable file (a `not_verified` note, never a
refusal). Do not move the check into a critic-side tool: it would depend on the model choosing to call it, and
the check is mechanical precisely because the prose rule it replaced resolved the same file two different ways
on two runs. Approval framing also stays out of the critic's context.

### Handoff confirmation gates

Three phase exits are gated so a `finished` transfer needs more than the model's own reading of the
conversation. Two use a per-turn flag; the third deliberately does not — read its entry before assuming a fourth
gate should copy either shape.

- **Construction → retrieval**: `graph_construction_agent`'s `finished` refuses until `HANDOFF_CONFIRMED_KEY`
  (`tools/construction_handoff_tools.py`) is set by an explicit `confirm_construction_handoff` call — never
  inferred from tone. `reset_construction_handoff_confirmation` (`before_agent_callback`) clears it every turn.
  On confirmation, transfer goes directly to `graphrag_agent_v2`, not back through the coordinator (the numbered
  sequence above simplifies this step).
- **Retrieval → coordinator**: the same shape on `graphrag_agent_v2` (`GRAPHRAG_HANDOFF_CONFIRMED_KEY`,
  `confirm_graphrag_handoff`, `reset_graphrag_handoff_confirmation` in `tools/graphrag_handoff_tools.py`), so it
  stays in retrieval across several questions. Deliberately not factored into a helper shared with the
  construction gate: each `finished`'s **docstring is the model-visible tool description** (ADK reads
  `__doc__`), and the two legitimately say different things (hand the user to retrieval vs. hand them back), which
  a shared factory would have to synthesise. `graphrag_agent_v1` has no gate and keeps its single-answer-then-eject
  behavior, for the A/B comparison below.
- **Intent → coordinator**: `user_intent_agent_v2`'s `finished` refuses until `approved_user_goal` is present
  **and equal to** `perceived_user_goal` (both written by `tools/user_goal_tools.py`). No new state key, tool,
  reset callback, or `before_agent_callback` — that is the point. The other gates guard something the user *said*
  (turn-scoped, goes stale, needs a flag and a reset); this one guards something the user *did*, already recorded
  durably. It is **not copy #3** of the flag/reset/confirm shape — do not factor the three together. Equality
  rather than presence, because a goal approved and then revised leaves an approved key that no longer describes
  what was asked. `finished` branches three ways (nothing recorded / never approved / stale since approval); the
  messages are read by the model, not the user, and each names the next tool to call — there is no escape hatch.
  The shared `finished` was split for this: `_transfer_to_coordinator` (ungated, `user_intent_agent_v1`'s; its
  `__name__` must stay `"finished"` since ADK derives the tool name from it) vs. the gated `finished` (v2's). v1
  uses `set_user_goal` and never writes `approved_user_goal`, so gating in place would leave it unable to exit.

**The `transfer_to_agent` bypass.** ADK injects a `transfer_to_agent` tool (plus an advertising instruction
block) into every sub-agent with a parent or peers, and it never consulted the gates. `graph_construction_agent`,
`graphrag_agent_v2` and `user_intent_agent_v2` (never `_v1`) therefore run `strip_transfer_to_agent`
(`common/adk_transfer.py`) as a `before_model_callback`, removing the tool from `tools_dict`, `config.tools` and
the system instruction. `disallow_transfer_to_parent` was avoided because it also kills phase stickiness
(`Runner._find_agent_to_run` would re-arbitrate every message through the coordinator). Instruction-block removal
matches two marker phrases; if a `google-adk` upgrade changes ADK's wording, `_without_transfer_block` logs a
warning rather than failing — check logs after any ADK bump.

**Always pair the strip with `drop_foreign_context`**: `before_model_callback=[drop_foreign_context,
strip_transfer_to_agent]`. Each stripped agent is entered by someone else's `transfer_to_agent` call, which ADK
rewrites into a `"For context: [kg_construction_agent_v1] called tool transfer_to_agent…"` turn — a worked
example of the call the model then copies, after the strip already removed the tool from `tools_dict`. ADK then
raises `ValueError` mid-turn: a dead turn with no response and no spinner (the swallowed-exception mode; debug
from the `adk web` stdout, per the *Two coordinators* section). `user_intent_agent` is the most exposed, since the
interview is the stickiest phase. Only the coordinator lacks `drop_foreign_context`, by design: its transfer tool
is never stripped, since that is how the workflow advances.

### Tool results

All tools return a `ToolResult` (`common/tool_result.py`): `{"status": "success", <key>: value}` or
`{"status": "error", "error_message": str}`. Use `tool_success(key, value)` / `tool_error(msg)` to construct these,
and `is_success`/`is_error`/`get_or_else`/`get_or_raise`/`map_result` to consume them — don't invent ad hoc dict
shapes for new tools.

### Neo4j access

All Cypher execution goes through the `Neo4jForADK` singleton (`common/neo4j_for_adk.get_graphdb()`), which wraps
the driver and returns `ToolResult`s via `result_to_adk`. The singleton's identity is permanent — the five
`graphdb = get_graphdb()` bindings taken at import time (one per module) stay valid forever, including across a
`close_graphdb()` or a transient outage, because `_ensure_connected()` transparently rebuilds the driver on next
use rather than requiring callers to re-fetch the singleton. Since `neo4j` 6.x, using a closed `Driver` raises
`DriverError("Driver closed")` (5.x only warned), so a call site that skips the heal fails loudly — but only when
its call is the *first* after a close; once any other path has healed, a regressed one runs on the healthy driver
and passes silently. Every entry point that hands out or uses the driver must therefore go through the heal, and a
new one needs its own step in `tests/integration/test_connection_recovery.py`, which documents how it covers them
and which heals it cannot reach. Config comes from `Neo4jDsn`/`Neo4jConfig`
(`common/pydantic_neo4j.py`), parsed from the `NEO4J_DSN` env var (local `bolt://` and Aura `neo4j+s://` DSNs both
work — see `.env.example` for the full list of allowed schemes). `tools/cypher_tools.py` builds on this for
higher-level operations like `get_physical_schema`, `create_uniqueness_constraint`, `neo4j_is_ready`. There is no
Neo4j import directory to manage: `tools/kg_construction_tools.py` reads CSVs client-side (via `common/file_source.py`
and `common/csv_reader.py`) and loads rows with parameterised `UNWIND` batches, since Aura forbids
`LOAD CSV FROM "file:///"`.

A construction rule may carry `property_types` (`{property_name: "integer"|"float"|"boolean"}`);
`common/value_types.py` converts those values in Python before the batch is sent, and the loaders'
Cypher gains two `FOREACH` passes for them — one writing converted values, one clearing values that
were blank or unreadable via a sentinel, since Cypher cannot distinguish a failed parse from a
ragged row's absent key. A typed column failing on more than half a batch's non-blank values stops
that rule with an error rather than half-typing the property — but only once at least
`TYPE_FAILURE_MIN_SAMPLE` (2) values are present, since one bad value out of one row cannot tell a
wrong type declaration from a single data-entry typo. Identifiers and any column a
relationship joins on may not be typed; `check_construction_plan_consistency` refuses such a plan at
approval time.

Labels and relationship types, which Cypher cannot parameterise, are validated with
`common/cypher_identifiers.checked()` and then interpolated into the query text — never passed as Cypher `$()`
dynamic labels, which cannot use a uniqueness index. The loaders' `ToolResult`s include `nodes_in_graph` /
`relationships_in_graph`, real `MATCH...count()` reads (not the row count `MERGE` was handed, which can
collapse duplicates) — but these counts are label/type-wide, not scoped to the rows the current call just
wrote, so a re-run against a non-empty graph will include prior data too.

### Grounding: `graphrag_agent_v2`

`graphrag_agent_v2` (shipped as `0.3.0`, PR #4) answers only from graph queries made in the current turn, not
from conversational recall. Three pieces make that possible:

- `common/adk_context.py`: `drop_foreign_context`, a `before_model_callback` that strips other agents' turns
  from the request. ADK rewrites another agent's output into a user-role message carrying a `"For context:"`
  sentinel before this callback ever sees it, so role alone can't distinguish it from a real user turn — the
  filter keys on the sentinel instead.
- `common/graph_profile.py`: turns `neo4j_graphrag`'s enriched schema into tri-state, always-present
  annotations (completeness, uniqueness, per-pattern degree, per-value distribution), cached via
  `get_cached_profile` — because the library's own report doesn't say whether a sampled property list is
  exhaustive or not.
- `tools/cypher_tools.py`: `get_physical_schema()` (no profile — output must stay byte-identical to before,
  since the coordinator, `graph_construction_agent`, and `single_agent`'s `cypher_agent` all depend on that
  exact shape) vs. `get_graph_schema_with_profile()` (adds the profile), both built on the shared
  `_physical_schema(include_data_profile: bool)`.

`graphrag_agent_v1` is kept unchanged alongside v2 for an A/B comparison; `agent.py` selects v2 via
`AGENT_NAME`. Design record: [PR #4](https://github.com/volkovyy-00/agentic-kg/pull/4) / `CHANGELOG.md`'s
`0.3.0` entry — the underlying spec/plan are gitignored local notes, not present in a fresh clone.

### LLM selection

`common/llm_catalog.get_llm(kind: LlmKind)` returns a lazily-constructed `LiteLlm` instance, cached per `LlmKind`
(`LlmKind.reasoning` / `LlmKind.conversational`) in a `dict`, so each kind gets its own instance instead of one call
site's model choice winning for the whole process. Every model runs through OpenRouter: settings hold the model name
in OpenRouter's spelling (`llm_model_conversational` / `llm_model_reasoning`, e.g. `"openai/gpt-4o"`), and
`_model_name()` derives the `"openrouter/"` prefix LiteLLM needs rather than having it configured separately.
Swapping a model means editing `LLM_MODEL_CONVERSATIONAL` / `LLM_MODEL_REASONING` in `.env`, not code.

Models in the maintainer's local, gitignored `.env` (the code default and `.env.example` are `openai/gpt-4o` /
`openai/gpt-4o-mini`): reasoning = `openai/gpt-5.6-luna`, conversational = `deepseek/deepseek-v4-flash-0731` (DeepSeek's
official V4-Flash release, 2026-07-31, superseding the preview build previously pinned here). The reasoning
slot moved off `openai/gpt-5` (2026-07-31) because its workload — `schema_proposal_agent`'s propose/critique/refine
trio and `graph_construction_agent` — is many small tool-orchestration steps at `reasoning_effort="low"`, which is
exactly Luna's target profile, at ~1/16th the output cost. LiteLLM has no `openrouter/openai/gpt-5.6-luna` entry in
`model_cost`, so its own cost estimate is 0 for this model; OpenRouter returns real `cost` / `cost_details` on the
response instead, which is what to read if you add cost tracking.

`get_llm()` also caps `max_tokens` at 8192: with no cap, OpenRouter pre-authorizes the full token ceiling
(e.g. ~$0.66 for a 65536-token `gpt-5` call) against account balance before the call runs. If that pre-auth
exceeds the balance, the call gets a 402 that ADK's dev UI shows as an indistinguishable hang — no spinner, no
error, no trace span, since telemetry only fires on a successful response. If reasoning-model calls silently
stop working, check account balance and the `adk web` server's own stdout (it logs the real exception) before
assuming a code regression.

### Domain models

Typed domain shapes (e.g. `UserIntent` in `domain/user_intent.py`) are `TypedDict` + a `pydantic.TypeAdapter` for
runtime validation (`validate_user_intent`, `is_valid_user_intent`), rather than full Pydantic `BaseModel`s — follow
that pattern for new domain types.

## Differences from the deeplearning.ai course

- Many agents use a `finished` tool (`tools/adk_tools.py`) to explicitly signal completion and transfer control back
  to the parent agent, rather than relying on implicit turn-ending.
