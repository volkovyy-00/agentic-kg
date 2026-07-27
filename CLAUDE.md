# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Companion project to the deeplearning.ai short course "Agentic Knowledge Graph Construction". It's a multi-agent
system, built on Google ADK (Agent Development Kit) with LiteLLM, that interviews a user, picks source files, proposes
a graph schema, and builds a knowledge graph in Neo4j. It is a reference/teaching implementation, not production code.

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
- Branch: `foundation-file-sources-and-models`

**All design decisions for sub-projects 2 and 3 are already settled** — chunking, extraction context,
resumability, identity model, approval posture, models, and definition of done — and are recorded with their
reasoning in the decisions document above. Read it before writing spec 2 or 3; do not re-derive them. Technical
constraints found while fact-checking Foundation (PdfLoader's `fs`/path contract, missing
`langchain-text-splitters`, `LongRunningFunctionTool`'s falsy-return behaviour, APOC Core-only on Aura) live in
the Foundation spec's *Follow-on work* section.

Target dataset for 2 and 3 is SEC 10-K filings plus `Company_Filings.csv` / `Asset_Manager_Holdings.csv`.
**Those files are not in this repo and must be sourced.** The bundled furniture example must keep working throughout.

**Write specs 2 and 3 against the post-Foundation codebase**, not against descriptions written before it landed.

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

# Integration tests (require Docker; spins up Neo4j via Testcontainers)
uv run pytest -q -m integration
```

- Python 3.12, dependency/venv management via `uv` (see `pyproject.toml`, `uv.lock`).
- `pytest` defaults to `-m 'not integration'` (see `[tool.pytest.ini_options]` in `pyproject.toml`), so plain
  `pytest`/`uv run pytest` never touches Docker.
- If using colima instead of Docker Desktop, integration tests need:
  `export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock` and `export TESTCONTAINERS_RYUK_DISABLED=true`.
- No linter/formatter is configured in this project.
- Source files are read by the application itself (via `fsspec`, `common/file_source.py`), not by the database, so
  nothing needs to be copied into a Neo4j import directory — this also works unchanged against Neo4j Aura, which
  has no such directory. Point `SOURCE_URI` in `.env` at a folder of source files; the bundled example works with
  `SOURCE_URI=./data/bom` (ask the running agent "Where are my files?" to confirm what it resolved to).

## Architecture

### Two coordinators, one shared tool/agent library

`adk web src/agentic_kg/coordinators/` discovers two independent top-level agents ("coordinators"):

- **`single_agent`** (`coordinators/single_agent/`) — one agent that talks to Neo4j directly via Cypher, delegating
  to `agents/cypher_agent` as a sub-agent for query execution.
- **`multi_agent`** (`coordinators/multi_agent/`) — a hierarchical `LlmAgent` (`full_workflow_agent`) that delegates,
  in strict sequence, through five sub-agents defined in `coordinators/multi_agent/sub_agents/`:
  1. `user_intent_agent` — establishes `kind_of_graph` / `graph_description`
  2. `file_suggestion_agent` — requires an approved user goal; suggests input files
  3. `schema_proposal_agent` — requires approved file suggestions; proposes a construction plan
  4. `graph_construction_agent` — requires an approved schema; builds the graph
  5. `graphrag_agent` — only usable once `get_physical_schema` shows the graph exists; answers questions over it

Note there are **two separate implementations of similarly-named agents**: `src/agentic_kg/agents/` (standalone
versions, e.g. `cypher_agent` — the one actually wired into `single_agent` — plus `user_intent_agent`, which is not
currently used by either coordinator) vs. `src/agentic_kg/coordinators/multi_agent/sub_agents/` (versions wired
into the full workflow, with richer instructions/tools). They are not interchangeable — check which coordinator
you're editing before reusing code between them. `agents/file_suggestion_agent/` used to be a third standalone
implementation here; Foundation deleted it (see the *variants* section below).

### The `variants` pattern

Every agent's prompt/tool wiring lives in a sibling `variants.py`, not in `agent.py`. Each `variants.py` defines a
`variants` dict keyed by version-suffixed agent names (e.g. `file_suggestion_agent_v1/_v2/_v3`), each holding an
`instruction` string and a `tools` list — these are successive course-chapter iterations of the same agent, growing
more capable (more tools, more validation) at each version. `agent.py` just picks one:

```python
AGENT_NAME = "file_suggestion_agent_v3"
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

### Tool results

All tools return a `ToolResult` (`common/tool_result.py`): `{"status": "success", <key>: value}` or
`{"status": "error", "error_message": str}`. Use `tool_success(key, value)` / `tool_error(msg)` to construct these,
and `is_success`/`is_error`/`get_or_else`/`get_or_raise`/`map_result` to consume them — don't invent ad hoc dict
shapes for new tools.

### Neo4j access

All Cypher execution goes through the `Neo4jForADK` singleton (`common/neo4j_for_adk.get_graphdb()`), which wraps
the driver and returns `ToolResult`s via `result_to_adk`. Config comes from `Neo4jDsn`/`Neo4jConfig`
(`common/pydantic_neo4j.py`), parsed from the `NEO4J_DSN` env var (local `bolt://` and Aura `neo4j+s://` DSNs both
work — see `.env.example` for the full list of allowed schemes). `tools/cypher_tools.py` builds on this for
higher-level operations like `get_physical_schema`, `create_uniqueness_constraint`, `neo4j_is_ready`. There is no
Neo4j import directory to manage: `tools/kg_construction_tools.py` reads CSVs client-side (via `common/file_source.py`
and `common/csv_reader.py`) and loads rows with parameterised `UNWIND` batches, since Aura forbids
`LOAD CSV FROM "file:///"`. Labels and relationship types, which Cypher cannot parameterise, are validated with
`common/cypher_identifiers.checked()` and then interpolated into the query text — never passed as Cypher `$()`
dynamic labels, which cannot use a uniqueness index.

### LLM selection

`common/llm_catalog.get_llm(kind: LlmKind)` returns a lazily-constructed `LiteLlm` instance, cached per `LlmKind`
(`LlmKind.reasoning` / `LlmKind.conversational`) in a `dict`, so each kind gets its own instance instead of one call
site's model choice winning for the whole process. Every model runs through OpenRouter: settings hold the model name
in OpenRouter's spelling (`llm_model_conversational` / `llm_model_reasoning`, e.g. `"openai/gpt-4o"`), and
`_model_name()` derives the `"openrouter/"` prefix LiteLLM needs rather than having it configured separately.
Swapping a model means editing `LLM_MODEL_CONVERSATIONAL` / `LLM_MODEL_REASONING` in `.env`, not code.

### Domain models

Typed domain shapes (e.g. `UserIntent` in `domain/user_intent.py`) are `TypedDict` + a `pydantic.TypeAdapter` for
runtime validation (`validate_user_intent`, `is_valid_user_intent`), rather than full Pydantic `BaseModel`s — follow
that pattern for new domain types.

## Differences from the deeplearning.ai course

- Many agents use a `finished` tool (`tools/adk_tools.py`) to explicitly signal completion and transfer control back
  to the parent agent, rather than relying on implicit turn-ending.
