# CLAUDE.md

A multi-agent system, built on Google ADK with LiteLLM, that interviews a user, picks source files,
proposes a graph schema and builds a knowledge graph in Neo4j. Forked from the deeplearning.ai course
"Agentic Knowledge Graph Construction", but **developed into a real program**: course-shaped structure
(the `variants` dicts) is history, not a constraint — prefer the choice that makes a working program.
The bundled example (`data/bom/`) must keep working, but it is one dataset, not the scope: designs and
fixes must hold for source files the program has never seen.

## Where knowledge lives

This file holds what every session needs, one bullet or short paragraph per rule — no status or ticket keys. Rules
for one area live in `.claude/rules/` and load when you read a matching file; why code is the way it
is lives in the docstring at that code. `CONTRIBUTING.md` (*Where project knowledge lives*) lists every
home; humans start at `docs/spec.md`.

## Before you change

- `google-adk` is pinned to one minor window (`pyproject.toml`). ADK docs, samples and posts describe
  1.x, or a 2.x newer than ours, as often as ours: check which line a source describes before trusting
  it, and check `uv.lock` for the resolved version when behaviour looks version-dependent.
- After any `google-adk` bump, read the `adk web` server log for transfer-block warnings.
- An integration run that reports "skipped" did not run: read "N passed", not "N skipped".
- Before adding a gated agent or a gate (the flag/reset/confirm shape), read
  `.claude/rules/handoff-gates.md`.
- Before adding a tool that writes construction-plan rules, read `.claude/rules/construction-plan.md`.
- Before adding a Neo4j driver entry point, read `.claude/rules/neo4j-access.md`.
- To debug a silent or failing `adk web` turn, read `.claude/skills/debug-adk-web/SKILL.md`.

## Commands

```bash
uv venv && uv sync
cp .env.example .env      # then set OPENROUTER_API_KEY and NEO4J_DSN

uv run adk web src/agentic_kg/coordinators/     # dev UI on http://localhost:8000; --port 8001 if busy

uv run pytest -q                                 # unit tests; never touches Docker
uv run pytest tests/unit/test_tool_result.py::test_tool_success -v   # one test
uv run pytest --cov --cov-report=term-missing    # coverage (CI sends coverage.xml to SonarCloud)
uv run pytest -q -m integration                  # Neo4j via Testcontainers; needs Docker, ~12 min

uv run ruff check . && uv run ruff format --check .   # both gate CI; drop --check to fix
uv run pyright                                   # src only; must report 0 errors
```

- Python 3.12, dependencies via `uv`. Plain `pytest` runs `-m 'not integration'` (`pyproject.toml`),
  which also holds the Ruff and pyright config.
- With colima instead of Docker Desktop, integration tests need
  `export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock` and `export TESTCONTAINERS_RYUK_DISABLED=true`.
- This repo is a GitHub fork of `neo4j-contrib/agentic-kg`, so `gh` targets the parent unless
  `gh repo set-default volkovyy-00/agentic-kg` has run in this clone (check: `gh repo set-default --view`);
  otherwise pass `--repo volkovyy-00/agentic-kg`.
- The application reads source files itself (`fsspec`, `src/agentic_kg/common/file_source.py`), so
  there is no Neo4j import directory and Aura works unchanged. `SOURCE_URI` in `.env` names the folder;
  the example uses `SOURCE_URI=./data/bom` (ask the running agent "Where are my files?" to confirm).

## Architecture

- `src/agentic_kg/coordinators/` — what `adk web` loads: `single_agent` and `multi_agent`
- `src/agentic_kg/agents/` — standalone `cypher_agent` and `user_intent_agent`
- `src/agentic_kg/tools/` — ADK tool functions; `src/agentic_kg/common/` — Neo4j, LLM, file sources,
  ADK callbacks; `src/agentic_kg/domain/` — typed shapes
- `tests/unit/`, `tests/integration/` (`integration` marker, Testcontainers)

### Two coordinators

- **`single_agent`** — one agent that queries Neo4j through Cypher, delegating execution to the
  standalone `src/agentic_kg/agents/cypher_agent/`.
- **`multi_agent`** — `full_workflow_agent`, registered as `kg_construction_agent_v1`
  (`MULTI_AGENT_COORDINATOR` in `src/agentic_kg/common/agent_names.py`), delegates in sequence through
  five sub-agents in `src/agentic_kg/coordinators/multi_agent/sub_agents/`:
  1. `user_intent_agent` — establishes `kind_of_graph` / `graph_description`
  2. `file_suggestion_agent` — needs an approved user goal; suggests input files
  3. `schema_proposal_agent` — needs approved files; proposes a construction plan
  4. `graph_construction_agent` — needs an approved plan; builds the graph, then on the user's
     confirmation hands them straight to `graphrag_agent_v2`, not back through the coordinator
  5. `graphrag_agent` — answers questions over the built graph

**Two sets of similarly named agents.** `src/agentic_kg/agents/` (standalone; its `cypher_agent` is the
one `single_agent` uses) and the workflow's sub-agents are not interchangeable — check which coordinator
you are editing before reusing code. `src/agentic_kg/agent.py` is a third root, outside both
coordinators; see its docstring.

### The `variants` pattern

Every agent's prompt and tool wiring lives in a sibling `variants.py`: a `variants` dict keyed by
version-suffixed names (`graphrag_agent_v1`, `graphrag_agent_v2`), each holding an `instruction` and a
`tools` list. `agent.py` selects one through `AGENT_NAME`. Add a capability to the **selected** variant;
add a new numbered one only when an A/B comparison is wanted. Keep the dict shape.

### State, tool results, Neo4j, models

- Agents pass data through ADK session state (`tool_context.state`), not return values. Tools follow
  get/set/approve naming per concept (`set_perceived_user_goal` → `approve_perceived_user_goal` →
  `get_approved_user_goal` in `src/agentic_kg/tools/user_goal_tools.py`), and a later stage's tool fails
  fast with `tool_error(...)` when an earlier key is missing — that is how "requires approved X" is
  enforced. When a bug crosses agents, check which state keys each tool reads and writes first.
- A new tool returns a `ToolResult` (`src/agentic_kg/common/tool_result.py`), never an ad hoc dict.
- All Cypher goes through the `get_graphdb()` singleton (`src/agentic_kg/common/neo4j_for_adk.py`);
  `NEO4J_DSN` takes a local `bolt://` or an Aura `neo4j+s://` DSN.
- `get_llm(kind)` (`src/agentic_kg/common/llm_catalog.py`) returns one cached LiteLLM instance per
  `LlmKind`, routed through OpenRouter. Change models in `.env` (`LLM_MODEL_CONVERSATIONAL`,
  `LLM_MODEL_REASONING`), not in code.

<!-- Budget: 150 lines, enforced by tests/unit/test_agent_context.py. The trap-bearing architecture map
     stays on purpose, whatever /doctor's trim check proposes. -->
