# Agentic Knowledge Graph Construction

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Built with Google ADK](https://img.shields.io/badge/built%20with-Google%20ADK-orange)

A multi-agent system that turns a folder of source files into a Neo4j knowledge graph, then answers
questions over it — no Cypher required from the user. Built on [Google ADK](https://github.com/google/adk-python)
with LiteLLM routing model calls through OpenRouter, talking to Neo4j (local or Aura) over Bolt.

Originally forked from the companion project to deeplearning.ai's *Agentic Knowledge Graph Construction*
course; developed independently since.

## Features

- A chain of agents interviews the user for a goal, picks input files, proposes a graph schema, builds
  the graph, and answers questions about it — pausing for explicit human approval between each stage.
- Works against local Neo4j or Neo4j Aura: source files are read by the application (via `fsspec` — local
  paths, `s3://`, `https://`) and loaded with parameterised queries, so there's no Cypher `LOAD CSV` /
  import-directory dependency.
- Retrieval grounding for the question-answering stage: schema profiling, stale-context filtering, and
  bounded reads, so it can't be misled by incomplete or stale context that merely *looks* complete.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12.

```bash
uv venv
uv sync
cp .env.example .env
```

Then edit `.env`:

- `OPENROUTER_API_KEY` — required, one key covers every model call.
- `NEO4J_DSN` — `bolt://neo4j:secret@localhost:7687/neo4j` for a local instance, or
  `neo4j+s://user:password@xxxxxxxx.databases.neo4j.io` for Aura.
- `SOURCE_URI` — where your source files live. The bundled example works out of the box:
  `SOURCE_URI=./data/bom` (relative paths resolve against the repo root, not your shell's cwd).

## Run the web UI

Google ADK ships a dev web interface that discovers agents automatically:

```bash
uv run adk web src/agentic_kg/coordinators/
```

Open `http://localhost:8000` (pass `--port 8001` if that's busy). Two coordinators are discovered:

- **`multi_agent`** — the product. A five-stage pipeline: `user_intent` → `file_suggestion` →
  `schema_proposal` → `graph_construction` → `graphrag`, with an approval gate between every stage.
- **`single_agent`** — a frozen exercise carried over from the original course: a single agent that talks
  to Neo4j directly via Cypher. Useful for ad-hoc queries; not under active development.

## Testing

273 tests (253 unit, 20 integration).

```bash
uv run pytest -q                # unit tests, fast, no external deps
uv run pytest -q -m integration # integration tests, need Docker (spins up Neo4j via Testcontainers)
```

If you use [colima](https://github.com/abiosoft/colima) instead of Docker Desktop, the integration run
needs two extra env vars first — see `CONTRIBUTING.md` for the exact workaround.

## Roadmap

See `docs/spec.md` (§6, "Current state") and `CHANGELOG.md`'s `[Unreleased]` section for what's shipped
and what's next.

## Contributing

See `CONTRIBUTING.md` for the branch/PR workflow and testing expectations. No CI is configured — run the
test suite yourself before opening a PR.

## License

MIT — see `LICENSE.txt`.
