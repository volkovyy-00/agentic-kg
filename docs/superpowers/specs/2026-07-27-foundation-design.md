# Foundation: file sources, driver-side loading, and model configuration

**Date:** 2026-07-27
**Status:** Approved
**Sub-project:** 1 of 3 (Foundation → Unstructured ingestion → Linking)

## Why this exists

The system cannot run against the configured database. `get_neo4j_import_dir()` calls
`dbms.listConfig()`, which Neo4j Aura forbids:

```
Neo.ClientError.Security.Forbidden
Executing admin procedure 'dbms.listConfig' permission has not been granted
```

Every file-facing tool chains through that function, and graph construction separately uses
`LOAD CSV FROM "file:///"`, a scheme Aura cannot serve. File discovery, schema proposal and
construction are all dead against Aura today.

Fixing this is not preparatory work for unstructured ingestion — it is the same requirement
as "not hardcoded to one example". A system whose only admissible data source is one directory
on the database host cannot let a user bring their own data.

## Scope

**In**

- A single owner for file access, backed by `fsspec`
- Spreadsheet loading without `file:///`
- OpenRouter as the sole provider; per-job model selection
- Narrow fix to `finished()` removing its private-API dependency
- Three defects found in the files being rewritten (see *Bugs fixed in passing*)
- Deletion of one dead function (`import_markdown_file`)

**Out**

- Upgrading `google-adk` (stays 1.10.0) — deferred to its own ticket
- Embedding configuration — deferred to sub-project 2, where something reads it
- Typed CSV fields — would change the existing graph's shape
- Unstructured ingestion of any kind

## Definition of done

The furniture/BOM example runs end to end against Aura — user intent through to a constructed
graph — driven from `adk web`, with the unit and integration suites passing.

## Architecture

`src/agentic_kg/common/file_source.py` is new and owns file access, playing the role for files
that `neo4j_for_adk.py` plays for the database: wrap a third-party library, expose a small surface.

It answers exactly four questions:

- where are the files (resolve the configured location into an `fsspec` filesystem plus a root)
- what is there (list names, relative, recursive)
- open one by relative name
- does one exist

The location is one setting, `SOURCE_URI`, accepting a local folder, a bucket or a URL
interchangeably — `fsspec.core.url_to_fs()` parses all three from a single string. `.env.example`
ships it as `SOURCE_URI=./data/bom`, so a fresh clone works with the bundled sample data and no
copying. There is no code-level default: unset means an actionable error at first use, not a
silent fallback to somewhere unexpected.

**The invariant.** Exactly one place resolves where files live; everything else uses names
relative to that root. This is already how the system behaves — `list_import_files` returns
`x.relative_to(import_dir)`, every other tool re-joins `import_dir / file_path`, and
`construction_plan_tools.py` stores whatever relative string it was handed. Construction plans
have never known about absolute locations. Only the supplier of the root changes.

Keeping plans relative means a plan built while files sat in a local folder still works when the
same files move to a bucket. The plan describes the data, not the storage — the same relationship
it already has with the database connection, which is likewise not recorded in the plan.

### Rejected alternatives

**Swap the plumbing, leave the shape alone.** Smallest possible diff: point the existing four
join-and-open sites at `fsspec` individually. Rejected because sub-projects 2 and 3 add a PDF
loader, a Markdown loader and content fingerprinting, each of which also opens source files.
Without a seam, that is three more copies of the same logic and the one-place invariant quietly
stops holding.

**A registry of named sources.** Genuinely useful eventually — filings and reference tables may
live in different places — but one root with subdirectories covers it, and listing is already
recursive. Building it now invents a requirement.

**Host files at a URL for Aura to fetch.** Keeps loading server-side, but requires the data be
reachable from the internet. Rejected: unacceptable for non-public source documents, and it would
leave two loading paths to test.

## Components

| File | Change |
|---|---|
| `common/file_source.py` | **new** — the seam |
| `common/config.py` | `SOURCE_URI`, `OPENROUTER_API_KEY`, `LLM_MODEL_CONVERSATIONAL`, `LLM_MODEL_REASONING`; `validate_env()` checks the OpenRouter key |
| `common/llm_catalog.py` | per-kind model resolution (two bugs, below) |
| `tools/file_tools.py` | `sample_file`, `search_file`, `search_csv_file` become thin callers; `import_markdown_file` deleted; `approve_suggested_files` returns a result |
| `tools/cypher_tools.py` | `get_neo4j_import_dir()` deleted |
| `tools/kg_construction_tools.py` | driver-side loading; result checking; identifier validation |
| `tools/adk_tools.py` | `finished()` becomes a factory |
| `tools/toolset.py` | delete the stray `from tkinter import Label` |
| `coordinators/multi_agent/agent.py` | swap the import-directory tool for a source-location tool; assign model kinds |
| `coordinators/multi_agent/names.py` | **new** — coordinator name constant, breaking the construction-time cycle |
| sub-agent `variants.py` / `agent.py` | build `finished` from the factory; assign model kinds |

**Deleted:** `import_markdown_file`. It is one of the four join-and-open copies and is already
broken — it imports `agentic_kg.sub_agents.cypher_agent.tools`, and no `sub_agents` package
exists. Sub-project 2 builds Markdown loading properly.

**Not touched:** `coordinators/multi_agent/prompts.py` is dead (it describes a `dataprep_agent`
that does not exist; the live instruction is inline in `agent.py`). Out of Foundation's path.

## Data flow

### Spreadsheets

Today the database opens the file:

```cypher
LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
CALL (row) { MERGE (n:$($label) {...}) ... } IN TRANSACTIONS OF 1000 ROWS
```

That mechanism — not the schema, not the plan — is what Aura cannot do. Replacement:

1. `file_source` opens the file by relative name
2. `clevercsv` parses with separator sniffing (already a dependency, already used this way in
   `search_csv_file`)
3. rows accumulate into batches of 1000
4. each batch is sent as one parameterised query

```cypher
UNWIND $rows AS row
MERGE (n:$($label) { <unique_column> : row[$unique_column_name] })
FOREACH (k IN $properties | SET n[k] = row[k])
```

```cypher
UNWIND $rows AS row
MATCH (from_node:$($from_label) { <from_col> : row[$from_node_column] }),
      (to_node:$($to_label)     { <to_col>   : row[$to_node_column] })
MERGE (from_node)-[r:$($relationship_type)]->(to_node)
FOREACH (k IN $properties | SET r[k] = row[k])
```

Reading-and-batching is a separate function from sending, so batching is testable without a
database.

**Unchanged:** `MERGE` semantics, so re-running updates rather than duplicates. Uniqueness
constraints created before loading. Nodes before relationships, since the relationship query
matches nodes that must already exist. Construction plan shape untouched.

**Documented, not a regression:** each batch commits separately, so a mid-file failure leaves
earlier batches in the graph. `IN TRANSACTIONS OF 1000 ROWS` already committed per batch. Same
guarantee, different implementation — recorded because "batched" is easily misread as "was
atomic, now isn't".

**Identifier validation.** `<unique_column>` above is written into the query text, not passed as
a parameter — Cypher cannot parameterise a property key inside a map literal. Today
`kg_construction_tools.py` interpolates `unique_column_name` (line 61) and
`from_node_column` / `to_node_column` (lines 106–107) with no validation, from values that
originate in an LLM proposal. `neo4j_for_adk.py` already provides `is_symbol()` and `sanitize()`
for exactly this and they are never called here. The rewrite validates before interpolating.

### Files

The three surviving file tools call `file_source` instead of computing a path and calling
`open()`. Behaviour is unchanged: `sample_file` reads up to 100 lines, `search_csv_file` sniffs
the dialect from the first 2048 bytes.

## Configuration and models

Model names are stored once in OpenRouter's spelling (`openai/gpt-4o`); the `openrouter/` prefix
LiteLLM requires is derived, not configured. One name, one place, no drift.

`llm_catalog.get_llm(kind)` has two independent bugs:

- it hardcodes `MODEL_GPT_4O_MINI` and ignores `settings.llm_model`, which is read only into a
  log line
- `_llm_instance` is a single module-level slot with no keying on `kind`, so the first caller's
  model would be returned to every later caller even if the hardcoding were fixed

Both are fixed together: cache per kind, resolve each kind's model from settings.

The eight `get_llm()` call sites then declare a kind. Five files hold one call each — the
coordinator, `file_suggestion_agent`, `graphrag_agent`, `graph_construction_agent`,
`user_intent_agent` — and `schema_proposal_agent/agent.py` holds three (proposal, critic, and the
coordinator wrapping the loop).

- **conversational** — coordinator, `user_intent`, `file_suggestion`, `graphrag`
- **reasoning** — `schema_proposal`, `schema_critic`, `graph_construction`, and the schema
  coordinator

**Embeddings are deliberately absent.** They belong in settings alongside their dimension, in
sub-project 2, where something reads them. Shipping a setting with no reader repeats the exact
failure mode `llm_model` already demonstrates.

## The `finished()` fix

Today:

```python
tool_context.actions.transfer_to_agent = tool_context._invocation_context.agent.parent_agent.name
```

The private lookup exists because the tool is constructed in `variants.py` at import time, before
the coordinator that owns it exists. There is no public alternative: `ToolContext` exposes only
`actions`, credential helpers and `search_memory`; `CallbackContext` only `state` and artifact
helpers; `ReadonlyContext.agent_name` is the *current* agent's name, not the parent's.

So the fix removes the need for runtime discovery. A constants module holds the coordinator's
name, imported by both the coordinator and the sub-agents' tool lists:

```python
def make_finished(parent_agent_name: str):
    def finished(tool_context: ToolContext):
        """Finish this phase and hand control back to the coordinator."""
        tool_context.actions.escalate = True
        tool_context.actions.transfer_to_agent = parent_agent_name
        return {}
    return finished
```

No import cycle, no private attributes, and the tool the model sees is unchanged — still named
`finished`, still zero-argument. That last property is the point: ADK auto-injects a public
`transfer_to_agent(agent_name, ...)` into every agent already, but it requires the model to
reproduce an agent name as an argument. A zero-argument tool is categorically more reliable,
particularly on smaller models.

`escalate` is retained. It is currently inert — ADK reads it in exactly one place,
`loop_agent.py:61`, and no site that calls `finished` runs inside a `LoopAgent` — but it is
correct if a future phase ever does.

## Bugs fixed in passing

Both sit inside functions this design already rewrites.

**`construct_domain_graph` always reports success** (`kg_construction_tools.py:124–139`). It calls
`import_nodes(...)` and `import_relationships(...)` in loops without capturing the results, then
unconditionally returns `tool_success`. If all five spreadsheets failed, the agent is told the
graph was built and reports that to the user. This directly undermines Foundation's own
done-condition. The rewrite collects per-file outcomes.

**`approve_suggested_files` returns `None`** (`file_tools.py:69–75`). It sets state and falls off
the end with no `return`, where every other tool returns a `ToolResult` shape.

**`from tkinter import Label` in `tools/toolset.py`** crashes four modules today in a venv without
Tk — `agentic_kg.agent`, `tools.user_intent_tools`, `tools.toolset`, and
`agents.file_suggestion_agent.agent` all fail to import. The `multi_agent` coordinator survives
only because it happens not to import `toolset`. One line, deleted.

## Error handling

| Situation | Behaviour |
|---|---|
| `SOURCE_URI` unset or unreachable | `tool_error` naming the configured value, raised at first use — not at import, which would break web UI startup |
| File not found | `tool_error`, as today |
| Batch fails mid-file | Error naming the file and rows loaded, so partial state is known rather than guessed |
| Column name fails validation | Rejected before any query is built |
| Database unreachable | Already handled — `send_query` wraps and returns `tool_error` |
| `OPENROUTER_API_KEY` missing | `validate_env()` checks it, as it checks the OpenAI key today |

## Testing

**Unit, no disk or database.** `fsspec` ships an in-memory filesystem, so `file_source` is
testable in isolation: listing, nested paths, reading, missing files. No file handling is testable
today without a real directory.

**Unit, CSV batching without a database.** A known spreadsheet through in-memory storage,
asserting the batches and parameters produced.

**Integration, existing container pattern.** Load `data/bom` into a containerised Neo4j and assert
node and relationship counts. This is trustworthy specifically because `file:///` is gone rather
than running in parallel — the container exercises the same code Aura will run. Two paths would
have made this test prove nothing about Aura.

**Import smoke test.** Import every module in the package. This is what would have caught the
`tkinter` breakage on the day it landed; the existing suites (`test_tool_result.py`,
`test_pydantic_neo4j.py`) never import the affected modules. About five lines.

**Manual acceptance.** Drive the furniture example end to end from `adk web` against Aura. Cannot
be automated — there is no Aura in CI — and should not be faked. It is the done-condition.

## Follow-on work

Sub-project 2 (unstructured ingestion) adds the two schema-proposal agents from Lesson 7, chunking
strategy as approved plan data, PDF and Markdown loaders, the extraction executor with
resumability, and document-scoped entity resolution. Sub-project 3 adds linking to reference-table
rows and cross-tier retrieval.

The `fsspec` filesystem this spec introduces is what sub-project 2's PDF loader consumes directly:
`neo4j_graphrag`'s `PdfLoader.run(filepath, metadata, fs)` already accepts an `AbstractFileSystem`,
so wiring it up is passing one argument rather than writing an adapter.

## Risks

- **No Aura in CI.** The done-condition is a manual run. Mitigated by removing the dual path, so
  container tests exercise production code.
- **`fsspec` for remote sources is untested here.** Only the local case is exercised by
  `data/bom`. Bucket and URL sources are supported by construction but unproven until someone
  uses one.
- **Batching performance for very large spreadsheets.** Rows now cross the network instead of
  being read locally by the database. Irrelevant at reference-table scale; would matter at
  millions of rows.
