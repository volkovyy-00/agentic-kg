# Foundation: file sources, driver-side loading, and model configuration

**Date:** 2026-07-27
**Status:** Approved (fact-checked 2026-07-27 — see *Verification*)
**Sub-project:** 1 of 3 (Foundation → Unstructured ingestion → Linking)

## Why this exists

The system cannot run against the configured database. `get_neo4j_import_dir()`
(`cypher_tools.py:205`) calls `dbms.listConfig()`, which the Aura application user cannot execute:

```
Neo.ClientError.Security.Forbidden
Executing admin procedure 'dbms.listConfig' permission has not been granted
for user '641447dc' with roles [PUBLIC, console_admin_free_641447dc] restricted to ACCESS
```

This is a privilege restriction, not an Aura feature ban — `dbms.listConfig()` requires admin
procedure-execution privileges, which Aura's console roles do not grant. The practical effect is
the same: the call fails, and every file-facing tool chains through it.

Separately, graph construction uses `LOAD CSV FROM "file:///"`, which Aura refuses outright:

```
Neo.ClientError.Statement.ExternalResourceFailed
Cannot load from URL 'file:///assemblies.csv':
configuration property 'dbms.security.allow_csv_import_from_file_urls' is false
```

Both errors were reproduced against the live instance (Neo4j 5.27-aura, enterprise).

Fixing this is not preparatory work for unstructured ingestion — it is the same requirement as
"not hardcoded to one example". A system whose only admissible data source is one directory on the
database host cannot let a user bring their own data.

## Scope

**In**

- A single owner for file access, backed by `fsspec`
- Spreadsheet loading without `file:///`, using index-backed MERGE
- OpenRouter as the sole provider; per-job model selection
- Narrow fix to `finished()` removing its private-API dependency
- Three import defects and two silent-failure bugs (see *Defects fixed*)
- Deletion of three dead functions

**Out**

- Upgrading `google-adk` — deferred, but see *Dependency hygiene*, which must still be done here
- Embedding configuration — deferred to sub-project 2, where something reads it
- Typed CSV fields — would change the existing graph's shape
- Unstructured ingestion of any kind

## Definition of done

The furniture/BOM example runs end to end against Aura — user intent through to a constructed
graph — driven from `adk web`, with unit and integration suites passing and the import smoke test
green.

## Architecture

`src/agentic_kg/common/file_source.py` is new and owns file access, playing the role for files
that `neo4j_for_adk.py` plays for the database: wrap a third-party library, expose a small surface.

It answers exactly four questions:

- where are the files (resolve the configured location into an `fsspec` filesystem plus a root)
- what is there (list names, relative, recursive)
- open one by relative name
- does one exist

The location is one setting, `SOURCE_URI`. `fsspec.core.url_to_fs()` parses a local path, a bucket
URL or an HTTP URL from a single string and returns an `(fs, path)` pair.

**Relative-path anchoring.** `url_to_fs("./data/bom")` absolutises against the *process* working
directory. `adk web src/agentic_kg/coordinators/` does not guarantee what that is. Relative
`SOURCE_URI` values are therefore resolved against the repository root, derived from
`agentic_kg.__file__`, not against the CWD. Absolute URIs and non-local schemes pass through
untouched.

**The invariant.** Exactly one place resolves where files live; everything else uses names
relative to that root. This is already how the system behaves — `list_import_files` returns
`x.relative_to(import_dir)` (`file_tools.py:39`), every other tool re-joins `import_dir /
file_path`, and `construction_plan_tools.py` stores whatever relative string it was handed
(lines 53, 130). Construction plans have never known about absolute locations. Only the supplier
of the root changes.

Keeping plans relative means a plan built while files sat in a local folder still works when the
same files move elsewhere. The plan describes the data, not the storage — the same relationship it
already has with the database connection, which is likewise not recorded in the plan.

### Rejected alternatives

**Swap the plumbing, leave the shape alone.** Smallest possible diff: point the existing
join-and-open sites at `fsspec` individually. Rejected because sub-projects 2 and 3 add a PDF
loader, a Markdown loader and content fingerprinting, each of which also opens source files.
Without a seam, that is three more copies of the same logic.

**A registry of named sources.** One root with subdirectories covers it — verified: listing
`./data/bom` returns 15 relative names including `product_reviews/*.md`. Building a registry now
invents a requirement.

**Host files at a URL for Aura to fetch.** Technically viable — verified, `LOAD CSV` from
`https://data.neo4j.com/northwind/products.csv` returns 77 rows on this Aura instance. Rejected
because it requires source data be reachable from the internet, which is unacceptable for
non-public documents, and it would leave two loading paths to test. Note that Aura supports only
`https`/`http`/`ftp` for `LOAD CSV`; native `s3://` and `gs://` are Enterprise-only and not
available on Aura. Reading client-side sidesteps this entirely.

## Components

| File | Change |
|---|---|
| `common/file_source.py` | **new** — the seam |
| `common/config.py` | `SOURCE_URI`, `OPENROUTER_API_KEY`, `LLM_MODEL_CONVERSATIONAL`, `LLM_MODEL_REASONING`; `validate_env()` extended **and actually wired in** |
| `common/llm_catalog.py` | per-kind model resolution |
| `tools/file_tools.py` | `sample_file`, `search_file`, `search_csv_file` become thin callers; `import_markdown_file` deleted; `approve_suggested_files` returns a result |
| `tools/cypher_tools.py` | `get_neo4j_import_dir()` deleted |
| `tools/kg_construction_tools.py` | driver-side loading; index-backed MERGE; result checking; symmetric identifier validation; delete `construct_node`/`construct_relationship` |
| `tools/adk_tools.py` | `finished()` becomes a factory |
| `tools/toolset.py` | delete `from tkinter import Label` |
| `tools/user_intent_tools.py` | fix `from .tool_result import …` → `common.tool_result` |
| `agents/file_suggestion_agent/variants.py` | resolve undefined `file_toolset` |
| `coordinators/multi_agent/agent.py` | swap the import-directory tool for a source-location tool; assign model kinds |
| `coordinators/multi_agent/names.py` | **new** — coordinator name constant, breaking the construction-time cycle |
| `.env.example` | **add** `SOURCE_URI`, `OPENROUTER_API_KEY`, the two model settings |
| `pyproject.toml` | declare `fsspec` and `aiohttp`; bound the `google-adk` constraint |
| sub-agent `variants.py` / `agent.py` | build `finished` from the factory; assign model kinds |

**Deleted:** `import_markdown_file` (`file_tools.py:305`) — imports
`agentic_kg.sub_agents.cypher_agent.tools`; no `sub_agents` package exists. `construct_node` and
`construct_relationship` (`kg_construction_tools.py:16–47`) — zero callers, and they use a
different query shape (hardcoded `id` property) than the live path.

**Not touched:** `coordinators/multi_agent/prompts.py` is dead (describes a `dataprep_agent` that
does not exist). Out of Foundation's path.

## Data flow

### Spreadsheets

Today the database opens the file:

```cypher
LOAD CSV WITH HEADERS FROM "file:///" + $source_file AS row
CALL (row) { MERGE (n:$($label) {...}) ... } IN TRANSACTIONS OF 1000 ROWS
```

Replacement:

1. `file_source` opens the file by relative name, in text mode (`clevercsv.reader` requires an
   iterable of `str`, not bytes)
2. `clevercsv` parses with separator sniffing — already a dependency, already used this way in
   `search_csv_file` (`file_tools.py:182`)
3. rows accumulate into batches of 1000
4. each batch is sent as one parameterised query

```cypher
UNWIND $rows AS row
MERGE (n:<Label> { <unique_column> : row[$unique_column_name] })
FOREACH (k IN $properties | SET n[k] = row[k])
```

```cypher
UNWIND $rows AS row
MATCH (from_node:<FromLabel> { <from_col> : row[$from_node_column] }),
      (to_node:<ToLabel>     { <to_col>   : row[$to_node_column] })
MERGE (from_node)-[r:<REL_TYPE>]->(to_node)
FOREACH (k IN $properties | SET r[k] = row[k])
```

Reading-and-batching is a separate function from sending, so batching is testable without a
database.

**Labels and types are interpolated, not passed as `$()` parameters — this is a deliberate change
from the current code.** Cypher's dynamic labels (`MERGE (n:$($label))`, introduced 5.26, GA)
work correctly on 5.27-aura, but the planner cannot use indexes with them until Neo4j 2025.11.
Verified on the live instance:

```
MERGE (n:$($label) { a_id : row.a_id })  ->  operator: Merge             (no index)
MERGE (n:__PerfA   { a_id : row.a_id })  ->  operator: MergeUniqueNode   (index-backed)
```

With dynamic labels every row triggers an all-nodes scan, so load time grows quadratically. Since
the property key already has to be interpolated (Cypher cannot parameterise a map key), and
`is_symbol()` validation is being made explicit anyway, interpolating the validated label costs
nothing and restores index-backed merging.

**Unchanged:** `MERGE` semantics — verified idempotent, two identical runs produce two nodes, not
four. Uniqueness constraints created before loading. Nodes before relationships, since the
relationship query matches nodes that must already exist. Construction plan shape untouched.

**Transaction semantics, verified.** Each batch is one auto-commit transaction. On failure,
completed batches remain committed and **the failing batch rolls back entirely** — tested with a
deliberate constraint violation; nothing from the failing batch persisted. This matches the
documented behaviour of the `CALL { … } IN TRANSACTIONS` construct it replaces, so it is not a
regression: *"any inner transactions that were successfully committed remain unchanged and are not
rolled back. However, any inner transactions that failed are fully rolled back."*
Worth recording because "batched" is easily misread as "was atomic, now isn't", and because the
guarantee is stronger than "partial state" suggests — there is no torn batch.

**Ragged rows, verified.** When a row lacks a column named in `properties`, `row[k]` returns null
(documented map-subscript behaviour, no error) and `SET n[k] = null` leaves the property absent
rather than writing a null — confirmed via `keys(n)`. This is the desired behaviour for uneven
CSVs, and it is silent, so it is recorded here.

**Identifier validation.** Node identifiers are *already* validated indirectly: `import_nodes`
calls `create_uniqueness_constraint`, which runs `is_symbol()` on the label and column
(`cypher_tools.py:142,145`). The relationship path has no such check — `from_node_column` and
`to_node_column` reach lines 106–107 unvalidated. The rewrite makes validation explicit and
symmetric across both paths, and extends it to the interpolated labels and relationship types.
(`sanitize()` exists at `neo4j_for_adk.py:39` and is called nowhere in `src/`.)

### Files

The three surviving file tools call `file_source` instead of computing a path and calling
`open()`. Behaviour is unchanged: `sample_file` reads up to 100 lines, `search_csv_file` sniffs the
dialect from the first 2048 bytes.

Note for the implementer: `clevercsv.Sniffer().sniff()` returns a degenerate
`SimpleDialect('','','')` on an empty or trivial sample rather than raising `clevercsv.Error`, so
the existing `except clevercsv.Error` fallback does not fire for empty files. Handle the degenerate
dialect explicitly.

## Configuration and models

Model names are stored once in OpenRouter's spelling (`openai/gpt-4o`); the `openrouter/` prefix
LiteLLM requires is derived, not configured. Verified: LiteLLM's
`get_llm_provider("openrouter/openai/gpt-4o")` returns `('openai/gpt-4o', 'openrouter')`, and it
reads `OPENROUTER_API_KEY` from the environment.

**The derived string must be wrapped in a `LiteLlm` instance, never passed to an agent as a bare
model string.** ADK deliberately does not register `LiteLlm` in its `LLMRegistry` — only `Gemini`
is registered — so `LLMRegistry.resolve("openrouter/openai/gpt-4o")` raises
`ValueError: Model … not found`. `llm_catalog` already returns a `LiteLlm` instance and must
continue to; the failure mode if someone "simplifies" this later is a startup error, not a silent
fallback, but it is worth knowing why the wrapper exists.

`llm_catalog.get_llm(kind)` has two independent bugs:

- it hardcodes `MODEL_GPT_4O_MINI` (line 52) and ignores `settings.llm_model`, which appears only
  in a log line (line 50)
- `_llm_instance` (line 38) is a single module-level slot with no keying on `kind`, so the first
  caller's model is returned to every later caller

Both are fixed together: cache per kind, resolve each kind's model from settings.

**There are twelve `get_llm()` call sites, not eight.** Eight are in the `multi_agent` tree:

- **conversational** — the coordinator, `user_intent`, `file_suggestion`, `graphrag`
- **reasoning** — `schema_proposal`, `schema_critic`, the schema coordinator,
  `graph_construction`

Four more sit outside it: `coordinators/single_agent/agent.py:12`, `agents/cypher_agent/agent.py:10`,
`agents/file_suggestion_agent/agent.py:12`, `agents/user_intent_agent/agent.py:13`.

**Two of those already pass `LlmKind.reasoning`, silently ignored today by the un-keyed cache.
Fixing the cache makes them live.** `single_agent` is a shipped coordinator that `adk web`
discovers, so this is a behavioural change, not a no-op. All four get explicit kinds:
`single_agent` and `cypher_agent` conversational, the two `agents/` variants reasoning as they
already request.

**Embeddings are deliberately absent.** They belong in settings alongside their dimension, in
sub-project 2, where something reads them. Shipping a setting with no reader repeats the exact
failure mode `llm_model` already demonstrates.

## The `finished()` fix

Today (`tools/adk_tools.py:7`):

```python
tool_context.actions.transfer_to_agent = tool_context._invocation_context.agent.parent_agent.name
```

The private lookup exists because the tool is constructed in `variants.py` at import time, before
the coordinator that owns it exists. There is no public alternative: searching `ReadonlyContext`,
`CallbackContext` and `ToolContext` for any member containing "parent" returns nothing.
`ReadonlyContext.agent_name` returns the *current* agent's name.

The fix removes the need for runtime discovery. A constants module holds the coordinator's name,
imported by both the coordinator and the sub-agents' tool lists:

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
`finished`, still zero-argument. That last property is the point: ADK's own public
`transfer_to_agent(agent_name, …)` requires the model to reproduce an agent name as an argument,
and a zero-argument tool is categorically more reliable, particularly on smaller models.

ADK does auto-inject `transfer_to_agent` — but **not universally**. Injection is conditional on
having transfer targets, and parent/peer targets are only added when the parent is itself an
`LlmAgent` (`agent_transfer.py:113–132`). An agent under a `SequentialAgent` or `LoopAgent` with no
sub-agents gets no transfer tool at all. That strengthens the case for `make_finished`: in exactly
that configuration there is no built-in fallback.

`escalate` is retained. It is currently inert — ADK reads it for control flow in exactly one place,
`loop_agent.py:61`, and no site that calls `finished` runs inside a `LoopAgent` — but it is correct
if a future phase ever does.

## Defects fixed

**Three independent import defects break seven modules** in a virtualenv without Tk (as here).
Fixing only the first leaves six broken; fixing the first two leaves two:

1. `from tkinter import Label` — `tools/toolset.py:1`
2. `from .tool_result import …` — `tools/user_intent_tools.py:11`; the module is
   `common/tool_result.py`
3. `file_toolset` referenced at `agents/file_suggestion_agent/variants.py:39` and defined nowhere
   in `src/`

All three must be fixed for the import smoke test to pass. Both coordinators import cleanly today,
which is why this has gone unnoticed.

**`construct_domain_graph` always reports success** (`kg_construction_tools.py:124–139`). It calls
`import_nodes(...)` (line 132) and `import_relationships(...)` (line 137) without capturing the
results, then unconditionally returns `tool_success` (line 139). If all five spreadsheets failed,
the agent is told the graph was built. This directly undermines Foundation's own done-condition.
The rewrite collects per-file outcomes.

**`approve_suggested_files` returns `None`** (`file_tools.py:69–75`). It sets state at line 75 and
falls off the end, where every other tool returns a `ToolResult` shape.

## Dependency hygiene

Foundation does not upgrade `google-adk`, but it must bound it. The declared constraint is
`google-adk>=1.10.0` — unbounded, while the latest published version is **2.5.0**. Any fresh lock
would cross a major version boundary silently. Change to `>=1.10,<2` so that "stays on 1.10" is
enforced rather than merely lockfile-incidental, leaving the 2.x migration as its own deliberate
ticket.

Two dependencies are used but undeclared:

- **`fsspec`** arrives transitively via `neo4j-graphrag` (`<2025.0.0`). Building a first-class
  module on an inherited pin is fragile — declare it directly.
- **`aiohttp`** is what makes `https://` sources work at all, and it arrives only as a transitive
  dependency of the LLM stack. Declare it, or accept that HTTP sources break if that stack changes.

**`s3://` does not work today.** `s3fs` is absent from the venv and the lockfile, so
`url_to_fs("s3://…")` raises `ImportError` at resolution time. Either add `s3fs` behind an extra
or state plainly that bucket sources require installing it. The earlier phrasing — "supported by
construction but unproven" — was wrong in the optimistic direction.

## Error handling

| Situation | Behaviour |
|---|---|
| `SOURCE_URI` unset or unreachable | `tool_error` naming the configured value, at first use — not at import, which would break web UI startup |
| `SOURCE_URI` uses an uninstalled scheme (e.g. `s3://`) | `tool_error` naming the missing package, rather than a raw `ImportError` |
| File not found | `tool_error`, as today |
| Batch fails mid-file | Error naming the file and rows committed; the failing batch is fully rolled back |
| Identifier fails validation | Rejected before any query is built |
| Database unreachable | Already handled — `send_query` wraps and returns `tool_error` |
| `OPENROUTER_API_KEY` missing | `validate_env()` checks it **and is invoked from a coordinator import path** |

The last row is a change, not a restatement. `validate_env()`'s only caller today is
`src/agentic_kg/agent.py:1` — itself one of the seven broken modules, and not what
`adk web src/agentic_kg/coordinators/` loads. The check has never run in the live path.

## Testing

**Unit, no disk or database.** `fsspec` ships `MemoryFileSystem` in core (no optional dependency),
so `file_source` is testable in isolation: listing, nested paths, reading, missing files. Note the
memory filesystem is a process-global singleton — tests must clear it between cases.

**Unit, CSV batching without a database.** A known spreadsheet through in-memory storage, asserting
the batches and parameters produced.

**Integration, existing container pattern.** Load `data/bom` into a containerised Neo4j and assert
node and relationship counts. Trustworthy specifically because `file:///` is gone rather than
running in parallel — the container exercises the same code Aura will run.

**Import smoke test.** Import every module in the package. Requires all three import defects
fixed. This is what would have caught the `tkinter` breakage on the day it landed; the existing
suites import only `common.pydantic_neo4j` and `common.tool_result`.

**Manual acceptance.** Drive the furniture example end to end from `adk web` against Aura. Cannot
be automated — there is no Aura in CI — and should not be faked.

## Verification

The following were confirmed by execution against the live Aura instance (5.27-aura, enterprise)
on 2026-07-27, not from documentation:

- `file:///` `LOAD CSV` fails; `https://` `LOAD CSV` succeeds (77 rows)
- `dbms.listConfig()` is denied to the Aura application user
- Dynamic labels, dynamic relationship types, `SET n[k] = row[k]` and `row[$col]` all execute
  correctly on 5.27
- Dynamic labels produce `Merge`; static labels produce `MergeUniqueNode` — the index difference
- The `UNWIND` node and relationship loads work, are idempotent, and set properties correctly
- A failing batch rolls back entirely while earlier batches persist
- Missing columns leave properties absent rather than null
- `url_to_fs("./data/bom")` → 15 relative names including `product_reviews/`; `clevercsv` parses
  `assemblies.csv` (64 rows), `products.csv` (10), `part_supplier_mapping.csv` (176)

All test data written during verification was removed; the database was left empty, as found.

## Follow-on work

Sub-project 2 (unstructured ingestion) adds the two schema-proposal agents from Lesson 7, chunking
strategy as approved plan data, PDF and Markdown loaders, the extraction executor with
resumability, and document-scoped entity resolution. Sub-project 3 adds linking to reference-table
rows and cross-tier retrieval.

Two things this spec's work hands forward, with caveats:

- `neo4j_graphrag`'s `PdfLoader.run(filepath, metadata, fs)` accepts an `fsspec` filesystem — but
  `fs` must be a filesystem object or a bare protocol name (`"memory"`, not `"memory://…"`), and
  `filepath` must be the filesystem-native path. That is exactly the `(fs, path)` pair `url_to_fs`
  returns, so `file_source` should expose both. Note the abstract `DataLoader.run` signature is
  `(filepath, metadata)` with no `fs`, so a custom Markdown loader must widen it.
- `langchain-text-splitters` is **not installed** and is not in the lockfile. It is an unselected
  extra of `neo4j-graphrag`, and `LangChainTextSplitterAdapter` does a top-level import of it, so
  that adapter raises `ImportError` today. Sub-project 2 must add the dependency;
  `FixedSizeSplitter` works without it.

Also relevant to sub-project 3: OpenRouter's rerank endpoint is Cohere-shaped, not OpenAI-shaped,
so the `openai` client cannot call it. That does not affect the single-key decision — one bearer
token covers chat, embeddings and rerank — but it does mean rerank needs a raw HTTP call.

Two further constraints confirmed during this spec's verification, recorded here because they bear
directly on sub-project 2's design and would be expensive to discover during implementation:

- **`LongRunningFunctionTool` treats any falsy return as "no response."** The framework check is
  `if not function_response` (`functions.py:290`, and again at `:427` for the async path — the
  contract is uniform, not branch-specific), so `{}`, `""` and `0` all suppress the function
  response event exactly as `None` does. The per-document progress payload must therefore always be
  a non-empty dict; returning an empty one on a no-op document would silently produce no progress
  at all.
- **Aura ships APOC Core only.** `apoc.text.*` (44 functions) and `apoc.refactor.mergeNodes` — the
  two families the entity-resolution design depends on — are confirmed present and pre-installed,
  requiring no opt-in. But the extended library is absent, so nothing in `apoc.load.*` or
  `apoc.periodic.*` is available. Any resolution or batching design that reaches for those needs a
  different approach.

## Risks

- **No Aura in CI.** The done-condition is a manual run. Mitigated by removing the dual path, so
  container tests exercise production code.
- **Bucket sources need `s3fs`, HTTP sources need `aiohttp`.** Neither is currently declared. Until
  addressed, only local sources are genuinely supported.
- **Dynamic-label index behaviour changes in Neo4j 2025.11.** If the project later moves to a
  version where dynamic labels use indexes, the interpolation decided here becomes optional rather
  than necessary — worth revisiting, not worth pre-empting.
- **Batching performance for very large spreadsheets.** Rows now cross the network instead of being
  read by the database. Irrelevant at reference-table scale; would matter at millions of rows.

## Acceptance

**Run on 2026-07-27 against Neo4j Aura instance `641447dc` and OpenRouter. Passed.**

Models: `openai/gpt-4o-mini` (conversational), `openai/gpt-4o` (reasoning). `SOURCE_URI=./data/bom`.
Database was empty (0 nodes) beforehand.

Driven through a fresh `adk web src/agentic_kg/coordinators/` on port 8081, via the server's own HTTP
API rather than the browser UI — the Chrome extension would not connect. Same server, same
coordinator, same sub-agents and tools; only the front end differed.

What ran, in order:

1. "Is Neo4j ready? And where are my files being read from?" → `neo4j_is_ready` returned "Neo4j is
   Ready!" against Aura, and `get_source_location` returned the resolved source root. This is the
   tool that replaced `get_neo4j_import_dir`, whose `dbms.listConfig()` call Aura forbids.
2. Stated a supply-chain goal → `user_intent_agent` set and approved it.
3. `file_suggestion_agent` listed all 15 source files through `fsspec`, sampled four through
   `open_source`, and suggested `part_supplier_mapping.csv` + `suppliers.csv`. Approved.
4. `schema_proposal_agent` proposed Supplier/Part nodes and a `Supplies` relationship. Approved.
5. `graph_construction_agent` created both uniqueness constraints, then
   `build_graph_from_construction_rules` loaded driver-side: Supplier 20 rows, Part 176 rows,
   Supplies 176 rows, all reported per-rule.

Resulting graph, verified independently of the agent:

| | |
|---|---|
| Supplier nodes | 20 |
| Part nodes | 88 (from 176 rows — `MERGE` deduplicated correctly) |
| `Supplies` relationships | 176 |
| Suppliers with `name` set | 20 |
| Relationships with `lead_time_days` set | 176 |
| Constraints | `Supplier_supplier_id_constraint`, `Part_part_id_constraint` |

These match the container integration test's expectations exactly.

**Handoffs behaved correctly and the coordinator never stalled.** The clearest evidence came from a
failure path: the coordinator routed to `file_suggestion_agent` before a goal existed, that agent's
`get_approved_user_goal` returned a `tool_error`, it called `finished()`, and control returned to the
coordinator, which then routed to `user_intent_agent`. Every phase afterwards handed back the same
way. Note `schema_proposal_agent` runs a `schema_refinement_loop` — a `LoopAgent`, the one context
where `finished()`'s `escalate` flag is not inert.

**Non-ASCII data survives the round trip.** `suppliers.csv` contains "São Paulo"; it is stored in
Aura intact, with no mojibake markers in any city value. This exercises the `encoding="utf-8"`
default in `open_source` end to end, not just in unit tests. (The bundled data is more non-ASCII than
it first appears: `products.csv` and `assemblies.csv` carry Malmö/Västerås/Örebro/Linköping/
Norrköping, and every `product_reviews/*.md` carries those plus ★ glyphs.)

### Observations, none blocking

- **This Aura instance uses its instance ID as both username and database name**, not `neo4j` for
  either. `.env.example` shows `/neo4j` as the database, which is right for a default AuraDB instance
  but not universal — worth knowing before assuming a connection failure is a code fault.
- **A malformed DSN scheme surfaces clearly.** A typo'd `bneo4j+s://` produced a precise pydantic
  error naming the allowed schemes, which is exactly what the restored "Allowed schemes" comment in
  `.env.example` is for.
- **The agent wrote one query with the relationship reversed** — `(:Part)<-[:Supplies]-(:Supplier)`
  where the graph has `(:Part)-[:Supplies]->(:Supplier)` — got zero rows, and reported "no
  single-source risks". A wrong conclusion from a wrong query, not a construction defect: the same
  data queried in the correct direction is present and correct. This is ordinary LLM behaviour and
  the sort of thing the graphrag phase will need guardrails for.
- **Stricter identifier validation was not hit.** The proposed `Supplies` relationship type and all
  column names are bare identifiers. A plan whose label or column contained a hyphen or leading digit
  would now return a `tool_error` where it previously reached the database.

### Second run, 2026-07-27: through the browser UI, confirms the first

The first run above used the HTTP API because the Chrome extension wasn't connecting yet. Once it
was, the same acceptance was repeated through `adk web`'s actual browser UI, against the same Aura
instance (cleared to 0 nodes and constraints dropped beforehand) — this is the closer approximation
of how a real user drives the system, and it surfaced one defect the first run couldn't have: **the
port-8080 server that was already running had been started before this session's final commits, so
it was serving stale code** — its trace showed a call to `get_neo4j_import_dir`, a tool this branch
deleted in Task 6. Confirmed via `grep` (zero matches in `src/`) and by the process's start time
predating the branch's last commits. Killed and restarted from the current tree before proceeding;
worth remembering whenever a long-lived dev server sits across a session boundary.

On the restarted server, the full five-phase workflow ran again through the browser, this time
proposing a richer schema — `Supplier`, `Component`, `Assembly` nodes with `SUPPLIES` and `PART_OF`
relationships (`components.csv` and `assemblies.csv` used in addition to the pair from the first
run) — which is expected variance from an LLM-proposed plan, not a regression. Resulting graph,
verified independently of the agent:

| | |
|---|---|
| Supplier nodes | 20 |
| Component nodes | 88 |
| Assembly nodes | 64 |
| `SUPPLIES` relationships | 176 |
| `PART_OF` relationships | 88 |
| Constraints | all three, one per node label |

Asking "which components have only a single supplier?" this time produced the *correct* answer (zero
single-source components), verified independently with a direct Cypher count — the first run's
backwards-relationship query was LLM variance, not a systematic defect, as its own write-up
concluded.
