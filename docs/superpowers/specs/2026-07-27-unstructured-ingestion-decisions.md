# Unstructured ingestion: settled decisions for sub-projects 2 and 3

**Date:** 2026-07-27
**Status:** Alignment complete. **This is not a spec** — it is the input to writing one.
**Applies to:** sub-project 2 (unstructured ingestion) and sub-project 3 (linking)

## How to use this document

These decisions were reached in a structured alignment interview and are settled. When writing
spec 2 or spec 3, treat them as constraints and do not re-open them. Each records *why*, because
the reasoning is what makes them defensible when implementation pressure suggests otherwise.

Write the spec against the **post-Foundation codebase**, not against what is described here as
current. Sub-project 1 (`2026-07-27-foundation-design.md`) deletes `get_neo4j_import_dir`, replaces
`LOAD CSV FROM "file:///"`, rewrites `llm_catalog`, and changes `finished()`. Verify the current
state of any file before relying on a claim about it.

Technical constraints discovered during Foundation's fact-check — the `PdfLoader` `fs`/path
contract, the missing `langchain-text-splitters` dependency, `LongRunningFunctionTool`'s
falsy-return behaviour, APOC Core-only on Aura, and OpenRouter's Cohere-shaped rerank endpoint —
are recorded in the Foundation spec's *Follow-on work* section. Read both.

---

## 1. Target dataset and definition of done

**Dataset:** SEC 10-K annual filings (PDF, sometimes Markdown; tens of thousands to 100k+ words
each, following the standard Item 1 / 1A / 7 / 8 outline), plus two reference tables —
`Company_Filings.csv` (company name, CIK, CUSIP, ticker, filing path) and
`Asset_Manager_Holdings.csv` (which institutional managers hold which companies, by quarter).

**These files are not in the repository and must be sourced.** They block the definition of done,
not the start of work.

**Done means:** a full end-to-end run over the filings and both reference tables, with the bundled
furniture/BOM example still working.

**Explicitly not claimed:** this dataset will *not* demonstrate that semantic entity matching is
necessary. Company-name variants (`"PAYPAL"` vs `"PAYPAL HLDGS INC"`) are shared-substring cases
that string-edit distance handles well. A genuinely non-lexical join (`"checkout service"` vs
`"checkout-api"`) would be needed to prove that, and this corpus does not contain one. Design the
matching step to be swappable; do not claim it proves more than it does.

**Why a second dataset at all:** genericity cannot be reviewed into existence. The course material
is tuned to the furniture reviews in ways that fail *silently* — a document with no `---` delimiter
becomes one giant unsplit chunk; a document with no H1 gets the literal title `"Untitled"`;
entities outside the hardcoded list are dropped. None of these raise an error. Only a full run on
structurally different documents surfaces them.

---

## 2. Chunking

**Decision:** split on section boundaries **first**, then run a recursive splitter within each
section at ~500 tokens with ~15% overlap. Attach metadata to every chunk, including the section.

**Why section-first is not optional:** the plan is to use the section (`Item 1A`, `Item 7`) as
chunk metadata so retrieval can filter to, say, risk factors before searching. If a chunk straddles
a section boundary it still gets one section label, so the label is *false*. Filtering on a field
that is sometimes wrong is worse than not having the field, because it will be trusted.
Section-first makes the label true by construction.

**The strategy is agent-proposed and user-approved, never hardcoded.** "Split on Item N" in the
code is the same category of error as the course's "split on `---`" — merely tuned to a different
dataset. The system offers general strategies (split on headings, on a repeated marker, by length),
an agent samples the actual documents and proposes one with parameters, and the choice lands in the
approved plan. This also converts an invisible failure into a visible decision: a bad split
currently produces one enormous chunk and no error.

**Fallback:** documents with no discernible structure get plain length-based splitting with
overlap, so an entity spanning a boundary is not lost.

**Dependency:** `langchain-text-splitters` is not installed and is not in the lockfile. It is an
unselected extra of `neo4j-graphrag`, and `LangChainTextSplitterAdapter` imports it at module
level, so that adapter raises `ImportError` today. `FixedSizeSplitter` works without it but
measures in **characters**, not tokens (default 4000/200).

---

## 3. Extraction

**Decision: one pass, not two.** A single chunking pass at ~500 tokens serves both embedding and
extraction. Extraction gets a **context header** injected into its prompt: document title, section
heading, and the company identity pulled from `Company_Filings.csv`.

**The problem this solves.** Chunk size that is right for retrieval is wrong for extraction. A 10-K
says "the Company" hundreds of times and rarely repeats the name. A 450-token slab from the middle
of Item 7 gives the model no idea which company it is reading, which segment is discussed, or what
"the prior period" refers to. This does not error — it produces thin, ungrounded entities that look
fine in the graph and are quietly wrong. The course notebook already hacks around this by
prepending a file's first few lines to every extraction prompt.

**Why the header rather than wide extraction windows.** The diagnosed failure is identity and
section *ambiguity*, which a header fixes at negligible cost: one chunk size, one embedding pass,
one clean `NEXT_CHUNK` chain. Two-pass extraction buys only the recovery of relationships spanning
more text than one chunk holds — a real but narrower and more speculative gap — while doubling
chunk storage, doubling what the plan schema must represent, and doubling what "agent-proposed and
approved" must cover. Treat wide-window extraction as a targeted escalation *if* real 10-K runs
show failures of the form "needed context that was in the same section but outside the chunk"
rather than "didn't know whose report this was."

**Pulling company identity from the reference table is the interesting part.** Extraction starts
*knowing* which company it is reading, so "the Company" resolves at extraction time instead of
becoming a matching problem afterwards. This is the gazetteer idea applied where it does most good.
It extends the course's own `get_well_known_types` pattern (which derives entity *types* from the
construction plan) one level down, to entity *values*.

**Fail open.** When a document matches no row in any reference table, the header carries title and
section only and ingestion proceeds. A missing reference table must not block ingestion; it just
means the header does less work. Plenty of corpora have no such table at all.

**Entity and fact types come from the two Lesson 7 agents** (`prototype/schema_proposal_unstructured.ipynb`),
derived per dataset. Their prompts reference tools that are not bound in that notebook — do not copy
them verbatim. The hardcoded `['Product', 'Issue', 'Feature', 'Location']` with
`additional_node_types: False` is exactly what makes the course version furniture-specific.

**Tables are out of scope.** `neo4j-graphrag`'s `PdfLoader` extracts text via
`pypdf.PdfReader(...).pages[n].extract_text()`, which is layout-blind and destroys table structure.
No chunking strategy recovers information already lost at extraction. Therefore:

- **The graph represents narrative content** — business description, risk factors, MD&A — and
  **not financial statement line items.** This is a stated boundary, not an accident.
- **Table-dense regions are detected and skipped, not extracted from.** Garbled table text does not
  produce nothing; it produces confident, plausible, wrong relationships between numbers that were
  never related. A gap in the graph is visible; a wrong figure is not.
- The detection heuristic must not be tuned to 10-K layouts.
- Partial compensation: `Asset_Manager_Holdings.csv` brings real numeric data in through the
  spreadsheet path, which is exact.

---

## 4. Execution and resumability

**Decision:** a `LongRunningFunctionTool` that processes **one document per invocation**, reports
progress, and lets the database remember state.

**Approval posture:** the user approves the **batch once**, then it works through documents one at
a time with a stop button. This mirrors the existing spreadsheet flow — approve the plan, then it
builds without asking per file — rather than asking per document, which would undercut leaving a
run unattended.

**Resumability:** each `Document` node carries a content fingerprint and a status. The ordering is
**clear status → delete that document's prior contribution → write → mark complete**, in that
order.

**Why the ordering, specifically.** `Neo4jForADK.send_query()` runs one statement per call, so
"delete the old contribution" and "write the new one" cannot be one atomic operation. A crash
between them would leave a document *worse* than not-done. Clearing the status first makes "not
marked complete" the single source of truth for "redo this one", regardless of where a crash lands.

**Why the database rather than a job system:** progress is answerable by querying the graph, by any
session, at any time, even after the original run died. That yields restartability without building
a queue. Close the browser and you lose the live progress display, not the work.

**The same mechanism makes re-import safe.** Spreadsheet loading is idempotent because it is
`MERGE`-based; AI extraction is not — run it twice and you get two overlapping sets of extracted
things. Unchanged document, already done, skip. Changed document, remove the prior contribution,
redo.

**Provenance comes free.** `neo4j-graphrag` already writes `Document`, `Chunk`,
`(Chunk)-[:FROM_DOCUMENT]->(Document)`, `(Chunk)-[:NEXT_CHUNK]->(Chunk)` and
`(Entity)-[:FROM_CHUNK]->(Chunk)`. That is a complete trail from any extracted thing back to its
source file, which is what makes document-level undo possible.

**Trap:** `LongRunningFunctionTool` treats *any* falsy return as "no response" — the framework check
is `if not function_response` (`functions.py:290`, and `:427` for the async path), so `{}` behaves
exactly like `None`. The progress payload must always be a non-empty dict, or a no-op document
silently reports no progress at all.

---

## 5. Identity and entity resolution

**Decision: merge within a document, never across documents.** One NVIDIA per filing, not two
hundred per filing and not one shared across five filings.

**Why this resolves a genuine conflict.** Document-level undo requires that nothing extracted is
shared between documents. If one NVIDIA node is built from five filings, "delete filing three's
contribution" stops being a delete and becomes reference counting — get it wrong and you either
orphan data or silently corrupt the other four. Merging only within a document dissolves the
conflict rather than managing it, while still fixing the 200-chunks-per-document problem, which is
the one that actually degrades the graph.

**Cross-document identity lives in the reference-table node.** Every per-filing extraction links to
the same row-derived company node via `CORRESPONDS_TO`. Queries traverse from that one node outward.

**What this preserves that full merging would destroy:** you can still see *which filing said what*.
"NVIDIA's 2024 filing flagged this risk; the 2025 one didn't" is exactly the question this dataset
is interesting for, and it is only answerable if per-filing extractions stay distinct.

**Usability cost, accepted:** the graph will contain several nodes a naive look would call
duplicates. Anyone querying must go through the reference-table node, not the extracted ones. This
needs to show up in how the query agent is instructed.

### The library default will destroy this if left alone

`SimpleKGPipeline(perform_entity_resolution=...)` **defaults to `True`**, and its
`SinglePropertyExactMatchResolver` runs `MATCH (entity:__Entity__)` with the filter appended only
if supplied — so by default it sweeps every extracted entity in the entire database and merges via
`apoc.refactor.mergeNodes`, which is **destructive and irreversible**. Left at its default,
ingesting a second filing fuses it into the first and document-level undo is not merely harder, it
is *gone* — the evidence of which document contributed what is consumed by the merge.

**Therefore:**

- Resolution runs **once per document, immediately after that document's extraction**, never as a
  single global pass.
- Scope it with a **temporary `__pending` marker property** set by our own parameterised query, so
  `filter_query` stays the constant literal `WHERE entity.__pending = true`. This matters:
  `filter_query` is raw string concatenation with **no parameter binding** — the resolver calls
  `execute_query(stat_query, database_=...)` with no parameters dict — so scoping by interpolating
  a file path would mean building Cypher by string-joining user-supplied input.
- **Cleanup is a blanket removal**: `MATCH (e:__Entity__) WHERE e.__pending IS NOT NULL REMOVE
  e.__pending`. `mergeNodes` *preserves properties*, so the marker survives onto the merged node
  while the others cease to exist — a targeted "remove the marker from the nodes I tagged" would
  leave stale flags on exactly the nodes that merged. The blanket form is still a constant,
  parameterless query.

**Exact-match resolution is not airtight, by construction.** The resolver groups by label plus the
literal value of `name`. Within one document, `"NVIDIA"` and `"NVIDIA Corporation"` will *not*
merge. They both link to the same reference-table node instead. Do not assume one entity per
company per document.

**The resolver merges; domain linking only adds.** `CORRESPONDS_TO` correlation never mutates
domain nodes, so it is safe to run graph-wide. The resolver must be tightly scoped. Same-sounding
operations, opposite blast radius — treat them differently.

---

## 6. Models and providers

**Single OpenRouter key** covers chat, extraction *and* embeddings. Verified: the embeddings
endpoint is `https://openrouter.ai/api/v1/embeddings`, OpenAI-compatible in both request and
response shape, so `OpenAIEmbeddings(model=..., base_url=..., api_key=...)` from `neo4j-graphrag`
works because it passes kwargs straight to `openai.OpenAI`.

**Per-job model selection lives in the settings file** (implemented in Foundation). Sub-project 2
adds the embedding model **and its dimension**, recorded together — Neo4j needs the dimension to
build a vector index. They were deliberately excluded from Foundation because nothing there reads
them, and a setting with no reader is the exact failure mode `llm_model` already demonstrated.

**Two model-naming conventions:** the agent side goes through LiteLLM and wants
`openrouter/openai/gpt-4o`; the extraction and embedding side talks to OpenRouter directly and
wants `openai/gpt-4o`. Foundation stores one and derives the other.

**Reranking is deferred**, and the reason is simply that no vector search exists yet for a reranker
to sit in front of. (An earlier objection — that reranking would reintroduce a second provider —
is wrong: OpenRouter added a rerank endpoint in April 2026. But it is Cohere-shaped, so the `openai`
client cannot call it; it needs a raw HTTP call.)

**Standing risk:** extraction quality on dense financial prose is unproven at any model size. The
project ran on `gpt-4o-mini` hardcoded; 10-K prose is far denser than product reviews, and weak
extraction produces plausible-looking thin results rather than obvious errors.

---

## 7. Sub-project 3: linking

Connect extracted entities to reference-table rows via `CORRESPONDS_TO`, and teach the query agent
to traverse from filing prose into the holdings data.

**Linking, not merging** — this is the load-bearing choice. Extraction never mutates spreadsheet-derived
data, so undoing a document can only remove what extraction added. That is what makes "redo the whole
document" safe rather than destructive, and it keeps "the model said this" separable from "the CSV
said this", so a bad extraction is revocable and answers stay traceable to a chunk.

**Key alignment should be a plan artifact, not a heuristic.** The course fuzzy-matches property
*key names* with rapidfuzz and blindly takes the top-scored pair — which produced a
`dimensions`↔`description` false positive at 0.57. Make it
`propose_correlation(label, entity_key, domain_key, method, threshold)` with human approval,
parallel to `propose_node_construction`.

**Value matching:** the course uses `apoc.text.jaroWinklerDistance` with a hand-picked global
cutoff over a full cartesian product. For anything beyond toy scale this needs blocking first (a
full-text or vector index over embeddings already computed). `neo4j-graphrag` 1.14 ships
`SinglePropertyExactMatchResolver`, `FuzzyMatchResolver`, `SpaCySemanticMatchResolver` and
`BasePropertySimilarityResolver` — prefer these over hand-rolled matching, noting they dedup
*within* the graph rather than performing cross-tier linking.

**Aura ships APOC Core only.** `apoc.text.*` (44 functions) and `apoc.refactor.mergeNodes` are
confirmed present and pre-installed. `apoc.load.*` and `apoc.periodic.*` are **not** available.

---

## 8. Deliberately left open

**Whether extraction becomes a new `construction_type` inside the existing
`approved_construction_plan`, or a separate approved plan of its own.** Both are workable and the
choice is reversible; it belongs to spec 2's design phase rather than to alignment.

---

## 9. Rejected alternatives, and why

| Rejected | Why |
|---|---|
| Two-pass extraction (wide windows for extraction, small chunks for retrieval) | Solves a narrower, more speculative problem than the context header, at double the storage, schema and approval surface. Escalate to it only if real runs show same-section-outside-chunk failures. |
| Merging entities across documents | Cleaner-looking graph, but makes document-level undo reference-counted with a nasty failure mode, and destroys the which-filing-said-what signal. |
| Full background job system with its own queue and state | The graph already answers "how far along is it?". A queue is a large thing to build and explain for the last 10%. |
| Asking approval per document | Ten filings becomes ten approvals, undercutting unattended runs. |
| Never asking at all | Breaks the propose-then-approve habit every other phase follows, and makes an hour of spend possible from one ambiguous instruction. |
| Hardcoding "split on Item N" | The same error as the course's `---`, tuned to a different dataset. |
| Relying on chunk traversal to fix extraction context | Traversal repairs **retrieval**, which happens fresh each query. Extraction happens once at ingest; if the model lacked context then, the entity is already wrong and no traversal recovers it. Retrieval is always fixable later; extraction is not cheaply. |
| A cross-encoder reranker now | Read-time concern with no vector search yet to front. |

---

## 10. Standing risks

- **SEC dataset is not in the repository** and must be sourced. Blocks the definition of done.
- **Table detection must not be tuned to 10-K layouts**, or it reintroduces the exact
  dataset-specific hardcoding this work exists to remove.
- **Extraction quality on dense financial prose is unproven**, and failures are quiet rather than
  loud.
- **PDF text extraction is layout-blind.** Accepted, and scoped around by excluding tables.
- **`google-adk` is pinned to 1.10** (Foundation bounds it `<2`); the latest is 2.5.0. Upgrading is
  a separate deliberate ticket, not something to fold into sub-project 2.
