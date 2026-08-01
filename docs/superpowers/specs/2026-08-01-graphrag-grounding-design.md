# Grounding `graphrag_agent` in the graph

**Date:** 2026-08-01
**Status:** design approved, ready for implementation planning
**Scope:** retrieval side only. The construction-side typing defect is documented separately in `docs/typing-defect.md` and must not be touched by this work.

## Problem

A live session produced three failures from `graphrag_agent_v1`. All arithmetic in its answers was correct; the errors were in what it knew and how it framed what it counted.

**1. Invented a set of records.** Asked about sourcing concentration, it stated which four suppliers had no supplier relationships. It never queried for this. It was recalling a warning emitted by the schema critique agent 3.5 hours earlier in the same conversation, and it named `SUP-018` — which has one relationship — among the four. The correct set is `SUP-007`, `SUP-010`, `SUP-017`, `SUP-020`.

**2. Counted a heterogeneous relationship set uniformly.** It answered "greatest sourcing concentration" with `MATCH (s:Supplier)-[:SUPPLIES]->(:Part) RETURN s.country, count(*)`, reporting Sweden 32 and Canada 32 as jointly the top concentration. Every `Part` has exactly two suppliers and exactly one flagged `preferred_supplier='yes'`, so the 176 relationships are 88 primary plus 88 backup. Filtered to primary, Sweden is the sole source for 32 parts and Canada for zero — Canada is purely the fallback. The answer inverted the risk story. `preferred_supplier` appeared in the schema the agent fetched and in none of its four queries.

**3. Reported worst-case quotes as lead times, and miscounted a superlative.** Asked for the longest lead times, it returned the maximum across all quotes. All 15 relationships at ≥28 days are `preferred_supplier='no'` — fallback options, not lead times anyone would experience. It observed that fact and reported it as an insight without noticing it invalidated the framing. Separately it claimed Malmö Desk was the most exposed product with 4 parts in range; Linköping Bed has 6. It reached that by counting ~15 returned rows by eye rather than aggregating.

### Provenance of the dataset claims

Two kinds of evidence appear in this document and they carry different weight.

**Code citations** (file and line) were verified by reading the installed source and can be re-checked by anyone at any time.

**Dataset claims** — the four disconnected supplier IDs, the 88/88 primary-backup split, Sweden 32 versus Canada 0, six `Part` nodes named "Screws", 16 names reused across 63 of 88 parts, 15 relationships at ≥28 days, Linköping Bed 6 versus Malmö 4, and `lead_time_days` returning 10 values against `distinct_count: 27` — were verified by direct Cypher against the live Aura instance on 2026-07-31, but the graph has since been wiped and rebuilt. They are **not** checkable from repository source alone. They are reproducible: all of them derive deterministically from the CSVs in `data/bom/`, so rebuilding the graph and re-running the queries will reproduce them.

**Not reproducible:** the session narrative itself — that the invented supplier list originated in a schema-critique warning issued 3.5 hours earlier in the same conversation. That comes from the session transcript, which is not preserved in the repository. It is the diagnosis behind the context filter, so it is worth knowing it rests on an observation rather than on an artifact anyone can re-inspect.

## Root causes

- **Failure 1** is a context-plumbing problem, not a memory-discipline problem. ADK rewrites other agents' output into the current agent's context with `role='user'`, so the critique's warning arrived looking like something the user had said.
- **Failures 2 and 3 share one cause:** the agent did not know the *grain* of the pattern it queried. `(Supplier)-[:SUPPLIES]->(Part)` is 2:1, so a result row is not a Part. Counting rows uniformly and taking `max()` over them both follow mechanically from assuming one row equals one Part.
- **Failure 3's miscount** is a tool-boundary problem: the query tool returns unbounded raw rows with no count, which invites manual counting.

## Scope

**In:** changes to the retrieval path — context filtering, enriched schema with completeness annotations, degree and value profiling, caching with invalidation, safe query execution and result bounding, four prompt rules, and `graphrag_agent_v2` packaging.

### Scope revision

An independent review against the goal — *a universal tool that queries any created graph without error* — found the first draft solving a narrower problem: grounding the agent on one small, single-pattern, CSV-built graph. Two groups of work were folded in, for two different reasons, and the distinction matters when judging whether scope grew.

**Group 1: corrections to mechanisms this document already committed to.** Gating `is_enhanced` (the draft enabled it globally while promising byte-identical output to other consumers — a plain contradiction); keying degree on `(start, type, end)` rather than type alone; server-enforced read-only and a query timeout in place of a text-matching guard; tri-state annotations so an uncomputable annotation is not silently read as safe. These are bugs in the draft, not new ambitions. Deferring them would mean writing a second document that reopens `send_query`, `is_write_query` and the profile's dict shape purely to fix the first one.

**Group 2: pulled forward on scheduling grounds, not genericity grounds.** The shape-based integration tests. Sub-project 2 is already committed with its design settled, and it will introduce exactly the multi-pattern and self-referencing relationship shapes where the keying bug lives. Validating against those shapes now costs one test file; discovering the bug mid-way through sub-project 2 costs unwinding a profile shape its consumers already depend on. This is not a retroactive widening of the "don't hardcode supplier names" constraint.

**Out:** the construction-side typing defect (every property stored as `STRING`, `unit_cost` as `'$42.73'`). See `docs/typing-defect.md`. **Also out:** any change to graphrag's model tier — see *Deliberately unchanged* below.

## Generality constraints

This work was diagnosed from one dataset, so the failure mode to guard against is a fix that only works on furniture. Three rules, in decreasing order of how mechanically they can be checked.

**1. No dataset vocabulary in `src/`.** No label, relationship type, property name, or literal value from `data/bom/` — or from any other dataset — may appear anywhere under `src/`, including inside prompt strings. Everything shipped reasons about *shapes*: entity counts, distinct counts, degree, completeness. Nothing reasons about suppliers.

This is enforced by a test that greps `src/` for a curated list of unambiguous dataset tokens: `preferred_supplier`, `supplier_id`, `assembly_id`, `part_id`, `lead_time_days`, `unit_cost`, `minimum_order_quantity`, `SUP-`, `Screws`. Deliberately excluded from that list are the bare label names `Part`, `Product` and `Assembly`, because `Part` collides with `google.genai.types.Part`, which this codebase legitimately uses — a naive token list would fail on correct code and get deleted by the next person. The curated list is the point; do not "improve" it into a broad match.

**2. Test fixtures must not use the demo domain.** Unit tests for `graph_profile.py` construct synthetic schema dicts with neutral names. A fixture that reproduces the furniture graph would let a fix that hardcodes furniture pass, which is exactly the failure this section exists to prevent. The furniture data appears in exactly one place: the manual acceptance run, which is not automated and asserts nothing.

**3. Rubric criteria are evidence, not specifications.** Each acceptance criterion exists because it caught a general defect. None of them may be built against directly. If a proposed change would make a criterion pass without a general principle behind it, it is the wrong change — even if it works.

## Design

### Module layout

New module `src/agentic_kg/common/graph_profile.py` owns enriched-schema post-processing, degree counting, value counting, and the cache. `cypher_tools.py` is already ~200 lines covering five unrelated jobs; the profiling work is one coherent responsibility with real internal logic and is the part most worth testing in isolation. It lives in `common/` rather than `tools/` to match the existing split — `tools/` holds what agents call, `common/` holds the machinery beneath (`neo4j_for_adk.py`, `cypher_identifiers.py`).

New module `src/agentic_kg/common/adk_context.py` owns the context filter and the sentinel constant. Not in `variants.py`, which is a prompt/tool registry and should not hold behaviour; not in `tools/adk_tools.py`, because it is not a tool.

`graph_profile.py` binds `graphdb` at module level exactly as `cypher_tools.py` does, so the existing `FakeGraphDb` monkeypatch pattern works unchanged.

`tests/unit/test_imports.py` discovers modules via `rglob("*.py")` (line 24), so both new modules are smoke-tested with no registration.

### Tool result shape

`tool_success(key, value)` produces `{"status": "success", <key>: value}`, and `_payload_key` (`tool_result.py:53-64`) raises when a success result carries more than one non-status key. Sibling keys are therefore forbidden. Both tools keep a single payload key:

```
get_physical_schema(include_data_profile: bool = False)
  -> tool_success("schema", { ...library keys..., "profile": {...} })

read_neo4j_cypher(query, params)
  -> tool_success("query_result", {
       "row_count": int, "records": [...], "truncated": bool, "note": str })
```

The profile nests *inside* the schema dict. With `include_data_profile=False` the `profile` key is **absent entirely** — not present-and-empty — and `is_enhanced` stays off, so the other three consumers receive a dict byte-identical to today's.

**One flag gates both the enrichment and the profile.** An earlier draft of this document enabled `is_enhanced=True` unconditionally while also promising byte-identical output to other consumers. Those cannot both hold: enrichment adds `values`, `distinct_count`, `min` and `max` to every property *and* costs a full scan per label. Broadcasting it would have made the coordinator's "is the database empty?" check (`agent.py:27`) scan the whole graph and would have regressed `graph_construction_agent`'s latency — the specific thing this document's risk section says must not happen. The parameter is named `include_data_profile` rather than `include_degree_profile` because it now governs both.

### Context filter

`before_model_callback` on graphrag drops any `Content` whose first part's text is exactly `'For context:'`.

Detection must use that sentinel, not `role` or `author`. `_convert_foreign_event` (`contents.py:304-358`) sets `content.role='user'` (322) and `author='user'` (355), so by callback time a foreign event is indistinguishable from a human turn by role. The sentinel is prepended unconditionally at line 323, before the parts loop. `_get_contents` performs no merging of adjacent same-role contents — each event becomes one deep-copied `Content` (255-260) — so the sentinel stays reliably at index 0.

Mutating `llm_request.contents` in place and returning `None` is correct: the callback contract passes `callback_context=`/`llm_request=` as keywords, a falsy return means proceed, and the same `llm_request` object flows on to the model with no snapshot taken beforehand.

Attached via `before_model_callback=variants[AGENT_NAME].get("before_model_callback")` in `agent.py`. This is a verified no-op for v1: the field defaults to `None` (`llm_agent.py:225`) and `canonical_before_model_callbacks` returns `[]` on falsy (lines 390-391).

`include_contents='none'` was rejected: `_get_current_turn_contents` (`contents.py:264`) walks back to the most recent event authored by `'user'` *or another agent*, which would also discard graphrag's own prior turns and break follow-up questions.

### Enriched schema and completeness annotations

When `include_data_profile=True`, call `get_structured_schema` with `is_enhanced=True`, `sanitize=True`, and a timeout — all existing parameters currently unused at `cypher_tools.py:39`. When it is `False`, call it exactly as today.

**No global size gate.** The library already decides exhaustive-versus-sampled per label and per relationship type (`schema.py:830`), so a whole-graph gate would downgrade a 20-row `Supplier` label the moment an unrelated `Chunk` label crossed 10,000.

Post-process instead, using a signal the response already carries:

| Condition | Meaning | Annotation |
|---|---|---|
| `distinct_count` absent | sampled branch, 5 rows, completeness unknowable (`schema.py:572-573`) | suppress `values`, `completeness: "unknown"` |
| `len(values) < distinct_count` | exhaustive but truncated to `DISTINCT_VALUE_LIMIT` | `completeness: "partial"` |
| `len(values) == distinct_count` | complete | `completeness: "complete"` |

**Every annotation is tri-state and always present.** An annotation is never omitted to mean "not applicable" — a missing key reads as *safe* to a model, and the entities where we cannot compute an annotation are exactly the large, unfamiliar ones where being wrong is most likely. `completeness`, `uniqueness` and any later annotation each take an explicit value including `"unknown"`, and the prompt rules are phrased over the tri-state rather than over presence. This is the single most important rule for graphs larger than the demo: above `EXHAUSTIVE_SEARCH_LIMIT` the library computes nothing, and a design that signals that by silence gets *more* permissive as the graph gets less familiar.

The middle row matters: `lead_time_days` returns 10 values with `distinct_count: 27`. Presented unlabelled, a truncated list reads as complete — the same failure shape this work exists to fix.

There is a third branch worth a comment in the implementation, though it needs no special case. When a property has a RANGE index whose own statistics report 10 or fewer distinct values, the library reads the complete distinct set from the index rather than sampling rows (`schema.py:546-564`) and emits `distinct_count` despite not being row-exhaustive. Because that path always yields `len(values) == distinct_count`, the comparison rule above classifies it as complete, which is correct by construction. It cannot arise in this codebase today — the only index creation is `create_uniqueness_constraint`, which runs on ID properties — but a reader comparing the table against `schema.py` will find a branch the table doesn't obviously cover, so `graph_profile.py` should say so in a comment.

`sanitize=True` is a generic backstop over the library's own query family, **not** an embedding guard. `LIST` properties only ever receive `min_size`/`max_size` (`schema.py:584-596`, dispatch at 758-763) and `BOOLEAN`/`POINT`/`DURATION` are skipped outright, so enriched schema structurally cannot emit a vector. The real embedding exposure is `read_neo4j_cypher` — see below.

### Degree and value profile

New `include_data_profile: bool = False` parameter. Only graphrag binds a wrapper that passes `True`.

Throughout this section **entity** means a node label or a relationship type, and **entity count** means that label's node count or that type's edge count. Both annotations below apply uniformly to node and relationship properties. This symmetry is deliberate and is the main thing to protect during implementation: the two failures that motivated these annotations happened to land on opposite sides of that split — `preferred_supplier` is a relationship property, `part_name` is a node property — and defining either annotation over only the half where its own bug lived would fit the design to the bugs rather than to the class.

Contents:
- **entity counts** — node count per label, edge count per relationship type
- **endpoint degree per `(start, type, end)` pattern** — for each end, the minimum, maximum and mean edges per distinct endpoint node, plus the distinct endpoint count. Min equal to max is the signal that a pattern has fixed grain
- **per-value counts** for any property, node or relationship, carrying a `distinct_count` of 10 or fewer (`VALUE_COUNT_MAX_DISTINCT = 10`)
- a **uniqueness** annotation on any property, node or relationship: `"unique"` when `distinct_count` equals its entity count, `"non_unique"` when below, `"unknown"` when `distinct_count` was not computed

**Degree is keyed on the `(start, type, end)` triple, never on the relationship type alone.** The library already returns triples (`schema.py:56`, `REL_QUERY` yields `{start, type, end}`), so this uses data we are handed rather than adding anything. On the demo graph every type spans exactly one label pair, which makes per-type and per-pattern indistinguishable — that is precisely why keying on type alone is easy to write and impossible to notice. On a graph where one type spans several pairs, pooled degree statistics describe no actual pattern, and `min == max` can hold across the pool while being false for every individual pattern. That would reintroduce the exact grain error this profile exists to prevent. Self-referencing types compound it: start and end endpoint sets overlap, so pooled distinct-endpoint counts describe neither in-degree nor out-degree.

The threshold of 10 is not arbitrary: it is the library's own `DISTINCT_VALUE_LIMIT` (`schema.py:29`). Above it the library truncates the `values` list, so per-value counts would be partial regardless — computing them past that point would produce exactly the misleading half-complete output the annotations exist to prevent.

*Why per-value counts.* A low-cardinality property partitions the entities carrying it. Where that property sits on a relationship, the partition divides the edge set into kinds, and any aggregation that counts edges uniformly across them is meaningless. `distinct_count` reveals that a partition exists; only the per-value counts reveal whether it is balanced, and therefore whether uniform counting is defensible. The library never computes per-value counts, so this is ours. Restricting it to properties that already carry a `distinct_count` confines the work to entities the library already scanned exhaustively — cheap where it is cheap, and skipped precisely where the sampled branch would make it a lie.

*Why the non-unique mark.* Grouping or ranking by a property that does not identify its entity silently merges rows, and the merge is invisible in the result. Comparing distinct values against entity count detects this for any property without knowing anything about what the property means.

Both principles are properties of graphs, not of any dataset. That they happen to close rubric criteria A1 and B.4 is a consequence, not the reason — a rubric criterion is evidence that a general defect is real, never a specification to build against.

Queries are plain Cypher, not APOC. Everything needed is standard aggregation; Aura guarantees only APOC Core, and the enriched schema already depends on APOC through the library. A second APOC dependency of our own would be a portability liability for no gain.

**Identifier quoting and error isolation.** Label and relationship-type names are interpolated into the profile's Cypher, and they come from the database, not from a model. They must be backtick-quoted the way the library does it (`schema.py:707-709`), **not** passed through `common/cypher_identifiers.checked()` — that helper rejects anything which is not a bare identifier, so a perfectly legal extracted label like `Legal Entity` or `10-K` would raise `InvalidIdentifier` and take down the entire profile, and with it `get_physical_schema`, the tool graphrag is instructed to call first.

Each entity's profile queries are wrapped so a failure degrades that one entry to `"profile_error"` rather than failing the call. The library already does this — `except CypherTypeError: return` at `schema.py:858-859`, inside `enhance_properties`, which `enhance_schema` invokes once per entity (lines 903 and 914). This design must match it or it is strictly less robust than the thing it wraps.

**Cold-start cost is bounded, not merely cached.** Writing the terms out, where **N** is node labels, **M** relationship types, **P** distinct `(start, type, end)` patterns (P ≥ M), and **Q** qualifying properties:

| Driver | Queries |
|---|---|
| Library enriched pass — one scan per label and per relationship type | N + M |
| Entity counts — one query for all node labels, one for all relationship types | 2 |
| Endpoint degree — two per pattern, one grouped at each end | 2P |
| Per-value counts — one per qualifying property | Q |
| **Cold total** | **N + M + 2P + Q + 2** |

On the demo graph (N=4, M=3, P=3) that is roughly eighteen. On an ingested corpus with tens of labels, tens of patterns and hundreds of properties it is hundreds, in one synchronous tool call, which in `adk web` is indistinguishable from a hang. So: a query budget. Profile the top-K entities by count, mark the remainder `"not_profiled"`, and carry a per-query timeout. The cache reduces how often this is paid; it does not bound what is paid, and per-document writes during ingestion invalidate it constantly.

Note also that the counts driving the library's exhaustive-versus-sampled decision come from `apoc.meta.graph({sample: 1000, maxRels: 100})` (`schema.py:65-70`) — they are themselves sampled, and `maxRels: 100` means graphs with more than 100 relationship patterns get some types silently omitted from enrichment. Treat those counts as estimates.

**Consumer check.** `get_physical_schema` has four consumers: the `multi_agent` coordinator (`agent.py:44`), `graph_construction_agent` (`variants.py:59`), `graphrag_agent`, and `single_agent`'s `cypher_agent` (both variants). Only graphrag can use a degree profile. `graph_construction_agent`'s latency was specifically tuned in a prior session and must not regress, and `single_agent` is a separate coordinator outside this work's scope. Hence the parameter, defaulting off.

The wrapper is an explicit named function, **not** `functools.partial`. ADK derives tool identity from the callable (`function_tool.py:42-58`): a partial has no `__name__`, so it falls through to `func.__class__.__name__`, which is the literal string `"partial"`, and its `__doc__` resolves to the `functools.partial` class docstring. The tool would register under a colliding name with a description of the wrong thing. `include_data_profile` is never exposed to any model as a parameter — a model-visible flag is an optional one, and optional is what we rejected.

### Caching and invalidation

The cache is **module-level in `graph_profile.py`, not session state.** One `adk web` process serves every session, and the thing cached is a property of the database, not of a conversation. Per-session caches would disagree with each other whenever one session rebuilt the graph. This matches the existing `graphdb = get_graphdb()` singleton.

Three lazily-computed values: base schema (always), degree profile and per-value counts (only when requested).

**Two invalidation layers, because one is not honest.**

*Counter.* An attribute on the `Neo4jForADK` singleton, incremented inside `send_query` when `is_write_query(cypher_query)` is true. Every write in the codebase funnels through this one method — `kg_construction_tools.py:83,128,207` and the `cypher_tools` DDL paths all call `graphdb.send_query` directly, none go through `write_neo4j_cypher`. Instrumenting the single chokepoint rather than four call sites means a fifth write path added later is covered with nothing to remember. The counter lives in `neo4j_for_adk.py`, not `graph_profile.py`, to preserve dependency direction: `graph_profile.py` already depends on `neo4j_for_adk.py`, and the reverse would invert it.

**`is_write_query` must be extended to match `DROP`.** The current regex (`neo4j_for_adk.py:74-79`) is `MERGE|CREATE|SET|DELETE|REMOVE|ADD`, so `reset_neo4j_data` drops constraints and indexes without bumping the counter. The same gap also lets graphrag run `DROP CONSTRAINT` through the read tool today, but that half is fixed properly by access mode (see *Query execution* below) rather than by the regex. Once the regex is only a cache hint, its error profile is benign in both directions: a false positive costs one recomputation, a false negative is caught by the fingerprint layer.

*Fingerprint.* Total node and relationship counts, one combined query, revalidated per graphrag turn. The counter structurally cannot see writes from outside the process, and those happen in this workflow — during the demo the graph was built through the UI and wiped from a script. A counter-only cache would have served a schema for a database that no longer existed and stated it as fact, which is the exact failure class this work addresses.

Cost: one query per turn instead of roughly twelve. In-process writes short-circuit on the counter with no query at all.

*Documented limit:* a fingerprint of counts will not notice a change that alters only property values without changing counts. The schema shape and cardinality we cache do not move under that kind of edit.

### Query result bounding

`read_neo4j_cypher` returns `row_count` (the true count, before truncation), a `records` list capped at **50 rows** (`MAX_RETURNED_ROWS = 50`), a `truncated` flag, and a note stating that counts must come from a Cypher aggregation rather than from the returned rows.

The cap exists to bound context growth from a result set of unknown size. Its value is a judgement about how many rows are worth reading individually before the honest answer is "aggregate this instead," which is why `row_count` is always present: the number of rows *returned* must never be load-bearing. 50 is a starting point, not a tuned constant, held in a module-level constant, and no behaviour may depend on its exact value.

It also summarises array-valued properties rather than returning them whole: `to_python` (`neo4j_for_adk.py:86`) recurses into lists, so `MATCH (c:Chunk) RETURN c` would return a full embedding vector. This path has no sanitize equivalent and is the real embedding exposure.

**A row cap alone does not make this path safe.** The cases the cap is named against — a cartesian product, a bare `MATCH (n) RETURN n` — are exactly the cases that exhaust time or memory *before* any capping can happen, because `send_query` runs untimed (`neo4j_for_adk.py:147-158`) and `result_to_adk` calls `to_eager_result()` (line 81-84), materialising every row in the driver process first. A hung tool call in `adk web` is indistinguishable from a routing bug, which this project has already been burned by twice. Three changes, all in `send_query`:

1. **Per-query timeout.** Wrap the text in `neo4j.Query(text, timeout=...)`. Verified available: the installed driver is `neo4j 5.28.2` and `Query.__init__` accepts `timeout`.
2. **Server-enforced read-only for the read path.** Open the session with `default_access_mode=neo4j.READ_ACCESS` (verified: `neo4j.READ_ACCESS == 'READ'`, and `SessionConfig` accepts `default_access_mode`). The server then rejects any write with a real error.
3. **Stream instead of materialise.** Iterate the result, retain the first `MAX_RETURNED_ROWS`, and keep counting beyond them up to `ROW_COUNT_CEILING`. Memory is bounded by the retained rows, not the result size. `row_count` is exact when the query completes under the ceiling; past it the payload reports `row_count_at_least` instead, because inventing an exact number we did not finish counting would be the same species of error as everything else in this document.

**`is_write_query` is demoted to a cache hint and must never be a security boundary.** Measured against the current regex (`neo4j_for_adk.py:74-79`):

```
MATCH (c:Chunk) WHERE c.text CONTAINS 'set forth' RETURN c   -> True   (a read, rejected)
CALL apoc.refactor.mergeNodes([a,b]) YIELD node RETURN node  -> False  (a write, allowed)
```

The first fails because the regex matches "set" inside a string literal; the second because `\bMERGE\b` finds no word boundary inside `mergeNodes` — which is the exact call sub-project 2's entity resolver makes. Text matching cannot decide this question, and a longer regex only moves the boundary. Access mode moves the decision to the server, where it belongs. The regex survives solely to decide whether to bump the cache counter, where a false positive costs one recomputation and a false negative is caught by the fingerprint layer. `DROP` is still added to it for the counter's benefit.

**This change is deliberately global, not gated.** Checked all three consumers — `graphrag_agent`, `graph_construction_agent` (referenced in its instruction steps 4, 5 and 7), and `single_agent`'s `cypher_agent` (both variants). All three read rows and draw conclusions, so all three benefit and none pays overhead for a capability it cannot use. That is the opposite of the degree profile, which only graphrag can use. Supporting evidence: `graph_construction_agent/variants.py:51` already instructs that agent to *"count the label or type yourself with `read_neo4j_cypher` before quoting a number"* — a hand-written prompt rule for the thing `row_count` fixes mechanically.

No existing test asserts on this tool's return shape.

### Prompt and packaging

Lands as `graphrag_agent_v2` in `variants.py` per `CLAUDE.md:121`, carrying three keys: `instruction`, `tools` (the profile wrapper, `read_neo4j_cypher`, `finished`), and `before_model_callback`. `agent.py` flips `AGENT_NAME` and adds the one `.get(...)` line.

v1 stays intact for the acceptance A/B, then is removed once that has served its purpose. Note that the rationale `CLAUDE.md` gives for the variants pattern — mirroring the course's progressive-exercise structure — no longer applies to this project; the mechanism is kept here because it enables the A/B, not for course fidelity.

Three prompt additions, none naming a supplier, part, or lead time:

1. Counts, rankings and superlatives come from a Cypher aggregation, never from counting returned rows. Report ties as ties rather than reading rank off row order.
2. Before querying, state what is being counted and over what.
3. Do not group or rank by a property whose profile `uniqueness` is `"non_unique"`. Where it is `"unknown"`, say so in the answer rather than proceeding as if it were unique.
4. Before ordering, comparing or aggregating a property numerically, check its schema type. A `STRING` requires an explicit cast — `'9'` sorts after `'30'` — and a value carrying a currency symbol or separator will not cast cleanly.

Rules 3 and 4 point at assertions the tool computes rather than asking the model to derive them live, consistent with the completeness annotations and the row-count guard.

Rule 4 exists because this design otherwise makes a deferred bug *more* likely to fire. Every property in a CSV-built graph is a `STRING` (`docs/typing-defect.md`), and rule 1 actively directs the agent toward `max()` and `ORDER BY` — over strings, which compare lexicographically. Acceptance criterion B is itself a lead-time ordering question that can satisfy B1 through B3 on framing while the ordering underneath it is wrong. The enriched schema already reports `type: STRING` per property, so the information is present; the profile additionally marks a `STRING` property whose sampled values all parse as numeric, which is the case where a cast is both necessary and safe. This is a retrieval-side mitigation only — the construction-side fix stays deferred.

## Deliberately unchanged

`graphrag_agent` stays on `LlmKind.conversational` (`deepseek/deepseek-v4-flash`). This is the experiment: whether better information alone fixes the framing errors. Changing information and model together would make the result uninterpretable and could mean paying for a higher tier indefinitely without evidence it was needed.

Moving it is also not a config flip. `_REASONING_EFFORT = "low"` (`llm_catalog.py:49`) is applied to every `LlmKind.reasoning` agent and was tuned for many-small-tool-call orchestration (comment at 43-48), a different workload from analytical retrieval. Doing it properly means a third `LlmKind` with its own model and effort setting.

## Testing

Seven files in `tests/unit/` needing no Docker, Neo4j or API key, plus one `integration`-marked file that does. No test anywhere asserts on model prose.

The unit tests cover every mechanism in isolation. The integration file exists for one reason the unit tests structurally cannot serve: it is the only place this work runs against a graph shape other than the demo's.

**`test_adk_context.py`** — drops a foreign `Content`; keeps real user messages; keeps graphrag's own model turns including function-call and function-response parts; survives `None` content and empty `parts`.

The canary builds a real `Event` authored by another agent, passes it through ADK's own `_convert_foreign_event`, and asserts the filter catches the result. Asserting on the literal string would only prove we still agree with ourselves; this fails on drift. Motivated by the wide pin `google-adk>=1.10,<2` (`pyproject.toml:14`), under which a routine `uv sync` could change the sentinel.

**`test_graph_profile.py`** — annotation logic as pure functions over dicts. `completeness`: `distinct_count` equal to `len(values)` → `"complete"`; greater → `"partial"`; absent → `"unknown"` with `values` suppressed. `uniqueness`: `distinct_count` equal to entity count → `"unique"`; below → `"non_unique"`; absent → `"unknown"`. One test asserts that **no annotation key is ever missing** from a profiled property, since omission-means-unknown is the specific regression this design forbids.

Every case is asserted **twice, once for a node property and once for a relationship property**, since the node/relationship symmetry is the specific thing most likely to be lost during implementation. Fixtures use neutral synthetic names per generality constraint 2 — no furniture vocabulary.

**`test_generality.py`** — greps `src/` for the curated dataset-token list and fails if any appears. Cheap and mechanical, but **necessary rather than sufficient**: it catches vocabulary overfitting only. It cannot see structural overfitting — assuming one pattern per relationship type, assuming entities below `EXHAUSTIVE_SEARCH_LIMIT`, assuming single-label nodes — nor threshold overfitting, nor anything in the prompt prose. Those are covered by the shape-based integration tests below, which is where the real guarantee lives.

**`test_graph_profile_cache.py`** — same counter and fingerprint → no recompute; counter bumped → recompute; counter unchanged but fingerprint moved → recompute (the external-write case).

**`test_neo4j_for_adk.py`** — new file. `FakeGraphDb` replaces the whole `graphdb` binding and therefore bypasses `send_query` entirely, so the cache test cannot verify the counter itself. This fakes one level lower, at the driver/session boundary, and asserts the counter increments on `MERGE`/`SET`/`DROP` and holds on `MATCH`. `neo4j_for_adk.py` currently has no unit coverage at all — only `tests/integration/test_neo4j_for_adk_integration.py`, which is excluded by default.

**`test_cypher_tools.py`** additions — row count present and exact under the ceiling, `row_count_at_least` past it, truncation flag set at the cap, array properties summarised rather than returned whole, `is_write_query` matches `DROP` (as a *cache hint*; rejection of writes is the server's job and is asserted in the integration file), and the byte-identical guard: `get_physical_schema()` with no argument returns a dict with no `profile` key **and no `values`/`distinct_count` keys**, proving `is_enhanced` stayed off for the other three consumers.

**`test_graphrag_context_filtering.py`** — extends the existing `ScriptedLlm` (`test_schema_refinement_loop_turn_cap.py:36`) with one line appending each `llm_request`. Seeds a session containing a foreign-agent event, runs both variants through `InMemoryRunner`, asserts v2's captured requests contain no sentinel **and that v1's do**. Without the negative control, a test passing because the fixture never produced foreign context would look identical to a working filter.

### `tests/integration/test_graph_profile_shapes.py` (marked `integration`)

The only test here that needs a database. It exists because the profile builds Cypher by interpolating names taken from the graph, and nothing else ever executes it. Three synthetic graphs built with Testcontainers, following the existing `tests/integration/` setup:

1. **Multi-pattern** — one relationship type spanning two different `(start, end)` label pairs
2. **Self-referencing** — a relationship type whose start and end are the same label
3. **Awkward names and shapes** — a multi-label node, and a label whose name is not a bare identifier (`Legal Entity`) to prove backtick quoting holds where `checked()` would have raised

Assertions, all shape-based and domain-free:
- the profile completes without error on all three, within the query budget
- degree is reported per `(start, type, end)` pattern and matches hand-computed ground truth
- annotations that cannot be computed are reported as `"unknown"`, never omitted
- a failure profiling one entity degrades that entry and leaves the rest intact
- a result larger than the cap reports a true `row_count` and `truncated`
- a deliberately malformed query returns a structured error, not a hang
- a write submitted through `read_neo4j_cypher` is rejected by the server, including `CALL apoc.refactor.mergeNodes(...)`, which the regex does not catch

This is the only evidence in the whole plan that any of this works on a graph other than the demo. Without it, "universal" is an assertion.

## Acceptance

Manual, after implementation. Two questions, 3 runs each, on both v1 and v2 — 12 runs, graded per criterion rather than collapsed to a verdict.

Repeats rather than a pinned temperature: there is no sampling control anywhere in the codebase (`get_llm` passes only `timeout`, `num_retries`, `max_tokens`, and reasoning-tier `reasoning_effort`), and `_llm_instances` is cached per `LlmKind`, so the conversational `LiteLlm` object is shared by four agents. There is no per-agent handle to pin.

`adk web` must be restarted between arms — a server started before an edit serves stale code (`SESSION_HANDOFF.md:14`). Neo4j persists across the restart, so "same graph" holds; "same session" does not.

**Question A — sourcing concentration**
- A1: uses the preferred-supplier flag in at least one aggregation
- A2: does not present Sweden and Canada as jointly concentrated — either ranks on primary sourcing or separates quotes-available from actually-sourced
- A3: any disconnected-supplier list is exactly `SUP-007`/`010`/`017`/`020`, or absent
- A4: every factual claim traces to a query in that turn

**Question B — longest lead times**
- B1: names the aggregation used (worst-case, soonest-possible, or preferred-supplier); all three are correct, leaving it unstated is not
- B2: does not present worst-case quotes as the lead time without flagging them non-preferred
- B3: any most-exposed-product claim matches a Cypher aggregation (Linköping Bed at 6, not Malmö at 4)
- B4: does not merge distinct parts sharing a name

## Risks and limitations

- **The ADK sentinel is an internal string under a wide version pin.** Mitigated by the behavioural canary, not eliminated.
- **The cache and query-tool changes are global**, so the v1/v2 A/B does not isolate them. Only the prompt, the profile wrapper and the context filter differ between arms.
- **`graph_construction_agent`'s latency must not regress.** It is the reason the degree profile is parameterized rather than broadcast; verify its schema payload is unchanged.
- **Fingerprint invalidation is blind to property-only edits** that leave node and relationship counts unchanged.
- **The profile costs `N + M + 2P + Q + 2` queries when cold** — see the cost table under *Degree and value profile* for the terms. Roughly eighteen on the demo graph; hundreds on an ingested corpus. The cache reduces how often that is paid and the query budget bounds what is paid; neither alone is sufficient, and this is the largest remaining performance risk in the design.
- **This work verifies the agent is well-informed, not that it reasons correctly.** If framing errors persist with the degree profile in context, that is evidence the model tier is the binding constraint — a useful outcome, not a failure.

## Known gaps, named rather than implied

These are not addressed here and should not be assumed covered by the words "universal" or "without error".

- **The graph carries no description of what it means.** The context filter removes every other agent's output, which is correct — the goal text from the demo session contained frozen counts stated as fact, and injecting it would rebuild the hallucination through a cleaner channel. But on a graph whose labels are not self-describing, graphrag will have no account of what the graph *is*. That is a real problem and a different one: semantic opacity, not stale facts. The right fix is for the graph to describe itself, not for conversational state to be replayed. Out of scope here.
- **No vector or hybrid retrieval.** Sub-project 2 produces chunked narrative text, and Cypher alone cannot answer a question about what a document *says*. `read_neo4j_cypher` is the only retrieval tool graphrag has. That is a separate spec.
- **The typing defect remains.** Rule 4 mitigates it at query time; it does not fix it. See `docs/typing-defect.md`.
- **The acceptance rubric is domain-specific by construction.** It grades reasoning on a dataset whose ground truth is known. The universality claim rests on the shape-based integration tests, not on the rubric — these two things measure different properties and neither substitutes for the other.
- **The cache has no lock**, and ADK may run turns concurrently. Worst case is duplicate computation, not incorrect results.
