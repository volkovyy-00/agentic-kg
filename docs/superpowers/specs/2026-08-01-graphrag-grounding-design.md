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

**In:** seven changes to the retrieval path — context filtering, enriched schema with completeness annotations, degree and value profiling, caching with invalidation, query-result bounding, three prompt rules, and `graphrag_agent_v2` packaging.

**Out:** the construction-side typing defect (every property stored as `STRING`, `unit_cost` as `'$42.73'`). See `docs/typing-defect.md`. **Also out:** any change to graphrag's model tier — see *Deliberately unchanged* below.

## Design

### Module layout

New module `src/agentic_kg/common/graph_profile.py` owns enriched-schema post-processing, degree counting, value counting, and the cache. `cypher_tools.py` is already ~200 lines covering five unrelated jobs; the profiling work is one coherent responsibility with real internal logic and is the part most worth testing in isolation. It lives in `common/` rather than `tools/` to match the existing split — `tools/` holds what agents call, `common/` holds the machinery beneath (`neo4j_for_adk.py`, `cypher_identifiers.py`).

New module `src/agentic_kg/common/adk_context.py` owns the context filter and the sentinel constant. Not in `variants.py`, which is a prompt/tool registry and should not hold behaviour; not in `tools/adk_tools.py`, because it is not a tool.

`graph_profile.py` binds `graphdb` at module level exactly as `cypher_tools.py` does, so the existing `FakeGraphDb` monkeypatch pattern works unchanged.

`tests/unit/test_imports.py` discovers modules via `rglob("*.py")` (line 24), so both new modules are smoke-tested with no registration.

### Tool result shape

`tool_success(key, value)` produces `{"status": "success", <key>: value}`, and `_payload_key` (`tool_result.py:53-64`) raises when a success result carries more than one non-status key. Sibling keys are therefore forbidden. Both tools keep a single payload key:

```
get_physical_schema(include_degree_profile: bool = False)
  -> tool_success("schema", { ...library keys..., "profile": {...} })

read_neo4j_cypher(query, params)
  -> tool_success("query_result", {
       "row_count": int, "records": [...], "truncated": bool, "note": str })
```

The profile nests *inside* the schema dict. With `include_degree_profile=False` the `profile` key is **absent entirely** — not present-and-empty — so the other consumers receive a byte-identical dict.

### Context filter

`before_model_callback` on graphrag drops any `Content` whose first part's text is exactly `'For context:'`.

Detection must use that sentinel, not `role` or `author`. `_convert_foreign_event` (`contents.py:304-358`) sets `content.role='user'` (322) and `author='user'` (355), so by callback time a foreign event is indistinguishable from a human turn by role. The sentinel is prepended unconditionally at line 323, before the parts loop. `_get_contents` performs no merging of adjacent same-role contents — each event becomes one deep-copied `Content` (255-260) — so the sentinel stays reliably at index 0.

Mutating `llm_request.contents` in place and returning `None` is correct: the callback contract passes `callback_context=`/`llm_request=` as keywords, a falsy return means proceed, and the same `llm_request` object flows on to the model with no snapshot taken beforehand.

Attached via `before_model_callback=variants[AGENT_NAME].get("before_model_callback")` in `agent.py`. This is a verified no-op for v1: the field defaults to `None` (`llm_agent.py:225`) and `canonical_before_model_callbacks` returns `[]` on falsy (lines 390-391).

`include_contents='none'` was rejected: `_get_current_turn_contents` (`contents.py:264`) walks back to the most recent event authored by `'user'` *or another agent*, which would also discard graphrag's own prior turns and break follow-up questions.

### Enriched schema and completeness annotations

Call `get_structured_schema` with `is_enhanced=True`, `sanitize=True`, and a timeout. All three are existing parameters currently unused at `cypher_tools.py:39`.

**No global size gate.** The library already decides exhaustive-versus-sampled per label and per relationship type (`schema.py:830`), so a whole-graph gate would downgrade a 20-row `Supplier` label the moment an unrelated `Chunk` label crossed 10,000.

Post-process instead, using a signal the response already carries:

| Condition | Meaning | Annotation |
|---|---|---|
| `distinct_count` absent | sampled branch, 5 rows, completeness unknowable (`schema.py:572-573`) | suppress `values`, mark non-exhaustive |
| `len(values) < distinct_count` | exhaustive but truncated to `DISTINCT_VALUE_LIMIT` | mark partial |
| `len(values) == distinct_count` | complete | leave as-is |

The middle row matters: `lead_time_days` returns 10 values with `distinct_count: 27`. Presented unlabelled, a truncated list reads as complete — the same failure shape this work exists to fix.

There is a third branch worth a comment in the implementation, though it needs no special case. When a property has a RANGE index whose own statistics report 10 or fewer distinct values, the library reads the complete distinct set from the index rather than sampling rows (`schema.py:546-564`) and emits `distinct_count` despite not being row-exhaustive. Because that path always yields `len(values) == distinct_count`, the comparison rule above classifies it as complete, which is correct by construction. It cannot arise in this codebase today — the only index creation is `create_uniqueness_constraint`, which runs on ID properties — but a reader comparing the table against `schema.py` will find a branch the table doesn't obviously cover, so `graph_profile.py` should say so in a comment.

`sanitize=True` is a generic backstop over the library's own query family, **not** an embedding guard. `LIST` properties only ever receive `min_size`/`max_size` (`schema.py:584-596`, dispatch at 758-763) and `BOOLEAN`/`POINT`/`DURATION` are skipped outright, so enriched schema structurally cannot emit a vector. The real embedding exposure is `read_neo4j_cypher` — see below.

### Degree and value profile

New `include_degree_profile: bool = False` parameter. Only graphrag binds a wrapper that passes `True`.

Contents:
- per-label node counts
- per-relationship-type edge count and endpoint cardinality (for `SUPPLIES`: 176 edges, every `Part` has exactly 2)
- per-value counts for properties that carry a `distinct_count` of **10 or fewer** (`VALUE_COUNT_MAX_DISTINCT = 10`)
- a **non-unique** mark on any property whose `distinct_count` is below its label's node count

The threshold of 10 is not arbitrary: it is the library's own `DISTINCT_VALUE_LIMIT` (`schema.py:29`). Above it the library truncates the `values` list, so per-value counts would be partial regardless — computing them past that point would produce exactly the misleading half-complete output the annotations exist to prevent.

The per-value counts are the fact that makes failure 2 legible: `preferred_supplier` splits 88/88. The library reports `distinct_count` (how many distinct values) but never per-value row counts, so this is ours to compute. Restricting it to properties that already carry a `distinct_count` means we only compute it where the library took the exhaustive branch and already scanned that label — cheap where it is cheap, skipped precisely where the sampled branch would make it a lie.

The non-unique mark closes rubric B.4 with data: six distinct `Part` nodes are named "Screws", and 16 names are reused across 63 of 88 parts.

Queries are plain Cypher, not APOC. Everything needed is standard aggregation; Aura guarantees only APOC Core, and the enriched schema already depends on APOC through the library. A second APOC dependency of our own would be a portability liability for no gain.

**Consumer check.** `get_physical_schema` has four consumers: the `multi_agent` coordinator (`agent.py:44`), `graph_construction_agent` (`variants.py:59`), `graphrag_agent`, and `single_agent`'s `cypher_agent` (both variants). Only graphrag can use a degree profile. `graph_construction_agent`'s latency was specifically tuned in a prior session and must not regress, and `single_agent` is a separate coordinator outside this work's scope. Hence the parameter, defaulting off.

The wrapper is an explicit named function, **not** `functools.partial`. ADK derives tool identity from the callable (`function_tool.py:42-58`): a partial has no `__name__`, so it falls through to `func.__class__.__name__`, which is the literal string `"partial"`, and its `__doc__` resolves to the `functools.partial` class docstring. The tool would register under a colliding name with a description of the wrong thing. `include_degree_profile` is never exposed to any model as a parameter — a model-visible flag is an optional one, and optional is what we rejected.

### Caching and invalidation

The cache is **module-level in `graph_profile.py`, not session state.** One `adk web` process serves every session, and the thing cached is a property of the database, not of a conversation. Per-session caches would disagree with each other whenever one session rebuilt the graph. This matches the existing `graphdb = get_graphdb()` singleton.

Three lazily-computed values: base schema (always), degree profile and per-value counts (only when requested).

**Two invalidation layers, because one is not honest.**

*Counter.* An attribute on the `Neo4jForADK` singleton, incremented inside `send_query` when `is_write_query(cypher_query)` is true. Every write in the codebase funnels through this one method — `kg_construction_tools.py:83,128,207` and the `cypher_tools` DDL paths all call `graphdb.send_query` directly, none go through `write_neo4j_cypher`. Instrumenting the single chokepoint rather than four call sites means a fifth write path added later is covered with nothing to remember. The counter lives in `neo4j_for_adk.py`, not `graph_profile.py`, to preserve dependency direction: `graph_profile.py` already depends on `neo4j_for_adk.py`, and the reverse would invert it.

**`is_write_query` must be extended to match `DROP`.** The current regex (`neo4j_for_adk.py:74-79`) is `MERGE|CREATE|SET|DELETE|REMOVE|ADD`. Two consequences: `reset_neo4j_data` drops constraints and indexes without bumping the counter, and — independently — `read_neo4j_cypher` uses this same function to reject writes, so `graphrag` can today execute `DROP CONSTRAINT` through the read-only tool. Pre-existing, in a function this design already modifies, and the counter's correctness depends on it. False positives on this regex cost only a cache miss; false negatives cost correctness.

*Fingerprint.* Total node and relationship counts, one combined query, revalidated per graphrag turn. The counter structurally cannot see writes from outside the process, and those happen in this workflow — during the demo the graph was built through the UI and wiped from a script. A counter-only cache would have served a schema for a database that no longer existed and stated it as fact, which is the exact failure class this work addresses.

Cost: one query per turn instead of roughly twelve. In-process writes short-circuit on the counter with no query at all.

*Documented limit:* a fingerprint of counts will not notice a change that alters only property values without changing counts. The schema shape and cardinality we cache do not move under that kind of edit.

### Query result bounding

`read_neo4j_cypher` returns `row_count` (the true count, before truncation), a `records` list capped at **50 rows** (`MAX_RETURNED_ROWS = 50`), a `truncated` flag, and a note stating that counts must come from a Cypher aggregation rather than from the returned rows.

50 is chosen against the observed failure: the queries behind failure 3 returned roughly 15 to 20 rows, so a normal exploratory query stays untruncated and the change is invisible in ordinary use, while `row_count` is always present to make eyeballing unnecessary. The cap bounds pathological results — an accidental cartesian product, or a bare `MATCH (n) RETURN n`. It also summarises array-valued properties rather than returning them whole: `to_python` (`neo4j_for_adk.py:86`) recurses into lists, so `MATCH (c:Chunk) RETURN c` would return a full embedding vector. This path has no sanitize equivalent and is the real embedding exposure.

**This change is deliberately global, not gated.** Checked all three consumers — `graphrag_agent`, `graph_construction_agent` (referenced in its instruction steps 4, 5 and 7), and `single_agent`'s `cypher_agent` (both variants). All three read rows and draw conclusions, so all three benefit and none pays overhead for a capability it cannot use. That is the opposite of the degree profile, which only graphrag can use. Supporting evidence: `graph_construction_agent/variants.py:51` already instructs that agent to *"count the label or type yourself with `read_neo4j_cypher` before quoting a number"* — a hand-written prompt rule for the thing `row_count` fixes mechanically.

No existing test asserts on this tool's return shape.

### Prompt and packaging

Lands as `graphrag_agent_v2` in `variants.py` per `CLAUDE.md:121`, carrying three keys: `instruction`, `tools` (the profile wrapper, `read_neo4j_cypher`, `finished`), and `before_model_callback`. `agent.py` flips `AGENT_NAME` and adds the one `.get(...)` line.

v1 stays intact for the acceptance A/B, then is removed once that has served its purpose. Note that the rationale `CLAUDE.md` gives for the variants pattern — mirroring the course's progressive-exercise structure — no longer applies to this project; the mechanism is kept here because it enables the A/B, not for course fidelity.

Three prompt additions, none naming a supplier, part, or lead time:

1. Counts, rankings and superlatives come from a Cypher aggregation, never from counting returned rows. Report ties as ties rather than reading rank off row order.
2. Before querying, state what is being counted and over what.
3. Do not group or rank by a property the profile marks non-unique.

Rule 3 points at an assertion the tool computes rather than asking the model to compare two numbers live — consistent with the completeness annotations and the row-count guard, which all move judgment from the model into code.

## Deliberately unchanged

`graphrag_agent` stays on `LlmKind.conversational` (`deepseek/deepseek-v4-flash`). This is the experiment: whether better information alone fixes the framing errors. Changing information and model together would make the result uninterpretable and could mean paying for a higher tier indefinitely without evidence it was needed.

Moving it is also not a config flip. `_REASONING_EFFORT = "low"` (`llm_catalog.py:49`) is applied to every `LlmKind.reasoning` agent and was tuned for many-small-tool-call orchestration (comment at 43-48), a different workload from analytical retrieval. Doing it properly means a third `LlmKind` with its own model and effort setting.

## Testing

Six files, all in `tests/unit/`, none marked `integration` — no Docker, Neo4j or API key required. No test asserts on model prose.

**`test_adk_context.py`** — drops a foreign `Content`; keeps real user messages; keeps graphrag's own model turns including function-call and function-response parts; survives `None` content and empty `parts`.

The canary builds a real `Event` authored by another agent, passes it through ADK's own `_convert_foreign_event`, and asserts the filter catches the result. Asserting on the literal string would only prove we still agree with ourselves; this fails on drift. Motivated by the wide pin `google-adk>=1.10,<2` (`pyproject.toml:14`), under which a routine `uv sync` could change the sentinel.

**`test_graph_profile.py`** — annotation logic as pure functions over dicts: `distinct_count` present and equal to `len(values)` → complete; present and greater → partial; absent → non-exhaustive with values suppressed; below the label's node count → marked non-unique; equal → not marked.

**`test_graph_profile_cache.py`** — same counter and fingerprint → no recompute; counter bumped → recompute; counter unchanged but fingerprint moved → recompute (the external-write case).

**`test_neo4j_for_adk.py`** — new file. `FakeGraphDb` replaces the whole `graphdb` binding and therefore bypasses `send_query` entirely, so the cache test cannot verify the counter itself. This fakes one level lower, at the driver/session boundary, and asserts the counter increments on `MERGE`/`SET`/`DROP` and holds on `MATCH`. `neo4j_for_adk.py` currently has no unit coverage at all — only `tests/integration/test_neo4j_for_adk_integration.py`, which is excluded by default.

**`test_cypher_tools.py`** additions — row count present, truncation flag set at the limit, array properties summarised, `is_write_query` matches `DROP` and `read_neo4j_cypher` rejects it, and `get_physical_schema()` with no argument returns a dict with no `profile` key.

**`test_graphrag_context_filtering.py`** — extends the existing `ScriptedLlm` (`test_schema_refinement_loop_turn_cap.py:36`) with one line appending each `llm_request`. Seeds a session containing a foreign-agent event, runs both variants through `InMemoryRunner`, asserts v2's captured requests contain no sentinel **and that v1's do**. Without the negative control, a test passing because the fixture never produced foreign context would look identical to a working filter.

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
- **The profile costs N+M aggregate queries** when cold (one per label, one per relationship type, plus one per qualifying property) — about a dozen on the demo graph. The cache is what makes this acceptable; without it the design would not be proposed.
- **This work verifies the agent is well-informed, not that it reasons correctly.** If framing errors persist with the degree profile in context, that is evidence the model tier is the binding constraint — a useful outcome, not a failure.
