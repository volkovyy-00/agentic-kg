# Defect: every graph property is a STRING

Every property written into the Neo4j graph by the CSV construction path is a Neo4j STRING, including
values that are semantically numeric or boolean. The CSV reader hands the loaders `str` values, the
construction plan carries no type information to override them with, and the loaders write them
verbatim. The result is that no numeric comparison, range filter, or aggregation works without an
explicit `toInteger()` / `toFloat()` cast (and for currency, a `replace()` first), and — worse —
sorting or comparing without a cast does not fail, it silently returns lexicographic order (`'9' > '30'`).
An LLM-driven retrieval agent will not reliably remember to cast on every query, so this shows up as
wrong answers rather than errors. **Nothing here is fixed by this document; it is a handoff note.**

## Evidence

From the live graph (reported, not re-verified in this session):

- `SUPPLIES.unit_cost` = `'$42.73'` — STRING, currency symbol retained
- `SUPPLIES.lead_time_days` = `'8'` — STRING
- `SUPPLIES.minimum_order_quantity`, `HAS_ASSEMBLY.quantity`, `HAS_PART.quantity`, `Product.price` — all STRING

Confirmed in the source CSVs under `data/bom/`:

- `part_supplier_mapping.csv`: `lead_time_days=8`, `unit_cost=$42.73`, `minimum_order_quantity=14`,
  and `preferred_supplier=yes` — the last is a BOOLEAN candidate the original report did not mention.
- `products.csv`: `price=$246` — currency symbol, and integral rather than decimal.
- `assemblies.csv` / `components.csv`: `quantity=3`, `quantity=1`.

Caveat on names: relationship types are chosen by the schema-proposal LLM per run, so `SUPPLIES` /
`HAS_ASSEMBLY` / `HAS_PART` are that run's names (the tests use e.g. `ASSEMBLY_OF`). The defect is in the
loading path, not tied to any particular type name.

## Root cause (verified against the code)

**1. `src/agentic_kg/common/csv_reader.py`** — `read_csv_batches` (definition at line 41) builds rows as
`{key: value for key, value in zip(header, row)}` (line 64) from a `clevercsv` reader. Every value is a
Python `str`; there is no dtype handling. This is explicit, not accidental — the module docstring
(lines 4–5) already says:

> Values stay strings, exactly as LOAD CSV produced them — typed fields are deliberately out of scope.

**2. `src/agentic_kg/tools/construction_plan_tools.py`** — *correction to the original description*: there
is no construction-plan **model**. The plan is a plain `dict` assembled inline by
`propose_node_construction` (rule literal at lines 74–83) and `propose_relationship_construction`
(lines 202–212), with `"properties": proposed_properties or []` where `proposed_properties: list[str]`
is a tool parameter. No Pydantic model, no `TypedDict`, no validator — so "bare `List[str]` of property
names, nowhere to record a type" is correct in effect, but the fix is *introducing* a shape, not editing
one. `check_construction_plan_consistency` also treats the list as bare names (line 319,
`known_columns = {unique_column, *(node_rule.get("properties") or [])}`).

**3. `src/agentic_kg/tools/kg_construction_tools.py`** — `load_nodes_from_csv` (line 43) sends:

```cypher
FOREACH (k IN [p IN $properties WHERE row[p] IS NOT NULL] | SET n[k] = row[k])   # line 63
```

and `import_relationships` (line 160; the original note's "~line 173" is where `properties` is read,
line 173) sends the same shape for `r` at line 179. Whatever string the reader produced is written as-is.

On the NULL filter, accurately: the comprehension drops property names whose row value **IS NULL**,
because `SET n[k] = null` *removes* an existing property rather than skipping it (see the comment at
lines 56–60). A value is NULL only when the key is absent from the row dict — which happens when a row
is shorter than the header (`csv_reader` omits missing keys rather than padding). **An empty CSV field is
`''`, not NULL**, so empty strings pass the filter today and are written as empty-string properties.
Any coercion step has to decide what `''` becomes for an INTEGER/FLOAT/BOOLEAN target.

**Precedent for typed values: none.** `grep` for `toInteger|toFloat|toBoolean|date(|datetime(` across
`src/` and `tests/` returns no hits. There is no existing typed-value handling anywhere to imitate.

## Blast radius

Code:

- `src/agentic_kg/common/csv_reader.py` (if inference or coercion happens at read time)
- `src/agentic_kg/tools/construction_plan_tools.py` (plan shape, both propose paths, both bulk paths,
  `check_construction_plan_consistency`)
- `src/agentic_kg/tools/kg_construction_tools.py` (both loaders' `FOREACH`)
- `src/agentic_kg/coordinators/multi_agent/sub_agents/schema_proposal_agent/agent.py` and its
  `variants.py` (instructions describe the plan's fields to the LLM; the propose/critique/refine loop
  reads and writes the plan)

Tests that actually assert on the plan's shape:

- `tests/unit/test_construction_plan_tools.py` — asserts `plan[...]["properties"] == [<str>, ...]` at
  lines 62, 77, 128, 178, 233, 349, 353, and passes literal `"properties"` lists at 218–226. **This is the
  file that breaks if the list stops being a list of strings.**
- `tests/unit/test_kg_construction_tools.py` — builds rules with `"properties": [...]` at lines 73, 105,
  125, 293, 447, 508, 529.
- `tests/integration/test_csv_loading_integration.py` — rules at lines 24, 33, 43, 50, including the
  typed-looking `["lead_time_days", "unit_cost"]`; this is where a coercion assertion would naturally go.
- *Correction*: `tests/unit/test_schema_refinement_loop.py`, `..._callbacks.py`, and `..._turn_cap.py`
  do **not** assert on the plan's shape at all — the only relevant line is
  `test_schema_refinement_loop.py:82`, asserting the string `"get_proposed_construction_plan"` appears in
  an instruction. The refinement loop is a blast-radius risk for *behaviour* (the LLM must now produce
  types), not for these tests.

## Open design questions

1. **How does the plan carry a type?** Parallel `property_types: dict[str, str]`, or `properties` becomes
   a list of objects (`{"name": ..., "type": ...}`)? The parallel-dict option keeps every existing
   `properties` assertion and `check_construction_plan_consistency` working and is additive; the
   list-of-objects option is cleaner but breaks every call site and test listed above. Undecided.
2. **Who decides the type?**
   - *Inferred from CSV sampling* — no LLM cost, works without user input, but a sample can be wrong
     (a leading-zero part code becomes an INTEGER; a column that is numeric for 1000 rows and free text
     at row 5000 fails late).
   - *Proposed by the schema-proposal LLM* — sees column names and user intent, so it can tell
     `part_id` from `quantity`, but it is nondeterministic, costs a round trip, and the critique loop
     then has to be able to challenge a type.
   - *Declared by the user* — authoritative, but pushes schema work onto a user the workflow is designed
     to keep at the approval level.
   - These are not exclusive: infer, let the LLM override, let the user override. Cost/complexity of the
     three-layer version is the thing to weigh. Undecided.
3. **Where does coercion run, and what does it do with dirty values?** Python-side in the loader before
   the batch is sent, or Cypher-side in the `FOREACH`. Cases that must have an answer: currency symbols
   (`$42.73`), thousands separators, empty string, and values that simply fail to parse. Options are
   skip the property, write it as a string, or fail the batch — each changes what "success" means in the
   loader's `ToolResult`. Note the existing NULL filter (above) does *not* already cover the empty-string
   case. Undecided.
4. **Which Neo4j types to target?** INTEGER, FLOAT, BOOLEAN, DATE are the ones the bundled data
   motivates (`quantity`, `unit_cost`, `preferred_supplier`, none yet for DATE). Whether to ship all four
   at once or start with INTEGER/FLOAT is open.
5. **Existing graphs.** Does the fix require a rebuild, or is a migration/re-run path expected? Loading is
   `MERGE`-based, so a re-run over a fixed plan would overwrite properties in place — worth confirming.

## Not in scope

A separate workstream is fixing the retrieval-side agent (`graphrag_agent`). This document covers
**construction-side typing only** — how values get into the graph, not how queries read them out. Do not
fold retrieval-side prompt or query changes into this work.
