# graphrag Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `graphrag_agent` answer from the graph rather than from conversational memory, and make the query path safe and shape-general rather than tuned to the bundled demo graph.

**Architecture:** Three layers, built bottom-up. `neo4j_for_adk.py` gains a write counter, a per-query timeout, server-enforced read-only access, and streamed row retention. A new `graph_profile.py` turns the library's enriched schema into annotated facts (completeness, uniqueness, degree per pattern, per-value counts) behind a two-layer cache. `cypher_tools.py` exposes both through gated tool functions, and a new `graphrag_agent_v2` variant binds them alongside a `before_model_callback` that strips other agents' output from the model's context.

**Tech Stack:** Python 3.12, Google ADK (`google-adk>=1.10,<2`), `neo4j` 5.28.2 driver, `neo4j-graphrag` schema helpers, LiteLLM via OpenRouter, pytest, `uv`.

**Spec:** `docs/superpowers/specs/2026-08-01-graphrag-grounding-design.md` — read it before starting. This plan implements it and does not revisit its decisions.

## Global Constraints

- **Single payload key.** `tool_success(key, value)` returns `{"status": "success", <key>: value}`. `_payload_key` (`src/agentic_kg/common/tool_result.py:53-64`) raises when a success result has more than one non-status key. Never add sibling keys — nest instead.
- **`include_data_profile=False` must return a dict byte-identical to today's**: no `profile` key, and no `values`/`distinct_count` keys. Three other consumers depend on this — the `multi_agent` coordinator, `graph_construction_agent` (latency previously tuned, must not regress), and `single_agent`'s `cypher_agent`.
- **Degree is keyed on `(start, type, end)` triples**, never on relationship type alone.
- **Every annotation is tri-state and always present.** Never omit a key to mean "not computable" — use `"unknown"`.
- **Backtick-quote** label and relationship-type names in profile Cypher. Do **not** use `common/cypher_identifiers.checked()`; these names come from the database, not from a model, and `checked()` raises on legal names like `Legal Entity`.
- **Per-entity error isolation.** One failing entity degrades to `"profile_error"` and never fails the whole call.
- **No dataset vocabulary anywhere in `src/`,** including prompt strings. Enforced by `tests/unit/test_generality.py` (Task 2).
- **`graphrag_agent` stays on `LlmKind.conversational`.** No model tier change.
- **Leave `graphrag_agent_v1` intact.** v2 is added alongside it for the acceptance A/B.
- **No test asserts on model prose.**
- **Baseline is 172 passed / 3 skipped.** `uv run pytest` must stay green after every task.
- Run tests with `uv run pytest`. `addopts = "-q -m 'not integration'"` already excludes integration tests; run those with `uv run pytest -m integration`.

## File Structure

| File | Responsibility |
|---|---|
| `src/agentic_kg/common/adk_context.py` | **New.** Foreign-context sentinel + the `before_model_callback` filter. Nothing else. |
| `src/agentic_kg/common/graph_profile.py` | **New.** Annotation of enriched-schema properties, profile queries, and the module-level cache. |
| `src/agentic_kg/common/neo4j_for_adk.py` | **Modify.** Write counter, `DROP` in `is_write_query`, query timeout, read-access session, streamed row retention. |
| `src/agentic_kg/tools/cypher_tools.py` | **Modify.** Gate `get_physical_schema`, add the graphrag wrapper, reshape `read_neo4j_cypher`. |
| `.../sub_agents/graphrag_agent/variants.py` | **Modify.** Add `graphrag_agent_v2` entry. |
| `.../sub_agents/graphrag_agent/agent.py` | **Modify.** Select v2, attach callback. |

---

### Task 1: Foreign-context filter

**Files:**
- Create: `src/agentic_kg/common/adk_context.py`
- Test: `tests/unit/test_adk_context.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `FOREIGN_CONTEXT_SENTINEL: str` and `drop_foreign_context(callback_context, llm_request) -> None`. Task 9 binds the function into `variants["graphrag_agent_v2"]["before_model_callback"]`; Task 11 asserts on its end-to-end effect.

Background: ADK rewrites another agent's event into the current agent's context with `content.role = 'user'` and `author = 'user'` (`contents.py:322, 355`), so role cannot distinguish it from a real human turn. It unconditionally prepends `Part(text='For context:')` at line 323, before the parts loop, and `_get_contents` deep-copies one `Content` per event with no merging (line 258). The sentinel at index 0 is therefore the only reliable signal.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_adk_context.py
"""Unit tests for the graphrag foreign-context filter.

The canary test deliberately drives ADK's own _convert_foreign_event rather
than asserting on our copy of the sentinel string: asserting our constant
equals our constant proves nothing. google-adk is pinned >=1.10,<2, so a
routine `uv sync` can change that wording; this test is what notices.
"""
from google.adk.events.event import Event
from google.adk.flows.llm_flows.contents import _convert_foreign_event
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from agentic_kg.common.adk_context import (
    FOREIGN_CONTEXT_SENTINEL,
    drop_foreign_context,
)


def _content(role, *parts):
    return types.Content(role=role, parts=list(parts))


def _request(*contents):
    return LlmRequest(contents=list(contents))


def test_drops_content_carrying_the_sentinel():
    foreign = _content("user", types.Part(text=FOREIGN_CONTEXT_SENTINEL),
                       types.Part(text="[other_agent] said: 4 suppliers are orphaned"))
    req = _request(foreign)
    drop_foreign_context(None, req)
    assert req.contents == []


def test_keeps_real_user_message():
    human = _content("user", types.Part(text="which countries dominate sourcing?"))
    req = _request(human)
    drop_foreign_context(None, req)
    assert req.contents == [human]


def test_keeps_own_model_turn_and_tool_parts():
    said = _content("model", types.Part(text="let me check the schema"))
    call = _content("model", types.Part(
        function_call=types.FunctionCall(name="read_neo4j_cypher", args={"query": "MATCH (n) RETURN n"})))
    resp = _content("user", types.Part(
        function_response=types.FunctionResponse(name="read_neo4j_cypher", response={"status": "success"})))
    req = _request(said, call, resp)
    drop_foreign_context(None, req)
    assert req.contents == [said, call, resp]


def test_drops_only_the_foreign_content_from_a_mixed_history():
    human = _content("user", types.Part(text="hello"))
    foreign = _content("user", types.Part(text=FOREIGN_CONTEXT_SENTINEL),
                       types.Part(text="[x] said: hi"))
    own = _content("model", types.Part(text="hi back"))
    req = _request(human, foreign, own)
    drop_foreign_context(None, req)
    assert req.contents == [human, own]


def test_survives_empty_and_none_content():
    empty = types.Content(role="user", parts=[])
    bare = types.Content(role="user")
    req = _request(empty, bare)
    drop_foreign_context(None, req)
    assert req.contents == [empty, bare]


def test_returns_none_so_the_model_call_proceeds():
    req = _request(_content("user", types.Part(text="hi")))
    assert drop_foreign_context(None, req) is None


def test_canary_adk_still_marks_foreign_events_with_our_sentinel():
    """Fails if a google-adk upgrade changes the foreign-event wording."""
    original = Event(
        author="schema_critic_agent",
        content=_content("model", types.Part(text="4 suppliers have no quote rows")),
    )
    converted = _convert_foreign_event(original)

    assert converted.content.parts[0].text == FOREIGN_CONTEXT_SENTINEL, (
        "ADK's _convert_foreign_event no longer emits our sentinel as part 0. "
        "The graphrag context filter is now a silent no-op. Check the installed "
        "google-adk version against the >=1.10,<2 pin in pyproject.toml."
    )

    req = _request(converted.content)
    drop_foreign_context(None, req)
    assert req.contents == []
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/unit/test_adk_context.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'agentic_kg.common.adk_context'`

- [ ] **Step 3: Write the implementation**

```python
# src/agentic_kg/common/adk_context.py
"""Context hygiene for agents that must not inherit other agents' claims.

ADK shows one agent the output of the others by rewriting each foreign event
into a user-role message (`_convert_foreign_event`, google/adk/flows/llm_flows/
contents.py). That is useful for an agent summarising a colleague's work and
actively harmful for one whose job is to report what the database says: a
warning another agent emitted hours earlier arrives wearing the user's role and
reads as ground truth.

Detection has to key on the sentinel text, not the role. By the time a
before_model_callback sees llm_request.contents, the converter has already set
both `role` and `author` to 'user' (contents.py:322, 355), so a foreign event
and a real human turn are indistinguishable by role -- filtering on role would
silently discard everything the user actually typed. The sentinel is prepended
unconditionally, before the parts loop (contents.py:323), and _get_contents
deep-copies one Content per event without merging adjacent ones (line 258), so
it reliably sits at index 0.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Must match google.adk.flows.llm_flows.contents._convert_foreign_event.
# tests/unit/test_adk_context.py drives ADK's real converter to detect drift.
FOREIGN_CONTEXT_SENTINEL = "For context:"


def _is_foreign(content: Any) -> bool:
    parts = getattr(content, "parts", None)
    if not parts:
        return False
    return getattr(parts[0], "text", None) == FOREIGN_CONTEXT_SENTINEL


def drop_foreign_context(callback_context: Any, llm_request: Any) -> Optional[None]:
    """Remove other agents' output from the request, in place.

    Returns None so ADK proceeds with the (now filtered) request; a non-None
    return would short-circuit the model call entirely.
    """
    contents = getattr(llm_request, "contents", None)
    if not contents:
        return None

    kept = [c for c in contents if not _is_foreign(c)]
    dropped = len(contents) - len(kept)
    if dropped:
        logger.debug("Dropped %d foreign-context message(s) before model call", dropped)
        llm_request.contents = kept
    return None
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/unit/test_adk_context.py -v`
Expected: 7 passed

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest`
Expected: 179 passed, 3 skipped

- [ ] **Step 6: Commit**

```bash
git add src/agentic_kg/common/adk_context.py tests/unit/test_adk_context.py
git commit -m "feat: add foreign-context filter for graphrag

Keys on ADK's 'For context:' sentinel rather than role, because
_convert_foreign_event rewrites both role and author to 'user' before a
before_model_callback ever sees the request. Includes a canary that drives
ADK's real converter, so a google-adk upgrade under the wide >=1.10,<2 pin
fails loudly instead of turning the filter into a no-op."
```

---

### Task 2: Generality guard

**Files:**
- Create: `tests/unit/test_generality.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing importable. Built early deliberately so it guards every task that follows.

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_generality.py
"""Guards against the demo dataset leaking into shipped code.

This work was diagnosed from one furniture supply-chain graph. Everything in
src/ must reason about graph *shapes* -- entity counts, distinct counts,
degree, completeness -- and never about suppliers. That includes prompt
strings, which are the largest and least reviewable overfitting surface.

This catches vocabulary overfitting only. It cannot see structural
overfitting (assuming one pattern per relationship type, assuming entities
below EXHAUSTIVE_SEARCH_LIMIT, assuming single-label nodes); that is what
tests/integration/test_graph_profile_shapes.py is for.
"""
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

# Curated and deliberately narrow. Bare label names (Part, Product, Assembly)
# are EXCLUDED because `Part` collides with google.genai.types.Part, which this
# codebase uses legitimately -- a broad match would fail on correct code and be
# deleted by the next person. Do not "improve" this into a substring sweep.
FORBIDDEN_TOKENS = [
    "preferred_supplier",
    "supplier_id",
    "assembly_id",
    "part_id",
    "lead_time_days",
    "unit_cost",
    "minimum_order_quantity",
    "SUP-",
    "Screws",
]


@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_dataset_vocabulary_absent_from_src(token):
    offenders = [
        str(path.relative_to(SRC))
        for path in sorted(SRC.rglob("*.py"))
        if token in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"Dataset token {token!r} found in src/: {offenders}. "
        "Shipped code must reason about graph shapes, not about one dataset. "
        "See 'Generality constraints' in the design spec."
    )
```

- [ ] **Step 2: Run it — it must PASS immediately**

Run: `uv run pytest tests/unit/test_generality.py -v`
Expected: 9 passed. This test starts green; it is a ratchet, not a to-do. If it fails now, something already leaked and must be removed before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_generality.py
git commit -m "test: guard src/ against demo-dataset vocabulary"
```

---

### Task 3: Write counter and DROP in `is_write_query`

**Files:**
- Modify: `src/agentic_kg/common/neo4j_for_adk.py:74-79` (`is_write_query`), `:123-158` (`Neo4jForADK`)
- Test: `tests/unit/test_neo4j_for_adk.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Neo4jForADK.write_count: int`, incremented inside `send_query` when `is_write_query()` is true. Task 6's cache reads `get_graphdb().write_count`.

Background: every write in the codebase funnels through `Neo4jForADK.send_query` — `kg_construction_tools.py:83,128,207` and the `cypher_tools` DDL paths all call it directly, none go through `write_neo4j_cypher`. Instrumenting that one chokepoint means a write path added later is covered with nothing to remember. `neo4j_for_adk.py` has no unit coverage today (only an integration file, excluded by default), so this task creates it and fakes one level below the existing `FakeGraphDb`, at the driver/session boundary.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_neo4j_for_adk.py
"""Unit tests for Neo4jForADK's write counter and query execution.

tests/unit/test_cypher_tools.py's FakeGraphDb replaces the whole `graphdb`
binding, which bypasses send_query entirely -- so it can never verify the
counter that lives inside it. These tests fake one level lower, at the
driver/session boundary, so the real send_query body runs.
"""
import pytest

from agentic_kg.common import neo4j_for_adk
from agentic_kg.common.neo4j_for_adk import Neo4jForADK, is_write_query


class FakeResult:
    def __init__(self, records):
        self._records = records

    def __iter__(self):
        return iter(self._records)


class FakeRecord:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class FakeSession:
    def __init__(self, recorder, records):
        self._recorder = recorder
        self._records = records

    def run(self, query, parameters=None, **kwargs):
        self._recorder.append((query, parameters, kwargs))
        return FakeResult(self._records)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    def __init__(self, records=()):
        self.sessions = []
        self.queries = []
        self._records = [FakeRecord(r) for r in records]

    def session(self, **config):
        self.sessions.append(config)
        return FakeSession(self.queries, self._records)


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(Neo4jForADK, "__init__", lambda self: None)
    instance = Neo4jForADK()
    instance._driver = FakeDriver()
    instance._neo4j_config = type("Cfg", (), {"database": "neo4j"})()
    instance.write_count = 0
    return instance


@pytest.mark.parametrize("query", [
    "MERGE (n:Thing {id: 1})",
    "CREATE (n:Thing)",
    "MATCH (n) SET n.x = 1",
    "MATCH (n) DETACH DELETE n",
    "MATCH (n) REMOVE n.x",
    "DROP CONSTRAINT some_constraint",
    "DROP INDEX some_index",
])
def test_is_write_query_detects_writes(query):
    assert is_write_query(query) is True


@pytest.mark.parametrize("query", [
    "MATCH (n:Thing) RETURN n",
    "MATCH (n) RETURN count(n)",
])
def test_is_write_query_passes_plain_reads(query):
    assert is_write_query(query) is False


def test_write_query_increments_the_counter(db):
    db.send_query("MERGE (n:Thing {id: 1})")
    assert db.write_count == 1
    db.send_query("DROP CONSTRAINT c")
    assert db.write_count == 2


def test_read_query_leaves_the_counter_alone(db):
    db.send_query("MATCH (n) RETURN n")
    assert db.write_count == 0


def test_failed_write_does_not_increment(db):
    def boom(*a, **k):
        raise RuntimeError("no")
    db._driver.session = boom
    result = db.send_query("MERGE (n:Thing)")
    assert result["status"] == "error"
    assert db.write_count == 0
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/unit/test_neo4j_for_adk.py -v`
Expected: `DROP CONSTRAINT` / `DROP INDEX` cases FAIL (regex lacks `DROP`); counter tests FAIL with `AttributeError: 'Neo4jForADK' object has no attribute 'write_count'`.

- [ ] **Step 3: Add `DROP` to the regex**

Replace `is_write_query` (`src/agentic_kg/common/neo4j_for_adk.py:74-79`):

```python
def is_write_query(query: str) -> bool:
    """Heuristic write detection, used ONLY as a cache-invalidation hint.

    This is deliberately not a security boundary and must never be used as
    one. It matches text, so it cannot tell a keyword from a string literal
    ("... CONTAINS 'set forth' ..." reads as a write) and it misses camelCase
    procedure calls (\\bMERGE\\b finds no boundary inside `mergeNodes`, which is
    what apoc.refactor.mergeNodes is). Read-only enforcement is the server's
    job via default_access_mode -- see send_read_query.

    As a cache hint both error directions are benign: a false positive costs
    one recomputation, and a false negative is caught by the fingerprint layer
    in graph_profile.
    """
    return (
        re.search(r"\b(MERGE|CREATE|SET|DELETE|REMOVE|ADD|DROP)\b", query, re.IGNORECASE)
        is not None
    )
```

- [ ] **Step 4: Add the counter**

In `Neo4jForADK.__init__` (after the `logger.debug(...)` line at `:136`), add:

```python
        # Bumped by send_query on every successful write. graph_profile's cache
        # reads this to invalidate without a round-trip. It counts in-process
        # writes only; writes from elsewhere are caught by the fingerprint.
        self.write_count = 0
```

Replace `send_query`:

```python
    def send_query(self, cypher_query, parameters=None) -> Dict[str, Any]:
        session = self._driver.session(database=self._neo4j_config.database)
        try:
            result = session.run(cypher_query, parameters or {})
            adk_result = result_to_adk(result)
            if is_write_query(cypher_query):
                self.write_count += 1
            return adk_result
        except Exception as e:
            return tool_error(str(e))
        finally:
            session.close()
```

- [ ] **Step 5: Run and confirm the tests pass**

Run: `uv run pytest tests/unit/test_neo4j_for_adk.py -v`
Expected: 14 passed

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest`
Expected: 193 passed, 3 skipped

- [ ] **Step 7: Commit**

```bash
git add src/agentic_kg/common/neo4j_for_adk.py tests/unit/test_neo4j_for_adk.py
git commit -m "feat: add write counter to Neo4jForADK, DROP to is_write_query

Counter goes in send_query, the single chokepoint every write in the codebase
already funnels through, so a write path added later is covered with nothing
to remember. DROP was missing from the regex, so reset_neo4j_data's constraint
drops were invisible. Documents that is_write_query is a cache hint only, not
a security boundary."
```

---

### Task 4: Safe read execution — timeout, read access mode, streaming

**Files:**
- Modify: `src/agentic_kg/common/neo4j_for_adk.py`
- Test: `tests/unit/test_neo4j_for_adk.py` (extend)

**Interfaces:**
- Consumes: `Neo4jForADK` from Task 3.
- Produces: module constants `QUERY_TIMEOUT_SECONDS = 30`, `MAX_RETURNED_ROWS = 50`, `ROW_COUNT_CEILING = 100_000`, and
  `Neo4jForADK.send_read_query(cypher_query, parameters=None, max_rows=MAX_RETURNED_ROWS) -> Dict[str, Any]`
  returning `tool_success("query_result", {...})` where the payload holds `records`, `truncated`, and either `row_count` (exact) or `row_count_at_least`. Task 5 calls it with `max_rows=None`; Task 8 calls it with the default.

Background: `send_query` runs untimed and `result_to_adk` calls `to_eager_result()`, materialising every row before any cap can apply — so a row cap alone cannot bound the runaway queries it is named against. Verified available in the installed `neo4j` 5.28.2: `neo4j.Query(text, timeout=...)`, `neo4j.READ_ACCESS == 'READ'`, and `SessionConfig.default_access_mode`.

- [ ] **Step 1: Write the failing tests (append to `tests/unit/test_neo4j_for_adk.py`)**

```python
from agentic_kg.common.neo4j_for_adk import MAX_RETURNED_ROWS, ROW_COUNT_CEILING


def _rows(n):
    return [{"i": i} for i in range(n)]


def test_read_query_opens_a_read_access_session(db):
    db._driver = FakeDriver(_rows(3))
    db.send_read_query("MATCH (n) RETURN n")
    assert db._driver.sessions[0]["default_access_mode"] == "READ"


def test_read_query_passes_a_timeout(db):
    db._driver = FakeDriver(_rows(1))
    db.send_read_query("MATCH (n) RETURN n")
    query_obj, _params, _kwargs = db._driver.queries[0]
    assert getattr(query_obj, "timeout", None) is not None


def test_read_query_returns_exact_row_count_under_the_cap(db):
    db._driver = FakeDriver(_rows(3))
    result = db.send_read_query("MATCH (n) RETURN n")
    payload = result["query_result"]
    assert payload["row_count"] == 3
    assert payload["truncated"] is False
    assert len(payload["records"]) == 3


def test_read_query_truncates_records_but_counts_them_all(db):
    db._driver = FakeDriver(_rows(MAX_RETURNED_ROWS + 25))
    payload = db.send_read_query("MATCH (n) RETURN n")["query_result"]
    assert len(payload["records"]) == MAX_RETURNED_ROWS
    assert payload["row_count"] == MAX_RETURNED_ROWS + 25
    assert payload["truncated"] is True
    assert "aggregation" in payload["note"]


def test_read_query_reports_a_floor_past_the_counting_ceiling(db, monkeypatch):
    monkeypatch.setattr(neo4j_for_adk, "ROW_COUNT_CEILING", 10)
    db._driver = FakeDriver(_rows(25))
    payload = db.send_read_query("MATCH (n) RETURN n")["query_result"]
    assert "row_count" not in payload
    assert payload["row_count_at_least"] == 10
    assert payload["truncated"] is True


def test_read_query_never_increments_the_write_counter(db):
    db._driver = FakeDriver(_rows(1))
    db.send_read_query("MATCH (n) SET n.x = 1")
    assert db.write_count == 0


def test_read_query_max_rows_none_retains_everything(db):
    db._driver = FakeDriver(_rows(120))
    payload = db.send_read_query("MATCH (n) RETURN n", max_rows=None)["query_result"]
    assert len(payload["records"]) == 120
    assert payload["truncated"] is False


def test_read_query_returns_structured_error_not_an_exception(db):
    def boom(**k):
        raise RuntimeError("syntax error at line 1")
    db._driver.session = boom
    result = db.send_read_query("MATCH bad")
    assert result["status"] == "error"
    assert "syntax error" in result["error_message"]


def test_long_lists_are_summarised_not_returned_whole(db):
    db._driver = FakeDriver([{"embedding": [0.1] * 1536, "name": "x"}])
    payload = db.send_read_query("MATCH (c) RETURN c")["query_result"]
    record = payload["records"][0]
    assert record["name"] == "x"
    assert isinstance(record["embedding"], str)
    assert "1536" in record["embedding"]
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/unit/test_neo4j_for_adk.py -v -k read or summarised`
Expected: `ImportError: cannot import name 'MAX_RETURNED_ROWS'`

- [ ] **Step 3: Implement**

Add to the imports at the top of `src/agentic_kg/common/neo4j_for_adk.py`:

```python
from neo4j import (
    GraphDatabase,
    Query,
    READ_ACCESS,
    Result,
)
```

Add module constants after `logger = logging.getLogger(__name__)`:

```python
# Bound on how long any single agent-issued query may run. A hung tool call in
# `adk web` is indistinguishable from a routing bug, which this project has
# been burned by before.
QUERY_TIMEOUT_SECONDS = 30

# How many rows are retained and shown. Not a tuned constant -- a judgement
# about how many rows are worth reading individually before the honest answer
# is "aggregate this instead". No behaviour may depend on its exact value.
MAX_RETURNED_ROWS = 50

# How far we keep counting past the cap before reporting a floor instead of an
# exact total. Counting is cheap (an int); claiming an exact number we did not
# finish counting would not be.
ROW_COUNT_CEILING = 100_000

# Lists longer than this are replaced by a summary string. Embedding vectors
# would otherwise be pasted into the model's context verbatim.
MAX_INLINE_LIST_LENGTH = 32

_TRUNCATION_NOTE = (
    "Records are capped. Counts, rankings and superlatives must come from a "
    "Cypher aggregation, never from counting these rows."
)
```

Add the list-summarising helper next to `to_python`:

```python
def summarise_long_lists(value):
    """Replace oversized lists with a description of their shape.

    to_python recurses into lists, so `MATCH (c:Chunk) RETURN c` would return a
    full embedding vector. get_structured_schema's `sanitize` does not reach
    this path -- it only covers the library's own query family.
    """
    if isinstance(value, dict):
        return {k: summarise_long_lists(v) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > MAX_INLINE_LIST_LENGTH:
            kind = type(value[0]).__name__ if value else "unknown"
            return f"<list of {len(value)} {kind} values, omitted>"
        return [summarise_long_lists(v) for v in value]
    return value
```

Add the method to `Neo4jForADK`, directly after `send_query`:

```python
    def send_read_query(
        self,
        cypher_query,
        parameters=None,
        max_rows: Optional[int] = MAX_RETURNED_ROWS,
    ) -> Dict[str, Any]:
        """Run a query read-only, timed, and with bounded row retention.

        Read-only is enforced by the *server* through default_access_mode, not
        by inspecting the query text -- text matching cannot distinguish a
        keyword from a string literal, and misses camelCase procedure calls
        like apoc.refactor.mergeNodes.

        Rows are streamed rather than materialised, so memory is bounded by
        max_rows instead of by the size of the result. Counting continues past
        max_rows up to ROW_COUNT_CEILING; beyond that the payload reports
        row_count_at_least rather than inventing an exact total.

        Pass max_rows=None to retain every row (used for internal aggregate
        queries whose results are already small).
        """
        session = self._driver.session(
            database=self._neo4j_config.database,
            default_access_mode=READ_ACCESS,
        )
        try:
            query = Query(cypher_query, timeout=QUERY_TIMEOUT_SECONDS)
            result = session.run(query, parameters or {})

            records = []
            counted = 0
            hit_ceiling = False
            for record in result:
                counted += 1
                if max_rows is None or len(records) < max_rows:
                    records.append(summarise_long_lists(to_python(record.data())))
                if counted >= ROW_COUNT_CEILING:
                    hit_ceiling = True
                    break

            truncated = max_rows is not None and counted > len(records)
            payload: Dict[str, Any] = {
                "records": records,
                "truncated": truncated or hit_ceiling,
            }
            if hit_ceiling:
                payload["row_count_at_least"] = counted
            else:
                payload["row_count"] = counted
            if payload["truncated"]:
                payload["note"] = _TRUNCATION_NOTE
            return tool_success("query_result", payload)
        except Exception as e:
            return tool_error(str(e))
        finally:
            session.close()
```

- [ ] **Step 4: Run and confirm the tests pass**

Run: `uv run pytest tests/unit/test_neo4j_for_adk.py -v`
Expected: 23 passed

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest`
Expected: 202 passed, 3 skipped

- [ ] **Step 6: Commit**

```bash
git add src/agentic_kg/common/neo4j_for_adk.py tests/unit/test_neo4j_for_adk.py
git commit -m "feat: add timed, read-only, streamed query execution

send_read_query enforces read-only at the server via default_access_mode
instead of guessing from query text, bounds runtime with neo4j.Query(timeout),
and streams rows so memory is bounded by the retention cap rather than by
result size. Reports an exact row_count under the counting ceiling and
row_count_at_least past it. Summarises oversized lists so embedding vectors
never reach the model's context."
```

---

### Task 5: Property annotations

**Files:**
- Create: `src/agentic_kg/common/graph_profile.py`
- Test: `tests/unit/test_graph_profile.py`

**Interfaces:**
- Consumes: nothing (pure functions over dicts).
- Produces: `VALUE_COUNT_MAX_DISTINCT = 10` and `annotate_property(prop: dict, entity_count: int) -> dict`, returning a new dict with `completeness`, `uniqueness` and `numeric_like` always present. Task 6 calls it for every property of every entity.

Background from the library (`neo4j_graphrag/schema.py`): the exhaustive branch emits `values` plus a true `distinct_count` (lines 566-577); the sampled branch emits `values` with **no** `distinct_count` (572-573). A third path reads a RANGE index and also emits `distinct_count` (546-564) — it always yields `len(values) == distinct_count`, so the comparison below classifies it as complete by construction, with no special case. `DISTINCT_VALUE_LIMIT` is 10 (`schema.py:29`), which is where `VALUE_COUNT_MAX_DISTINCT` comes from: above it the library truncates `values`, so per-value counts would be partial regardless.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_graph_profile.py
"""Unit tests for enriched-schema annotation.

Every case is asserted twice, once for a node property and once for a
relationship property. The node/relationship symmetry is the thing most
likely to be lost during implementation: the two bugs that motivated these
annotations landed on opposite sides of that split, so defining either one
over only its own half would fit the design to the bugs instead of the class.
"""
import pytest

from agentic_kg.common.graph_profile import (
    VALUE_COUNT_MAX_DISTINCT,
    annotate_property,
)

# Same payload shape for both kinds -- the library returns identical property
# dicts under "node_props" and "rel_props", which is why one function serves.
KINDS = ["node", "relationship"]


@pytest.mark.parametrize("kind", KINDS)
def test_complete_when_values_match_distinct_count(kind):
    prop = {"property": "flag", "type": "STRING", "values": ["a", "b"], "distinct_count": 2}
    out = annotate_property(prop, entity_count=100)
    assert out["completeness"] == "complete"
    assert out["values"] == ["a", "b"]


@pytest.mark.parametrize("kind", KINDS)
def test_partial_when_values_are_truncated(kind):
    prop = {"property": "n", "type": "STRING",
            "values": [str(i) for i in range(10)], "distinct_count": 27}
    out = annotate_property(prop, entity_count=100)
    assert out["completeness"] == "partial"
    assert out["values"] == [str(i) for i in range(10)]


@pytest.mark.parametrize("kind", KINDS)
def test_unknown_and_values_suppressed_when_sampled(kind):
    prop = {"property": "n", "type": "STRING", "values": ["7", "9"]}
    out = annotate_property(prop, entity_count=50_000)
    assert out["completeness"] == "unknown"
    assert "values" not in out


@pytest.mark.parametrize("kind", KINDS)
def test_unique_when_distinct_count_equals_entity_count(kind):
    prop = {"property": "id", "type": "STRING", "values": [], "distinct_count": 88}
    out = annotate_property(prop, entity_count=88)
    assert out["uniqueness"] == "unique"


@pytest.mark.parametrize("kind", KINDS)
def test_non_unique_when_distinct_count_is_below_entity_count(kind):
    prop = {"property": "label", "type": "STRING", "values": [], "distinct_count": 72}
    out = annotate_property(prop, entity_count=88)
    assert out["uniqueness"] == "non_unique"


@pytest.mark.parametrize("kind", KINDS)
def test_uniqueness_unknown_without_distinct_count(kind):
    prop = {"property": "label", "type": "STRING", "values": ["x"]}
    out = annotate_property(prop, entity_count=88)
    assert out["uniqueness"] == "unknown"


@pytest.mark.parametrize("kind", KINDS)
def test_uniqueness_unknown_when_entity_count_unavailable(kind):
    prop = {"property": "label", "type": "STRING", "values": [], "distinct_count": 5}
    out = annotate_property(prop, entity_count=None)
    assert out["uniqueness"] == "unknown"


@pytest.mark.parametrize("kind", KINDS)
def test_numeric_like_string_is_flagged(kind):
    prop = {"property": "days", "type": "STRING",
            "values": ["8", "12", "30"], "distinct_count": 3}
    out = annotate_property(prop, entity_count=10)
    assert out["numeric_like"] is True


@pytest.mark.parametrize("kind", KINDS)
def test_non_numeric_string_is_not_flagged(kind):
    prop = {"property": "city", "type": "STRING",
            "values": ["Berlin", "Lisbon"], "distinct_count": 2}
    out = annotate_property(prop, entity_count=10)
    assert out["numeric_like"] is False


@pytest.mark.parametrize("kind", KINDS)
def test_numeric_like_false_for_already_numeric_types(kind):
    prop = {"property": "n", "type": "INTEGER", "min": 1, "max": 9}
    out = annotate_property(prop, entity_count=10)
    assert out["numeric_like"] is False


@pytest.mark.parametrize("kind", KINDS)
def test_every_annotation_key_is_always_present(kind):
    """Omission-means-unknown is the regression this design forbids.

    A missing key reads as *safe* to a model, and the entities we cannot
    annotate are exactly the large unfamiliar ones where being wrong is most
    likely -- so silence would make the agent more confident the less it knows.
    """
    sparse = {"property": "mystery"}
    out = annotate_property(sparse, entity_count=None)
    for key in ("completeness", "uniqueness", "numeric_like"):
        assert key in out, f"{key} must always be present, never omitted"


@pytest.mark.parametrize("kind", KINDS)
def test_input_is_not_mutated(kind):
    prop = {"property": "flag", "type": "STRING", "values": ["a"], "distinct_count": 1}
    before = dict(prop)
    annotate_property(prop, entity_count=1)
    assert prop == before


def test_value_count_threshold_matches_the_library_limit():
    from neo4j_graphrag.schema import DISTINCT_VALUE_LIMIT
    assert VALUE_COUNT_MAX_DISTINCT == DISTINCT_VALUE_LIMIT
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/unit/test_graph_profile.py -v`
Expected: `ModuleNotFoundError: No module named 'agentic_kg.common.graph_profile'`

- [ ] **Step 3: Implement**

```python
# src/agentic_kg/common/graph_profile.py
"""Turns the enriched Neo4j schema into annotated, honestly-labelled facts.

neo4j_graphrag's get_structured_schema(is_enhanced=True) reports what values a
property holds, but says nothing about whether that report is complete. Its
exhaustive branch emits `values` plus a true `distinct_count`; its sampled
branch (anything above EXHAUSTIVE_SEARCH_LIMIT = 10000 entities) emits `values`
from five rows with no `distinct_count` at all. Handed to a model unlabelled,
five sampled values read exactly like the whole truth.

Every annotation here is therefore tri-state and ALWAYS present. An absent key
would read as "fine", and the entities we cannot annotate are precisely the
large, unfamiliar ones where a confident wrong answer is most likely.
"""
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Mirrors neo4j_graphrag.schema.DISTINCT_VALUE_LIMIT. Above it the library
# truncates its own `values` list, so per-value counts would be partial
# regardless -- exactly the misleading half-complete output these annotations
# exist to prevent.
VALUE_COUNT_MAX_DISTINCT = 10

_NUMERIC_TYPES = {"INTEGER", "FLOAT"}

# Tolerates a leading currency symbol and thousands separators, because the
# point of this flag is to spot values that *look* numeric while being stored
# as text -- which is where silent lexicographic ordering comes from.
_NUMERIC_LIKE = re.compile(r"^\s*[^\d\-+.]?\s*[-+]?[\d,]*\.?\d+\s*$")


def _is_numeric_like(values) -> bool:
    if not values:
        return False
    return all(isinstance(v, str) and _NUMERIC_LIKE.match(v) for v in values)


def annotate_property(prop: Dict[str, Any], entity_count: Optional[int]) -> Dict[str, Any]:
    """Annotate one property dict from the enriched schema.

    Works identically for node and relationship properties -- the library
    returns the same shape under "node_props" and "rel_props". `entity_count`
    is the node count for a label or the edge count for a relationship type;
    pass None when it is unavailable, which yields "unknown" rather than a
    guess.

    Returns a new dict; the input is never mutated.
    """
    out = dict(prop)
    values = prop.get("values")
    distinct_count = prop.get("distinct_count")

    if distinct_count is None:
        # Sampled branch: five arbitrary rows, completeness unknowable. Drop
        # the values rather than present a sample as if it were the set.
        out.pop("values", None)
        out["completeness"] = "unknown"
    elif values is not None and len(values) < distinct_count:
        out["completeness"] = "partial"
    else:
        out["completeness"] = "complete"

    if distinct_count is None or entity_count is None:
        out["uniqueness"] = "unknown"
    elif distinct_count >= entity_count:
        out["uniqueness"] = "unique"
    else:
        out["uniqueness"] = "non_unique"

    prop_type = prop.get("type")
    out["numeric_like"] = (
        prop_type not in _NUMERIC_TYPES and _is_numeric_like(values)
    )

    return out
```

- [ ] **Step 4: Run and confirm the tests pass**

Run: `uv run pytest tests/unit/test_graph_profile.py -v`
Expected: 25 passed

- [ ] **Step 5: Commit**

```bash
git add src/agentic_kg/common/graph_profile.py tests/unit/test_graph_profile.py
git commit -m "feat: annotate enriched-schema properties with tri-state facts

completeness, uniqueness and numeric_like are always present, never omitted:
an absent key reads as safe, and the entities we cannot annotate are the large
unfamiliar ones where that is most dangerous. Identical handling for node and
relationship properties, asserted twice in tests, because the two bugs that
motivated these annotations landed on opposite sides of that split."
```

---

### Task 6: Profile queries — entity counts, degree per pattern, value counts

**Files:**
- Modify: `src/agentic_kg/common/graph_profile.py`
- Test: `tests/unit/test_graph_profile.py` (extend)

**Interfaces:**
- Consumes: `annotate_property` (Task 5), `graphdb.send_read_query` (Task 4).
- Produces: `MAX_PROFILED_ENTITIES = 25`, `quote(name: str) -> str`, and `build_profile(schema: dict) -> dict` returning
  `{"entity_counts": {...}, "patterns": [...], "properties": {...}, "budget": {...}}`. Task 7 calls `build_profile`; Task 8 is unaffected.

Background: labels and relationship-type names come from the *database*, not from a model, so they must be backtick-quoted the way the library does (`schema.py:707-709`) and must **not** go through `common/cypher_identifiers.checked()` — that helper rejects anything which is not a bare identifier, so a legal extracted label like `Legal Entity` would raise and take down the whole profile, and with it `get_physical_schema`, the tool graphrag is told to call first. The library isolates failures per entity (`except CypherTypeError: return`, `schema.py:858-859`, invoked per entity at 903 and 914); this must match that or be strictly less robust than what it wraps.

- [ ] **Step 1: Write the failing tests (append to `tests/unit/test_graph_profile.py`)**

```python
from agentic_kg.common import graph_profile
from agentic_kg.common.graph_profile import build_profile, quote


class FakeGraphDbForProfile:
    """Answers profile queries from a scripted table keyed by substring."""

    def __init__(self, responses=None, fail_on=None):
        self.responses = responses or {}
        self.fail_on = fail_on or ()
        self.queries = []

    def send_read_query(self, query, parameters=None, max_rows=None):
        self.queries.append(query)
        for needle in self.fail_on:
            if needle in query:
                return {"status": "error", "error_message": "boom"}
        for needle, records in self.responses.items():
            if needle in query:
                return {"status": "success",
                        "query_result": {"records": records, "row_count": len(records),
                                         "truncated": False}}
        return {"status": "success",
                "query_result": {"records": [], "row_count": 0, "truncated": False}}


@pytest.fixture
def fake_profile_db(monkeypatch):
    db = FakeGraphDbForProfile()
    monkeypatch.setattr(graph_profile, "graphdb", db)
    return db


SCHEMA = {
    "node_props": {"Alpha": [{"property": "code", "type": "STRING"}]},
    "rel_props": {"LINKS": [{"property": "kind", "type": "STRING",
                             "values": ["x", "y"], "distinct_count": 2}]},
    "relationships": [
        {"start": "Alpha", "type": "LINKS", "end": "Beta"},
        {"start": "Alpha", "type": "LINKS", "end": "Gamma"},
    ],
}


def test_quote_backticks_names_and_escapes_embedded_backticks():
    assert quote("Alpha") == "`Alpha`"
    assert quote("Legal Entity") == "`Legal Entity`"
    assert quote("we`ird") == "`we``ird`"


def test_quote_accepts_names_that_checked_would_reject():
    """checked() is for model-supplied identifiers; these come from the DB."""
    from agentic_kg.common.cypher_identifiers import InvalidIdentifier, checked
    with pytest.raises(InvalidIdentifier):
        checked("label", "Legal Entity")
    assert quote("Legal Entity") == "`Legal Entity`"


def test_degree_is_keyed_per_start_type_end_pattern(fake_profile_db):
    profile = build_profile(SCHEMA)
    keys = {p["pattern"] for p in profile["patterns"]}
    assert keys == {"Alpha-[LINKS]->Beta", "Alpha-[LINKS]->Gamma"}


def test_one_failing_entity_degrades_only_itself(fake_profile_db, monkeypatch):
    db = FakeGraphDbForProfile(fail_on=("`Alpha`",))
    monkeypatch.setattr(graph_profile, "graphdb", db)
    profile = build_profile(SCHEMA)
    assert profile["properties"]["Alpha"] == "profile_error"
    assert profile["properties"]["LINKS"] != "profile_error"


def test_value_counts_only_for_small_distinct_counts(fake_profile_db):
    schema = {
        "node_props": {"Alpha": [
            {"property": "small", "type": "STRING", "values": ["a"], "distinct_count": 2},
            {"property": "big", "type": "STRING", "values": ["a"], "distinct_count": 900},
        ]},
        "rel_props": {}, "relationships": [],
    }
    build_profile(schema)
    counted = [q for q in fake_profile_db.queries if "count(*)" in q and "`small`" in q]
    not_counted = [q for q in fake_profile_db.queries if "count(*)" in q and "`big`" in q]
    assert counted and not not_counted


def test_budget_marks_unprofiled_entities_rather_than_dropping_them(monkeypatch):
    monkeypatch.setattr(graph_profile, "MAX_PROFILED_ENTITIES", 1)
    db = FakeGraphDbForProfile()
    monkeypatch.setattr(graph_profile, "graphdb", db)
    schema = {
        "node_props": {"Alpha": [], "Beta": [], "Gamma": []},
        "rel_props": {}, "relationships": [],
    }
    profile = build_profile(schema)
    statuses = set(profile["properties"].values())
    assert "not_profiled" in statuses
    assert profile["budget"]["profiled"] == 1
    assert profile["budget"]["skipped"] == 2
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/unit/test_graph_profile.py -v -k "quote or degree or failing or value_counts or budget"`
Expected: `ImportError: cannot import name 'build_profile'`

- [ ] **Step 3: Implement**

Add to the imports at the top of `graph_profile.py`:

```python
from .neo4j_for_adk import get_graphdb
from .tool_result import is_success

graphdb = get_graphdb()
```

Add constants next to `VALUE_COUNT_MAX_DISTINCT`:

```python
# Cold profile cost is N + M + P + Q + 2 queries (labels, relationship types,
# patterns, qualifying properties, plus two count queries). On a large ingested
# corpus that is hundreds, in one synchronous tool call -- indistinguishable
# from a hang in `adk web`. Profile the largest entities and mark the rest.
MAX_PROFILED_ENTITIES = 25
```

Append the implementation:

```python
def quote(name: str) -> str:
    """Backtick-quote an identifier that came from the database.

    Deliberately NOT common.cypher_identifiers.checked(): that guards against
    injection from model-supplied names and rejects anything which is not a
    bare identifier. These names are read out of the graph, so `Legal Entity`
    and `10-K` are perfectly legal and must survive. Escaping doubles any
    embedded backtick, which is Cypher's own convention.
    """
    return "`" + name.replace("`", "``") + "`"


def _records(result) -> list:
    if not is_success(result):
        return []
    return result.get("query_result", {}).get("records", [])


def _entity_counts(labels, rel_types) -> Dict[str, Optional[int]]:
    """One grouped query per direction rather than one query per entity."""
    counts: Dict[str, Optional[int]] = {name: None for name in [*labels, *rel_types]}

    for row in _records(graphdb.send_read_query(
        "MATCH (n) UNWIND labels(n) AS label "
        "RETURN label AS name, count(*) AS n", max_rows=None)):
        counts[row["name"]] = row["n"]

    for row in _records(graphdb.send_read_query(
        "MATCH ()-[r]->() RETURN type(r) AS name, count(r) AS n", max_rows=None)):
        counts[row["name"]] = row["n"]

    return counts


def _pattern_degree(start: str, rel_type: str, end: str) -> Dict[str, Any]:
    """Degree statistics for ONE (start, type, end) pattern.

    Keyed per pattern, never per relationship type. On a graph where one type
    spans several label pairs, pooled statistics describe no actual pattern,
    and min == max can hold across the pool while being false for every
    pattern in it -- reintroducing the exact grain error this profile exists
    to prevent.
    """
    query = (
        f"MATCH (a:{quote(start)})-[r:{quote(rel_type)}]->(b:{quote(end)}) "
        "WITH a, b, r "
        "WITH collect({a: id(a), b: id(b)}) AS pairs, count(r) AS edges "
        "UNWIND pairs AS p "
        "WITH edges, p.a AS a_id, p.b AS b_id "
        "WITH edges, "
        "     count(DISTINCT a_id) AS distinct_start, "
        "     count(DISTINCT b_id) AS distinct_end "
        "RETURN edges, distinct_start, distinct_end"
    )
    rows = _records(graphdb.send_read_query(query, max_rows=None))
    if not rows:
        return {"edges": 0, "start_degree": "unknown", "end_degree": "unknown"}

    row = rows[0]
    edges = row["edges"]

    def _degree(query_text):
        got = _records(graphdb.send_read_query(query_text, max_rows=None))
        if not got:
            return "unknown"
        r = got[0]
        return {"min": r["lo"], "max": r["hi"], "mean": round(r["avg"], 2)}

    start_degree = _degree(
        f"MATCH (a:{quote(start)})-[r:{quote(rel_type)}]->(:{quote(end)}) "
        "WITH a, count(r) AS d "
        "RETURN min(d) AS lo, max(d) AS hi, avg(d) AS avg")
    end_degree = _degree(
        f"MATCH (:{quote(start)})-[r:{quote(rel_type)}]->(b:{quote(end)}) "
        "WITH b, count(r) AS d "
        "RETURN min(d) AS lo, max(d) AS hi, avg(d) AS avg")

    return {
        "edges": edges,
        "distinct_start": row["distinct_start"],
        "distinct_end": row["distinct_end"],
        "start_degree": start_degree,
        "end_degree": end_degree,
    }


def _value_counts(entity: str, prop_name: str, is_relationship: bool) -> Any:
    if is_relationship:
        query = (
            f"MATCH ()-[r:{quote(entity)}]->() "
            f"WITH r.{quote(prop_name)} AS value, count(*) AS n "
            "WHERE value IS NOT NULL RETURN value, n ORDER BY n DESC"
        )
    else:
        query = (
            f"MATCH (n:{quote(entity)}) "
            f"WITH n.{quote(prop_name)} AS value, count(*) AS n "
            "WHERE value IS NOT NULL RETURN value, n ORDER BY n DESC"
        )
    rows = _records(graphdb.send_read_query(query, max_rows=None))
    if not rows:
        return "unknown"
    return {str(r["value"]): r["n"] for r in rows}


def _profile_entity(entity, props, entity_count, is_relationship):
    annotated = []
    for prop in props:
        out = annotate_property(prop, entity_count)
        distinct_count = prop.get("distinct_count")
        if distinct_count is not None and distinct_count <= VALUE_COUNT_MAX_DISTINCT:
            out["value_counts"] = _value_counts(
                entity, prop["property"], is_relationship)
        else:
            out["value_counts"] = "unknown"
        annotated.append(out)
    return annotated


def build_profile(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Compute entity counts, per-pattern degree, and annotated properties.

    Never raises. A failure profiling one entity marks that entry
    "profile_error" and leaves the rest intact, matching what the library
    already does per entity (neo4j_graphrag/schema.py:858-859).
    """
    node_props = schema.get("node_props", {}) or {}
    rel_props = schema.get("rel_props", {}) or {}
    relationships = schema.get("relationships", []) or []

    counts = _entity_counts(list(node_props), list(rel_props))

    entities = [(name, props, False) for name, props in node_props.items()]
    entities += [(name, props, True) for name, props in rel_props.items()]
    # Largest first, so a budget cut drops the entities we could say least
    # about anyway rather than an arbitrary slice.
    entities.sort(key=lambda e: counts.get(e[0]) or 0, reverse=True)

    properties: Dict[str, Any] = {}
    profiled = 0
    for name, props, is_rel in entities:
        if profiled >= MAX_PROFILED_ENTITIES:
            properties[name] = "not_profiled"
            continue
        try:
            properties[name] = _profile_entity(name, props, counts.get(name), is_rel)
        except Exception:
            logger.exception("Profiling failed for entity %s; continuing", name)
            properties[name] = "profile_error"
        profiled += 1

    patterns = []
    for rel in relationships:
        start, rel_type, end = rel.get("start"), rel.get("type"), rel.get("end")
        if not (start and rel_type and end):
            continue
        entry = {"pattern": f"{start}-[{rel_type}]->{end}",
                 "start": start, "type": rel_type, "end": end}
        try:
            entry.update(_pattern_degree(start, rel_type, end))
        except Exception:
            logger.exception("Degree profiling failed for %s; continuing", entry["pattern"])
            entry["start_degree"] = "profile_error"
            entry["end_degree"] = "profile_error"
        patterns.append(entry)

    return {
        "entity_counts": counts,
        "patterns": patterns,
        "properties": properties,
        "budget": {
            "profiled": min(profiled, MAX_PROFILED_ENTITIES),
            "skipped": max(0, len(entities) - MAX_PROFILED_ENTITIES),
            "limit": MAX_PROFILED_ENTITIES,
        },
    }
```

- [ ] **Step 4: Run and confirm the tests pass**

Run: `uv run pytest tests/unit/test_graph_profile.py -v`
Expected: 31 passed

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest`
Expected: 208 passed, 3 skipped

- [ ] **Step 6: Commit**

```bash
git add src/agentic_kg/common/graph_profile.py tests/unit/test_graph_profile.py
git commit -m "feat: profile entity counts, per-pattern degree, and value counts

Degree is keyed on (start, type, end) triples, never on relationship type
alone: on the demo graph each type spans one label pair, which is exactly why
keying on type is invisible there and wrong everywhere else. Names from the
database are backtick-quoted rather than passed through checked(), which would
raise on a legal label like 'Legal Entity' and take down the whole profile.
Per-entity error isolation and a query budget bound the cold-start cost."
```

---

### Task 7: The cache

**Files:**
- Modify: `src/agentic_kg/common/graph_profile.py`
- Test: `tests/unit/test_graph_profile_cache.py`

**Interfaces:**
- Consumes: `build_profile` (Task 6), `Neo4jForADK.write_count` (Task 3).
- Produces: `get_cached_profile(schema_loader) -> dict` and `reset_cache() -> None`. `schema_loader` is a zero-argument callable returning the enriched schema dict; Task 8 passes one that calls `get_structured_schema`.

Background: the cache is module-level, not session state — one `adk web` process serves every session and the thing cached is a property of the *database*, not of a conversation. Two invalidation layers: the write counter catches in-process writes with no query at all, and a node/relationship-count fingerprint catches writes from outside the process, which genuinely happen (the demo graph was built through the UI and wiped from a script).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_graph_profile_cache.py
"""Unit tests for profile cache invalidation.

Two layers, because one is not honest. The counter cannot see writes made
outside this process, and those happen in this workflow -- so a counter-only
cache would serve a schema for a database that no longer exists and state it
as fact, which is the exact failure class this work exists to fix.
"""
import pytest

from agentic_kg.common import graph_profile


class FakeDb:
    def __init__(self, nodes=10, rels=5):
        self.write_count = 0
        self.nodes = nodes
        self.rels = rels
        self.fingerprint_calls = 0

    def send_read_query(self, query, parameters=None, max_rows=None):
        if "AS nodes" in query:
            self.fingerprint_calls += 1
            records = [{"nodes": self.nodes, "rels": self.rels}]
        else:
            records = []
        return {"status": "success",
                "query_result": {"records": records, "row_count": len(records),
                                 "truncated": False}}


@pytest.fixture
def db(monkeypatch):
    fake = FakeDb()
    monkeypatch.setattr(graph_profile, "graphdb", fake)
    monkeypatch.setattr(graph_profile, "build_profile", lambda schema: {"built": True})
    graph_profile.reset_cache()
    yield fake
    graph_profile.reset_cache()


def _loader(counter):
    def load():
        counter.append(1)
        return {"node_props": {}, "rel_props": {}, "relationships": []}
    return load


def test_second_call_is_served_from_cache(db):
    calls = []
    graph_profile.get_cached_profile(_loader(calls))
    graph_profile.get_cached_profile(_loader(calls))
    assert len(calls) == 1


def test_write_counter_change_invalidates(db):
    calls = []
    graph_profile.get_cached_profile(_loader(calls))
    db.write_count += 1
    graph_profile.get_cached_profile(_loader(calls))
    assert len(calls) == 2


def test_fingerprint_change_invalidates_without_a_counter_change(db):
    """The out-of-process write case -- the reason one layer is not enough."""
    calls = []
    graph_profile.get_cached_profile(_loader(calls))
    db.nodes = 0          # someone wiped the graph from a script
    graph_profile.get_cached_profile(_loader(calls))
    assert len(calls) == 2


def test_cache_hit_still_costs_exactly_one_fingerprint_query(db):
    graph_profile.get_cached_profile(_loader([]))
    graph_profile.get_cached_profile(_loader([]))
    assert db.fingerprint_calls == 2


def test_reset_cache_forces_recompute(db):
    calls = []
    graph_profile.get_cached_profile(_loader(calls))
    graph_profile.reset_cache()
    graph_profile.get_cached_profile(_loader(calls))
    assert len(calls) == 2
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/unit/test_graph_profile_cache.py -v`
Expected: `AttributeError: module 'agentic_kg.common.graph_profile' has no attribute 'reset_cache'`

- [ ] **Step 3: Implement — append to `graph_profile.py`**

```python
# Module-level, deliberately not session state: one adk web process serves
# every session, and what is cached is a property of the database, not of a
# conversation. Per-session caches would disagree with each other the moment
# one session rebuilt the graph.
_cache: Dict[str, Any] = {}

_FINGERPRINT_QUERY = (
    "MATCH (n) WITH count(n) AS nodes "
    "OPTIONAL MATCH ()-[r]->() "
    "RETURN nodes, count(r) AS rels"
)


def reset_cache() -> None:
    """Discard the cached profile. For tests and for explicit invalidation."""
    _cache.clear()


def _fingerprint():
    """Node and relationship totals, or None when they cannot be read.

    Catches writes the counter structurally cannot see -- anything done to the
    database from outside this process. Blind to edits that change property
    values without changing counts; the shape and cardinality we cache do not
    move under those.
    """
    rows = _records(graphdb.send_read_query(_FINGERPRINT_QUERY, max_rows=None))
    if not rows:
        return None
    return (rows[0].get("nodes"), rows[0].get("rels"))


def get_cached_profile(schema_loader) -> Dict[str, Any]:
    """Return {"schema": ..., "profile": ...}, recomputing only when stale.

    schema_loader is a zero-argument callable returning the enriched schema.
    It is only invoked on a miss, so the expensive enriched pass is skipped
    entirely on a hit.
    """
    write_count = getattr(graphdb, "write_count", None)
    fingerprint = _fingerprint()

    if (
        _cache
        and _cache.get("write_count") == write_count
        and _cache.get("fingerprint") == fingerprint
        and fingerprint is not None
    ):
        return _cache["value"]

    schema = schema_loader()
    value = {"schema": schema, "profile": build_profile(schema)}
    _cache.clear()
    _cache.update(
        {"write_count": write_count, "fingerprint": fingerprint, "value": value}
    )
    return value
```

- [ ] **Step 4: Run and confirm the tests pass**

Run: `uv run pytest tests/unit/test_graph_profile_cache.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest`
Expected: 213 passed, 3 skipped

- [ ] **Step 6: Commit**

```bash
git add src/agentic_kg/common/graph_profile.py tests/unit/test_graph_profile_cache.py
git commit -m "feat: cache the graph profile behind counter and fingerprint

Module-level rather than session state: one process serves every session and
the cached thing belongs to the database, not a conversation. The counter
catches in-process writes with no query; the fingerprint catches writes made
from outside the process, which a counter structurally cannot see and which
happen in this workflow."
```

---

### Task 8: Gate `get_physical_schema` and reshape `read_neo4j_cypher`

**Files:**
- Modify: `src/agentic_kg/tools/cypher_tools.py:26-63`
- Test: `tests/unit/test_cypher_tools.py` (extend)

**Interfaces:**
- Consumes: `get_cached_profile` (Task 7), `send_read_query` (Task 4).
- Produces: `get_physical_schema(include_data_profile: bool = False)`, `get_graph_schema_with_profile()` (the graphrag-only wrapper, bound by Task 9), and the reshaped `read_neo4j_cypher`.

Critical: with `include_data_profile=False` the returned dict must be byte-identical to today's — no `profile` key **and** no `values`/`distinct_count` keys, because `is_enhanced` stays off. Three other consumers depend on that: the coordinator (`agent.py:44`), `graph_construction_agent` (`variants.py:59`, latency previously tuned), and `single_agent`'s `cypher_agent`.

The wrapper must be a named function, never `functools.partial`: ADK derives tool identity from the callable (`function_tool.py:42-58`), and a partial has no `__name__`, so it falls through to `func.__class__.__name__` — the literal string `"partial"` — with the `functools.partial` class docstring as its description.

- [ ] **Step 1: Write the failing tests (append to `tests/unit/test_cypher_tools.py`)**

```python
from agentic_kg.common.neo4j_for_adk import MAX_RETURNED_ROWS


class FakeReadDb(FakeGraphDb):
    def __init__(self, payload=None):
        super().__init__()
        self.payload = payload or {"records": [], "row_count": 0, "truncated": False}
        self.read_queries = []

    def send_read_query(self, query, parameters=None, max_rows=MAX_RETURNED_ROWS):
        self.read_queries.append((query, parameters, max_rows))
        return {"status": "success", "query_result": self.payload}


def test_read_neo4j_cypher_returns_a_single_nested_payload_key(monkeypatch):
    db = FakeReadDb({"records": [{"n": 1}], "row_count": 1, "truncated": False})
    monkeypatch.setattr(cypher_tools, "graphdb", db)
    result = cypher_tools.read_neo4j_cypher("MATCH (n) RETURN n")
    assert set(result) == {"status", "query_result"}
    assert result["query_result"]["row_count"] == 1


def test_read_neo4j_cypher_goes_through_the_read_only_path(monkeypatch):
    db = FakeReadDb()
    monkeypatch.setattr(cypher_tools, "graphdb", db)
    cypher_tools.read_neo4j_cypher("MATCH (n) RETURN n")
    assert db.read_queries, "must use send_read_query, not send_query"
    assert not db.queries, "must not fall through to the unrestricted path"


def test_get_physical_schema_without_profile_is_unchanged(monkeypatch):
    captured = {}

    def fake_structured_schema(driver, **kwargs):
        captured.update(kwargs)
        return {"node_props": {"A": [{"property": "x", "type": "STRING"}]},
                "rel_props": {}, "relationships": [], "metadata": {}}

    monkeypatch.setattr(cypher_tools, "get_structured_schema", fake_structured_schema)
    monkeypatch.setattr(cypher_tools, "graphdb", FakeReadDb())

    result = cypher_tools.get_physical_schema()
    schema = result["schema"]

    assert captured.get("is_enhanced") in (None, False), "is_enhanced must stay off"
    assert "profile" not in schema
    prop = schema["node_props"]["A"][0]
    assert "values" not in prop and "distinct_count" not in prop


def test_get_physical_schema_with_profile_enriches_and_profiles(monkeypatch):
    captured = {}

    def fake_structured_schema(driver, **kwargs):
        captured.update(kwargs)
        return {"node_props": {}, "rel_props": {}, "relationships": [], "metadata": {}}

    monkeypatch.setattr(cypher_tools, "get_structured_schema", fake_structured_schema)
    monkeypatch.setattr(cypher_tools, "graphdb", FakeReadDb())
    from agentic_kg.common import graph_profile
    graph_profile.reset_cache()
    monkeypatch.setattr(graph_profile, "graphdb", FakeReadDb())

    result = cypher_tools.get_physical_schema(include_data_profile=True)

    assert captured["is_enhanced"] is True
    assert captured["sanitize"] is True
    assert captured.get("timeout") is not None
    assert "profile" in result["schema"]
    graph_profile.reset_cache()


def test_graphrag_wrapper_is_a_named_function_not_a_partial():
    """ADK derives tool identity from the callable; a partial registers as
    'partial' with functools' own docstring as its description."""
    fn = cypher_tools.get_graph_schema_with_profile
    assert fn.__name__ == "get_graph_schema_with_profile"
    assert fn.__doc__ and "partial" not in fn.__doc__.lower()
    import inspect
    assert not inspect.signature(fn).parameters, "must take no model-visible args"
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/unit/test_cypher_tools.py -v -k "read_neo4j or physical_schema or wrapper"`
Expected: FAIL — `read_neo4j_cypher` still returns `records` at the top level; `get_graph_schema_with_profile` does not exist.

- [ ] **Step 3: Implement**

Update the imports at the top of `src/agentic_kg/tools/cypher_tools.py`:

```python
from agentic_kg.common.graph_profile import get_cached_profile
from agentic_kg.common.neo4j_for_adk import (
    get_graphdb,
    is_write_query,
    close_graphdb,
    QUERY_TIMEOUT_SECONDS,
)
```

Replace `get_physical_schema` (`:26-42`):

```python
def get_physical_schema(include_data_profile: bool = False) -> Dict[str, Any]:
    """Tool to get the physical schema of a Neo4j graph database.

    Args:
        include_data_profile: when True, additionally enrich the schema with
            value information and attach a data profile (entity counts,
            per-pattern degree, per-value counts, completeness and uniqueness
            annotations). Defaults to False so the coordinator, the
            construction agent and single_agent's cypher agent keep receiving
            exactly the dict they receive today -- enrichment costs a full scan
            per label and one of those consumers is latency-tuned.

    Returns:
        A dictionary containing:
        - "status": "success" or "error"
        - "schema": the schema as a JSON object if "success"
        - "error_message": the error message if "error"
    """
    driver = graphdb.get_driver()
    database_name = graphdb.get_config().database

    try:
        if not include_data_profile:
            return tool_success("schema", get_structured_schema(driver, database=database_name))

        def load_enriched_schema():
            return get_structured_schema(
                driver,
                is_enhanced=True,
                database=database_name,
                timeout=QUERY_TIMEOUT_SECONDS,
                sanitize=True,
            )

        cached = get_cached_profile(load_enriched_schema)
        schema = dict(cached["schema"])
        schema["profile"] = cached["profile"]
        return tool_success("schema", schema)
    except Exception as e:
        return tool_error(str(e))


def get_graph_schema_with_profile() -> Dict[str, Any]:
    """Get the graph schema together with a profile of the data it holds.

    Returns the node labels, relationship types and properties, plus for each
    property whether its reported values are complete, whether it uniquely
    identifies its entity, and how its values are distributed; and for each
    relationship pattern how many edges it has and how they spread across the
    nodes at each end. Use this before writing any query: it tells you the
    grain of a pattern, which determines whether counting rows is meaningful.
    """
    return get_physical_schema(include_data_profile=True)
```

Replace `read_neo4j_cypher` (`:44-63`):

```python
def read_neo4j_cypher(
    query: str,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Submits a read-only Cypher query to a Neo4j database.

    Args:
        query: The Cypher query string to execute.
        params: Optional parameters to pass to the query.

    Returns:
        A dictionary with "status" and, on success, "query_result" holding:
        - "records": the rows, capped in number
        - "row_count" (or "row_count_at_least" for very large results)
        - "truncated": whether records were capped
        - "note": present when truncated

        Counts and rankings must come from a Cypher aggregation, never from
        counting the returned records.
    """
    return graphdb.send_read_query(query, params)
```

- [ ] **Step 4: Run and confirm the tests pass**

Run: `uv run pytest tests/unit/test_cypher_tools.py -v`
Expected: all pass

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest`
Expected: 219 passed, 3 skipped

- [ ] **Step 6: Commit**

```bash
git add src/agentic_kg/tools/cypher_tools.py tests/unit/test_cypher_tools.py
git commit -m "feat: gate schema enrichment, reshape read_neo4j_cypher

One flag gates both is_enhanced and the profile, so the three non-graphrag
consumers get a dict byte-identical to today's -- enrichment costs a full scan
per label and would have made the coordinator's empty-database check scan the
whole graph. The graphrag wrapper is a named function because ADK derives tool
identity from the callable and a partial registers as 'partial'. Query results
nest under one payload key, since _payload_key raises on siblings."
```

---

### Task 9: `graphrag_agent_v2`

**Files:**
- Modify: `src/agentic_kg/coordinators/multi_agent/sub_agents/graphrag_agent/variants.py`, `.../graphrag_agent/agent.py`
- Test: `tests/unit/test_graphrag_context_filtering.py`

**Interfaces:**
- Consumes: `drop_foreign_context` (Task 1), `get_graph_schema_with_profile` (Task 8).
- Produces: `variants["graphrag_agent_v2"]` with keys `instruction`, `tools`, `before_model_callback`.

v1 stays exactly as it is, for the acceptance A/B. `.get("before_model_callback")` returning `None` is a verified no-op: the field defaults to `None` (`llm_agent.py:225`) and `canonical_before_model_callbacks` returns `[]` on falsy (390-391).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_graphrag_context_filtering.py
"""End-to-end proof that graphrag_agent_v2 never sees another agent's output.

Asserts on what reached the model, never on what the model said. The negative
control matters as much as the positive one: without asserting that v1 DOES
receive the sentinel, a test passing because the fixture never produced
foreign context would look identical to a working filter.
"""
import asyncio

from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from pydantic import Field

from agentic_kg.common.adk_context import FOREIGN_CONTEXT_SENTINEL
from agentic_kg.coordinators.multi_agent.sub_agents.graphrag_agent.variants import variants


class CapturingLlm(BaseLlm):
    """Scripted LLM that also records every request it was handed.

    Extends the pattern in test_schema_refinement_loop_turn_cap.py:36.
    """
    responses: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        self.requests.append(llm_request)
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        yield self.responses[index]


def _text(text):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _build_agent(variant_name):
    from google.adk.agents import Agent
    spec = variants[variant_name]
    return Agent(
        name=variant_name,
        model=CapturingLlm(model="scripted", responses=[_text("ok")]),
        description="test",
        instruction=spec["instruction"],
        tools=spec["tools"],
        before_model_callback=spec.get("before_model_callback"),
    )


async def _run_with_foreign_history(agent):
    runner = InMemoryRunner(agent=agent, app_name="ctx_test")
    session = await runner.session_service.create_session(app_name="ctx_test", user_id="u1")

    # A message shaped exactly as ADK reshapes another agent's output.
    foreign = types.Content(role="user", parts=[
        types.Part(text=FOREIGN_CONTEXT_SENTINEL),
        types.Part(text="[schema_critic] said: 4 suppliers have no quote rows"),
    ])
    from google.adk.events.event import Event
    await runner.session_service.append_event(
        session=session, event=Event(author="user", content=foreign))

    async for _ in runner.run_async(
        user_id="u1", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="which are orphaned?")]),
    ):
        pass
    return agent.model.requests


def _sentinel_present(requests):
    for req in requests:
        for content in (req.contents or []):
            for part in (content.parts or []):
                if getattr(part, "text", None) == FOREIGN_CONTEXT_SENTINEL:
                    return True
    return False


def test_v2_never_receives_foreign_context():
    # asyncio.run inside a sync test, matching
    # test_schema_refinement_loop_turn_cap.py:120 -- this repo drives async ADK
    # code without taking a pytest-asyncio dependency.
    requests = asyncio.run(_run_with_foreign_history(_build_agent("graphrag_agent_v2")))
    assert requests, "the model was never called"
    assert not _sentinel_present(requests)


def test_v1_still_receives_it_negative_control():
    """Proves the fixture actually produces foreign context."""
    requests = asyncio.run(_run_with_foreign_history(_build_agent("graphrag_agent_v1")))
    assert requests, "the model was never called"
    assert _sentinel_present(requests)


def test_v1_is_left_intact_for_the_acceptance_ab():
    assert "graphrag_agent_v1" in variants
    assert "before_model_callback" not in variants["graphrag_agent_v1"]


def test_v2_binds_the_profile_wrapper_not_the_bare_schema_tool():
    from agentic_kg.tools.cypher_tools import get_graph_schema_with_profile, get_physical_schema
    tools = variants["graphrag_agent_v2"]["tools"]
    assert get_graph_schema_with_profile in tools
    assert get_physical_schema not in tools
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/unit/test_graphrag_context_filtering.py -v`
Expected: `KeyError: 'graphrag_agent_v2'`

Do **not** add `pytest-asyncio`. It is not installed and is not needed: this repo drives async ADK code from sync tests via `asyncio.run()` (see `tests/unit/test_schema_refinement_loop_turn_cap.py:120`), and the tests above follow that pattern.

- [ ] **Step 3: Add the v2 variant**

In `.../graphrag_agent/variants.py`, add these imports at the top:

```python
from agentic_kg.common.adk_context import drop_foreign_context
from agentic_kg.tools.cypher_tools import get_graph_schema_with_profile
```

Add a new entry to the `variants` dict, after the `graphrag_agent_v1` entry (leave v1 untouched):

```python
    # graphrag_agent_v2 -- grounded in the graph rather than in the transcript.
    # v1 is retained unchanged so the acceptance run can A/B the two.
    "graphrag_agent_v2": {
        "instruction": """
        You are an expert at information retrieval from a knowledge graph.
        Your goal is to answer the user's questions using only what the graph says.

        Tools:
        - get_graph_schema_with_profile: the graph's structure plus a profile of
          its data -- how many of each entity, how relationship patterns spread
          across the nodes at each end, and for each property whether its values
          are complete, whether it uniquely identifies its entity, and how those
          values are distributed
        - read_neo4j_cypher: run a read-only Cypher query
        - finished: signal that the user is done

        The graph is the only source of truth about the data. Every count, name,
        membership and ranking you state must come from a query result in this
        turn. If you find yourself recalling a fact from earlier in the
        conversation, query for it again instead -- it is cheap, and what you
        remember may describe a graph that has since changed.

        For each question:
        1. Call 'get_graph_schema_with_profile' first.
        2. Say what you are counting and over what before you query. If a
           relationship pattern's degree shows more than one edge per node, a
           result row is not one node -- decide which subset you mean and say so.
        3. Counts, rankings and superlatives must come from a Cypher aggregation,
           never from counting the rows you got back. Report ties as ties rather
           than reading a ranking off row order.
        4. Do not group or rank by a property whose 'uniqueness' is 'non_unique'
           -- it will silently merge rows. Where it is 'unknown', say so in your
           answer rather than proceeding as if it were unique.
        5. Before ordering, comparing or aggregating a property numerically,
           check its type. A STRING needs an explicit cast: '9' sorts after '30'
           without one, and a value carrying a currency symbol or separator will
           not cast cleanly. The profile flags such properties as 'numeric_like'.
        6. Where an annotation reads 'unknown' or 'not_profiled', treat it as
           missing information to disclose, never as permission to assume.
        """,
        "tools": [
            get_graph_schema_with_profile,
            read_neo4j_cypher,
            finished,
        ],
        "before_model_callback": drop_foreign_context,
    },
```

- [ ] **Step 4: Select v2 in `agent.py`**

Replace the body of `.../graphrag_agent/agent.py`:

```python
from google.adk.agents import Agent


from agentic_kg.common.llm_catalog import get_llm, LlmKind

from .variants import variants

AGENT_NAME = "graphrag_agent_v2"
graphrag_agent = Agent(
    name=AGENT_NAME,
    # Stays on the conversational tier deliberately: the experiment is whether
    # better information alone fixes the framing errors. Changing information
    # and model together would make the result uninterpretable.
    model=get_llm(LlmKind.conversational),
    description="Information retrieval from a knowledge graph using a range of query tools.", # Crucial for delegation later
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"],
    # .get() so v1, which has no callback, stays a no-op: the field defaults to
    # None (llm_agent.py:225) and canonical_before_model_callbacks returns []
    # on falsy (390-391).
    before_model_callback=variants[AGENT_NAME].get("before_model_callback"),
)

root_agent = graphrag_agent
```

- [ ] **Step 5: Run and confirm the tests pass**

Run: `uv run pytest tests/unit/test_graphrag_context_filtering.py -v`
Expected: 4 passed

- [ ] **Step 6: Run the whole suite, including the generality guard**

Run: `uv run pytest`
Expected: 223 passed, 3 skipped. If `test_generality.py` fails, a dataset word reached the new prompt — remove it.

- [ ] **Step 7: Commit**

```bash
git add src/agentic_kg/coordinators/multi_agent/sub_agents/graphrag_agent/ tests/unit/test_graphrag_context_filtering.py
git commit -m "feat: add graphrag_agent_v2 with profile tool and context filter

v1 is left untouched so the acceptance run can A/B the two by flipping one
constant. Four prompt rules, none naming any dataset: aggregate in Cypher,
state what you are counting, respect the uniqueness annotation, and cast
STRING properties before comparing them numerically."
```

---

### Task 10: Shape-based integration tests

**Files:**
- Create: `tests/integration/test_graph_profile_shapes.py`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing importable.

This is the only place any of this work runs against a graph other than the demo's. Without it, "universal" is an assertion. It follows the existing container pattern in `tests/integration/test_neo4j_integration.py`: module-level `pytestmark`, a Docker reachability check that skips the module, and `Neo4jContainer(image="neo4j:5")`.

- [ ] **Step 1: Write the tests**

```python
# tests/integration/test_graph_profile_shapes.py
"""Runs the profile against graph shapes the demo graph does not have.

The demo graph has one (start, end) pair per relationship type, every label
under 10k, single-label nodes and bare-identifier names. Every one of those is
a property of that dataset, not of graphs in general, and each one hides a
different bug. These tests are the only evidence this work generalises.
"""
import pytest

pytestmark = pytest.mark.integration

try:
    import docker  # type: ignore
    docker.from_env().ping()
except Exception as e:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {e}", allow_module_level=True)


@pytest.fixture(scope="module")
def graphdb_against_container():
    from testcontainers.neo4j import Neo4jContainer
    from agentic_kg.common import graph_profile
    from agentic_kg.common.neo4j_for_adk import Neo4jForADK

    with Neo4jContainer(image="neo4j:5") as container:
        url = container.get_connection_url()
        try:
            auth = container.get_auth()
        except AttributeError:
            auth = ("neo4j", "password")

        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(url, auth=auth)

        db = Neo4jForADK.__new__(Neo4jForADK)
        db._driver = driver
        db._neo4j_config = type("Cfg", (), {"database": "neo4j"})()
        db.write_count = 0

        original = graph_profile.graphdb
        graph_profile.graphdb = db
        graph_profile.reset_cache()

        db.send_query("""
            CREATE (a:Alpha {code: 'a1'})
            CREATE (b:Beta {code: 'b1'})
            CREATE (g:Gamma {code: 'g1'})
            // one relationship type spanning two (start, end) pairs
            CREATE (a)-[:LINKS {kind: 'x'}]->(b)
            CREATE (a)-[:LINKS {kind: 'y'}]->(g)
            // self-referencing type
            CREATE (a2:Alpha {code: 'a2'})
            CREATE (a)-[:FOLLOWS]->(a2)
            // multi-label node and a name that is not a bare identifier
            CREATE (m:Alpha:Archived {code: 'm1'})
            CREATE (le:`Legal Entity` {code: 'le1'})
            CREATE (m)-[:LINKS {kind: 'x'}]->(le)
        """)

        yield db, graph_profile

        graph_profile.graphdb = original
        graph_profile.reset_cache()
        driver.close()


def _schema_for(db):
    from neo4j_graphrag.schema import get_structured_schema
    return get_structured_schema(db.get_driver(), is_enhanced=True,
                                 database="neo4j", sanitize=True)


def test_profile_completes_on_all_shapes(graphdb_against_container):
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    assert profile["patterns"], "no patterns profiled"
    for entry in profile["properties"].values():
        assert entry != "profile_error", "an entity failed to profile"


def test_degree_is_reported_per_pattern_not_pooled(graphdb_against_container):
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    patterns = {p["pattern"] for p in profile["patterns"]}
    # LINKS spans Alpha->Beta, Alpha->Gamma and Alpha->Legal Entity. Pooling
    # them under one "LINKS" key is the bug this asserts against.
    links = {p for p in patterns if "[LINKS]" in p}
    assert len(links) >= 3, f"LINKS pooled instead of split per pattern: {patterns}"


def test_self_referencing_pattern_is_profiled(graphdb_against_container):
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    follows = [p for p in profile["patterns"] if p["type"] == "FOLLOWS"]
    assert follows and follows[0]["start"] == follows[0]["end"] == "Alpha"


def test_non_identifier_label_survives_quoting(graphdb_against_container):
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    assert "Legal Entity" in profile["entity_counts"]
    assert profile["properties"].get("Legal Entity") != "profile_error"


def test_annotations_are_always_present(graphdb_against_container):
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    for entity, props in profile["properties"].items():
        if not isinstance(props, list):
            continue
        for prop in props:
            for key in ("completeness", "uniqueness", "numeric_like", "value_counts"):
                assert key in prop, f"{entity}.{prop.get('property')} missing {key}"


def test_result_larger_than_the_cap_reports_a_true_row_count(graphdb_against_container):
    db, _ = graphdb_against_container
    from agentic_kg.common.neo4j_for_adk import MAX_RETURNED_ROWS
    payload = db.send_read_query(
        f"UNWIND range(1, {MAX_RETURNED_ROWS + 20}) AS i RETURN i")["query_result"]
    assert payload["row_count"] == MAX_RETURNED_ROWS + 20
    assert len(payload["records"]) == MAX_RETURNED_ROWS
    assert payload["truncated"] is True


def test_malformed_query_returns_a_structured_error(graphdb_against_container):
    db, _ = graphdb_against_container
    result = db.send_read_query("MATCH ( RETURN")
    assert result["status"] == "error"
    assert result["error_message"]


@pytest.mark.parametrize("write_query", [
    "CREATE (n:ShouldNotExist)",
    "MATCH (n:Alpha) SET n.tampered = true",
    "CALL apoc.refactor.mergeNodes([]) YIELD node RETURN node",
])
def test_writes_are_rejected_by_the_server_on_the_read_path(
        graphdb_against_container, write_query):
    """The last case is the one is_write_query's regex does not catch."""
    db, _ = graphdb_against_container
    result = db.send_read_query(write_query)
    assert result["status"] == "error", f"{write_query!r} was not rejected"
```

- [ ] **Step 2: Run the integration tests**

Run: `uv run pytest -m integration tests/integration/test_graph_profile_shapes.py -v`
Expected: all pass, or the module skips cleanly if Docker is not running.

If `apoc.refactor.mergeNodes` errors with "unknown procedure" rather than an access-mode violation, that still satisfies the assertion (the write did not happen), but note it in the commit message — APOC is not installed in the stock `neo4j:5` image.

- [ ] **Step 3: Confirm the default suite is unaffected**

Run: `uv run pytest`
Expected: 223 passed, 3 skipped — the new file is excluded by `addopts`.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_graph_profile_shapes.py
git commit -m "test: verify the profile against non-demo graph shapes

Multi-pattern relationship types, a self-referencing type, a multi-label node
and a label that is not a bare identifier. Each one hides a bug the demo graph
cannot expose. Also asserts the server rejects writes on the read path,
including the APOC call the is_write_query regex misses."
```

---

## Manual acceptance (not automated)

After Task 10, before merging. Twelve runs: two questions x three repeats x v1 and v2, graded per criterion rather than collapsed into a verdict.

Repeats rather than a pinned temperature: there is no sampling control anywhere in the codebase, and `_llm_instances` is cached per `LlmKind`, so the conversational `LiteLlm` object is shared by four agents — there is no per-agent handle to pin.

1. Rebuild the demo graph from `data/bom` through the normal flow.
2. Set `AGENT_NAME = "graphrag_agent_v1"`, **restart `adk web`** (a server started before an edit serves stale code — `SESSION_HANDOFF.md:14`), run both questions three times each in fresh sessions.
3. Set `AGENT_NAME = "graphrag_agent_v2"`, restart again, repeat.
4. Grade each run against the A1–A4 / B1–B4 rubric in the spec's *Acceptance* section.

Note that the cache and the query-result reshaping are global, so the A/B isolates only the prompt, the profile tool and the context filter.

If framing errors persist in v2 with the degree profile in context, that is evidence the model tier is the binding constraint — a useful result, not a failed one. Only then consider a third `LlmKind`.

## Self-review notes

**Spec coverage.** Context filter → Task 1. Generality guard → Task 2. Write counter and `DROP` → Task 3. Timeout, read access mode, streaming, list summarising → Task 4. Completeness/uniqueness/numeric_like annotations → Task 5. Entity counts, per-pattern degree, value counts, quoting, error isolation, budget → Task 6. Two-layer cache → Task 7. `include_data_profile` gating, byte-identical guarantee, named wrapper, nested payload → Task 8. v2 variant, four prompt rules, v1 preserved → Task 9. Shape-based verification → Task 10. Acceptance rubric → manual section.

**Type consistency.** `send_read_query(query, parameters=None, max_rows=MAX_RETURNED_ROWS)` is called with `max_rows=None` throughout Tasks 6 and 7 and with the default in Task 8. `annotate_property(prop, entity_count)` takes `Optional[int]` and is called with `counts.get(name)`, which may be `None`. `build_profile(schema)` returns the four keys Task 7 wraps and Task 10 asserts on. `get_cached_profile(schema_loader)` returns `{"schema", "profile"}`, which is exactly what Task 8 unpacks.

**Known deviation to watch during implementation.** The two degree queries in `_pattern_degree` scan the pattern twice; if that shows up as a cost problem on a large graph, combine them — but only with the integration test in place to prove the combined form still reports per-pattern figures.
