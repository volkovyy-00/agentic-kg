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

    def to_eager_result(self):
        # DEVIATION from the brief's verbatim FakeResult: added this method.
        # result_to_adk (neo4j_for_adk.py) calls result.to_eager_result().records
        # -- without this, every successful send_query call raises AttributeError
        # internally (caught by send_query's except, turned into a tool_error),
        # so the write-counter increment line is never reached and
        # test_write_query_increments_the_counter fails even with the full
        # implementation in place. This mirrors the real neo4j.Result API just
        # enough for that call to work.
        return type("EagerResult", (), {"records": self._records})()


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
        # Counts close() calls, so a test can tell "closed once" from
        # "closed twice" -- which is the whole point of close() being idempotent.
        self.closed = 0
        self._records = [FakeRecord(r) for r in records]

    def session(self, **config):
        self.sessions.append(config)
        return FakeSession(self.queries, self._records)

    def close(self):
        self.closed += 1


class FakeConfig:
    """Stands in for Neo4jConfig. `uri` exists because the reconnect log line
    (Task 5) reads it; `database` because send_query passes it to session()."""
    database = "neo4j"
    uri = "bolt://fake:7687"


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


def test_summarised_values_are_declared_not_silently_omitted(db):
    """Would catch: reporting a payload as complete while data was withheld.

    Every row fits under the cap, so `truncated` is legitimately False -- but a
    1536-element list inside one of them was replaced by a summary. Without a
    separate signal the payload positively asserts a completeness it does not
    have, which is the exact failure class this work exists to remove.
    """
    db._driver = FakeDriver([{"embedding": [0.1] * 1536, "name": "x"}])
    payload = db.send_read_query("MATCH (c) RETURN c")["query_result"]
    assert payload["truncated"] is False, "no ROWS were dropped"
    assert payload["values_summarised"] is True
    assert "note" in payload


def test_values_summarised_is_false_when_nothing_was_omitted(db):
    """The negative control: the flag must not be always-on."""
    db._driver = FakeDriver([{"name": "x", "tags": ["a", "b"]}])
    payload = db.send_read_query("MATCH (c) RETURN c")["query_result"]
    assert payload["values_summarised"] is False
    assert payload["records"][0]["tags"] == ["a", "b"]
    assert "note" not in payload


def test_close_shuts_the_driver_and_marks_the_instance_closed(db):
    db.close()
    assert db._driver.closed == 1
    assert db._closed is True


def test_close_twice_does_not_close_the_driver_twice(db):
    db.close()
    db.close()
    assert db._driver.closed == 1
    assert db._closed is True


def test_send_query_reconnects_after_close(db, monkeypatch):
    rebuilt = FakeDriver()
    monkeypatch.setattr(neo4j_for_adk, "make_driver", lambda cfg: rebuilt)
    monkeypatch.setattr(
        neo4j_for_adk, "load_neo4j_config_from_settings", lambda: FakeConfig())

    db.close()
    result = db.send_query("MATCH (n) RETURN n")

    assert result["status"] == "success", result.get("error_message")
    assert db._driver is rebuilt
    assert db._closed is False


def test_send_read_query_reconnects_after_close(db, monkeypatch):
    rebuilt = FakeDriver()
    monkeypatch.setattr(neo4j_for_adk, "make_driver", lambda cfg: rebuilt)
    monkeypatch.setattr(
        neo4j_for_adk, "load_neo4j_config_from_settings", lambda: FakeConfig())

    db.close()
    result = db.send_read_query("MATCH (n) RETURN n")

    assert result["status"] == "success", result.get("error_message")
    assert db._driver is rebuilt


def test_reconnect_rederives_config_instead_of_reusing_it(db, monkeypatch):
    """The integration fixture swaps NEO4J_DSN and calls reset_settings()
    before closing. Reusing the config captured at construction would silently
    reconnect to the old database -- see the spec's "Config is re-derived"
    section."""
    calls = []
    fresh = FakeConfig()
    fresh.database = "swapped"

    def load():
        calls.append(1)
        return fresh

    monkeypatch.setattr(neo4j_for_adk, "load_neo4j_config_from_settings", load)
    monkeypatch.setattr(neo4j_for_adk, "make_driver", lambda cfg: FakeDriver())

    db.close()
    db.send_query("RETURN 1")

    assert calls == [1]
    assert db._neo4j_config is fresh
    # The re-derived config is the one actually used, not just stored.
    assert db._driver.sessions[0]["database"] == "swapped"


def test_a_failed_reconnect_returns_a_tool_error_rather_than_raising(db, monkeypatch):
    """_ensure_connected sits INSIDE the existing try, for the reason those
    methods already document for session creation: a failure must reach the
    agent as a structured error, not raise mid-turn."""
    def boom():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(neo4j_for_adk, "load_neo4j_config_from_settings", boom)

    db.close()
    result = db.send_query("RETURN 1")

    assert result["status"] == "error"
    assert "settings unavailable" in result["error_message"]


def test_get_driver_reconnects_after_close(db, monkeypatch):
    """_physical_schema hands this driver straight to neo4j_graphrag, so it
    never passes through send_query. Covering only the query methods would
    leave the profiling path broken."""
    rebuilt = FakeDriver()
    monkeypatch.setattr(neo4j_for_adk, "make_driver", lambda cfg: rebuilt)
    monkeypatch.setattr(
        neo4j_for_adk, "load_neo4j_config_from_settings", lambda: FakeConfig())

    db.close()

    assert db.get_driver() is rebuilt
    assert db._closed is False
