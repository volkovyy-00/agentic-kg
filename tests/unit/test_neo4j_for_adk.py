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
