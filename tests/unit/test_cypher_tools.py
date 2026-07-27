"""Unit tests for query construction in cypher_tools.

Mirrors the injection-payload style in test_kg_construction_tools.py: the
database is faked so these tests assert what Cypher gets built (or, for
rejected input, that nothing is sent) without a Neo4j instance.
"""
import pytest

from agentic_kg.tools import cypher_tools


class FakeGraphDb:
    def __init__(self):
        self.queries = []

    def send_query(self, query, parameters=None):
        self.queries.append((query, parameters or {}))
        return {"status": "success", "records": []}


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeGraphDb()
    monkeypatch.setattr(cypher_tools, "graphdb", db)
    return db


INJECTION_PAYLOAD = "Person)\nDETACH\nDELETE\nn\n//"


def test_create_uniqueness_constraint_builds_expected_query(fake_db):
    result = cypher_tools.create_uniqueness_constraint("Person", "id")
    assert result["status"] == "success"
    query, _params = fake_db.queries[0]
    assert "FOR (n:Person)" in query
    assert "REQUIRE n.id IS UNIQUE" in query


def test_create_uniqueness_constraint_rejects_label_injection_payload_before_any_query(fake_db):
    """create_uniqueness_constraint used to guard interpolation with
    is_symbol() alone, which lets newline/paren payloads through -- it must
    use the same identifier regex kg_construction_tools.py uses."""
    result = cypher_tools.create_uniqueness_constraint(INJECTION_PAYLOAD, "id")
    assert result["status"] == "error"
    assert fake_db.queries == []


def test_create_uniqueness_constraint_rejects_property_injection_payload_before_any_query(fake_db):
    result = cypher_tools.create_uniqueness_constraint("Person", INJECTION_PAYLOAD)
    assert result["status"] == "error"
    assert fake_db.queries == []
