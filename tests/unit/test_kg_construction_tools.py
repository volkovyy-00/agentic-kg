"""Unit tests for query construction and result collection.

The database is faked: these tests assert what Cypher gets built and how
failures propagate, without a Neo4j instance.
"""
import pytest

from agentic_kg.tools import kg_construction_tools as kg


class FakeGraphDb:
    def __init__(self, responses=None):
        self.queries = []
        self.responses = responses or []

    def send_query(self, query, parameters=None):
        self.queries.append((query, parameters or {}))
        if self.responses:
            return self.responses.pop(0)
        return {"status": "success", "records": []}


@pytest.fixture
def fake_db(monkeypatch):
    db = FakeGraphDb()
    monkeypatch.setattr(kg, "graphdb", db)
    return db


@pytest.fixture
def one_batch(monkeypatch):
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "name"], [{"id": "1", "name": "Ada"}]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)


def test_node_query_interpolates_label_not_dynamic(fake_db, one_batch):
    kg.load_nodes_from_csv("people.csv", "Person", "id", ["name"])
    query, params = fake_db.queries[0]
    assert "MERGE (n:Person" in query
    assert "$($label)" not in query
    assert params["rows"] == [{"id": "1", "name": "Ada"}]


def test_node_query_uses_unwind_not_load_csv(fake_db, one_batch):
    kg.load_nodes_from_csv("people.csv", "Person", "id", ["name"])
    query, _params = fake_db.queries[0]
    assert query.strip().startswith("UNWIND $rows AS row")
    assert "LOAD CSV" not in query
    assert "file:///" not in query


def test_invalid_label_is_rejected_before_any_query(fake_db, one_batch):
    result = kg.load_nodes_from_csv("people.csv", "Not A Label", "id", ["name"])
    assert result["status"] == "error"
    assert fake_db.queries == []


def test_invalid_column_is_rejected_before_any_query(fake_db, one_batch):
    result = kg.load_nodes_from_csv("people.csv", "Person", "not a column", ["name"])
    assert result["status"] == "error"
    assert fake_db.queries == []


def test_relationship_columns_are_validated(fake_db, one_batch):
    rule = {
        "source_file": "knows.csv",
        "relationship_type": "KNOWS",
        "from_node_label": "Person",
        "from_node_column": "bad column",
        "to_node_label": "Person",
        "to_node_column": "to_id",
        "properties": [],
    }
    result = kg.import_relationships(rule)
    assert result["status"] == "error"
    assert fake_db.queries == []


INJECTION_PAYLOAD = "Person)\nDETACH\nDELETE\nn\n//"


def test_node_label_injection_payload_is_rejected_before_any_query(fake_db, one_batch):
    """is_symbol() alone lets newline/paren payloads through; the identifier
    regex in _checked() must catch what is_symbol() misses."""
    result = kg.load_nodes_from_csv("people.csv", INJECTION_PAYLOAD, "id", ["name"])
    assert result["status"] == "error"
    assert fake_db.queries == []


def test_node_column_injection_payload_is_rejected_before_any_query(fake_db, one_batch):
    result = kg.load_nodes_from_csv("people.csv", "Person", INJECTION_PAYLOAD, ["name"])
    assert result["status"] == "error"
    assert fake_db.queries == []


def test_relationship_type_injection_payload_is_rejected_before_any_query(fake_db, one_batch):
    rule = {
        "source_file": "knows.csv",
        "relationship_type": INJECTION_PAYLOAD,
        "from_node_label": "Person",
        "from_node_column": "from_id",
        "to_node_label": "Person",
        "to_node_column": "to_id",
        "properties": [],
    }
    result = kg.import_relationships(rule)
    assert result["status"] == "error"
    assert fake_db.queries == []


def test_import_nodes_rejects_injection_payload_before_creating_constraint(fake_db, monkeypatch):
    """import_nodes must validate before calling create_uniqueness_constraint,
    which interpolates the same identifiers itself."""
    constraint_calls = []
    monkeypatch.setattr(
        kg,
        "create_uniqueness_constraint",
        lambda label, column: constraint_calls.append((label, column)) or {"status": "success"},
    )
    rule = {
        "label": INJECTION_PAYLOAD,
        "unique_column_name": "id",
        "source_file": "people.csv",
        "properties": ["name"],
    }
    result = kg.import_nodes(rule)
    assert result["status"] == "error"
    assert constraint_calls == [], "create_uniqueness_constraint must not be reached"
    assert fake_db.queries == []


def test_batch_failure_reports_rows_committed(monkeypatch, one_batch):
    db = FakeGraphDb(responses=[{"status": "error", "error_message": "boom"}])
    monkeypatch.setattr(kg, "graphdb", db)
    result = kg.load_nodes_from_csv("people.csv", "Person", "id", ["name"])
    assert result["status"] == "error"
    assert "people.csv" in result["error_message"]
    assert "boom" in result["error_message"]


def test_construct_domain_graph_reports_failure(monkeypatch):
    monkeypatch.setattr(kg, "import_nodes", lambda rule: {"status": "error", "error_message": "nope"})
    monkeypatch.setattr(kg, "import_relationships", lambda rule: {"status": "success"})
    plan = {"Person": {"construction_type": "node", "label": "Person"}}
    result = kg.construct_domain_graph(plan)
    assert result["status"] == "error", "a failed import must not be reported as success"


def test_construct_domain_graph_loads_nodes_before_relationships(monkeypatch):
    order = []
    monkeypatch.setattr(kg, "import_nodes", lambda rule: order.append("node") or {"status": "success"})
    monkeypatch.setattr(kg, "import_relationships", lambda rule: order.append("rel") or {"status": "success"})
    plan = {
        "KNOWS": {"construction_type": "relationship", "relationship_type": "KNOWS"},
        "Person": {"construction_type": "node", "label": "Person"},
    }
    kg.construct_domain_graph(plan)
    assert order == ["node", "rel"]
