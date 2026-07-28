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


def test_construct_domain_graph_reports_tool_error_for_a_rule_missing_a_key(fake_db, one_batch):
    """construction_plan is LLM-produced; a rule missing a required key must
    surface as a tool_error, not an unhandled KeyError escaping into ADK."""
    plan = {"Person": {"construction_type": "node", "label": "Person"}}  # no unique_column_name etc.
    result = kg.construct_domain_graph(plan)
    assert result["status"] == "error"
    assert "Person" in result["error_message"]


def test_construct_domain_graph_reports_tool_error_for_a_relationship_rule_missing_a_key(fake_db, one_batch):
    plan = {"KNOWS": {"construction_type": "relationship", "relationship_type": "KNOWS"}}
    result = kg.construct_domain_graph(plan)
    assert result["status"] == "error"
    assert "KNOWS" in result["error_message"]


def test_construct_domain_graph_reports_successes_alongside_failures(monkeypatch):
    """On partial failure the agent needs to know which rules already
    committed, both to report accurately and to avoid redoing loaded work on
    retry -- not just see the concatenated failure text."""
    def fake_import_nodes(rule):
        if rule["label"] == "Product":
            return {"status": "success", "rows_loaded": {"source_file": "products.csv", "rows": 10}}
        return {"status": "error", "error_message": "boom"}

    monkeypatch.setattr(kg, "import_nodes", fake_import_nodes)
    monkeypatch.setattr(
        kg,
        "import_relationships",
        lambda rule: {"status": "success", "rows_loaded": {"source_file": "knows.csv", "rows": 5}},
    )
    plan = {
        "Product": {"construction_type": "node", "label": "Product"},
        "Supplier": {"construction_type": "node", "label": "Supplier"},
        "KNOWS": {"construction_type": "relationship", "relationship_type": "KNOWS"},
    }
    result = kg.construct_domain_graph(plan)
    assert result["status"] == "error"
    assert "Product (10 rows)" in result["error_message"]
    assert "KNOWS (5 rows)" in result["error_message"]
    assert "boom" in result["error_message"]


REL_RULE = {
    "source_file": "knows.csv",
    "relationship_type": "KNOWS",
    "from_node_label": "Person",
    "from_node_column": "id",
    "to_node_label": "Person",
    "to_node_column": "name",
    "properties": [],
}


def test_import_relationships_warns_when_no_rows_match(monkeypatch, one_batch):
    """A join that matches nothing raises no Cypher error -- it must not look
    like a clean success."""
    db = FakeGraphDb(responses=[{"status": "success", "records": [{"rows_matched": 0}]}])
    monkeypatch.setattr(kg, "graphdb", db)
    result = kg.import_relationships(dict(REL_RULE))
    assert result["status"] == "success"
    assert result["rows_loaded"]["rows_matched"] == 0
    assert "warning" in result["rows_loaded"]
    assert "knows.csv" in result["rows_loaded"]["warning"]


def test_import_relationships_has_no_warning_when_every_row_matches(monkeypatch, one_batch):
    db = FakeGraphDb(responses=[{"status": "success", "records": [{"rows_matched": 1}]}])
    monkeypatch.setattr(kg, "graphdb", db)
    result = kg.import_relationships(dict(REL_RULE))
    assert result["rows_loaded"]["rows_matched"] == 1
    assert "warning" not in result["rows_loaded"]


@pytest.fixture
def two_batches(monkeypatch):
    """Two batches of two rows each, so the warning threshold and the
    cross-batch summation of matched counts are both exercised."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "name"], [{"id": "1", "name": "Ada"}, {"id": "2", "name": "Grace"}]
        yield ["id", "name"], [{"id": "3", "name": "Alan"}, {"id": "4", "name": "Edsger"}]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)


def test_import_relationships_sums_matches_across_batches_and_warns_at_half(
        monkeypatch, two_batches):
    """2 of 4 rows matched is not < 4/2, so the threshold must NOT warn --
    and the two per-batch counts must be summed, not overwritten."""
    db = FakeGraphDb(responses=[
        {"status": "success", "records": [{"rows_matched": 1}]},
        {"status": "success", "records": [{"rows_matched": 1}]},
    ])
    monkeypatch.setattr(kg, "graphdb", db)
    result = kg.import_relationships(dict(REL_RULE))
    assert result["rows_loaded"]["rows"] == 4
    assert result["rows_loaded"]["rows_matched"] == 2
    assert "warning" not in result["rows_loaded"]


def test_import_relationships_warns_below_half_across_batches(monkeypatch, two_batches):
    db = FakeGraphDb(responses=[
        {"status": "success", "records": [{"rows_matched": 1}]},
        {"status": "success", "records": [{"rows_matched": 0}]},
    ])
    monkeypatch.setattr(kg, "graphdb", db)
    result = kg.import_relationships(dict(REL_RULE))
    assert result["rows_loaded"]["rows_matched"] == 1
    warning = result["rows_loaded"]["warning"]
    assert "only 1 of 4 rows matched both endpoints" in warning


def test_import_relationships_no_warning_when_most_rows_match(monkeypatch, two_batches):
    db = FakeGraphDb(responses=[
        {"status": "success", "records": [{"rows_matched": 2}]},
        {"status": "success", "records": [{"rows_matched": 1}]},
    ])
    monkeypatch.setattr(kg, "graphdb", db)
    result = kg.import_relationships(dict(REL_RULE))
    assert result["rows_loaded"]["rows_matched"] == 3
    assert "warning" not in result["rows_loaded"]


def test_construct_domain_graph_surfaces_warnings_on_success(monkeypatch):
    monkeypatch.setattr(kg, "import_nodes", lambda rule: {
        "status": "success", "rows_loaded": {"source_file": "people.csv", "rows": 3}})
    monkeypatch.setattr(kg, "import_relationships", lambda rule: {
        "status": "success",
        "rows_loaded": {"source_file": "knows.csv", "rows": 88,
                        "rows_matched": 0,
                        "warning": "only 0 of 88 rows matched both endpoints"},
    })
    plan = {
        "Person": {"construction_type": "node", "label": "Person"},
        "KNOWS": {"construction_type": "relationship", "relationship_type": "KNOWS"},
    }
    result = kg.construct_domain_graph(plan)
    assert result["status"] == "success"
    assert result["warnings"] == ["only 0 of 88 rows matched both endpoints"]


# Header validation before any query is sent

def test_missing_key_column_is_rejected_before_any_query(fake_db, one_batch):
    """row[$unique_column_name] is null when the column is absent, and MERGE
    then collapses the whole file onto one null-keyed node and reports success.
    Nothing may be sent in that case."""
    result = kg.load_nodes_from_csv("people.csv", "Person", "employee_id", ["name"])
    assert result["status"] == "error"
    assert "employee_id" in result["error_message"]
    assert "id" in result["error_message"]  # lists what is actually available
    assert fake_db.queries == []


def test_missing_join_column_is_rejected_before_any_query(fake_db, one_batch):
    """A null join value matches no node, so the load silently builds zero
    relationships instead of failing."""
    rule = {
        "source_file": "knows.csv",
        "relationship_type": "KNOWS",
        "from_node_label": "Person",
        "from_node_column": "id",
        "to_node_label": "Person",
        "to_node_column": "friend_id",
        "properties": [],
    }
    result = kg.import_relationships(rule)
    assert result["status"] == "error"
    assert "friend_id" in result["error_message"]
    assert fake_db.queries == []


def test_present_columns_still_load(fake_db, one_batch):
    result = kg.load_nodes_from_csv("people.csv", "Person", "id", ["name"])
    assert result["status"] == "success"
    assert len(fake_db.queries) == 1


def test_header_is_only_checked_once(fake_db, monkeypatch):
    """The check must not re-run per batch, and must not stop a valid load."""
    def two_batches(relative_path, batch_size=1000):
        yield ["id", "name"], [{"id": "1", "name": "Ada"}]
        yield ["id", "name"], [{"id": "2", "name": "Grace"}]
    monkeypatch.setattr(kg, "read_csv_batches", two_batches)
    result = kg.load_nodes_from_csv("people.csv", "Person", "id", ["name"])
    assert result["status"] == "success"
    assert len(fake_db.queries) == 2
