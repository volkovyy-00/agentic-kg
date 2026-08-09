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


# Rows processed vs. what the MERGE actually left in the graph

@pytest.fixture
def duplicate_keys(monkeypatch):
    """Four rows keyed on two distinct assembly names, as a bill-of-materials
    file with one row per component of an assembly legitimately is."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["assembly_name", "part"], [
            {"assembly_name": "chair", "part": "leg"},
            {"assembly_name": "chair", "part": "seat"},
            {"assembly_name": "desk", "part": "top"},
            {"assembly_name": "desk", "part": "drawer"},
        ]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)


class MergeSimulatingGraphDb:
    """Actually dedupes on the MERGE key, instead of a canned count response.

    A hardcoded {"count": 2} response would pass this test whether or not
    load_nodes_from_csv sent the right label or the right key column -- it
    proves nothing about the count query beyond "some number came back". This
    tracks the same key-uniqueness MERGE gives a real database, so the count
    query result actually depends on how many distinct keys the load sent.
    """
    def __init__(self):
        self.queries = []
        self._keys_by_label: dict[str, set] = {}

    def send_query(self, query, parameters=None):
        parameters = parameters or {}
        self.queries.append((query, parameters))
        if query.strip().startswith("UNWIND") and "MERGE (n:" in query:
            label = query.split("MERGE (n:", 1)[1].split(" ", 1)[0].split("{")[0]
            keys = self._keys_by_label.setdefault(label, set())
            unique_column_name = parameters["unique_column_name"]
            for row in parameters["rows"]:
                keys.add(row[unique_column_name])
            return {"status": "success", "records": []}
        if query.startswith("MATCH (n:") and "count(n)" in query:
            label = query.split("MATCH (n:", 1)[1].split(")", 1)[0]
            return {"status": "success",
                    "records": [{"count": len(self._keys_by_label.get(label, set()))}]}
        return {"status": "success", "records": []}


def test_node_count_reflects_merged_nodes_not_rows(monkeypatch, duplicate_keys):
    """MERGE collapses rows sharing a key, so 'rows' overstates the graph --
    the reported node count must come from the database, not the file."""
    db = MergeSimulatingGraphDb()
    monkeypatch.setattr(kg, "graphdb", db)
    result = kg.load_nodes_from_csv("assemblies.csv", "Assembly", "assembly_name", ["part"])
    assert result["status"] == "success"
    assert result["rows_loaded"]["rows"] == 4, "row count stays available"
    assert result["rows_loaded"]["nodes_in_graph"] == 2, (
        "two distinct assembly_name values (chair, desk) among four rows"
    )


def test_node_count_is_read_back_from_the_label(monkeypatch, duplicate_keys):
    db = FakeGraphDb()
    monkeypatch.setattr(kg, "graphdb", db)
    kg.load_nodes_from_csv("assemblies.csv", "Assembly", "assembly_name", ["part"])
    count_query, _params = db.queries[-1]
    assert count_query == "MATCH (n:Assembly) RETURN count(n) AS count"


def test_a_failed_count_does_not_fail_a_committed_load(monkeypatch, duplicate_keys):
    """The rows are already committed, so an unavailable count must leave the
    field off rather than report the load as failed."""
    db = FakeGraphDb(responses=[
        {"status": "success", "records": []},
        {"status": "error", "error_message": "boom"},
    ])
    monkeypatch.setattr(kg, "graphdb", db)
    result = kg.load_nodes_from_csv("assemblies.csv", "Assembly", "assembly_name", ["part"])
    assert result["status"] == "success"
    assert "nodes_in_graph" not in result["rows_loaded"]
    assert result["rows_loaded"]["rows"] == 4


REL_RULE = {
    "source_file": "knows.csv",
    "relationship_type": "KNOWS",
    "from_node_label": "Person",
    "from_node_column": "id",
    "to_node_label": "Person",
    "to_node_column": "name",
    "properties": [],
}


def test_relationship_count_reflects_merged_edges_not_rows(monkeypatch, one_batch):
    """rows_matched counts matched rows, which collapse on MERGE the same way
    node rows do, so the edge total is counted separately."""
    db = FakeGraphDb(responses=[
        {"status": "success", "records": [{"rows_matched": 1}]},
        {"status": "success", "records": [{"count": 27}]},
    ])
    monkeypatch.setattr(kg, "graphdb", db)
    result = kg.import_relationships(dict(REL_RULE))
    assert result["rows_loaded"]["rows_matched"] == 1, "existing field is unchanged"
    assert result["rows_loaded"]["relationships_in_graph"] == 27
    count_query, _params = db.queries[-1]
    assert count_query == "MATCH ()-[r:KNOWS]->() RETURN count(r) AS count"


def test_partial_failure_summary_reports_nodes_not_rows(monkeypatch):
    """This string is what the agent parrots back, so it must not call 64 rows
    64 nodes."""
    def fake_import_nodes(rule):
        if rule["label"] == "Assembly":
            return {"status": "success", "rows_loaded": {
                "source_file": "assemblies.csv", "rows": 64, "nodes_in_graph": 10}}
        return {"status": "error", "error_message": "boom"}

    monkeypatch.setattr(kg, "import_nodes", fake_import_nodes)
    monkeypatch.setattr(kg, "import_relationships", lambda rule: {
        "status": "success", "rows_loaded": {
            "source_file": "knows.csv", "rows": 88, "rows_matched": 88,
            "relationships_in_graph": 27}})
    plan = {
        "Assembly": {"construction_type": "node", "label": "Assembly"},
        "Supplier": {"construction_type": "node", "label": "Supplier"},
        "KNOWS": {"construction_type": "relationship", "relationship_type": "KNOWS"},
    }
    result = kg.construct_domain_graph(plan)
    assert result["status"] == "error"
    assert "Assembly (10 nodes now in graph, 64 rows read)" in result["error_message"]
    assert "KNOWS (27 relationships now in graph, 88 rows read)" in result["error_message"]


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
    """Neo4j does reject a null MERGE key, but only once the batch has been
    sent, and its message names the property alone. Failing here instead names
    the file and the columns it has, which is what an agent needs to correct
    the plan."""
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


def _load_queries(fake_db):
    """Only the batch loads, excluding the post-load count query."""
    return [q for q, _params in fake_db.queries if q.startswith("UNWIND")]


def test_present_columns_still_load(fake_db, one_batch):
    result = kg.load_nodes_from_csv("people.csv", "Person", "id", ["name"])
    assert result["status"] == "success"
    assert len(_load_queries(fake_db)) == 1


def test_header_is_only_checked_once(fake_db, monkeypatch):
    """The check must not re-run per batch, and must not stop a valid load."""
    def two_batches(relative_path, batch_size=1000):
        yield ["id", "name"], [{"id": "1", "name": "Ada"}]
        yield ["id", "name"], [{"id": "2", "name": "Grace"}]
    monkeypatch.setattr(kg, "read_csv_batches", two_batches)
    result = kg.load_nodes_from_csv("people.csv", "Person", "id", ["name"])
    assert result["status"] == "success"
    assert len(_load_queries(fake_db)) == 2


# Read failures are reported, not raised

def _undecodable_csv(monkeypatch, tmp_path):
    """Point the source at a CSV that is not UTF-8, as an Excel export often is."""
    (tmp_path / "latin1.csv").write_bytes(b"id,name\n1,Bj\xf6rk\n")
    monkeypatch.setenv("SOURCE_URI", str(tmp_path))
    from agentic_kg.common.config import reset_settings
    reset_settings()


def test_a_non_utf8_source_is_reported_not_raised(fake_db, monkeypatch, tmp_path):
    """UnicodeDecodeError escaped read_csv_batches and crashed the run instead
    of reaching the agent, which broke the contract that every tool returns a
    ToolResult."""
    _undecodable_csv(monkeypatch, tmp_path)
    result = kg.load_nodes_from_csv("latin1.csv", "Person", "id", ["name"])
    assert result["status"] == "error"
    assert "latin1.csv" in result["error_message"]
    assert "UnicodeDecodeError" in result["error_message"]


def test_a_non_utf8_source_is_reported_by_the_relationship_loader(
        fake_db, monkeypatch, tmp_path):
    _undecodable_csv(monkeypatch, tmp_path)
    rule = {
        "source_file": "latin1.csv",
        "relationship_type": "KNOWS",
        "from_node_label": "Person",
        "from_node_column": "id",
        "to_node_label": "Person",
        "to_node_column": "name",
        "properties": [],
    }
    result = kg.import_relationships(rule)
    assert result["status"] == "error"
    assert "UnicodeDecodeError" in result["error_message"]


def test_a_read_failure_does_not_escape_the_agent_facing_tool(
        fake_db, monkeypatch, tmp_path):
    """build_graph_from_construction_rules is the tool the agent actually calls,
    so it is the one that must not raise into ADK."""
    _undecodable_csv(monkeypatch, tmp_path)
    import agentic_kg.tools.cypher_tools as cypher_tools
    monkeypatch.setattr(cypher_tools, "graphdb", fake_db)

    class FakeToolContext:
        def __init__(self, state):
            self.state = state

    context = FakeToolContext({kg.APPROVED_CONSTRUCTION_PLAN: {"Person": {
        "construction_type": "node", "source_file": "latin1.csv", "label": "Person",
        "unique_column_name": "id", "properties": ["name"]}}})

    result = kg.build_graph_from_construction_rules(context)
    assert result["status"] == "error"
    assert "UnicodeDecodeError" in result["error_message"]


def test_a_missing_source_file_names_itself_once(fake_db, monkeypatch, tmp_path):
    """The dedicated FileNotFoundError clause keeps the message from reading
    "ghost.csv: FileNotFoundError: No such source file: ghost.csv"."""
    monkeypatch.setenv("SOURCE_URI", str(tmp_path))
    from agentic_kg.common.config import reset_settings
    reset_settings()
    result = kg.load_nodes_from_csv("ghost.csv", "Person", "id", ["name"])
    assert result["status"] == "error"
    assert result["error_message"].lower().count("ghost.csv") == 1


# --- typed properties -------------------------------------------------------

@pytest.fixture
def typed_batch(monkeypatch):
    """One batch carrying a currency value, a count and a flag."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "cost", "days", "preferred"], [
            {"id": "1", "cost": "$42.73", "days": "8", "preferred": "yes"},
        ]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)


def test_typed_values_are_converted_before_the_batch_is_sent(fake_db, typed_batch):
    """Without conversion the graph stores '$42.73' and every aggregation over it
    is wrong -- the defect itself."""
    kg.load_nodes_from_csv("p.csv", "P", "id", ["cost", "days", "preferred"],
                           {"cost": "float", "days": "integer", "preferred": "boolean"})

    _query, params = fake_db.queries[0]
    assert params["rows"][0]["cost"] == 42.73
    assert params["rows"][0]["days"] == 8
    assert params["rows"][0]["preferred"] is True


def test_untyped_properties_keep_the_original_parameter(fake_db, typed_batch):
    """$properties must carry untyped names only: the existing FOREACH over it is
    the ragged-row guard, and a typed name in both lists would be written twice."""
    kg.load_nodes_from_csv("p.csv", "P", "id", ["cost", "days"], {"cost": "float"})

    _query, params = fake_db.queries[0]
    assert params["properties"] == ["days"]
    assert params["typed_properties"] == ["cost"]


def test_the_query_has_a_write_pass_and_a_clear_pass_for_typed_properties(
        fake_db, typed_batch):
    """One pass cannot do both: SET n[k] = null deletes a property, so writing
    and clearing have to be separate FOREACHes over different filters."""
    kg.load_nodes_from_csv("p.csv", "P", "id", ["cost"], {"cost": "float"})

    query, params = fake_db.queries[0]
    assert "$typed_properties" in query
    assert "SET n[k] = null" in query
    assert params["clear"] == kg.CLEAR_SENTINEL


def test_an_unconvertible_typed_value_becomes_the_sentinel(fake_db, monkeypatch):
    """It must not stay a string (that is the untyped graph) and must not become
    a plain null (Cypher cannot tell that from a ragged row's absent key)."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "1", "cost": "N/A"}]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)

    kg.load_nodes_from_csv("p.csv", "P", "id", ["cost"], {"cost": "float"})

    _query, params = fake_db.queries[0]
    assert params["rows"][0]["cost"] == kg.CLEAR_SENTINEL


def test_a_blank_typed_value_becomes_the_sentinel(fake_db, monkeypatch):
    """A blank clears a stale value on re-run exactly as an unparseable one does;
    only the counting differs."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "1", "cost": ""}]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)

    kg.load_nodes_from_csv("p.csv", "P", "id", ["cost"], {"cost": "float"})

    _query, params = fake_db.queries[0]
    assert params["rows"][0]["cost"] == kg.CLEAR_SENTINEL


def test_a_ragged_row_keeps_a_typed_key_absent(fake_db, monkeypatch):
    """An absent key must stay absent, not become the sentinel: the sentinel
    clears, and a ragged row must never erase an earlier row's value."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "1", "cost": "1"}, {"id": "1"}]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)

    kg.load_nodes_from_csv("p.csv", "P", "id", ["cost"], {"cost": "float"})

    _query, params = fake_db.queries[0]
    assert "cost" not in params["rows"][1]


def test_a_typed_property_missing_from_the_header_fails_loudly(fake_db, typed_batch):
    """A typed property whose column does not exist would clear that property on
    every row, silently, for the whole file."""
    result = kg.load_nodes_from_csv("p.csv", "P", "id", ["csot"], {"csot": "float"})

    assert result["status"] == "error"
    assert "csot" in result["error_message"]
    assert fake_db.queries == []


def test_a_mostly_unconvertible_column_stops_the_rule(fake_db, monkeypatch):
    """Three of four non-blank values failing is a wrong type, not dirty data --
    and continuing would clear real values row by row."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "cost"], [
            {"id": "1", "cost": "N/A"}, {"id": "2", "cost": "N/A"},
            {"id": "3", "cost": "N/A"}, {"id": "4", "cost": "5"},
        ]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)

    result = kg.load_nodes_from_csv("p.csv", "P", "id", ["cost"], {"cost": "float"})

    assert result["status"] == "error"
    assert "cost" in result["error_message"]
    assert "float" in result["error_message"]
    assert "was not sent" in result["error_message"]
    assert fake_db.queries == []


def test_a_sparse_column_does_not_trip_the_gate(fake_db, monkeypatch):
    """Blanks are absence, not a wrong type. Counting them would abort a correct
    load of any optional column."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "cost"], [
            {"id": "1", "cost": ""}, {"id": "2", "cost": ""},
            {"id": "3", "cost": ""}, {"id": "4", "cost": "5"},
        ]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)

    result = kg.load_nodes_from_csv("p.csv", "P", "id", ["cost"], {"cost": "float"})

    assert result["status"] == "success", result.get("error_message")


def test_blank_and_unconvertible_counts_stay_separate(fake_db, monkeypatch):
    """Merged into one 'N of M failed' figure, a sparse column and a mistyped one
    read identically to whoever gets the result."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "cost"], [
            {"id": "1", "cost": ""}, {"id": "2", "cost": "N/A"}, {"id": "3", "cost": "5"},
        ]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)

    result = kg.load_nodes_from_csv("p.csv", "P", "id", ["cost"], {"cost": "float"})

    tally = result["rows_loaded"]["type_conversion"]["cost"]
    assert tally == {"converted": 1, "blank": 1, "unconvertible": 1, "examples": ["N/A"]}


def test_several_flagged_properties_join_into_one_warning(fake_db, monkeypatch):
    """'warning' is a single string with six existing assertions against it; a
    second flagged property must not overwrite the first."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "cost", "days"], [
            {"id": "1", "cost": "N/A", "days": "x"}, {"id": "2", "cost": "5", "days": "8"},
        ]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)

    result = kg.load_nodes_from_csv("p.csv", "P", "id", ["cost", "days"],
                                    {"cost": "float", "days": "integer"})

    warning = result["rows_loaded"]["warning"]
    assert "cost" in warning and "days" in warning


def test_the_gate_runs_on_every_batch_not_only_the_first(fake_db, monkeypatch):
    """Every bundled file is under DEFAULT_BATCH_SIZE, so no test built from
    data/bom can tell 'checked once' from 'checked every batch'. A first-batch-
    only gate would let a file whose later rows drift keep clearing values."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "1", "cost": "1"}, {"id": "2", "cost": "2"}]
        yield ["id", "cost"], [{"id": "3", "cost": "N/A"}, {"id": "4", "cost": "N/A"}]
        yield ["id", "cost"], [{"id": "5", "cost": "5"}, {"id": "6", "cost": "6"}]

    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)

    result = kg.load_nodes_from_csv("p.csv", "P", "id", ["cost"], {"cost": "float"})

    assert result["status"] == "error"
    assert "cost" in result["error_message"]
    # Batch 1 was sent and committed; batches 2 and 3 were not.
    assert len(fake_db.queries) == 1
    assert [row["id"] for row in fake_db.queries[0][1]["rows"]] == ["1", "2"]
    assert "2 rows committed" in result["error_message"]


def test_import_nodes_passes_property_types_from_the_rule(monkeypatch, fake_db, typed_batch):
    """A rule may legitimately not carry the key at all (a plan proposed before
    this change), so the read must be defensive."""
    monkeypatch.setattr(kg, "create_uniqueness_constraint",
                        lambda label, column: {"status": "success"})

    kg.import_nodes({"source_file": "p.csv", "label": "P", "unique_column_name": "id",
                     "properties": ["cost"], "property_types": {"cost": "float"}})

    _query, params = fake_db.queries[0]
    assert params["rows"][0]["cost"] == 42.73


def test_relationship_typed_values_are_converted(fake_db, typed_batch):
    """Relationship properties carry the same per-row data (lead times, costs)
    and need the same treatment; only the loader differs."""
    kg.import_relationships({
        "source_file": "p.csv", "relationship_type": "R",
        "from_node_label": "A", "from_node_column": "id",
        "to_node_label": "B", "to_node_column": "id",
        "properties": ["cost"], "property_types": {"cost": "float"},
    })

    _query, params = fake_db.queries[0]
    assert params["rows"][0]["cost"] == 42.73
    assert params["typed_properties"] == ["cost"]


def test_relationship_gate_stops_the_rule(fake_db, monkeypatch):
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "1", "cost": "N/A"}, {"id": "2", "cost": "N/A"}]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)

    result = kg.import_relationships({
        "source_file": "p.csv", "relationship_type": "R",
        "from_node_label": "A", "from_node_column": "id",
        "to_node_label": "B", "to_node_column": "id",
        "properties": ["cost"], "property_types": {"cost": "float"},
    })

    assert result["status"] == "error"
    assert fake_db.queries == []


def test_relationship_type_warning_and_join_warning_combine(fake_db, monkeypatch):
    """import_relationships already sets loaded["warning"] for its join
    under-match case; a second `loaded["warning"] = ...` for the type warning
    would silently overwrite that assignment instead of joining with it."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "cost"], [
            {"id": "1", "cost": "N/A"}, {"id": "2", "cost": "5"},
        ]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)

    result = kg.import_relationships({
        "source_file": "p.csv", "relationship_type": "R",
        "from_node_label": "A", "from_node_column": "id",
        "to_node_label": "B", "to_node_column": "id",
        "properties": ["cost"], "property_types": {"cost": "float"},
    })

    warning = result["rows_loaded"]["warning"]
    assert "cost" in warning
    assert "rows matched both endpoints" in warning


def test_a_type_declared_for_a_name_not_in_properties_is_ignored_by_the_loader(
        fake_db, monkeypatch):
    """Refused at approval time, so the loader only sees this via a hand-built
    plan or a direct call -- but coercing off the raw map would let such a name
    trip the gate and abort the whole rule for a property no FOREACH would have
    written, and if the name were the key column the row's key would be replaced
    by the sentinel and MERGE would key the node on it."""
    def fake_batches(relative_path, batch_size=1000):
        yield ["id", "q"], [{"id": "1", "q": "x"}, {"id": "2", "q": "y"}]
    monkeypatch.setattr(kg, "read_csv_batches", fake_batches)

    result = kg.load_nodes_from_csv("p.csv", "P", "id", ["id"], {"q": "integer"})

    assert result["status"] == "success", result.get("error_message")
    _query, params = fake_db.queries[0]
    assert params["typed_properties"] == []
    assert params["rows"][0]["q"] == "x"


@pytest.fixture
def header_only(monkeypatch):
    """A valid empty export: a header, no data rows. read_csv_batches yields
    nothing for one, so the header has to come from somewhere else."""
    def no_batches(relative_path, batch_size=1000):
        return
        yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(kg, "read_csv_batches", no_batches)
    monkeypatch.setattr(kg, "read_csv_header", lambda path: ["id", "name"], raising=False)


def test_a_header_only_file_still_refuses_a_column_it_does_not_have(fake_db, header_only):
    """Both loaders checked their columns inside the batch loop, which a
    header-only file never enters -- so a rule naming a column the file does not
    have came back as a clean zero-row success, and the promised missing-column
    error never fired. Nothing was written, but the plan was wrong and the run
    said it worked."""
    missing_key = kg.load_nodes_from_csv("people.csv", "Person", "person_id", ["name"])
    assert missing_key["status"] == "error"
    assert "person_id" in missing_key["error_message"]

    missing_typed = kg.load_nodes_from_csv(
        "people.csv", "Person", "id", ["name", "aeg"], {"aeg": "integer"})
    assert missing_typed["status"] == "error"
    assert "aeg" in missing_typed["error_message"]

    assert fake_db.queries == []


def test_a_header_only_file_loads_zero_rows_when_the_columns_are_right(fake_db, header_only):
    """An empty export is not an error. Refusing one would block a load whose
    rows simply have not landed yet."""
    result = kg.load_nodes_from_csv("people.csv", "Person", "id", ["name"])
    assert result["status"] == "success", result.get("error_message")
    assert result["rows_loaded"]["rows"] == 0
