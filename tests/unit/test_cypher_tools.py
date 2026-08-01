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


from agentic_kg.common.neo4j_for_adk import MAX_RETURNED_ROWS


class FakeReadDb(FakeGraphDb):
    def __init__(self, payload=None):
        super().__init__()
        self.payload = payload or {"records": [], "row_count": 0, "truncated": False}
        self.read_queries = []

    # _physical_schema reads both of these before its try block, so a fake
    # without them raises AttributeError rather than exercising the tool.
    def get_driver(self):
        return object()

    def get_config(self):
        return type("Cfg", (), {"database": "neo4j"})()

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

    result = cypher_tools.get_graph_schema_with_profile()

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


@pytest.mark.parametrize("tool_name", [
    "get_physical_schema", "get_graph_schema_with_profile", "read_neo4j_cypher",
])
def test_no_tool_exposes_the_profile_flag_to_a_model(tool_name):
    """ADK cannot express a default in a tool declaration, so any parameter
    with one is advertised as REQUIRED. A model-visible include_data_profile
    would make the profile optional -- and could trigger a full scan per label
    on the latency-tuned construction agent."""
    import inspect
    from google.adk.tools.function_tool import FunctionTool

    fn = getattr(cypher_tools, tool_name)
    assert "include_data_profile" not in inspect.signature(fn).parameters

    declared = FunctionTool(fn)._get_declaration()
    props = (declared.parameters.properties or {}) if declared.parameters else {}
    assert "include_data_profile" not in props
