"""Unit tests for query construction in cypher_tools.

Mirrors the injection-payload style in test_kg_construction_tools.py: the
database is faked so these tests assert what Cypher gets built (or, for
rejected input, that nothing is sent) without a Neo4j instance.
"""
import pytest

from agentic_kg.tools import cypher_tools


from fakes import RecordingGraphDb

# Shared with the rest of the unit suite; see tests/unit/fakes.py.
FakeGraphDb = RecordingGraphDb


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
    """Returns one fixed payload from every read. get_driver/get_config come
    from the shared base, which provides them for exactly this reason."""

    def __init__(self, payload=None):
        super().__init__()
        self.payload = payload or {"records": [], "row_count": 0, "truncated": False}

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


def test_profiled_payload_states_each_property_once_and_leads_with_the_profile(
        monkeypatch):
    """Would catch: returning the library's schema with `profile` bolted on.

    That shape carried the raw `values`/`distinct_count` for every property the
    profile also describes, and the raw copy came first. On the library's
    sampled branch the two disagree outright -- the raw copy lists five
    arbitrary values while the profile says completeness "unknown" and
    withholds them -- so the payload asserted precisely what the profile exists
    to deny. `metadata` (constraints, indexes) describes write-time guarantees
    a retrieval agent cannot ask about.
    """
    def fake_structured_schema(driver, **kwargs):
        return {"node_props": {"A": [{"property": "x", "type": "STRING",
                                      "values": ["1", "2"], "distinct_count": 9}]},
                "rel_props": {}, "relationships": [{"start": "A", "type": "R", "end": "A"}],
                "metadata": {"constraint": [], "index": []}}

    monkeypatch.setattr(cypher_tools, "get_structured_schema", fake_structured_schema)
    monkeypatch.setattr(cypher_tools, "graphdb", FakeReadDb())
    from agentic_kg.common import graph_profile
    graph_profile.reset_cache()
    monkeypatch.setattr(graph_profile, "graphdb", FakeReadDb())

    schema = cypher_tools.get_graph_schema_with_profile()["schema"]

    assert list(schema) == ["profile", "relationships"]
    assert schema["relationships"] == [{"start": "A", "type": "R", "end": "A"}]
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


class FakeDdlDb(FakeGraphDb):
    """Records queries and answers the SHOW CONSTRAINTS/INDEXES listings."""

    def __init__(self, constraints=(), indexes=()):
        super().__init__()
        self.constraints = list(constraints)
        self.indexes = list(indexes)

    def send_query(self, query, parameters=None):
        self.queries.append((query, parameters or {}))
        if "SHOW CONSTRAINTS" in query:
            return {"status": "success",
                    "records": [{"name": n} for n in self.constraints]}
        if "SHOW INDEXES" in query:
            return {"status": "success",
                    "records": [{"name": n} for n in self.indexes]}
        return {"status": "success", "records": []}


def test_reset_drops_ddl_by_name_not_by_parameter(monkeypatch):
    """Would catch: `DROP CONSTRAINT $name`.

    Cypher does not accept a parameter in a DDL name position, so the
    parameterised form is rejected by the server on every call -- meaning
    reset_neo4j_data reported success while dropping nothing at all. The name
    must be interpolated (backtick-quoted, since it comes from the database).
    """
    db = FakeDdlDb(constraints=["unique_person_id"], indexes=["idx_person_name"])
    monkeypatch.setattr(cypher_tools, "graphdb", db)

    result = cypher_tools.reset_neo4j_data()
    assert result["status"] == "success"

    drops = [q for q, _ in db.queries if q.startswith("DROP")]
    assert "DROP CONSTRAINT `unique_person_id`" in drops
    assert "DROP INDEX `idx_person_name`" in drops
    for query, params in db.queries:
        if query.startswith("DROP"):
            assert not params, "DDL names cannot be passed as query parameters"
            assert "$" not in query


def test_reset_quotes_ddl_names_that_are_not_bare_identifiers(monkeypatch):
    """Generated constraint names can contain characters a bare identifier
    cannot; unquoted interpolation would produce a syntax error."""
    db = FakeDdlDb(constraints=["constraint 1-of 2"])
    monkeypatch.setattr(cypher_tools, "graphdb", db)
    cypher_tools.reset_neo4j_data()
    assert "DROP CONSTRAINT `constraint 1-of 2`" in [q for q, _ in db.queries]


def test_reset_surfaces_a_failed_listing_instead_of_crashing(monkeypatch):
    """Would catch: `if (list_constraints == "error")`.

    That compares a dict to a string and is never true, so a failed listing
    fell through to result["records"] and raised KeyError/TypeError inside the
    tool instead of returning the error to the agent.
    """
    db = FakeDdlDb()

    def failing(query, parameters=None):
        db.queries.append((query, parameters or {}))
        if "SHOW CONSTRAINTS" in query:
            return {"status": "error", "error_message": "boom"}
        return {"status": "success", "records": []}

    monkeypatch.setattr(db, "send_query", failing)
    monkeypatch.setattr(cypher_tools, "graphdb", db)

    result = cypher_tools.reset_neo4j_data()
    assert result["status"] == "error"
    assert result["error_message"] == "boom"
