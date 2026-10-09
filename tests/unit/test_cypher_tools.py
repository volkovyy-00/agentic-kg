"""Unit tests for query construction in cypher_tools.

Mirrors the injection-payload style in test_kg_construction_tools.py: the
database is faked so these tests assert what Cypher gets built (or, for
rejected input, that nothing is sent) without a Neo4j instance.
"""

import pytest
from fakes import RecordingGraphDb

from agentic_kg.common.cypher_identifiers import quote
from agentic_kg.tools import cypher_tools

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
    assert "FOR (n:`Person`)" in query
    assert "REQUIRE n.`id` IS UNIQUE" in query


def test_create_uniqueness_constraint_quotes_a_keyword_label_and_key(fake_db):
    """KG-44: Order and END are ordinary names once quoted."""
    result = cypher_tools.create_uniqueness_constraint("Order", "END")
    assert result["status"] == "success"
    query, _params = fake_db.queries[0]
    assert "FOR (n:`Order`)" in query
    assert "REQUIRE n.`END` IS UNIQUE" in query


def test_the_constraint_statement_carries_no_name(fake_db):
    """KG-52: a given name makes IF NOT EXISTS match on the name, so two pairs
    that join to the same text (A_b + c, A + b_c) shared one and the second
    constraint was skipped silently. With no name it matches on label and key.
    A name creeping back must fail here."""
    cypher_tools.create_uniqueness_constraint("Person", "id")
    query, _params = fake_db.queries[0]
    assert " ".join(query.split()) == (
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:`Person`) REQUIRE n.`id` IS UNIQUE"
    )


def test_create_uniqueness_constraint_rejects_label_injection_payload_before_any_query(
    fake_db,
):
    """create_uniqueness_constraint checks its label with checked(), the same
    character rule kg_construction_tools.py uses, so a newline/paren payload
    is refused before any query is sent."""
    result = cypher_tools.create_uniqueness_constraint(INJECTION_PAYLOAD, "id")
    assert result["status"] == "error"
    assert fake_db.queries == []


FIELD_RULE = "It must be 1 to 16,383 characters of text, with no NUL."
UNICODE_ESCAPE_PAYLOAD = "x\\u0060: 1}) SET n.pwned = true //"


def _only_quoted_names_remain(query, label, key):
    """The query with every quoted name removed: nothing of a payload is left."""
    return query.replace(quote(key), "").replace(quote(label), "")


@pytest.mark.parametrize(
    "key",
    [
        INJECTION_PAYLOAD,
        UNICODE_ESCAPE_PAYLOAD,
        "Order ID",
        "customer-id",
        "Straße",
        "C:\\users",
    ],
    ids=["newline-payload", "escape-payload", "space", "hyphen", "accent", "backslash"],
)
def test_create_uniqueness_constraint_quotes_any_key_text(fake_db, key):
    result = cypher_tools.create_uniqueness_constraint("Person", key)
    assert result["status"] == "success"
    query, _params = fake_db.queries[0]
    assert f"REQUIRE n.{quote(key)} IS UNIQUE" in query
    remaining = _only_quoted_names_remain(query, "Person", key)
    assert " ".join(remaining.split()) == (
        "CREATE CONSTRAINT IF NOT EXISTS FOR (n:) REQUIRE n. IS UNIQUE"
    )


def test_create_uniqueness_constraint_still_refuses_a_label_with_a_space(fake_db):
    result = cypher_tools.create_uniqueness_constraint("Not A Label", "Order ID")
    assert result["status"] == "error"
    assert fake_db.queries == []


@pytest.mark.parametrize(
    "key, shown",
    [("", ""), ("a\x00b", "a\x00b"), (5, "5")],
    ids=["empty", "nul", "non-text"],
)
def test_create_uniqueness_constraint_refuses_an_empty_nul_or_non_text_key(
    fake_db, key, shown
):
    result = cypher_tools.create_uniqueness_constraint("Person", key)
    assert result["status"] == "error"
    assert result["error_message"] == f"Invalid property key: '{shown}'. {FIELD_RULE}"
    assert fake_db.queries == []


FOUND_NONE = {"status": "success", "records": [{"found": 0}]}


def test_a_successful_create_with_a_matching_constraint_returns_the_creates_result(
    fake_db,
):
    result = cypher_tools.create_uniqueness_constraint("Person", "id")
    assert result == {"status": "success", "records": []}


def test_the_check_sends_label_and_key_as_parameters_never_in_the_text(fake_db):
    cypher_tools.create_uniqueness_constraint("Order", "we`ird")
    check, params = fake_db.queries[1]
    assert "SHOW UNIQUENESS CONSTRAINTS" in check
    assert "entityType = 'NODE'" in check
    assert params == {"label": "Order", "key": "we`ird"}
    assert "Order" not in check and "we`ird" not in check
    assert len(fake_db.queries) == 2


def test_a_create_that_leaves_no_matching_constraint_is_an_error(fake_db):
    """KG-52 SC2: the create reported success but the database lists no
    single-key uniqueness constraint on this label and key."""
    fake_db.constraint_listing = FOUND_NONE
    result = cypher_tools.create_uniqueness_constraint("A", "b_c")
    assert result == {
        "status": "error",
        "error_message": (
            "No single-key uniqueness constraint on A/b_c was found after the create."
        ),
    }


@pytest.mark.parametrize(
    "listing",
    [
        {"status": "success", "records": []},
        {"status": "success", "records": [{}]},
        {"status": "success"},
        {"status": "success", "records": [{"found": "1"}]},
    ],
    ids=["no-record", "no-found", "no-records-key", "found-not-a-number"],
)
def test_an_unusable_listing_is_an_error_not_an_exception(fake_db, listing):
    fake_db.constraint_listing = listing
    result = cypher_tools.create_uniqueness_constraint("A", "b_c")
    assert result["status"] == "error"
    assert "A/b_c" in result["error_message"]


def test_a_failing_listing_returns_its_own_error(fake_db):
    fake_db.constraint_listing = {"status": "error", "error_message": "no privilege"}
    result = cypher_tools.create_uniqueness_constraint("A", "b_c")
    assert result == {"status": "error", "error_message": "no privilege"}


def test_a_failed_create_is_returned_before_any_check(fake_db, monkeypatch):
    """Data that already breaks uniqueness still fails the create, and the
    build reports that error, as before."""

    def failing(query, parameters=None):
        fake_db.queries.append((query, parameters or {}))
        return {"status": "error", "error_message": "duplicate values"}

    monkeypatch.setattr(fake_db, "send_query", failing)
    result = cypher_tools.create_uniqueness_constraint("Person", "id")
    assert result == {"status": "error", "error_message": "duplicate values"}
    assert len(fake_db.queries) == 1


def test_the_error_shows_at_most_80_characters_of_a_key(fake_db):
    fake_db.constraint_listing = FOUND_NONE
    key = "k" * 16383
    message = cypher_tools.create_uniqueness_constraint("Person", key)["error_message"]
    assert key not in message
    assert f"Person/{'k' * 80}..." in message


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
        return {
            "node_props": {"A": [{"property": "x", "type": "STRING"}]},
            "rel_props": {},
            "relationships": [],
            "metadata": {},
        }

    monkeypatch.setattr(cypher_tools, "get_structured_schema", fake_structured_schema)
    monkeypatch.setattr(cypher_tools, "graphdb", FakeReadDb())

    result = cypher_tools.get_physical_schema()
    schema = result["schema"]

    assert captured.get("is_enhanced") in (None, False), "is_enhanced must stay off"
    assert "profile" not in schema
    prop = schema["node_props"]["A"][0]
    assert "values" not in prop
    assert "distinct_count" not in prop


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
    monkeypatch,
):
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
        return {
            "node_props": {
                "A": [
                    {
                        "property": "x",
                        "type": "STRING",
                        "values": ["1", "2"],
                        "distinct_count": 9,
                    }
                ]
            },
            "rel_props": {},
            "relationships": [{"start": "A", "type": "R", "end": "A"}],
            "metadata": {"constraint": [], "index": []},
        }

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
    assert fn.__doc__
    assert "partial" not in fn.__doc__.lower()


def _declared_parameter_names(fn) -> set[str]:
    """The parameter names ADK declares to the model for fn.

    google-adk 2.x declares tools as a JSON schema (parameters_json_schema)
    and leaves declaration.parameters None -- the field this lookup used to
    read alone. Reading both keeps it honest whichever one ADK fills.
    """
    from google.adk.tools.function_tool import FunctionTool

    declared = FunctionTool(fn)._get_declaration()
    schema = declared.parameters_json_schema or {}
    props = (declared.parameters.properties or {}) if declared.parameters else {}
    return set(schema.get("properties", {})) | set(props)


def test_the_declaration_lookup_sees_real_parameters():
    """Guards the lookup below against going vacuous, as it was on google-adk
    2.x while it read only declaration.parameters: it must find the
    parameters a real tool does declare."""
    assert {"query", "params"} <= _declared_parameter_names(
        cypher_tools.read_neo4j_cypher
    )


@pytest.mark.parametrize(
    "tool_name",
    [
        "get_physical_schema",
        "get_graph_schema_with_profile",
        "read_neo4j_cypher",
    ],
)
def test_no_tool_exposes_the_profile_flag_to_a_model(tool_name):
    """A model-visible include_data_profile would put the choice in the
    model's hands: google-adk 2.9 declares it optional with its default, but a
    model may still pass True and trigger a full scan per label on the
    latency-tuned construction agent."""
    import inspect

    fn = getattr(cypher_tools, tool_name)
    assert "include_data_profile" not in inspect.signature(fn).parameters
    assert "include_data_profile" not in _declared_parameter_names(fn)


class FakeDdlDb(FakeGraphDb):
    """Records queries and answers the SHOW CONSTRAINTS/INDEXES listings."""

    def __init__(self, constraints=(), indexes=()):
        super().__init__()
        self.constraints = list(constraints)
        self.indexes = list(indexes)

    def send_query(self, query, parameters=None):
        self.queries.append((query, parameters or {}))
        if "SHOW CONSTRAINTS" in query:
            return {
                "status": "success",
                "records": [{"name": n} for n in self.constraints],
            }
        if "SHOW INDEXES" in query:
            return {"status": "success", "records": [{"name": n} for n in self.indexes]}
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
