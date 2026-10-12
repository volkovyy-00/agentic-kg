"""The shared test database cannot carry one test's leftovers into the next.

Every integration test runs against one Neo4j container (tests/integration/conftest.py), so the
step that runs before each test is the only thing between one test's leftovers and the next test's
result. These tests call that step directly rather than relying on test order: a test that only
checked "the database is empty" after an earlier test would pass when run alone, or in reverse,
whether or not the step works.
"""

import pytest
from shared_database import (
    check_container_health,
    fresh_lookup_indexes,
    prepare_for_test,
)

pytestmark = pytest.mark.integration

try:
    import docker

    docker.from_env().ping()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {exc}", allow_module_level=True)


def _count(graphdb, query):
    return len(graphdb.send_query(query)["records"])


def test_preparing_for_a_test_clears_everything_a_test_can_leave_behind(
    shared_container, monkeypatch
):
    from agentic_kg.common import graph_profile
    from agentic_kg.common.config import get_settings

    graphdb = prepare_for_test(monkeypatch, shared_container)
    graphdb.send_query("CREATE (:Leftover {id: 1})-[:LEFT]->(:Leftover {id: 2})")
    graphdb.send_query(
        "CREATE CONSTRAINT leftover_id FOR (n:Leftover) REQUIRE n.id IS UNIQUE"
    )
    graphdb.send_query("CREATE INDEX leftover_name FOR (n:Leftover) ON (n.name)")
    graph_profile._cache["leftover"] = "stale"
    settings_before = get_settings()
    driver_before, _ = graphdb._connection()

    graphdb = prepare_for_test(monkeypatch, shared_container)

    # Checked from outside prepare_for_test, so these hold even if its own emptiness check goes.
    assert _count(graphdb, "MATCH (n) RETURN n") == 0
    assert _count(graphdb, "SHOW CONSTRAINTS") == 0
    assert (
        _count(graphdb, "SHOW INDEXES YIELD type WHERE type <> 'LOOKUP' RETURN type")
        == 0
    )
    assert graph_profile._cache == {}
    assert get_settings() is not settings_before
    # The connection reopens lazily, so it shows only after a query has run.
    driver_after, _ = graphdb._connection()
    assert driver_after is not driver_before


def test_preparing_for_a_test_refuses_to_wipe_any_other_database(
    shared_container, monkeypatch
):
    """The client reads its address from the cached settings. If they were not reset, it would
    reconnect to whatever they last named -- in a real run, .env's database -- and the wipe would
    erase it. Stale settings are made here by caching an address no container listens on."""
    import agentic_kg.common.neo4j_for_adk as neo4j_for_adk
    from agentic_kg.common import config

    monkeypatch.setenv(
        "NEO4J_DSN", "bolt://neo4j:password@not-the-test-database.invalid:7687/neo4j"
    )
    config.reset_settings()
    config.get_settings()
    neo4j_for_adk.close_graphdb()
    monkeypatch.setattr(config, "reset_settings", lambda: None)

    with pytest.raises(
        pytest.fail.Exception, match="Refusing to wipe not-the-test-database.invalid"
    ):
        prepare_for_test(monkeypatch, shared_container)


def test_preparing_for_a_test_fails_with_the_wipes_own_error(
    shared_container, monkeypatch
):
    from agentic_kg.common.tool_result import tool_error
    from agentic_kg.tools import cypher_tools

    monkeypatch.setattr(
        cypher_tools,
        "reset_neo4j_data",
        lambda: tool_error("constraint x would not drop"),
    )

    with pytest.raises(pytest.fail.Exception, match="constraint x would not drop"):
        prepare_for_test(monkeypatch, shared_container)


def test_preparing_for_a_test_fails_when_a_built_in_lookup_index_is_gone(
    shared_container, monkeypatch
):
    """Neo4j's two lookup indexes are not erased by the wipe and not recreated by anything, so a
    wipe that dropped one would leave every later test on a database no fresh one looks like.
    Compared with the names read once, when the container was new."""
    graphdb = prepare_for_test(monkeypatch, shared_container)
    rel_lookup = graphdb.send_query(
        "SHOW INDEXES YIELD name, type, entityType "
        "WHERE type = 'LOOKUP' AND entityType = 'RELATIONSHIP' RETURN name"
    )["records"][0]["name"]
    assert rel_lookup in fresh_lookup_indexes(shared_container)

    graphdb.send_query(f"DROP INDEX `{rel_lookup}`")
    try:
        with pytest.raises(pytest.fail.Exception, match="built-in lookup indexes"):
            prepare_for_test(monkeypatch, shared_container)
    finally:
        graphdb.send_query(
            f"CREATE LOOKUP INDEX `{rel_lookup}` IF NOT EXISTS FOR ()-[r]-() ON EACH type(r)"
        )

    prepare_for_test(monkeypatch, shared_container)


def test_the_shared_database_stops_when_it_runs_out_of_memory(neo4j_graph):
    """A Java heap overflow is not a Docker memory kill: without this flag Neo4j can run on
    after one, and the end-of-run health check would see a running container."""
    records = neo4j_graph.send_query(
        "SHOW SETTINGS YIELD name, value WHERE name = 'server.jvm.additional' RETURN value"
    )["records"]

    assert "-XX:+ExitOnOutOfMemoryError" in records[0]["value"]


class _FakeWrapped:
    def __init__(self, state):
        self.attrs = {"State": state}

    def reload(self):
        pass


class _FakeContainer:
    def __init__(self, state):
        self._wrapped = _FakeWrapped(state)

    def get_wrapped_container(self):
        return self._wrapped


@pytest.mark.parametrize(
    "state, reason",
    [
        ({"Status": "exited", "ExitCode": 3, "OOMKilled": False}, "exited"),
        ({"Status": "running", "ExitCode": 0, "OOMKilled": True}, "out of memory"),
    ],
)
def test_the_health_check_reports_a_database_that_stopped_or_ran_out_of_memory(
    state, reason
):
    with pytest.raises(pytest.fail.Exception, match=reason):
        check_container_health(_FakeContainer(state))


def test_the_health_check_passes_a_running_database():
    check_container_health(
        _FakeContainer({"Status": "running", "ExitCode": 0, "OOMKilled": False})
    )
