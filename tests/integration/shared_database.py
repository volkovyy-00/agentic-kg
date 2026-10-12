"""The one Neo4j container every integration test shares, and the step that resets it before each test.

Not a test module. conftest.py builds its fixtures from it, and test_shared_database_isolation.py
calls prepare_for_test directly, so the fixture and its test run the same code with the same inputs.

The agentic_kg imports sit inside the functions on purpose: pytest imports this module while it
collects tests, unit runs included, and importing cypher_tools builds the Neo4j client from
whatever settings are ambient at that moment.
"""

import pytest

# Every neo4j_graphrag.get_structured_schema path is APOC-only (CALL apoc.meta.data /
# apoc.meta.graph), so any test that reads the physical or profiled schema needs it; the one
# container carries it for every test.
_APOC = '["apoc"]'
# Neo4j's defaults size memory from the host. Capped so one long-lived container stays small; a full
# run used about 1.3 GB with these values (2026-10-12).
_HEAP = "512m"
_PAGE_CACHE = "256m"


def start_container():
    """A Neo4j 5 container with APOC, its memory capped and its credentials pinned.

    Pin credentials explicitly rather than letting testcontainers fall back to
    NEO4J_USER/NEO4J_PASSWORD from the ambient environment. Neo4jContainer's own
    default is `password or os.environ.get("NEO4J_PASSWORD", "password")`, and this
    repo's .env sets a real NEO4J_PASSWORD (for the Aura instance) that can leak into
    os.environ mid-test-session (e.g. a transitively-imported library calling
    dotenv.load_dotenv() at import time) -- so a container left to pick its own default
    can silently come up with a password other than "password".

    ExitOnOutOfMemoryError: a Java heap overflow is not a Docker memory kill, so without it Neo4j
    could run on after one and check_container_health would see a running container.
    """
    from testcontainers.neo4j import Neo4jContainer

    return (
        Neo4jContainer(image="neo4j:5", username="neo4j", password="password")
        .with_env("NEO4J_PLUGINS", _APOC)
        .with_env("NEO4J_server_memory_heap_initial__size", _HEAP)
        .with_env("NEO4J_server_memory_heap_max__size", _HEAP)
        .with_env("NEO4J_server_memory_pagecache_size", _PAGE_CACHE)
        .with_env("NEO4J_server_jvm_additional", "-XX:+ExitOnOutOfMemoryError")
    )


def container_address(container):
    """The (host, port) the container's Bolt port is reachable at."""
    return container.get_container_host_ip(), int(container.get_exposed_port(7687))


def container_dsn(container):
    # Built from container.username/container.password rather than written as a literal, so the
    # two can never drift even though we know their values -- don't "simplify" this to a literal.
    host, port = container_address(container)
    return f"bolt://{container.username}:{container.password}@{host}:{port}/neo4j"


def prepare_for_test(monkeypatch, container):
    """Point the shared client at the container and return it emptied, with fresh caches.

    Order matters: the client re-derives its address from settings when it reopens, so the
    environment is set and the settings reset before it is closed, and the address is checked
    before anything is erased. A failure stops the test rather than letting it run on leftovers.
    """
    import agentic_kg.common.neo4j_for_adk as neo4j_for_adk
    from agentic_kg.common import config, graph_profile
    from agentic_kg.tools import cypher_tools

    monkeypatch.setenv("NEO4J_DSN", container_dsn(container))
    monkeypatch.setenv("SOURCE_URI", "./data/bom")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    config.reset_settings()
    neo4j_for_adk.close_graphdb()
    graph_profile.reset_cache()

    graphdb = neo4j_for_adk.get_graphdb()
    _, connected_to = graphdb._connection()
    if (connected_to.host, connected_to.port) != container_address(container):
        pytest.fail(
            f"Refusing to wipe {connected_to.host}:{connected_to.port}: "
            "it is not the shared test database"
        )

    wiped = cypher_tools.reset_neo4j_data()
    if wiped["status"] != "success":
        pytest.fail(
            f"Could not wipe the shared test database: {wiped['error_message']}"
        )
    _fail_unless_empty(graphdb)
    _fail_unless_lookups_are_fresh(graphdb, container)
    return graphdb


def _fail_unless_empty(graphdb):
    """Checked with its own queries, not cypher_tools.is_empty: an oracle that shares the wipe's
    code could share its bugs."""
    left = {
        "nodes": "MATCH (n) RETURN n LIMIT 1",
        "constraints": "SHOW CONSTRAINTS",
        "indexes": "SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name",
    }
    for what, query in left.items():
        result = graphdb.send_query(query)
        if result["status"] != "success" or result["records"]:
            pytest.fail(
                f"The shared test database still has {what} after the wipe: {result}"
            )


def _fail_unless_lookups_are_fresh(graphdb, container):
    """The wipe keeps Neo4j's built-in lookup indexes, and nothing recreates one it drops, so
    _fail_unless_empty, which skips them, would let every later test run without it."""
    result = graphdb.send_query(
        "SHOW INDEXES YIELD name, type WHERE type = 'LOOKUP' RETURN name"
    )
    now = sorted(r["name"] for r in result.get("records", []))
    if result["status"] != "success" or now != fresh_lookup_indexes(container):
        pytest.fail(
            "The shared test database's built-in lookup indexes changed: "
            f"{now} now, {fresh_lookup_indexes(container)} when it was new"
        )


# Neo4j's built-in lookup indexes as read when each container was new, keyed by container id.
_fresh_lookups: dict[str, list[str]] = {}


def fresh_lookup_indexes(container):
    """The names of Neo4j's built-in lookup indexes on this container when it was new.

    conftest.py reads them right after the container starts, before any test or wipe runs; later
    calls return that first reading. Read with a driver of its own, not the shared client, which
    every test re-points.
    """
    from neo4j import GraphDatabase

    key = container.get_wrapped_container().id
    if key not in _fresh_lookups:
        with GraphDatabase.driver(
            container.get_connection_url(),
            auth=(container.username, container.password),
        ) as driver:
            records = driver.execute_query(
                "SHOW INDEXES YIELD name, type WHERE type = 'LOOKUP' RETURN name"
            ).records
        _fresh_lookups[key] = sorted(r["name"] for r in records)
    return _fresh_lookups[key]


def check_container_health(container):
    """Fail if the container stopped or was killed for memory during the run.

    Must run before the container is stopped. No restart count: the container has no restart
    policy, so it never restarts on its own.
    """
    wrapped = container.get_wrapped_container()
    wrapped.reload()
    state = wrapped.attrs["State"]
    if state.get("OOMKilled"):
        pytest.fail("The shared test database was killed for running out of memory")
    if state.get("Status") != "running":
        pytest.fail(
            f"The shared test database {state.get('Status')} during the run "
            f"(exit code {state.get('ExitCode')})"
        )
