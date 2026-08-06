"""Shared fixtures for integration tests.

The container fixture lives here rather than in one test module because two
files need it: the CSV loading tests and the connection-recovery regression
test.
"""
import pytest


def _pinned_container(*, apoc=False):
    """A throwaway Neo4j 5 container with its credentials pinned.

    Pin credentials explicitly rather than letting testcontainers fall back to
    NEO4J_USER/NEO4J_PASSWORD from the ambient environment. Neo4jContainer's own
    default is `password or os.environ.get("NEO4J_PASSWORD", "password")`, and this
    repo's .env sets a real NEO4J_PASSWORD (for the Aura instance) that can leak into
    os.environ mid-test-session (e.g. a transitively-imported library calling
    dotenv.load_dotenv() at import time) -- so a container left to pick its own default
    can silently come up with a password other than "password". Pinning here removes
    that dependency entirely.

    Written once and shared: two copies of this rationale could be edited apart,
    leaving a reader unable to tell whether the difference was meaningful.
    """
    from testcontainers.neo4j import Neo4jContainer

    container = Neo4jContainer(image="neo4j:5", username="neo4j", password="password")
    if apoc:
        # Every neo4j_graphrag.get_structured_schema path is APOC-only (CALL
        # apoc.meta.data / apoc.meta.graph), so any test that reads the physical or
        # profiled schema needs it. See tests/integration/test_graph_profile_shapes.py,
        # which documents and relies on the same constraint with its own container.
        container = container.with_env("NEO4J_PLUGINS", '["apoc"]')
    return container


def _neo4j_graph(monkeypatch, container):
    # No `as`: DockerContainer.__enter__ returns self, so binding it would just
    # shadow the parameter with the identical object.
    with container:
        url = container.get_connection_url()
        host_port = url.split("//")[1]
        # The DSN below is still built from container.username/container.password
        # rather than written as a literal, so the two can never drift even though
        # we know their values -- don't "simplify" this back to a literal DSN.
        monkeypatch.setenv(
            "NEO4J_DSN",
            f"bolt://{container.username}:{container.password}@{host_port}/neo4j",
        )
        monkeypatch.setenv("SOURCE_URI", "./data/bom")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        from agentic_kg.common.config import reset_settings
        import agentic_kg.common.neo4j_for_adk as neo4j_for_adk
        # reset_settings() then close: the singleton reconnects lazily on next
        # use and re-derives its config, picking up the container DSN set above.
        reset_settings()
        neo4j_for_adk.close_graphdb()

        yield neo4j_for_adk.get_graphdb()

        neo4j_for_adk.close_graphdb()


@pytest.fixture
def neo4j_graph(monkeypatch):
    yield from _neo4j_graph(monkeypatch, _pinned_container())


@pytest.fixture
def neo4j_graph_with_apoc(monkeypatch):
    """As `neo4j_graph`, but with APOC installed -- needed by any test that
    reads the physical or profiled schema. See `_pinned_container`."""
    yield from _neo4j_graph(monkeypatch, _pinned_container(apoc=True))
