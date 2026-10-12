"""Shared fixtures for integration tests.

One Neo4j container serves the whole run: starting one per test made a full run take about 17
minutes and grow the Docker disk by several GB (KG-62). Isolation comes from prepare_for_test,
which wipes the database and resets the process-wide caches before every test; see
shared_database.py and test_shared_database_isolation.py.

Neither fixture is autouse, so a unit run never starts Docker. A run killed by hand can leave the
container running, because Ryuk is disabled under colima; `docker ps` shows it.
"""

import pytest
from shared_database import (
    check_container_health,
    fresh_lookup_indexes,
    prepare_for_test,
    start_container,
)


@pytest.fixture(scope="session")
def shared_container():
    # No `as`: DockerContainer.__enter__ returns self.
    container = start_container()
    with container:
        fresh_lookup_indexes(container)
        yield container
        check_container_health(container)


@pytest.fixture
def neo4j_graph(shared_container, monkeypatch):
    """The shared client, pointed at the shared container, emptied and with fresh caches."""
    import agentic_kg.common.neo4j_for_adk as neo4j_for_adk

    yield prepare_for_test(monkeypatch, shared_container)

    neo4j_for_adk.close_graphdb()
