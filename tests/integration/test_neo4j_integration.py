import pytest

# Mark entire module as integration tests
pytestmark = pytest.mark.integration

# Sanity check: ensure Docker daemon is reachable; otherwise skip at module level
try:
    import docker  # type: ignore

    client = docker.from_env()
    client.ping()
except Exception as e:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {e}", allow_module_level=True)


def test_neo4j_container_and_driver_roundtrip(shared_container):
    """The plain driver reaches the shared container. Opens and closes its own driver, never the
    shared client's."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        shared_container.get_connection_url(),
        auth=(shared_container.username, shared_container.password),
    )
    try:
        with driver.session(database="neo4j") as session:
            result = session.run("RETURN 1 AS ok").single()
            assert result["ok"] == 1
    finally:
        driver.close()
