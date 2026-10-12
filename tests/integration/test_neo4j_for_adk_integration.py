import pytest

pytestmark = pytest.mark.integration

# Skip module if Docker isn't available
try:
    import docker  # type: ignore

    _client = docker.from_env()
    _client.ping()
except Exception as e:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {e}", allow_module_level=True)


def test_neo4j_for_adk_roundtrip_with_testcontainers(shared_container):
    """A Neo4jForADK built from an explicit config reaches the shared container. It is its own
    client, not the shared one, so closing it leaves every other test's connection alone."""
    from shared_database import container_dsn

    from agentic_kg.common.neo4j_for_adk import Neo4jForADK
    from agentic_kg.common.pydantic_neo4j import Neo4jConfig

    client = Neo4jForADK(Neo4jConfig(dsn=container_dsn(shared_container)))
    try:
        # Simple read query to validate roundtrip and ADK result envelope
        result = client.send_query("RETURN 1 AS ok")
        assert result["status"] == "success"
        rows = result["records"]
        assert isinstance(rows, list)
        assert len(rows) == 1
        assert rows[0] == {"ok": 1}
    finally:
        client.close()
