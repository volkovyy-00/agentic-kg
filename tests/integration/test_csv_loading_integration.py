"""Load the bundled BOM CSVs into a real Neo4j and assert the result.

This exercises the same code that runs against Aura. It is representative
precisely because the file:/// path was removed rather than kept alongside —
there is only one loading implementation to test.
"""
import pytest

pytestmark = pytest.mark.integration

try:
    import docker
    docker.from_env().ping()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {exc}", allow_module_level=True)


PLAN = {
    "Product": {
        "construction_type": "node",
        "source_file": "products.csv",
        "label": "Product",
        "unique_column_name": "product_id",
        "properties": ["product_name", "price", "description"],
    },
    "Supplier": {
        "construction_type": "node",
        "source_file": "suppliers.csv",
        "label": "Supplier",
        "unique_column_name": "supplier_id",
        # suppliers.csv columns are: supplier_id,name,specialty,city,country,
        # website,contact_email — note "name", not "supplier_name"
        "properties": ["name", "specialty", "city", "country"],
    },
    "SUPPLIED_BY": {
        "construction_type": "relationship",
        "source_file": "part_supplier_mapping.csv",
        "relationship_type": "SUPPLIED_BY",
        "from_node_label": "Part",
        "from_node_column": "part_id",
        "to_node_label": "Supplier",
        "to_node_column": "supplier_id",
        "properties": ["lead_time_days", "unit_cost"],
    },
    "Part": {
        "construction_type": "node",
        "source_file": "part_supplier_mapping.csv",
        "label": "Part",
        "unique_column_name": "part_id",
        "properties": ["part_name"],
    },
}


@pytest.fixture
def neo4j_graph(monkeypatch):
    from testcontainers.neo4j import Neo4jContainer

    # Pin credentials explicitly rather than letting testcontainers fall back to
    # NEO4J_USER/NEO4J_PASSWORD from the ambient environment. Neo4jContainer's own
    # default is `password or os.environ.get("NEO4J_PASSWORD", "password")`, and this
    # repo's .env sets a real NEO4J_PASSWORD (for the Aura instance) that can leak into
    # os.environ mid-test-session (e.g. a transitively-imported library calling
    # dotenv.load_dotenv() at import time) — so a container left to pick its own default
    # can silently come up with a password other than "password". Pinning here removes
    # that dependency entirely. The DSN below is still built from container.username/
    # container.password rather than written as a literal, so the two can never drift
    # even though we know their values — don't "simplify" this back to a literal DSN.
    with Neo4jContainer(image="neo4j:5", username="neo4j", password="password") as container:
        url = container.get_connection_url()
        host_port = url.split("//")[1]
        monkeypatch.setenv(
            "NEO4J_DSN",
            f"bolt://{container.username}:{container.password}@{host_port}/neo4j",
        )
        monkeypatch.setenv("SOURCE_URI", "./data/bom")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        from agentic_kg.common.config import reset_settings
        import agentic_kg.common.neo4j_for_adk as neo4j_for_adk
        reset_settings()
        neo4j_for_adk.close_graphdb()

        yield neo4j_for_adk.get_graphdb()

        neo4j_for_adk.close_graphdb()


def test_loads_bom_csvs_into_the_graph(neo4j_graph, monkeypatch):
    import agentic_kg.tools.kg_construction_tools as kg
    monkeypatch.setattr(kg, "graphdb", neo4j_graph)
    import agentic_kg.tools.cypher_tools as cypher_tools
    monkeypatch.setattr(cypher_tools, "graphdb", neo4j_graph)

    result = kg.construct_domain_graph(PLAN)
    assert result["status"] == "success", result.get("error_message")

    # Counts come from the bundled data: products.csv has 10 data rows,
    # suppliers.csv has 20, and part_supplier_mapping.csv has 176 rows over
    # 88 distinct part_id values.
    products = neo4j_graph.send_query("MATCH (p:Product) RETURN count(p) AS c")
    assert products["records"][0]["c"] == 10

    suppliers = neo4j_graph.send_query("MATCH (s:Supplier) RETURN count(s) AS c")
    assert suppliers["records"][0]["c"] == 20

    parts = neo4j_graph.send_query("MATCH (p:Part) RETURN count(p) AS c")
    assert parts["records"][0]["c"] == 88

    rels = neo4j_graph.send_query(
        "MATCH (:Part)-[r:SUPPLIED_BY]->(:Supplier) RETURN count(r) AS c")
    assert rels["records"][0]["c"] > 0

    # A property from suppliers.csv must actually have landed
    named = neo4j_graph.send_query(
        "MATCH (s:Supplier) WHERE s.name IS NOT NULL RETURN count(s) AS c")
    assert named["records"][0]["c"] == 20


def test_loading_twice_is_idempotent(neo4j_graph, monkeypatch):
    import agentic_kg.tools.kg_construction_tools as kg
    monkeypatch.setattr(kg, "graphdb", neo4j_graph)
    import agentic_kg.tools.cypher_tools as cypher_tools
    monkeypatch.setattr(cypher_tools, "graphdb", neo4j_graph)

    kg.construct_domain_graph(PLAN)
    first = neo4j_graph.send_query("MATCH (n) RETURN count(n) AS c")["records"][0]["c"]
    kg.construct_domain_graph(PLAN)
    second = neo4j_graph.send_query("MATCH (n) RETURN count(n) AS c")["records"][0]["c"]

    assert first == second, "MERGE should update rather than duplicate"
