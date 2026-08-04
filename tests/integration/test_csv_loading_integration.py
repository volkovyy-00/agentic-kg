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


def test_a_row_without_a_column_does_not_erase_what_an_earlier_row_loaded(
        neo4j_graph, monkeypatch):
    """SET n[k] = null removes a property rather than skipping it.

    read_csv_batches omits the key for a row shorter than the header, so before
    the properties list was filtered, a ragged row -- or a re-run against a file
    that had lost a column -- silently erased values an earlier row had loaded,
    and which value survived depended on row order.
    """
    import agentic_kg.tools.kg_construction_tools as kg
    monkeypatch.setattr(kg, "graphdb", neo4j_graph)

    def two_rows(relative_path, batch_size=1000):
        # Second row is ragged: same entity, "city" absent entirely.
        yield ["id", "name", "city"], [
            {"id": "1", "name": "Ada", "city": "London"},
            {"id": "1", "name": "Ada"},
        ]

    monkeypatch.setattr(kg, "read_csv_batches", two_rows)
    result = kg.load_nodes_from_csv("people.csv", "Person", "id", ["name", "city"])
    assert result["status"] == "success", result.get("error_message")

    kept = neo4j_graph.send_query(
        "MATCH (n:Person {id:'1'}) RETURN n.city AS city")["records"][0]["city"]
    assert kept == "London"


def test_an_empty_cell_is_still_stored(neo4j_graph, monkeypatch):
    """Only an absent key is skipped. An empty string is a value the CSV
    actually carried, so it must still reach the graph."""
    import agentic_kg.tools.kg_construction_tools as kg
    monkeypatch.setattr(kg, "graphdb", neo4j_graph)

    def one_row(relative_path, batch_size=1000):
        yield ["id", "city"], [{"id": "2", "city": ""}]

    monkeypatch.setattr(kg, "read_csv_batches", one_row)
    assert kg.load_nodes_from_csv("people.csv", "Person", "id", ["city"])["status"] == "success"

    city = neo4j_graph.send_query(
        "MATCH (n:Person {id:'2'}) RETURN n.city AS city")["records"][0]["city"]
    assert city == ""
