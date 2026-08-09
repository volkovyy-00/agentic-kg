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
        "MATCH (:Part)-[r:SUPPLIED_BY]->(:Supplier) RETURN count(r) AS c"
    )
    assert rels["records"][0]["c"] > 0

    # A property from suppliers.csv must actually have landed
    named = neo4j_graph.send_query(
        "MATCH (s:Supplier) WHERE s.name IS NOT NULL RETURN count(s) AS c"
    )
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
    neo4j_graph, monkeypatch
):
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
        yield (
            ["id", "name", "city"],
            [
                {"id": "1", "name": "Ada", "city": "London"},
                {"id": "1", "name": "Ada"},
            ],
        )

    monkeypatch.setattr(kg, "read_csv_batches", two_rows)
    result = kg.load_nodes_from_csv("people.csv", "Person", "id", ["name", "city"])
    assert result["status"] == "success", result.get("error_message")

    kept = neo4j_graph.send_query("MATCH (n:Person {id:'1'}) RETURN n.city AS city")[
        "records"
    ][0]["city"]
    assert kept == "London"


def test_an_empty_cell_is_still_stored_for_a_text_property(neo4j_graph, monkeypatch):
    """Only an absent key is skipped. For an UNTYPED property an empty string is
    a value the CSV actually carried, so it must still reach the graph. A typed
    property is the opposite case -- see the test below."""
    import agentic_kg.tools.kg_construction_tools as kg

    monkeypatch.setattr(kg, "graphdb", neo4j_graph)

    def one_row(relative_path, batch_size=1000):
        yield ["id", "city"], [{"id": "2", "city": ""}]

    monkeypatch.setattr(kg, "read_csv_batches", one_row)
    assert (
        kg.load_nodes_from_csv("people.csv", "Person", "id", ["city"])["status"]
        == "success"
    )

    city = neo4j_graph.send_query("MATCH (n:Person {id:'2'}) RETURN n.city AS city")[
        "records"
    ][0]["city"]
    assert city == ""


def test_an_empty_cell_leaves_a_typed_property_unset(neo4j_graph, monkeypatch):
    """There is no empty number. Storing "" in a property declared float would put
    a string back into exactly the property this ticket exists to type."""
    import agentic_kg.tools.kg_construction_tools as kg

    monkeypatch.setattr(kg, "graphdb", neo4j_graph)

    def one_row(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "e1", "cost": ""}]

    monkeypatch.setattr(kg, "read_csv_batches", one_row)
    assert (
        kg.load_nodes_from_csv("p.csv", "Priced", "id", ["cost"], {"cost": "float"})[
            "status"
        ]
        == "success"
    )

    record = neo4j_graph.send_query("MATCH (n:Priced {id:'e1'}) RETURN n.cost AS cost")[
        "records"
    ][0]
    assert record["cost"] is None


def test_a_ragged_row_does_not_erase_a_typed_property(neo4j_graph, monkeypatch):
    """The typed counterpart of the untyped ragged-row guard. An absent key must
    still leave an earlier row's value alone -- the sentinel exists precisely so
    that clearing does not also cover this case."""
    import agentic_kg.tools.kg_construction_tools as kg

    monkeypatch.setattr(kg, "graphdb", neo4j_graph)

    def two_rows(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "r1", "cost": "10"}, {"id": "r1"}]

    monkeypatch.setattr(kg, "read_csv_batches", two_rows)
    assert (
        kg.load_nodes_from_csv("p.csv", "Ragged", "id", ["cost"], {"cost": "float"})[
            "status"
        ]
        == "success"
    )

    kept = neo4j_graph.send_query("MATCH (n:Ragged {id:'r1'}) RETURN n.cost AS cost")[
        "records"
    ][0]["cost"]
    assert kept == 10.0


def test_an_unreadable_value_clears_a_previously_loaded_one(neo4j_graph, monkeypatch):
    """Leaving the old value behind would produce a property holding numbers on
    most nodes and stale text on a few -- worse than uniform text, because an
    aggregation across the mix misbehaves rather than failing."""
    import agentic_kg.tools.kg_construction_tools as kg

    monkeypatch.setattr(kg, "graphdb", neo4j_graph)

    def good(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "c1", "cost": "10"}]

    def bad(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "c1", "cost": "N/A"}]

    monkeypatch.setattr(kg, "read_csv_batches", good)
    kg.load_nodes_from_csv("p.csv", "Cleared", "id", ["cost"], {"cost": "float"})
    monkeypatch.setattr(kg, "read_csv_batches", bad)
    assert (
        kg.load_nodes_from_csv("p.csv", "Cleared", "id", ["cost"], {"cost": "float"})[
            "status"
        ]
        == "success"
    )

    record = neo4j_graph.send_query(
        "MATCH (n:Cleared {id:'c1'}) RETURN n.cost AS cost"
    )["records"][0]
    assert record["cost"] is None


def test_a_blanked_source_cell_clears_a_previously_loaded_value(
    neo4j_graph, monkeypatch
):
    """Editing a cell to empty means the value is gone; leaving the old number in
    the graph would report data the source no longer has."""
    import agentic_kg.tools.kg_construction_tools as kg

    monkeypatch.setattr(kg, "graphdb", neo4j_graph)

    def good(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "b1", "cost": "10"}]

    def blanked(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "b1", "cost": ""}]

    monkeypatch.setattr(kg, "read_csv_batches", good)
    kg.load_nodes_from_csv("p.csv", "Blanked", "id", ["cost"], {"cost": "float"})
    monkeypatch.setattr(kg, "read_csv_batches", blanked)
    assert (
        kg.load_nodes_from_csv("p.csv", "Blanked", "id", ["cost"], {"cost": "float"})[
            "status"
        ]
        == "success"
    )

    record = neo4j_graph.send_query(
        "MATCH (n:Blanked {id:'b1'}) RETURN n.cost AS cost"
    )["records"][0]
    assert record["cost"] is None


def test_a_re_run_retypes_a_property_that_was_stored_as_a_string(
    neo4j_graph, monkeypatch
):
    """Acceptance criterion 6, directly: a graph built before this change holds
    '$42.73' as a STRING, and a re-run with a corrected plan must leave a real
    FLOAT behind rather than needing a manual rebuild. Asserting on the driver's
    Python type is what catches a coercion path that only runs for newly created
    nodes and no-ops on a MERGE hit."""
    import agentic_kg.tools.kg_construction_tools as kg

    monkeypatch.setattr(kg, "graphdb", neo4j_graph)

    def one_row(relative_path, batch_size=1000):
        yield ["id", "cost"], [{"id": "t1", "cost": "$42.73"}]

    monkeypatch.setattr(kg, "read_csv_batches", one_row)

    # First build: the old, untyped plan.
    kg.load_nodes_from_csv("p.csv", "Retyped", "id", ["cost"])
    before = neo4j_graph.send_query(
        "MATCH (n:Retyped {id:'t1'}) RETURN n.cost AS cost"
    )["records"][0]["cost"]
    assert before == "$42.73"

    # Same source, corrected plan.
    assert (
        kg.load_nodes_from_csv("p.csv", "Retyped", "id", ["cost"], {"cost": "float"})[
            "status"
        ]
        == "success"
    )

    after = neo4j_graph.send_query("MATCH (n:Retyped {id:'t1'}) RETURN n.cost AS cost")[
        "records"
    ][0]["cost"]
    assert isinstance(after, float)
    assert after == 42.73


TYPED_PLAN = {
    "Product": {
        "construction_type": "node",
        "source_file": "products.csv",
        "label": "TypedProduct",
        "unique_column_name": "product_id",
        "properties": ["product_name", "price"],
        "property_types": {"price": "float"},
    },
    "Part": {
        "construction_type": "node",
        "source_file": "part_supplier_mapping.csv",
        "label": "TypedPart",
        "unique_column_name": "part_id",
        "properties": ["part_name"],
        "property_types": {},
    },
    "Supplier": {
        "construction_type": "node",
        "source_file": "suppliers.csv",
        "label": "TypedSupplier",
        "unique_column_name": "supplier_id",
        "properties": ["name"],
        "property_types": {},
    },
    "TYPED_SUPPLIED_BY": {
        "construction_type": "relationship",
        "source_file": "part_supplier_mapping.csv",
        "relationship_type": "TYPED_SUPPLIED_BY",
        "from_node_label": "TypedPart",
        "from_node_column": "part_id",
        "to_node_label": "TypedSupplier",
        "to_node_column": "supplier_id",
        "properties": ["lead_time_days", "unit_cost", "preferred_supplier"],
        "property_types": {
            "lead_time_days": "integer",
            "unit_cost": "float",
            "preferred_supplier": "boolean",
        },
    },
}


def test_typed_bom_graph_answers_numeric_questions_without_casting(
    neo4j_graph, monkeypatch
):
    """Acceptance criteria 1-3 end to end: filter, compare, aggregate and sort on
    the bundled data with no toFloat() and no string cleaning in the query. Before
    this change every one of these either errored or sorted lexicographically."""
    import agentic_kg.tools.kg_construction_tools as kg

    monkeypatch.setattr(kg, "graphdb", neo4j_graph)
    import agentic_kg.tools.cypher_tools as cypher_tools

    monkeypatch.setattr(cypher_tools, "graphdb", neo4j_graph)

    result = kg.construct_domain_graph(TYPED_PLAN)
    assert result["status"] == "success", result.get("error_message")

    # Currency formatting is gone and the value is a number.
    price = neo4j_graph.send_query(
        "MATCH (p:TypedProduct {product_id:'P-1000'}) RETURN p.price AS price"
    )["records"][0]["price"]
    assert isinstance(price, (int, float)) and price == 246

    # Range comparison, no cast.
    quick = neo4j_graph.send_query(
        "MATCH ()-[r:TYPED_SUPPLIED_BY]->() WHERE r.lead_time_days < 10 "
        "RETURN count(r) AS c"
    )["records"][0]["c"]
    assert quick > 0

    # Aggregation, no cast.
    total = neo4j_graph.send_query(
        "MATCH ()-[r:TYPED_SUPPLIED_BY]->() RETURN sum(r.unit_cost) AS total"
    )["records"][0]["total"]
    assert total > 0

    # Numeric order, not lexicographic: '9' must not sort after '30'.
    longest = neo4j_graph.send_query(
        "MATCH ()-[r:TYPED_SUPPLIED_BY]->() RETURN r.lead_time_days AS d "
        "ORDER BY d DESC LIMIT 1"
    )["records"][0]["d"]
    shortest = neo4j_graph.send_query(
        "MATCH ()-[r:TYPED_SUPPLIED_BY]->() RETURN r.lead_time_days AS d "
        "ORDER BY d ASC LIMIT 1"
    )["records"][0]["d"]
    assert longest >= shortest
    assert isinstance(longest, int)

    # The yes/no column is a real boolean.
    preferred = neo4j_graph.send_query(
        "MATCH ()-[r:TYPED_SUPPLIED_BY]->() WHERE r.preferred_supplier = true "
        "RETURN count(r) AS c"
    )["records"][0]["c"]
    assert preferred > 0
