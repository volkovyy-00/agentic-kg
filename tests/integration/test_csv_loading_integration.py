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

    # A correctly keyed plan must be silent. Both SUPPLIED_BY join columns are
    # their endpoints' unique keys, so matching is capped at one pair per row.
    assert "warnings" not in result, result.get("warnings")

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
    assert rels["records"][0]["c"] == 176, "one per part_supplier_mapping.csv row"

    # KG-44 AC4: nothing was written beyond the labels and type counted above.
    total_nodes = neo4j_graph.send_query("MATCH (n) RETURN count(n) AS c")
    assert total_nodes["records"][0]["c"] == sum(
        r["records"][0]["c"] for r in (products, suppliers, parts)
    )
    total_rels = neo4j_graph.send_query("MATCH ()-[r]->() RETURN count(r) AS c")
    assert total_rels["records"][0]["c"] == rels["records"][0]["c"]

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


# KG-44: every name position holds a Cypher keyword -- labels Order and Match,
# key columns END and null, relationship type SET joining END to null -- plus a
# property column TRUE, which stays an ordinary parameter value.
KEYWORD_PLAN = {
    "Order": {
        "construction_type": "node",
        "source_file": "orders.csv",
        "label": "Order",
        "unique_column_name": "END",
        "properties": ["TRUE"],
    },
    "Match": {
        "construction_type": "node",
        "source_file": "matches.csv",
        "label": "Match",
        "unique_column_name": "null",
        "properties": ["name"],
    },
    "SET": {
        "construction_type": "relationship",
        "source_file": "set.csv",
        "relationship_type": "SET",
        "from_node_label": "Order",
        "from_node_column": "END",
        "to_node_label": "Match",
        "to_node_column": "null",
        "properties": [],
    },
}


@pytest.fixture
def keyword_sources(neo4j_graph, tmp_path, monkeypatch):
    """Point SOURCE_URI at CSVs whose columns are keywords.

    Depends on neo4j_graph so it runs after that fixture sets SOURCE_URI to
    data/bom, and wins.
    """
    (tmp_path / "orders.csv").write_text("END,TRUE\n1,yes\n2,no\n3,yes\n")
    (tmp_path / "matches.csv").write_text("null,name\na,Alpha\nb,Beta\n")
    (tmp_path / "set.csv").write_text("END,null\n1,a\n2,a\n3,b\n1,b\n")
    from agentic_kg.common.config import reset_settings

    monkeypatch.setenv("SOURCE_URI", str(tmp_path))
    reset_settings()
    yield
    # monkeypatch restores the env var, but not the cached settings, which would
    # still point at this deleted tmp_path. reset_settings() only clears the cache.
    reset_settings()


def test_names_that_are_cypher_keywords_build(
    neo4j_graph, keyword_sources, monkeypatch
):
    """KG-44: a plan with an Order node was approved, then the build refused
    Order and dropped its nodes, constraint and relationships. Neo4j accepts
    these names unquoted too, so this proves the quoted queries run; the unit
    tests prove they are quoted."""
    import agentic_kg.tools.cypher_tools as cypher_tools
    import agentic_kg.tools.kg_construction_tools as kg

    monkeypatch.setattr(kg, "graphdb", neo4j_graph)
    monkeypatch.setattr(cypher_tools, "graphdb", neo4j_graph)

    def count(query):
        return neo4j_graph.send_query(query)["records"][0]["c"]

    result = kg.construct_domain_graph(KEYWORD_PLAN)
    assert result["status"] == "success", result.get("error_message")
    assert "warnings" not in result, result.get("warnings")

    assert count("MATCH (n:`Order`) RETURN count(n) AS c") == 3
    assert count("MATCH (n:`Match`) RETURN count(n) AS c") == 2
    assert count("MATCH (:`Order`)-[r:`SET`]->(:`Match`) RETURN count(r) AS c") == 4
    # A keyword property column is an ordinary value.
    assert count("MATCH (n:`Order`) WHERE n.`TRUE` = 'yes' RETURN count(n) AS c") == 2

    constraints = neo4j_graph.send_query(
        "SHOW CONSTRAINTS YIELD type, labelsOrTypes, properties"
    )["records"]
    uniques = {
        (row["labelsOrTypes"][0], row["properties"][0])
        for row in constraints
        if "UNIQUENESS" in row["type"]  # neo4j:5 says UNIQUENESS; tolerate a variant
    }
    assert {("Order", "END"), ("Match", "null")} <= uniques

    # A re-run adds nothing.
    again = kg.construct_domain_graph(KEYWORD_PLAN)
    assert again["status"] == "success", again.get("error_message")
    assert count("MATCH (n) RETURN count(n) AS c") == 5
    assert count("MATCH ()-[r]->() RETURN count(r) AS c") == 4
    assert len(neo4j_graph.send_query("SHOW CONSTRAINTS")["records"]) == len(
        constraints
    )


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
    assert isinstance(price, (int, float))
    assert price == 246

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


# A deliberately miskeyed plan, kept separate from PLAN so the correctly-keyed
# tests above keep their counts and their no-warnings assertion.
OVER_MATCHING_PLAN = {
    "Product": {
        "construction_type": "node",
        "source_file": "products.csv",
        "label": "Product",
        "unique_column_name": "product_id",
        "properties": ["product_name"],
    },
    "Assembly": {
        "construction_type": "node",
        "source_file": "assemblies.csv",
        "label": "Assembly",
        "unique_column_name": "assembly_id",
        # product_id is retained as a property, which is what makes the
        # miskeyed join below match at all.
        "properties": ["assembly_name", "product_id"],
    },
    "ASSEMBLY_OF": {
        "construction_type": "relationship",
        "source_file": "assemblies.csv",
        "relationship_type": "ASSEMBLY_OF",
        # The defect: Assembly is keyed by assembly_id, so joining it on
        # product_id matches every assembly sharing the row's product.
        "from_node_label": "Assembly",
        "from_node_column": "product_id",
        "to_node_label": "Product",
        "to_node_column": "product_id",
        "properties": [],
    },
}


def test_over_matching_join_warns_without_changing_what_is_written(
    neo4j_graph, monkeypatch
):
    """The reported defect, against the real data that produced it.

    64 rows fan out to hundreds of endpoint-match combinations because
    product_id is not Assembly's key. The distinct endpoint pairs still number
    only 64 -- which is the whole trap: MERGE writes one relationship per
    distinct pair, so the graph looks right and the warning is the only signal.
    """
    import agentic_kg.tools.kg_construction_tools as kg

    monkeypatch.setattr(kg, "graphdb", neo4j_graph)
    import agentic_kg.tools.cypher_tools as cypher_tools

    monkeypatch.setattr(cypher_tools, "graphdb", neo4j_graph)

    result = kg.construct_domain_graph(OVER_MATCHING_PLAN)
    assert result["status"] == "success", result.get("error_message")

    assert len(result["warnings"]) == 1, result["warnings"]
    warning = result["warnings"][0]
    assert "assemblies.csv" in warning
    assert "ASSEMBLY_OF matched" in warning
    assert "(Assembly.product_id -> Product.product_id)" in warning

    loaded = result["domain_graph_constructed"]["ASSEMBLY_OF"]["rows_loaded"]
    assert loaded["rows"] == 64
    assert loaded["rows_matched"] > 64, "the fan-out is the whole point"

    # The warning changes nothing about what was written: one relationship per
    # distinct (assembly, product) pair, i.e. one per assembly.
    rels = neo4j_graph.send_query(
        "MATCH (:Assembly)-[r:ASSEMBLY_OF]->(:Product) RETURN count(r) AS c"
    )
    assert rels["records"][0]["c"] == 64
    assert loaded["relationships_in_graph"] == 64
