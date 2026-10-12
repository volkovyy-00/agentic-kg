"""numeric_like for every property of the bundled BOM graph, pinned to its value before the profile
switched to the converter's numeric rule.

A regression guard only. BOM's numeric columns hold no blanks, brackets or negatives, so the switch must
leave every answer here unchanged, and nothing here can show the fix working --
test_numeric_cast_integration.py does that.

Uses only APIs that existed before the switch (e12cb28), so it can be run there to re-capture BASELINE.
"""

import pytest

pytestmark = pytest.mark.integration

try:
    import docker

    docker.from_env().ping()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {exc}", allow_module_level=True)


# Every column of every CSV in data/bom (the product_reviews folder is unstructured text), untyped. Each
# node rule is keyed on a column that is unique in its file; part_supplier_mapping.csv's other six
# columns ride on SUPPLIED_BY.
BOM_PLAN = {
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
        "properties": [
            "name",
            "specialty",
            "city",
            "country",
            "website",
            "contact_email",
        ],
    },
    "Assembly": {
        "construction_type": "node",
        "source_file": "assemblies.csv",
        "label": "Assembly",
        "unique_column_name": "assembly_id",
        "properties": ["assembly_name", "component_name", "quantity", "product_id"],
    },
    "Part": {
        "construction_type": "node",
        "source_file": "components.csv",
        "label": "Part",
        "unique_column_name": "part_id",
        "properties": ["sub_assembly_name", "part_name", "quantity", "assembly_id"],
    },
    "SUPPLIED_BY": {
        "construction_type": "relationship",
        "source_file": "part_supplier_mapping.csv",
        "relationship_type": "SUPPLIED_BY",
        "from_node_label": "Part",
        "from_node_column": "part_id",
        "to_node_label": "Supplier",
        "to_node_column": "supplier_id",
        "properties": [
            "part_name",
            "supplier_name",
            "lead_time_days",
            "unit_cost",
            "minimum_order_quantity",
            "preferred_supplier",
        ],
    },
}

# Captured on e12cb28, 2026-10-11.
BASELINE = {
    "Assembly": {
        "assembly_id": "no",
        "assembly_name": "no",
        "component_name": "no",
        "product_id": "no",
        "quantity": "yes",
    },
    "Part": {
        "assembly_id": "no",
        "part_id": "no",
        "part_name": "no",
        "quantity": "yes",
        "sub_assembly_name": "no",
    },
    "Product": {
        "description": "no",
        "price": "numeric_after_cleaning",
        "product_id": "no",
        "product_name": "no",
    },
    "SUPPLIED_BY": {
        "lead_time_days": "yes",
        "minimum_order_quantity": "yes",
        "part_name": "no",
        "preferred_supplier": "no",
        "supplier_name": "no",
        "unit_cost": "numeric_after_cleaning",
    },
    "Supplier": {
        "city": "no",
        "contact_email": "no",
        "country": "no",
        "name": "no",
        "specialty": "no",
        "supplier_id": "no",
        "website": "no",
    },
}


def test_every_bom_property_keeps_its_numeric_like_answer(
    neo4j_graph_with_apoc, monkeypatch
):
    import agentic_kg.tools.cypher_tools as cypher_tools
    import agentic_kg.tools.kg_construction_tools as kg
    from agentic_kg.common import graph_profile

    monkeypatch.setattr(kg, "graphdb", neo4j_graph_with_apoc)
    monkeypatch.setattr(cypher_tools, "graphdb", neo4j_graph_with_apoc)
    monkeypatch.setattr(graph_profile, "graphdb", neo4j_graph_with_apoc)
    # The profile cache is process-wide and keyed on node and edge totals, which another test's
    # graph could share. Start and end clean.
    graph_profile.reset_cache()
    try:
        built = kg.construct_domain_graph(BOM_PLAN)
        assert built["status"] == "success", built.get("error_message")

        result = cypher_tools.get_graph_schema_with_profile()
        assert result["status"] == "success", result.get("error_message")
        properties = result["schema"]["profile"]["properties"]
        actual = {
            entity: {prop["property"]: prop["numeric_like"] for prop in props}
            for entity, props in properties.items()
        }
        assert actual == BASELINE
    finally:
        graph_profile.reset_cache()
