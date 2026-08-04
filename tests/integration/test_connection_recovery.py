"""Every graph tool keeps working after close_graphdb(), without a restart.

Deliberately does NO monkeypatching of any module's `graphdb` attribute. The
defect this guards against is precisely that the module-level bindings go
stale, so a test that substitutes its own object verifies the substitute
instead of the code path production takes. In particular
common/graph_profile.py has its own binding that no other integration test
exercises -- reintroducing the defect there would otherwise be invisible.
"""
import warnings

import pytest

pytestmark = pytest.mark.integration

try:
    import docker
    docker.from_env().ping()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {exc}", allow_module_level=True)


SUPPLIER_RULE = {
    "construction_type": "node",
    "source_file": "suppliers.csv",
    "label": "Supplier",
    "unique_column_name": "supplier_id",
    "properties": ["name", "specialty", "city", "country"],
}

PART_RULE = {
    "construction_type": "node",
    "source_file": "part_supplier_mapping.csv",
    "label": "Part",
    "unique_column_name": "part_id",
    "properties": ["part_name"],
}

SUPPLIED_BY_RULE = {
    "construction_type": "relationship",
    "source_file": "part_supplier_mapping.csv",
    "relationship_type": "SUPPLIED_BY",
    "from_node_label": "Part",
    "from_node_column": "part_id",
    "to_node_label": "Supplier",
    "to_node_column": "supplier_id",
    "properties": ["lead_time_days", "unit_cost"],
}


def test_every_graph_tool_works_after_a_close_and_recover_cycle(neo4j_graph_with_apoc):
    import agentic_kg.common.neo4j_for_adk as neo4j_for_adk
    import agentic_kg.tools.cypher_tools as cypher_tools
    import agentic_kg.tools.kg_construction_tools as kg
    from agentic_kg.common import graph_profile

    # Something to describe, loaded before the break. Counts come from the
    # bundled data: suppliers.csv has 20 data rows, part_supplier_mapping.csv
    # has 88 distinct part_id values.
    assert kg.import_nodes(SUPPLIER_RULE)["status"] == "success"
    assert kg.import_nodes(PART_RULE)["status"] == "success"
    assert kg.import_relationships(SUPPLIED_BY_RULE)["status"] == "success"

    # The profile cache is module-level and outlives a container, so a stale
    # entry from an earlier test could mask a broken profiling path.
    graph_profile.reset_cache()

    # The break: exactly what neo4j_is_ready does on a transient failure.
    neo4j_for_adk.close_graphdb()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        # 1. Schema read -- the get_driver() path.
        schema = cypher_tools.get_physical_schema()
        assert schema["status"] == "success", schema.get("error_message")
        assert "Supplier" in schema["schema"]["node_props"]

        # 2. Profiled schema -- graph_profile's own binding plus cache
        #    invalidation. Real data, not merely absence of error.
        profiled = cypher_tools.get_graph_schema_with_profile()
        assert profiled["status"] == "success", profiled.get("error_message")
        assert profiled["schema"]["profile"]["entity_counts"]["Supplier"] == 20
        assert profiled["schema"]["profile"]["properties"]

        # 3. Ad-hoc read query.
        rows = cypher_tools.read_neo4j_cypher("MATCH (p:Part) RETURN count(p) AS c")
        assert rows["status"] == "success", rows.get("error_message")
        assert rows["query_result"]["records"][0]["c"] == 88

        # 4. Both loaders, re-run after the break.
        assert kg.import_nodes(SUPPLIER_RULE)["status"] == "success"
        assert kg.import_relationships(SUPPLIED_BY_RULE)["status"] == "success"

        rels = neo4j_graph_with_apoc.send_query(
            "MATCH (:Part)-[r:SUPPLIED_BY]->(:Supplier) RETURN count(r) AS c")
        assert rels["records"][0]["c"] > 0

    # neo4j 5.x's Driver tolerates use after close() -- _check_state in
    # neo4j/_sync/driver.py only warns, with a literal "# TODO: 6.0 - raise
    # the error" above it -- so every assertion above passes whether or not a
    # reconnection call site actually ran: a stale, closed Driver silently
    # reopens its own connection pool on the next session/execute_query call.
    # The DeprecationWarning it emits on the way is the only signal, today,
    # that distinguishes "reconnected on purpose" from "reused a closed
    # handle that happened to still work" -- and at the 6.0 bump this becomes
    # a hard error on all four production call paths this test exercises, so
    # asserting its absence now is what actually catches a regressed
    # reconnection call site rather than the driver's own leniency.
    closed_warnings = [w for w in caught if "closed" in str(w.message).lower()]
    assert not closed_warnings, [str(w.message) for w in closed_warnings]


def test_repeated_close_and_recover_cycles_keep_working(neo4j_graph_with_apoc):
    import agentic_kg.common.neo4j_for_adk as neo4j_for_adk
    import agentic_kg.tools.cypher_tools as cypher_tools

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        for _ in range(3):
            neo4j_for_adk.close_graphdb()
            result = cypher_tools.read_neo4j_cypher("RETURN 1 AS ok")
            assert result["status"] == "success", result.get("error_message")
            assert result["query_result"]["records"][0]["ok"] == 1

    # See the comment on the equivalent assertion in
    # test_every_graph_tool_works_after_a_close_and_recover_cycle: neo4j 5.x's
    # Driver tolerates use after close() and only warns, so a passing
    # assertion above proves nothing about whether reconnection actually ran
    # -- only the absence of this warning does.
    closed_warnings = [w for w in caught if "closed" in str(w.message).lower()]
    assert not closed_warnings, [str(w.message) for w in closed_warnings]
