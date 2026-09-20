"""Every graph tool keeps working after close_graphdb(), without a restart.

Deliberately does NO monkeypatching of any module's `graphdb` attribute. The
defect this guards against is precisely that the module-level bindings go
stale, so a test that substitutes its own object verifies the substitute
instead of the code path production takes. In particular
common/graph_profile.py has its own binding that no other integration test
exercises -- reintroducing the defect there would otherwise be invisible.
"""

import logging

import pytest

pytestmark = pytest.mark.integration

# How these tests catch a closed driver being reused: since neo4j 6.0,
# Driver._check_state raises DriverError("Driver closed") on any use after
# close(), and every graph tool turns that into a `status: error` result. So a
# call site that lost its heal fails its tool's success assertion -- but only
# if it is the FIRST call after a close; once any path heals, the rest run on
# the new driver. Hence the close before every step below, and hence a new
# entry point needing its own close-then-call step to be covered at all.
#
# The logger _ensure_connected writes its rebuild line to. The rebuild count
# pins that each heal actually ran, as opposed to the step succeeding some
# other way.
RECONNECT_LOGGER = "agentic_kg.common.neo4j_for_adk"

try:
    import docker

    docker.from_env().ping()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {exc}", allow_module_level=True)


def _rebuild_log_lines(caplog):
    """The rebuild lines _ensure_connected emitted -- one per heal."""
    return [r.getMessage() for r in caplog.records if "rebuilding" in r.getMessage()]


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


def test_every_graph_tool_works_after_a_close_and_recover_cycle(
    neo4j_graph_with_apoc, caplog
):
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

    # Each step below opens with a break -- exactly what neo4j_is_ready does on
    # a transient failure. See RECONNECT_LOGGER for why every step needs its own.
    #
    # The exception is a heal that still does not run first within its own step.
    # get_config() is the case: _physical_schema calls get_driver() one line
    # earlier, which clears _closed, so dropping get_config's heal passes every
    # test here. tests/unit/test_neo4j_for_adk.py covers it instead -- and it
    # needs covering, since a stale config returns success-shaped data rather
    # than failing (see get_config's docstring).
    with caplog.at_level(logging.INFO, logger=RECONNECT_LOGGER):
        # 1. Schema read -- the get_driver() path.
        neo4j_for_adk.close_graphdb()
        schema = cypher_tools.get_physical_schema()
        assert schema["status"] == "success", schema.get("error_message")
        assert "Supplier" in schema["schema"]["node_props"]

        # 2. Profiled schema -- the profiling path end to end, including
        #    graph_profile's own binding and cache invalidation. Real data, not
        #    merely absence of error. Like get_config above, that binding is
        #    not what meets the closed driver here (get_driver runs first, and
        #    graph_profile reaches the database through send_read_query, which
        #    step 3 covers); what this step pins is that the path still works
        #    across a break, which is what the module docstring is about.
        neo4j_for_adk.close_graphdb()
        profiled = cypher_tools.get_graph_schema_with_profile()
        assert profiled["status"] == "success", profiled.get("error_message")
        assert profiled["schema"]["profile"]["entity_counts"]["Supplier"] == 20
        assert profiled["schema"]["profile"]["properties"]

        # 3. Ad-hoc read query -- the send_read_query path.
        neo4j_for_adk.close_graphdb()
        rows = cypher_tools.read_neo4j_cypher("MATCH (p:Part) RETURN count(p) AS c")
        assert rows["status"] == "success", rows.get("error_message")
        assert rows["query_result"]["records"][0]["c"] == 88

        # 4. Both loaders -- the send_query path. import_nodes opens with
        #    create_uniqueness_constraint, import_relationships with its first
        #    UNWIND batch; either way a write is the first call after the break.
        neo4j_for_adk.close_graphdb()
        assert kg.import_nodes(SUPPLIER_RULE)["status"] == "success"
        neo4j_for_adk.close_graphdb()
        assert kg.import_relationships(SUPPLIED_BY_RULE)["status"] == "success"

        rels = neo4j_graph_with_apoc.send_query(
            "MATCH (:Part)-[r:SUPPLIED_BY]->(:Supplier) RETURN count(r) AS c"
        )
        assert rels["records"][0]["c"] > 0

    # One heal per break, five breaks: each close was followed by a rebuild,
    # not merely by a step that happened to succeed.
    assert len(_rebuild_log_lines(caplog)) == 5, _rebuild_log_lines(caplog)


def test_repeated_close_and_recover_cycles_keep_working(neo4j_graph, caplog):
    """Uses the plain fixture, not the APOC one its sibling needs: this test
    only runs `RETURN 1`, so paying for a plugin install would add container
    boot time to every suite run for nothing."""
    import agentic_kg.common.neo4j_for_adk as neo4j_for_adk
    import agentic_kg.tools.cypher_tools as cypher_tools

    with caplog.at_level(logging.INFO, logger=RECONNECT_LOGGER):
        for _ in range(3):
            neo4j_for_adk.close_graphdb()
            result = cypher_tools.read_neo4j_cypher("RETURN 1 AS ok")
            assert result["status"] == "success", result.get("error_message")
            assert result["query_result"]["records"][0]["ok"] == 1

    # One heal per cycle, three cycles: proves each close was actually followed
    # by a rebuild, not just that the first one was.
    assert len(_rebuild_log_lines(caplog)) == 3, _rebuild_log_lines(caplog)
