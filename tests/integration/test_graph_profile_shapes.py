# tests/integration/test_graph_profile_shapes.py
"""Runs the profile against graph shapes the demo graph does not have.

The demo graph has one (start, end) pair per relationship type, every label
under 10k, single-label nodes and bare-identifier names. Every one of those is
a property of that dataset, not of graphs in general, and each one hides a
different bug. These tests are the only evidence this work generalises.
"""

import pytest

pytestmark = pytest.mark.integration

try:
    import docker  # type: ignore

    docker.from_env().ping()
except Exception as e:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {e}", allow_module_level=True)


@pytest.fixture(scope="module")
def graphdb_against_container():
    from testcontainers.neo4j import Neo4jContainer

    from agentic_kg.common import graph_profile
    from agentic_kg.common.neo4j_for_adk import Neo4jForADK

    # APOC is mandatory, not optional. Every get_structured_schema path is
    # APOC-only -- NODE_PROPERTIES_QUERY, REL_PROPERTIES_QUERY, REL_QUERY all
    # open with CALL apoc.meta.data and SCHEMA_COUNTS_QUERY uses
    # apoc.meta.graph (neo4j_graphrag/schema.py:30-69). Stock neo4j:5 ships no
    # plugins, so without this every test in this file dies with
    # Neo.ClientError.Procedure.ProcedureNotFound rather than skipping.
    container = Neo4jContainer(image="neo4j:5").with_env("NEO4J_PLUGINS", '["apoc"]')
    with container:
        url = container.get_connection_url()
        # testcontainers 4.12 has no get_auth(); username/password are set in
        # __init__ (testcontainers/neo4j/__init__.py:51-52) and honour
        # NEO4J_USER / NEO4J_PASSWORD, which a hardcoded fallback would ignore.
        auth = (container.username, container.password)

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(url, auth=auth)

        db = Neo4jForADK.__new__(Neo4jForADK)
        db._driver = driver
        db._neo4j_config = type("Cfg", (), {"database": "neo4j"})()
        db.write_count = 0

        original = graph_profile.graphdb
        graph_profile.graphdb = db
        graph_profile.reset_cache()

        # Degrees are deliberately UNEVEN. A fixture where every degree is 1
        # cannot distinguish per-pattern keying from pooled keying, because
        # both produce identical numbers -- it would look like a passing test
        # while proving nothing.
        #
        # Ground truth this builds:
        #   Alpha-[LINKS]->Beta          edges=3  start={a1:2, m1:1}  end={b1:2, b2:1}
        #   Alpha-[LINKS]->Gamma         edges=1  start={a1:1}        end={g1:1}
        #   Alpha-[LINKS]->Legal Entity  edges=1  start={m1:1}        end={le1:1}
        #   Alpha-[FOLLOWS]->Alpha       edges=2  start={a1:1, a2:1}  end={a2:1, m1:1}
        # LINKS totals 5 edges across three Alpha-rooted patterns, so a pooled
        # implementation reports edges=5 where a correct one reports 3.
        #
        # Note m1 carries :Alpha AND :Archived, so the enriched schema also
        # emits Archived-rooted patterns (Archived->Beta, Archived->Legal
        # Entity, Alpha->Archived FOLLOWS). Seven patterns total, not four.
        # That is correct -- a multi-label node genuinely participates in
        # patterns under each label -- and it is why assertions below match on
        # named patterns rather than on list length or position.
        db.send_query("""
            CREATE (a1:Alpha {code: 'a1'})
            CREATE (a2:Alpha {code: 'a2'})
            CREATE (m:Alpha:Archived {code: 'm1'})
            CREATE (b1:Beta {code: 'b1'})
            CREATE (b2:Beta {code: 'b2'})
            CREATE (g1:Gamma {code: 'g1'})
            CREATE (le:`Legal Entity` {code: 'le1'})
            CREATE (a1)-[:LINKS {kind: 'x'}]->(b1)
            CREATE (a1)-[:LINKS {kind: 'x'}]->(b2)
            CREATE (m)-[:LINKS {kind: 'y'}]->(b1)
            CREATE (a1)-[:LINKS {kind: 'y'}]->(g1)
            CREATE (m)-[:LINKS {kind: 'x'}]->(le)
            CREATE (a1)-[:FOLLOWS]->(a2)
            CREATE (a2)-[:FOLLOWS]->(m)
        """)

        yield db, graph_profile

        graph_profile.graphdb = original
        graph_profile.reset_cache()
        driver.close()


def _schema_for(db):
    from neo4j_graphrag.schema import get_structured_schema

    return get_structured_schema(
        db.get_driver(), is_enhanced=True, database="neo4j", sanitize=True
    )


def test_profile_completes_on_all_shapes(graphdb_against_container):
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    assert profile["patterns"], "no patterns profiled"
    for entry in profile["properties"].values():
        assert entry != "profile_error", "an entity failed to profile"


def test_degree_is_reported_per_pattern_not_pooled(graphdb_against_container):
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    patterns = {p["pattern"] for p in profile["patterns"]}
    # LINKS spans Alpha->Beta, Alpha->Gamma and Alpha->Legal Entity. Pooling
    # them under one "LINKS" key is the bug this asserts against.
    links = {p for p in patterns if "[LINKS]" in p}
    assert len(links) >= 3, f"LINKS pooled instead of split per pattern: {patterns}"


def _pattern(profile, key):
    for entry in profile["patterns"]:
        if entry["pattern"] == key:
            return entry
    raise AssertionError(
        f"pattern {key} missing from {[p['pattern'] for p in profile['patterns']]}"
    )


def test_degree_numbers_match_hand_computed_ground_truth(graphdb_against_container):
    """The assertion the spec actually requires: real numbers, not just keys.

    Checking only that distinct pattern keys exist would pass against a
    completely wrong degree calculation. These figures are hand-derived from
    the fixture topology documented in the fixture above.
    """
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))

    beta = _pattern(profile, "Alpha-[LINKS]->Beta")
    # 5 LINKS edges exist in total; a pooled implementation reports 5 here.
    assert beta["edges"] == 3
    assert beta["distinct_start"] == 2  # a1, m1
    assert beta["distinct_end"] == 2  # b1, b2
    assert beta["start_degree"] == {"min": 1, "max": 2, "mean": 1.5}
    assert beta["end_degree"] == {"min": 1, "max": 2, "mean": 1.5}

    gamma = _pattern(profile, "Alpha-[LINKS]->Gamma")
    assert gamma["edges"] == 1
    assert gamma["distinct_start"] == 1
    assert gamma["distinct_end"] == 1
    assert gamma["start_degree"] == {"min": 1, "max": 1, "mean": 1.0}

    legal = _pattern(profile, "Alpha-[LINKS]->Legal Entity")
    assert legal["edges"] == 1
    assert legal["distinct_start"] == 1  # m1 only

    follows = _pattern(profile, "Alpha-[FOLLOWS]->Alpha")
    assert follows["edges"] == 2
    assert follows["distinct_start"] == 2  # a1, a2
    assert follows["distinct_end"] == 2  # a2, m1


def test_fixed_grain_signal_is_trustworthy_per_pattern(graphdb_against_container):
    """min == max means fixed grain. It must hold per pattern, not per type.

    Across all LINKS edges pooled, start degrees are {a1:3, m1:2} -- min != max.
    Per pattern, Alpha->Gamma is a clean 1:1. A pooled implementation loses
    that, which is precisely the grain information the agent needs.
    """
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    gamma = _pattern(profile, "Alpha-[LINKS]->Gamma")
    assert gamma["start_degree"]["min"] == gamma["start_degree"]["max"] == 1


def test_entity_counts_match_ground_truth(graphdb_against_container):
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    counts = profile["entity_counts"]
    assert counts["Alpha"] == 3  # a1, a2, m1 (m1 is multi-label)
    assert counts["Beta"] == 2
    assert counts["Legal Entity"] == 1
    assert counts["LINKS"] == 5
    assert counts["FOLLOWS"] == 2


def test_self_referencing_pattern_is_profiled(graphdb_against_container):
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    follows = [p for p in profile["patterns"] if p["type"] == "FOLLOWS"]
    # Membership, not follows[0]: the multi-label node means FOLLOWS yields
    # both Alpha->Alpha and Alpha->Archived, and apoc.meta.data guarantees no
    # ordering between them.
    assert any(p["start"] == p["end"] == "Alpha" for p in follows)


def test_non_identifier_label_survives_quoting(graphdb_against_container):
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    assert "Legal Entity" in profile["entity_counts"]
    assert profile["properties"].get("Legal Entity") != "profile_error"


def test_annotations_are_always_present(graphdb_against_container):
    db, graph_profile = graphdb_against_container
    profile = graph_profile.build_profile(_schema_for(db))
    for entity, props in profile["properties"].items():
        if not isinstance(props, list):
            continue
        for prop in props:
            for key in ("completeness", "uniqueness", "numeric_like", "value_counts"):
                assert key in prop, f"{entity}.{prop.get('property')} missing {key}"


def _count_profile_queries(db, graph_profile):
    issued = []
    original = db.send_read_query

    def counting(query, parameters=None, max_rows=None):
        issued.append(query)
        return original(query, parameters, max_rows)

    db.send_read_query = counting
    try:
        schema = _schema_for(db)
        profile = graph_profile.build_profile(schema)
    finally:
        db.send_read_query = original
    return issued, schema, profile


def test_profile_query_count_is_exact(graphdb_against_container):
    """Would catch: a stray or duplicated query per entity or pattern.

    An inequality cannot -- `<=` passes just as happily on a profile that
    issues fewer queries than it should, including one whose cap silently
    swallowed work it was supposed to do.
    """
    db, graph_profile = graphdb_against_container
    issued, schema, _ = _count_profile_queries(db, graph_profile)

    p = min(len(schema.get("relationships", [])), graph_profile.MAX_PROFILED_PATTERNS)
    all_props = list(schema.get("node_props", {}).values()) + list(
        schema.get("rel_props", {}).values()
    )
    q = sum(
        1
        for props in all_props
        for prop in props
        if 0
        < (prop.get("distinct_count") or 0)
        <= graph_profile.VALUE_COUNT_MAX_DISTINCT
    )
    # 2 per profiled pattern + 1 per qualifying property + 2 entity counts.
    # The library's own N+M enriched scans happen in _schema_for, above.
    assert len(issued) == 2 * p + q + 2, (
        f"{len(issued)} queries issued; expected exactly {2 * p + q + 2} "
        f"for P={p} Q={q}"
    )


def test_pattern_cap_engages_and_marks_rather_than_drops(
    graphdb_against_container, monkeypatch
):
    """Would catch: a cap that is declared but never applied, and a cap that
    bounds cost by silently dropping patterns from the output.

    The first is invisible to any `<=` assertion. The second is the failure the
    profile exists to prevent, reintroduced by the mechanism meant to bound it:
    a pattern that vanishes has unknowable grain, and nothing says so.
    """
    db, graph_profile = graphdb_against_container

    uncapped, schema, full = _count_profile_queries(db, graph_profile)
    total_patterns = len(schema.get("relationships", []))
    assert total_patterns > 1, "fixture must have enough patterns to cap"

    monkeypatch.setattr(graph_profile, "MAX_PROFILED_PATTERNS", 1)
    capped, _, limited = _count_profile_queries(db, graph_profile)

    # The cap actually reduced work.
    assert len(capped) < len(uncapped)
    assert limited["budget"]["patterns_profiled"] == 1
    assert limited["budget"]["patterns_skipped"] == total_patterns - 1

    # ...and never-omit applies to the pattern list, not just to annotations.
    assert len(limited["patterns"]) == len(full["patterns"]) == total_patterns
    skipped = [p for p in limited["patterns"] if p["start_degree"] == "not_profiled"]
    assert len(skipped) == total_patterns - 1
    for entry in skipped:
        assert entry["end_degree"] == "not_profiled"
        assert entry["pattern"]  # still identifiable, just unprofiled


def test_one_failing_entity_degrades_only_that_entry(graphdb_against_container):
    """Spec assertion: a failure profiling one entity leaves the rest intact.

    Against a real driver rather than a fake, because the unit-level version of
    this test was passing vacuously.
    """
    db, graph_profile = graphdb_against_container
    original = db.send_read_query

    def flaky(query, parameters=None, max_rows=None):
        if "`Legal Entity`" in query and "count(*)" in query:
            return {"status": "error", "error_message": "simulated failure"}
        return original(query, parameters, max_rows)

    db.send_read_query = flaky
    try:
        profile = graph_profile.build_profile(_schema_for(db))
    finally:
        db.send_read_query = original

    assert profile["properties"]["Legal Entity"] == "profile_error"
    assert profile["properties"]["Alpha"] != "profile_error"


def test_result_larger_than_the_cap_reports_a_true_row_count(graphdb_against_container):
    db, _ = graphdb_against_container
    from agentic_kg.common.neo4j_for_adk import MAX_RETURNED_ROWS

    payload = db.send_read_query(
        f"UNWIND range(1, {MAX_RETURNED_ROWS + 20}) AS i RETURN i"
    )["query_result"]
    assert payload["row_count"] == MAX_RETURNED_ROWS + 20
    assert len(payload["records"]) == MAX_RETURNED_ROWS
    assert payload["truncated"] is True


def test_malformed_query_returns_a_structured_error(graphdb_against_container):
    db, _ = graphdb_against_container
    result = db.send_read_query("MATCH ( RETURN")
    assert result["status"] == "error"
    assert result["error_message"]


@pytest.mark.parametrize(
    "write_query",
    [
        "CREATE (n:ShouldNotExist)",
        "MATCH (n:Alpha) SET n.tampered = true",
        "CALL apoc.refactor.mergeNodes([]) YIELD node RETURN node",
    ],
)
def test_writes_are_rejected_by_the_server_on_the_read_path(
    graphdb_against_container, write_query
):
    """The last case is the one is_write_query's regex does not catch."""
    db, _ = graphdb_against_container
    result = db.send_read_query(write_query)
    assert result["status"] == "error", f"{write_query!r} was not rejected"
