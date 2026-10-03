"""Any header text works as a key, join column and matched property (KG-51).

Each name goes through the whole path against a real Neo4j: the propose tools
(which read a generated CSV through the real reader), the node build with its
uniqueness constraint, and the relationship build, where a join column is read as a
parameter and a matched property is written as a key inside the MATCH map. A name
either builds or is refused at proposal in the build's own words; none is accepted
at proposal and refused at build.
"""

import csv
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

try:
    import docker

    docker.from_env().ping()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {exc}", allow_module_level=True)

NAMES = {
    "space": "Order ID",
    "hyphen": "customer-id",
    "accent": "Straße",
    "backtick": "we`ird",
    "line break": "line\nbreak",
    "whitespace only": "   ",
    "backslash path": "C:\\users",
    "unicode escape": "a\\u0041b",
    "injection": "x\\u0060: 1}) SET n.pwned = true //",
}


def _write(directory, filename, header, rows):
    with (directory / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def db(neo4j_graph, monkeypatch, tmp_path):
    import agentic_kg.tools.cypher_tools as cypher_tools
    import agentic_kg.tools.kg_construction_tools as kg
    from agentic_kg.common.config import reset_settings

    monkeypatch.setattr(kg, "graphdb", neo4j_graph)
    monkeypatch.setattr(cypher_tools, "graphdb", neo4j_graph)
    monkeypatch.setenv("SOURCE_URI", str(tmp_path))
    reset_settings()
    return neo4j_graph


def test_every_name_goes_through_proposal_and_build(db, tmp_path):
    import agentic_kg.tools.kg_construction_tools as kg
    from agentic_kg.common.cypher_identifiers import quote
    from agentic_kg.tools.construction_plan_tools import (
        PROPOSED_CONSTRUCTION_PLAN,
        propose_node_construction,
        propose_relationship_construction,
    )

    for index, (case, name) in enumerate(NAMES.items()):
        label, rel = f"Node{index}", f"LINKS{index}"
        _write(
            tmp_path, f"nodes{index}.csv", [name, "note"], [["k1", "a"], ["k2", "b"]]
        )
        _write(tmp_path, f"links{index}.csv", [name, "other"], [["k1", "k2"]])
        ctx = SimpleNamespace(state={})

        node = propose_node_construction(
            f"nodes{index}.csv", label, name, ["note"], ctx
        )
        link = propose_relationship_construction(
            f"links{index}.csv",
            rel,
            label,
            name,
            label,
            "other",
            [],
            ctx,
            from_node_property=name,
            to_node_property=name,
        )
        assert node["status"] == "success", (case, node)
        assert link["status"] == "success", (case, link)

        built = kg.construct_domain_graph(ctx.state[PROPOSED_CONSTRUCTION_PLAN])
        assert built["status"] == "success", (case, built)

        nodes = db.send_query(
            f"MATCH (n:{quote(label)}) RETURN n[$key] AS key, keys(n) AS props "
            "ORDER BY key",
            {"key": name},
        )
        assert [r["key"] for r in nodes["records"]] == ["k1", "k2"], case
        assert all(set(r["props"]) == {name, "note"} for r in nodes["records"]), case

        links = db.send_query(
            f"MATCH (:{quote(label)})-[r:{quote(rel)}]->(:{quote(label)}) "
            "RETURN count(r) AS c"
        )
        assert links["records"][0]["c"] == 1, case

        constraint = db.send_query(
            "SHOW CONSTRAINTS YIELD labelsOrTypes, properties "
            "WHERE labelsOrTypes = [$label] RETURN properties",
            {"label": label},
        )
        assert [r["properties"] for r in constraint["records"]] == [[name]], case

    injected = db.send_query("MATCH (n) WHERE n.pwned IS NOT NULL RETURN count(n) AS c")
    assert injected["records"][0]["c"] == 0, "a header ran Cypher"


@pytest.mark.parametrize(
    "name, expected",
    [("k" * 16383, True), ("k" * 16384, False), ("a\x00b", False)],
    ids=["16383-characters", "16384-characters", "nul"],
)
def test_every_name_checked_field_accepts_builds_and_every_refused_build_is_refused_by_it(
    db, name, expected
):
    """checked_field refuses NUL because both Neo4j targets do, and more than
    16,383 characters because Aura does (TokenLengthError, verified 2026-10-03),
    although the Community neo4j:5 container accepts longer names (verified up to
    70,000 on 5.26.31). A name checked_field accepts must therefore build
    everywhere, and a name the container refuses must be refused by checked_field.
    If either fails after a Neo4j upgrade, the rule must follow it."""
    from agentic_kg.common.cypher_identifiers import (
        InvalidIdentifier,
        checked_field,
        quote,
    )

    try:
        checked_field("column name", name)
        accepted = True
    except InvalidIdentifier:
        accepted = False
    assert accepted is expected

    try:
        result = db.send_query(f"MERGE (n:`Kg51Limit` {{ {quote(name)} : 1 }})")
        built = result["status"] == "success"
        assert built or not accepted, result
        # A name over 16,383 characters: no assertion on the container outcome,
        # it builds on Community and is refused on Aura.
    finally:
        db.send_query("MATCH (n:`Kg51Limit`) DETACH DELETE n")


def test_a_16383_character_key_also_works_as_a_constraint_key(db):
    """The constraint's name is `<label>_<key>_constraint`, so it is longer than
    the key and longer than 16,383 characters. Neo4j accepted this on Aura on
    2026-10-03; this pins it on the container, because a build that failed here
    would fail after the proposal accepted the name."""
    import agentic_kg.tools.cypher_tools as cypher_tools

    key = "k" * 16383
    result = cypher_tools.create_uniqueness_constraint("Kg51Limit", key)
    try:
        assert result["status"] == "success", result.get("error_message")
        shown = db.send_query(
            "SHOW CONSTRAINTS YIELD labelsOrTypes, properties "
            "WHERE labelsOrTypes = ['Kg51Limit'] RETURN properties"
        )
        assert [r["properties"] for r in shown["records"]] == [[key]]
    finally:
        names = db.send_query(
            "SHOW CONSTRAINTS YIELD name, labelsOrTypes "
            "WHERE labelsOrTypes = ['Kg51Limit'] RETURN name"
        )
        for row in names["records"]:
            db.send_query(f"DROP CONSTRAINT `{row['name']}` IF EXISTS")


def test_a_refused_name_gives_the_same_text_at_proposal_and_at_build(db):
    import agentic_kg.tools.kg_construction_tools as kg
    from agentic_kg.tools.construction_plan_tools import propose_node_construction

    ctx = SimpleNamespace(state={})
    for name in ("a\x00b", "k" * 16384):
        proposed = propose_node_construction("nodes.csv", "Node", name, [], ctx)
        rule = {
            "construction_type": "node",
            "source_file": "nodes.csv",
            "label": "Node",
            "unique_column_name": name,
            "properties": [],
            "property_types": {},
        }
        assert proposed["status"] == "error"
        assert proposed["error_message"] == kg.import_nodes(rule)["error_message"]
