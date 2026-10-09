"""A node rule gets its own uniqueness constraint, or the build says it failed (KG-52).

Runs against a real Neo4j. The state a 0.8.x build left is made with raw DDL
under the old names, then the current build runs over it.
"""

import csv

import pytest

pytestmark = pytest.mark.integration

try:
    import docker

    docker.from_env().ping()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {exc}", allow_module_level=True)

# The query create_uniqueness_constraint checks with (cypher_tools), written out
# here on purpose instead of imported, so the test's oracle cannot share a bug with
# the production check. Keep the two in step by hand.
FOUND = (
    "SHOW UNIQUENESS CONSTRAINTS YIELD entityType, labelsOrTypes, properties "
    "WHERE entityType = 'NODE' AND labelsOrTypes = [$label] AND properties = [$key] "
    "RETURN count(*) AS found"
)


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


def _found(db, label, key):
    return db.send_query(FOUND, {"label": label, "key": key})["records"][0]["found"]


def _total(db):
    return len(db.send_query("SHOW CONSTRAINTS")["records"])


def _rule(directory, filename, label, key):
    _write(directory, filename, [key, "note"], [["k1", "a"], ["k2", "b"]])
    return {
        "source_file": filename,
        "label": label,
        "unique_column_name": key,
        "properties": ["note"],
    }


def test_a_graph_built_by_0_8_x_gets_the_constraint_its_name_hid(db, tmp_path):
    """KG-52 SC3. 0.8.x named both constraints of the colliding pair
    A_b_c_constraint, so only the first existed. After a build each pair has
    exactly one constraint and nothing errors; Person is the control; a second
    build adds none."""
    import agentic_kg.tools.kg_construction_tools as kg

    for statement in (
        "CREATE CONSTRAINT `A_b_c_constraint` IF NOT EXISTS "
        "FOR (n:`A_b`) REQUIRE n.`c` IS UNIQUE",
        "CREATE CONSTRAINT `Person_id_constraint` IF NOT EXISTS "
        "FOR (n:`Person`) REQUIRE n.`id` IS UNIQUE",
    ):
        assert db.send_query(statement)["status"] == "success"
    assert _found(db, "A_b", "c") == 1
    assert _found(db, "A", "b_c") == 0, "the start state must be the 0.8.x one"

    rules = [
        _rule(tmp_path, "ab.csv", "A_b", "c"),
        _rule(tmp_path, "a.csv", "A", "b_c"),
        _rule(tmp_path, "person.csv", "Person", "id"),
    ]
    pairs = [("A_b", "c"), ("A", "b_c"), ("Person", "id")]

    for rule in rules:
        built = kg.import_nodes(rule)
        assert built["status"] == "success", built.get("error_message")
    assert [_found(db, *pair) for pair in pairs] == [1, 1, 1]
    assert _total(db) == 3

    for rule in rules:
        again = kg.import_nodes(rule)
        assert again["status"] == "success", again.get("error_message")
    assert [_found(db, *pair) for pair in pairs] == [1, 1, 1]
    assert _total(db) == 3


def test_a_composite_constraint_neither_counts_nor_blocks(db):
    import agentic_kg.tools.cypher_tools as cypher_tools

    made = db.send_query(
        "CREATE CONSTRAINT `comp` FOR (n:`Kg52C`) REQUIRE (n.`x`, n.`y`) IS UNIQUE"
    )
    assert made["status"] == "success"
    assert _found(db, "Kg52C", "x") == 0, "a composite constraint must not count"

    created = cypher_tools.create_uniqueness_constraint("Kg52C", "x")
    assert created["status"] == "success", created.get("error_message")
    assert _found(db, "Kg52C", "x") == 1
    assert _found(db, "Kg52C", "y") == 0
    assert _total(db) == 2


def test_a_hand_made_constraint_under_another_name_is_enough(db):
    import agentic_kg.tools.cypher_tools as cypher_tools

    made = db.send_query(
        "CREATE CONSTRAINT `mine` FOR (n:`Kg52H`) REQUIRE n.`k` IS UNIQUE"
    )
    assert made["status"] == "success"

    created = cypher_tools.create_uniqueness_constraint("Kg52H", "k")
    assert created["status"] == "success", created.get("error_message")
    assert _found(db, "Kg52H", "k") == 1
    assert _total(db) == 1, "no second constraint under a generated name"
