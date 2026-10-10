"""The shared erase leaves a fresh-database state (KG-46 SC6).

Runs against a real Neo4j, which is what proves the facts the unit fakes
assume: the two LOOKUP indexes are the built-in set, a constraint's backing
index is not listed, and db.labels() keeps a label a constraint refers to.
"""

import csv
import logging
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

try:
    import docker

    docker.from_env().ping()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Docker not available/running: {exc}", allow_module_level=True)

logger = logging.getLogger(__name__)

# Written out rather than imported, so the oracle cannot share a bug with
# cypher_tools.NON_BUILTIN_INDEXES.
LOOKUPS = "SHOW INDEXES YIELD name, type WHERE type = 'LOOKUP' RETURN name"


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


def _lookups(db):
    return sorted(r["name"] for r in db.send_query(LOOKUPS)["records"])


def _contents():
    from agentic_kg.tools.cypher_tools import database_contents

    result = database_contents()
    assert result["status"] == "success", result
    return result["contents"]


def test_build_then_clear_leaves_a_fresh_database(db, tmp_path):
    import agentic_kg.tools.kg_construction_tools as kg
    from agentic_kg.tools.construction_plan_tools import (
        PROPOSED_CONSTRUCTION_PLAN,
        propose_node_construction,
        propose_relationship_construction,
    )
    from agentic_kg.tools.cypher_tools import is_empty, reset_neo4j_data

    fresh = _lookups(db)
    logger.warning("fresh database LOOKUP indexes: %d %s", len(fresh), fresh)
    assert fresh
    assert is_empty(_contents())

    _write(tmp_path, "people.csv", ["id", "name"], [["p1", "A"], ["p2", "B"]])
    _write(tmp_path, "knows.csv", ["from", "to"], [["p1", "p2"]])
    ctx = SimpleNamespace(state={})
    assert (
        propose_node_construction("people.csv", "Person", "id", ["name"], ctx)["status"]
        == "success"
    )
    assert (
        propose_relationship_construction(
            "knows.csv",
            "KNOWS",
            "Person",
            "from",
            "Person",
            "to",
            [],
            ctx,
            from_node_property="id",
            to_node_property="id",
        )["status"]
        == "success"
    )
    built = kg.construct_domain_graph(ctx.state[PROPOSED_CONSTRUCTION_PLAN])
    assert built["status"] == "success", built

    contents = _contents()
    assert contents["labels"] == {"Person": 2}
    assert contents["relationship_types"] == {"KNOWS": 1}
    assert len(contents["constraints"]) == 1
    assert contents["indexes"] == [], "a constraint's backing index was listed"

    assert reset_neo4j_data()["status"] == "success"
    after = _contents()
    assert is_empty(after), after
    assert _lookups(db) == fresh

    assert reset_neo4j_data()["status"] == "success"
    assert is_empty(_contents())
    assert _lookups(db) == fresh


def test_an_unlabelled_node_counts_and_is_erased(db):
    from agentic_kg.tools.cypher_tools import is_empty, reset_neo4j_data

    db.send_query("CREATE ()")
    contents = _contents()
    assert contents["nodes"] == 1 and contents["labels"] == {}
    assert not is_empty(contents)
    assert reset_neo4j_data()["status"] == "success"
    assert is_empty(_contents())


def test_a_label_kept_listed_by_a_user_index_is_not_reported(db):
    from agentic_kg.tools.cypher_tools import reset_neo4j_data

    db.send_query("CREATE INDEX FOR (n:Old) ON (n.name)")
    db.send_query("CREATE (:Old {name: 'x'})")
    db.send_query("MATCH (n) DETACH DELETE n")
    contents = _contents()
    assert contents["labels"] == {}, "a label with no nodes was reported"
    assert len(contents["indexes"]) == 1
    assert reset_neo4j_data()["status"] == "success"
