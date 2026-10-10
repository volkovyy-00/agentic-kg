"""What the database holds, as the clear question and the erase read it.

The fake below behaves like the Neo4j 5.26 facts recorded in the spec: a label
stays listed by db.labels() while a constraint or index refers to it, a
constraint's backing index is listed with owningConstraint set, and the two
LOOKUP indexes are built in. The integration test re-proves them on a real
server.
"""

import pytest

from agentic_kg.tools import cypher_tools


class FakeNeo4j:
    def __init__(
        self, nodes=None, rels=None, unlabelled=0, constraints=(), indexes=(), tokens=()
    ):
        self.nodes = dict(nodes or {})  # label -> count
        self.rels = dict(rels or {})  # type -> count
        self.unlabelled = unlabelled
        self.constraints = list(constraints)
        self.indexes = list(indexes)  # user indexes (not LOOKUP, not owned)
        self.tokens = set(tokens)  # labels still listed with no nodes
        self.queries = []

    def send_query(self, query, parameters=None):
        self.queries.append(query)

        def ok(rows):
            return {"status": "success", "records": rows}

        if query.startswith("MATCH (n) CALL"):
            self.nodes, self.rels, self.unlabelled = {}, {}, 0
            return ok([])
        if query == "MATCH (n) RETURN count(n) AS count":
            return ok([{"count": sum(self.nodes.values()) + self.unlabelled}])
        if query == "MATCH ()-[r]->() RETURN count(r) AS count":
            return ok([{"count": sum(self.rels.values())}])
        if "db.labels()" in query:
            listed = set(self.nodes) | (
                self.tokens if self.constraints or self.indexes else set()
            )
            return ok([{"label": label} for label in sorted(listed)])
        if "db.relationshipTypes()" in query:
            return ok([{"relationshipType": t} for t in sorted(self.rels)])
        if query.startswith("MATCH (n:"):
            label = (
                query[len("MATCH (n:") :].split(")")[0].strip("`").replace("``", "`")
            )
            return ok([{"count": self.nodes.get(label, 0)}])
        if query.startswith("MATCH ()-[r:"):
            rel = (
                query[len("MATCH ()-[r:") :].split("]")[0].strip("`").replace("``", "`")
            )
            return ok([{"count": self.rels.get(rel, 0)}])
        if query.startswith("SHOW CONSTRAINTS"):
            return ok([{"name": n} for n in self.constraints])
        if query == cypher_tools.NON_BUILTIN_INDEXES:
            return ok([{"name": n} for n in self.indexes])
        if query.startswith("DROP CONSTRAINT"):
            self.constraints = [n for n in self.constraints if f"`{n}`" not in query]
            return ok([])
        if query.startswith("DROP INDEX"):
            self.indexes = [n for n in self.indexes if f"`{n}`" not in query]
            return ok([])
        raise AssertionError(f"unexpected query: {query}")


@pytest.fixture
def db(monkeypatch):
    def install(**kwargs):
        fake = FakeNeo4j(**kwargs)
        monkeypatch.setattr(cypher_tools, "graphdb", fake)
        return fake

    return install


def _contents():
    result = cypher_tools.database_contents()
    assert result["status"] == "success", result
    return result["contents"]


def test_a_fresh_database_is_empty(db):
    db()
    contents = _contents()
    assert contents == {
        "nodes": 0,
        "relationships": 0,
        "labels": {},
        "relationship_types": {},
        "constraints": [],
        "indexes": [],
    }
    assert cypher_tools.is_empty(contents)


def test_counts_per_label_and_type(db):
    db(nodes={"Person": 2, "My Label": 1}, rels={"KNOWS": 1})
    contents = _contents()
    assert contents["labels"] == {"Person": 2, "My Label": 1}
    assert contents["relationship_types"] == {"KNOWS": 1}
    assert contents["nodes"] == 3 and contents["relationships"] == 1


def test_a_name_needing_quotes_is_counted(db):
    fake = db(nodes={"a`b": 4})
    assert _contents()["labels"] == {"a`b": 4}
    assert "MATCH (n:`a``b`) RETURN count(n) AS count" in fake.queries


def test_an_unlabelled_node_makes_the_database_non_empty(db):
    db(unlabelled=1)
    contents = _contents()
    assert contents["labels"] == {}
    assert not cypher_tools.is_empty(contents)


def test_a_label_with_no_nodes_is_not_reported(db):
    db(constraints=["c1"], tokens={"Old"})
    contents = _contents()
    assert contents["labels"] == {}
    assert contents["constraints"] == ["c1"]
    assert not cypher_tools.is_empty(contents)


def test_the_index_listing_excludes_lookup_and_constraint_backed_indexes():
    """The whole predicate, not substrings: an OR, or a dropped clause, must fail
    here rather than only in the integration test."""
    assert cypher_tools.NON_BUILTIN_INDEXES == (
        "SHOW INDEXES YIELD name, type, owningConstraint "
        "WHERE type <> 'LOOKUP' AND owningConstraint IS NULL RETURN name"
    )


def test_a_failed_read_returns_the_error(db, monkeypatch):
    fake = db()
    monkeypatch.setattr(
        fake,
        "send_query",
        lambda q, p=None: {"status": "error", "error_message": "down"},
    )
    assert cypher_tools.database_contents() == {
        "status": "error",
        "error_message": "down",
    }


def test_describe_lists_what_is_there():
    text = cypher_tools.describe_contents(
        {
            "nodes": 3,
            "relationships": 1,
            "labels": {"Person": 3},
            "relationship_types": {"KNOWS": 1},
            "constraints": ["c1"],
            "indexes": ["i1"],
        }
    )
    for fragment in ("3 nodes", "Person: 3", "KNOWS: 1", "c1", "i1"):
        assert fragment in text


def test_the_erase_leaves_the_database_empty_and_keeps_lookup_indexes(db):
    fake = db(
        nodes={"Person": 2},
        rels={"KNOWS": 1},
        constraints=["c1"],
        indexes=["i1"],
        tokens={"Person"},
    )
    result = cypher_tools.reset_neo4j_data()
    assert result["status"] == "success", result
    assert cypher_tools.is_empty(_contents())
    drops = [q for q in fake.queries if q.startswith("DROP INDEX")]
    assert drops == ["DROP INDEX `i1`"]


def test_the_erase_reports_what_remains(db, monkeypatch):
    fake = db(constraints=["stuck"])
    original = fake.send_query

    def drop_nothing(query, parameters=None):
        if query.startswith("DROP CONSTRAINT"):
            fake.queries.append(query)
            return {"status": "success", "records": []}
        return original(query, parameters)

    monkeypatch.setattr(fake, "send_query", drop_nothing)
    result = cypher_tools.reset_neo4j_data()
    assert result["status"] == "error"
    assert "stuck" in result["error_message"]
