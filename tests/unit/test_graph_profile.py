"""Unit tests for enriched-schema annotation.

Every case is asserted twice, once for a node property and once for a
relationship property. The node/relationship symmetry is the thing most
likely to be lost during implementation: the two bugs that motivated these
annotations landed on opposite sides of that split, so defining either one
over only its own half would fit the design to the bugs instead of the class.
"""
import pytest

from agentic_kg.common.graph_profile import (
    VALUE_COUNT_MAX_DISTINCT,
    annotate_property,
)

# NOTE ON KIND SYMMETRY.
# An earlier draft parametrized every test below over ["node", "relationship"]
# and then never used the parameter -- the same assertion run twice, which
# proves nothing and reads as coverage. annotate_property takes no kind
# argument at all: the library hands back identical property dicts under
# "node_props" and "rel_props", so kind-symmetry here is structural, not
# behavioural, and cannot be tested by varying an unused variable.
#
# The place kind genuinely changes behaviour is _value_counts, which emits a
# different MATCH clause for each. That is covered in test_graph_profile.py's
# Task 6 section by test_value_counts_query_shape_differs_by_kind.


def test_complete_when_values_match_distinct_count():
    prop = {"property": "flag", "type": "STRING", "values": ["a", "b"], "distinct_count": 2}
    out = annotate_property(prop, entity_count=100)
    assert out["completeness"] == "complete"
    assert out["values"] == ["a", "b"]


def test_partial_when_values_are_truncated():
    prop = {"property": "n", "type": "STRING",
            "values": [str(i) for i in range(10)], "distinct_count": 27}
    out = annotate_property(prop, entity_count=100)
    assert out["completeness"] == "partial"
    assert out["values"] == [str(i) for i in range(10)]


def test_unknown_and_values_suppressed_when_sampled():
    prop = {"property": "n", "type": "STRING", "values": ["7", "9"]}
    out = annotate_property(prop, entity_count=50_000)
    assert out["completeness"] == "unknown"
    assert "values" not in out


def test_numeric_like_is_unknown_when_completeness_is_unknown():
    """Sampled values were discarded as untrustworthy; they cannot then be
    used as evidence that a property is numeric. Prompt rule 5 acts on this."""
    prop = {"property": "n", "type": "STRING", "values": ["7", "9"]}
    out = annotate_property(prop, entity_count=50_000)
    assert out["numeric_like"] == "unknown"


def test_unique_when_distinct_count_equals_entity_count():
    prop = {"property": "id", "type": "STRING", "values": [], "distinct_count": 88}
    out = annotate_property(prop, entity_count=88)
    assert out["uniqueness"] == "unique"


def test_non_unique_when_distinct_count_is_below_entity_count():
    prop = {"property": "label", "type": "STRING", "values": [], "distinct_count": 72}
    out = annotate_property(prop, entity_count=88)
    assert out["uniqueness"] == "non_unique"


def test_uniqueness_unknown_without_distinct_count():
    prop = {"property": "label", "type": "STRING", "values": ["x"]}
    out = annotate_property(prop, entity_count=88)
    assert out["uniqueness"] == "unknown"


def test_uniqueness_unknown_when_entity_count_unavailable():
    prop = {"property": "label", "type": "STRING", "values": [], "distinct_count": 5}
    out = annotate_property(prop, entity_count=None)
    assert out["uniqueness"] == "unknown"


def test_numeric_like_string_is_flagged():
    prop = {"property": "days", "type": "STRING",
            "values": ["8", "12", "30"], "distinct_count": 3}
    out = annotate_property(prop, entity_count=10)
    assert out["numeric_like"] == "yes"


def test_non_numeric_string_is_not_flagged():
    prop = {"property": "city", "type": "STRING",
            "values": ["Berlin", "Lisbon"], "distinct_count": 2}
    out = annotate_property(prop, entity_count=10)
    assert out["numeric_like"] == "no"


def test_numeric_like_no_for_already_numeric_types():
    prop = {"property": "n", "type": "INTEGER", "min": 1, "max": 9}
    out = annotate_property(prop, entity_count=10)
    assert out["numeric_like"] == "no"


def test_every_annotation_uses_the_same_string_vocabulary():
    """Catches a bool/str mix, where a truthy "unknown" reads as consent."""
    out = annotate_property({"property": "x"}, entity_count=None)
    for key in ("completeness", "uniqueness", "numeric_like"):
        assert isinstance(out[key], str), f"{key} must be a string, got {type(out[key])}"
    assert out["numeric_like"] == "unknown"


def test_every_annotation_key_is_always_present():
    """Omission-means-unknown is the regression this design forbids.

    A missing key reads as *safe* to a model, and the entities we cannot
    annotate are exactly the large unfamiliar ones where being wrong is most
    likely -- so silence would make the agent more confident the less it knows.
    """
    sparse = {"property": "mystery"}
    out = annotate_property(sparse, entity_count=None)
    for key in ("completeness", "uniqueness", "numeric_like"):
        assert key in out, f"{key} must always be present, never omitted"


def test_input_is_not_mutated():
    prop = {"property": "flag", "type": "STRING", "values": ["a"], "distinct_count": 1}
    before = dict(prop)
    annotate_property(prop, entity_count=1)
    assert prop == before


def test_value_count_threshold_matches_the_library_limit():
    from neo4j_graphrag.schema import DISTINCT_VALUE_LIMIT
    assert VALUE_COUNT_MAX_DISTINCT == DISTINCT_VALUE_LIMIT


from agentic_kg.common import graph_profile
from agentic_kg.common.graph_profile import build_profile, quote


class FakeGraphDbForProfile:
    """Answers profile queries from a scripted table keyed by substring."""

    def __init__(self, responses=None, fail_on=None):
        self.responses = responses or {}
        self.fail_on = fail_on or ()
        self.queries = []

    def send_read_query(self, query, parameters=None, max_rows=None):
        self.queries.append(query)
        for needle in self.fail_on:
            if needle in query:
                return {"status": "error", "error_message": "boom"}
        for needle, records in self.responses.items():
            if needle in query:
                return {"status": "success",
                        "query_result": {"records": records, "row_count": len(records),
                                         "truncated": False}}
        return {"status": "success",
                "query_result": {"records": [], "row_count": 0, "truncated": False}}


@pytest.fixture
def fake_profile_db(monkeypatch):
    db = FakeGraphDbForProfile()
    monkeypatch.setattr(graph_profile, "graphdb", db)
    return db


SCHEMA = {
    "node_props": {"Alpha": [{"property": "code", "type": "STRING"}]},
    "rel_props": {"LINKS": [{"property": "kind", "type": "STRING",
                             "values": ["x", "y"], "distinct_count": 2}]},
    "relationships": [
        {"start": "Alpha", "type": "LINKS", "end": "Beta"},
        {"start": "Alpha", "type": "LINKS", "end": "Gamma"},
    ],
}


def test_quote_backticks_names_and_escapes_embedded_backticks():
    assert quote("Alpha") == "`Alpha`"
    assert quote("Legal Entity") == "`Legal Entity`"
    assert quote("we`ird") == "`we``ird`"


def test_quote_accepts_names_that_checked_would_reject():
    """checked() is for model-supplied identifiers; these come from the DB."""
    from agentic_kg.common.cypher_identifiers import InvalidIdentifier, checked
    with pytest.raises(InvalidIdentifier):
        checked("label", "Legal Entity")
    assert quote("Legal Entity") == "`Legal Entity`"


def test_degree_is_keyed_per_start_type_end_pattern(fake_profile_db):
    profile = build_profile(SCHEMA)
    keys = {p["pattern"] for p in profile["patterns"]}
    assert keys == {"Alpha-[LINKS]->Beta", "Alpha-[LINKS]->Gamma"}


FAILING_SCHEMA = {
    # Alpha needs a qualifying property (distinct_count <= 10) or no per-entity
    # query is ever issued for it and fail_on has nothing to match -- the
    # earlier version of this test passed vacuously.
    "node_props": {"Alpha": [{"property": "code", "type": "STRING",
                              "values": ["a"], "distinct_count": 2}]},
    "rel_props": {"LINKS": [{"property": "kind", "type": "STRING",
                             "values": ["x", "y"], "distinct_count": 2}]},
    "relationships": [],
}


def test_one_failing_entity_degrades_only_itself(monkeypatch):
    db = FakeGraphDbForProfile(fail_on=("`Alpha`",))
    monkeypatch.setattr(graph_profile, "graphdb", db)
    profile = build_profile(FAILING_SCHEMA)
    assert profile["properties"]["Alpha"] == "profile_error"
    assert profile["properties"]["LINKS"] != "profile_error"


def test_value_counts_only_for_small_distinct_counts(fake_profile_db):
    schema = {
        "node_props": {"Alpha": [
            {"property": "small", "type": "STRING", "values": ["a"], "distinct_count": 2},
            {"property": "big", "type": "STRING", "values": ["a"], "distinct_count": 900},
        ]},
        "rel_props": {}, "relationships": [],
    }
    build_profile(schema)
    counted = [q for q in fake_profile_db.queries if "count(*)" in q and "`small`" in q]
    not_counted = [q for q in fake_profile_db.queries if "count(*)" in q and "`big`" in q]
    assert counted and not not_counted


@pytest.mark.parametrize("is_relationship,expected,forbidden", [
    (False, "MATCH (n:`Thing`)", "-[r:`Thing`]->"),
    (True, "MATCH ()-[r:`Thing`]->()", "MATCH (n:`Thing`)"),
])
def test_value_counts_query_shape_differs_by_kind(
        fake_profile_db, is_relationship, expected, forbidden):
    """The one place node vs relationship genuinely changes the emitted Cypher.

    This is the real content behind the spec's "asserted twice, once per
    property kind" commitment -- annotate_property cannot carry it, because it
    has no kind parameter to vary.
    """
    graph_profile._value_counts("Thing", "flag", is_relationship=is_relationship)
    issued = " ".join(fake_profile_db.queries)
    assert expected in issued
    assert forbidden not in issued


@pytest.mark.parametrize("collection", ["node_props", "rel_props"])
def test_both_property_collections_are_annotated_identically(fake_profile_db, collection):
    """Same property dict, either collection, same annotations."""
    prop = {"property": "flag", "type": "STRING", "values": ["a", "b"], "distinct_count": 2}
    schema = {"node_props": {}, "rel_props": {}, "relationships": []}
    schema[collection] = {"Thing": [dict(prop)]}
    profile = build_profile(schema)
    annotated = profile["properties"]["Thing"][0]
    assert annotated["completeness"] == "complete"
    assert annotated["uniqueness"] in ("unique", "non_unique", "unknown")
    assert "value_counts" in annotated


def test_budget_marks_unprofiled_entities_rather_than_dropping_them(monkeypatch):
    monkeypatch.setattr(graph_profile, "MAX_PROFILED_ENTITIES", 1)
    db = FakeGraphDbForProfile()
    monkeypatch.setattr(graph_profile, "graphdb", db)
    schema = {
        "node_props": {"Alpha": [], "Beta": [], "Gamma": []},
        "rel_props": {}, "relationships": [],
    }
    profile = build_profile(schema)
    # list(), not set(): a profiled entity's value is a (possibly empty) list
    # of annotated properties, which is unhashable, so set() would raise
    # before this assertion ever ran. `in` on a list only needs `==`.
    statuses = list(profile["properties"].values())
    assert "not_profiled" in statuses
    assert profile["budget"]["entities_profiled"] == 1
    assert profile["budget"]["entities_skipped"] == 2


def test_budget_also_caps_the_pattern_loop(monkeypatch):
    """Gating only entities would leave the 2P degree queries unbounded."""
    monkeypatch.setattr(graph_profile, "MAX_PROFILED_PATTERNS", 1)
    db = FakeGraphDbForProfile()
    monkeypatch.setattr(graph_profile, "graphdb", db)
    profile = build_profile(SCHEMA)
    degrees = [p["start_degree"] for p in profile["patterns"]]
    assert "not_profiled" in degrees
    assert profile["budget"]["patterns_profiled"] == 1
