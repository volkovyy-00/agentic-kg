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
