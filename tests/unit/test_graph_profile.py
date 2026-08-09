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
    prop = {
        "property": "flag",
        "type": "STRING",
        "values": ["a", "b"],
        "distinct_count": 2,
    }
    out = annotate_property(prop, entity_count=100)
    assert out["completeness"] == "complete"
    assert out["values"] == ["a", "b"]


def test_partial_when_values_are_truncated():
    prop = {
        "property": "n",
        "type": "STRING",
        "values": [str(i) for i in range(10)],
        "distinct_count": 27,
    }
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
    prop = {
        "property": "days",
        "type": "STRING",
        "values": ["8", "12", "30"],
        "distinct_count": 3,
    }
    out = annotate_property(prop, entity_count=10)
    assert out["numeric_like"] == "yes"


def test_non_numeric_string_is_not_flagged():
    prop = {
        "property": "city",
        "type": "STRING",
        "values": ["Berlin", "Lisbon"],
        "distinct_count": 2,
    }
    out = annotate_property(prop, entity_count=10)
    assert out["numeric_like"] == "no"


@pytest.mark.parametrize(
    "values",
    [
        ["$42.73", "$1,299.00"],  # currency symbol
        ["1,234", "9,000"],  # thousands separator
        ["8", "$42.73"],  # mixed: one value needs cleaning, so all do
    ],
)
def test_numeric_like_separates_castable_text_from_text_needing_cleaning(values):
    """Would catch: one "yes" covering both bare numerals and currency.

    Prompt rule 6 reads "yes" as "a plain cast works". It does not here:
    toFloat('$42.73') and toFloat('1,234') both return null in Neo4j, and an
    aggregation over nulls reports a confident wrong number rather than failing.
    """
    prop = {
        "property": "unit_cost",
        "type": "STRING",
        "values": values,
        "distinct_count": len(values),
    }
    out = annotate_property(prop, entity_count=10)
    assert out["numeric_like"] == "numeric_after_cleaning"


def test_identifier_shaped_strings_are_still_not_numeric():
    """The currency class stays explicit rather than "any non-digit prefix":
    a wildcard would flag 'Q3'/'#5'/'a1' and invite a cast on a key column."""
    prop = {
        "property": "code",
        "type": "STRING",
        "values": ["Q3", "#5", "a1"],
        "distinct_count": 3,
    }
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
        assert isinstance(out[key], str), (
            f"{key} must be a string, got {type(out[key])}"
        )
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


from fakes import ScriptedGraphDb

from agentic_kg.common import graph_profile
from agentic_kg.common.graph_profile import build_profile, quote

# Shared with the rest of the unit suite; see tests/unit/fakes.py.
FakeGraphDbForProfile = ScriptedGraphDb


@pytest.fixture
def fake_profile_db(monkeypatch):
    db = FakeGraphDbForProfile()
    monkeypatch.setattr(graph_profile, "graphdb", db)
    return db


SCHEMA = {
    "node_props": {"Alpha": [{"property": "code", "type": "STRING"}]},
    "rel_props": {
        "LINKS": [
            {
                "property": "kind",
                "type": "STRING",
                "values": ["x", "y"],
                "distinct_count": 2,
            }
        ]
    },
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
    "node_props": {
        "Alpha": [
            {"property": "code", "type": "STRING", "values": ["a"], "distinct_count": 2}
        ]
    },
    "rel_props": {
        "LINKS": [
            {
                "property": "kind",
                "type": "STRING",
                "values": ["x", "y"],
                "distinct_count": 2,
            }
        ]
    },
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
        "node_props": {
            "Alpha": [
                {
                    "property": "small",
                    "type": "STRING",
                    "values": ["a"],
                    "distinct_count": 2,
                },
                {
                    "property": "big",
                    "type": "STRING",
                    "values": ["a"],
                    "distinct_count": 900,
                },
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    build_profile(schema)
    counted = [q for q in fake_profile_db.queries if "count(*)" in q and "`small`" in q]
    not_counted = [
        q for q in fake_profile_db.queries if "count(*)" in q and "`big`" in q
    ]
    assert counted and not not_counted


@pytest.mark.parametrize(
    "is_relationship,expected,forbidden",
    [
        (False, "MATCH (n:`Thing`)", "-[r:`Thing`]->"),
        (True, "MATCH ()-[r:`Thing`]->()", "MATCH (n:`Thing`)"),
    ],
)
def test_value_counts_query_shape_differs_by_kind(
    fake_profile_db, is_relationship, expected, forbidden
):
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
def test_both_property_collections_are_annotated_identically(
    fake_profile_db, collection
):
    """Same property dict, either collection, same annotations."""
    prop = {
        "property": "flag",
        "type": "STRING",
        "values": ["a", "b"],
        "distinct_count": 2,
    }
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
        "rel_props": {},
        "relationships": [],
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


def test_label_and_relationship_type_sharing_a_name_do_not_overwrite(monkeypatch):
    """Would catch: keying entity counts and profiles on the bare name.

    Neo4j keeps node labels and relationship types in separate namespaces, so
    a graph may hold both a `FOLLOWS` label and a `FOLLOWS` relationship type.
    Keyed on name alone the relationship pass runs second and overwrites the
    label's node count, so annotate_property receives an EDGE count as the
    entity count for a LABEL's properties -- a wrong `uniqueness` verdict that
    prompt rule 4 then acts on -- and one entity's profile vanishes from the
    output entirely while budget.entities_profiled still counts both.
    """
    db = FakeGraphDbForProfile(
        responses={
            "UNWIND labels(n)": [{"name": "FOLLOWS", "n": 7}],
            "type(r) AS name": [{"name": "FOLLOWS", "n": 900}],
        }
    )
    monkeypatch.setattr(graph_profile, "graphdb", db)
    schema = {
        "node_props": {"FOLLOWS": [{"property": "a", "type": "STRING"}]},
        "rel_props": {"FOLLOWS": [{"property": "b", "type": "STRING"}]},
        "relationships": [],
    }
    profile = build_profile(schema)

    counts = profile["entity_counts"]
    assert counts["FOLLOWS (node)"] == 7
    assert counts["FOLLOWS (relationship)"] == 900

    props = profile["properties"]
    assert props["FOLLOWS (node)"][0]["property"] == "a"
    assert props["FOLLOWS (relationship)"][0]["property"] == "b"
    assert profile["budget"]["entities_profiled"] == 2


def test_non_colliding_names_keep_their_bare_keys(fake_profile_db):
    """The disambiguation must not fire on ordinary graphs.

    Every consumer -- and the integration suite's ground-truth assertions --
    indexes entity_counts and properties by the bare name.
    """
    profile = build_profile(SCHEMA)
    assert "Alpha" in profile["properties"]
    assert "LINKS" in profile["properties"]
    assert not [k for k in profile["entity_counts"] if "(" in k]


def test_value_counts_keep_distinct_values_of_different_types_apart(monkeypatch):
    """Would catch: keying the distribution on str(value).

    Neo4j allows heterogeneous types on one property key. With str() keys the
    integer 1 and the string "1" collapse onto a single entry, the second
    silently overwriting the first -- leaving a distribution whose counts do
    not sum to the entity count and whose missing value is invisible.
    """
    # Needle is the quoted property name, which appears only in the
    # value-counts query. "count(*)" would also match the entity-count query,
    # whose rows carry different columns.
    db = FakeGraphDbForProfile(
        responses={
            "`status`": [{"value": 1, "n": 700}, {"value": "1", "n": 300}],
        }
    )
    monkeypatch.setattr(graph_profile, "graphdb", db)
    schema = {
        "node_props": {
            "Thing": [
                {
                    "property": "status",
                    "type": "STRING",
                    "values": ["1"],
                    "distinct_count": 2,
                }
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    counts = build_profile(schema)["properties"]["Thing"][0]["value_counts"]
    assert counts == [{"value": 1, "count": 700}, {"value": "1", "count": 300}]
    assert sum(c["count"] for c in counts) == 1000


def test_value_counts_empty_result_is_an_empty_distribution_not_unknown(
    fake_profile_db,
):
    """A successful query returning no rows means "no non-null values" -- a
    real answer. Reporting it as "unknown" conflates it with "could not
    determine", which is what a FAILED query yields (profile_error)."""
    schema = {
        "node_props": {
            "Thing": [
                {
                    "property": "flag",
                    "type": "STRING",
                    "values": [],
                    "distinct_count": 1,
                }
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    profile = build_profile(schema)
    assert profile["properties"]["Thing"][0]["value_counts"] == []


class FakeSummarisingDb(FakeGraphDbForProfile):
    """Reports values_summarised on the value-counts query, as send_read_query
    does when a property value is a list longer than MAX_INLINE_LIST_LENGTH."""

    def send_read_query(self, query, parameters=None, max_rows=None):
        result = super().send_read_query(query, parameters, max_rows)
        if "`tags`" in query:
            result["query_result"]["values_summarised"] = True
        return result


def test_summarised_property_values_are_declared_in_the_profile(monkeypatch):
    """Would catch: unwrapping send_read_query straight to `records`.

    send_read_query reports values_summarised when it replaces an oversized
    list value with a placeholder string. If the profile discards that flag,
    the model is shown "<list of N str values, omitted>" inside value_counts
    as though it were the real value -- the same silent-omission failure the
    flag exists to signal, reappearing one layer up in the tool graphrag is
    told to call first.
    """
    db = FakeSummarisingDb(
        responses={
            "`tags`": [{"value": "<list of 40 str values, omitted>", "n": 5}],
        }
    )
    monkeypatch.setattr(graph_profile, "graphdb", db)
    schema = {
        "node_props": {
            "Thing": [
                {"property": "tags", "type": "LIST", "values": [], "distinct_count": 1}
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    prop = build_profile(schema)["properties"]["Thing"][0]
    assert prop["value_counts_complete"] == "no"


def test_value_counts_complete_is_yes_when_nothing_was_summarised(fake_profile_db):
    """Negative control: the annotation must not be always-'no'."""
    schema = {
        "node_props": {
            "Thing": [
                {
                    "property": "flag",
                    "type": "STRING",
                    "values": ["a"],
                    "distinct_count": 2,
                }
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    prop = build_profile(schema)["properties"]["Thing"][0]
    assert prop["value_counts_complete"] == "yes"


def test_value_counts_complete_is_unknown_when_counts_were_not_computed(
    fake_profile_db,
):
    """Never silently 'yes' for a property the profile did not count."""
    schema = {
        "node_props": {
            "Thing": [
                {
                    "property": "big",
                    "type": "STRING",
                    "values": ["a"],
                    "distinct_count": 900,
                }
            ]
        },
        "rel_props": {},
        "relationships": [],
    }
    prop = build_profile(schema)["properties"]["Thing"][0]
    assert prop["value_counts"] == "unknown"
    assert prop["value_counts_complete"] == "unknown"


# --- partitioned_by: the partition stated where the aggregation happens ------

PARTITIONED_SCHEMA = {
    "node_props": {},
    "rel_props": {
        "SUPPLIES": [
            {
                "property": "preferred",
                "type": "STRING",
                "values": ["yes", "no"],
                "distinct_count": 2,
            }
        ]
    },
    "relationships": [{"start": "Supplier", "type": "SUPPLIES", "end": "Part"}],
}

PARTITION_ROWS = [{"value": "yes", "n": 88}, {"value": "no", "n": 88}]


def test_pattern_names_the_property_that_divides_its_edges(monkeypatch):
    """Would catch: the distribution living only under properties[type].

    That is where it lived, and the agent read past it -- it computed
    `COUNT(r)` over all 176 SUPPLIES edges when 88 were primary and 88 were
    fallback, and reported a country that supplies nothing as joint top. The
    evidence was in the payload; joining it to the pattern whose degree it
    invalidates was left to the model. Stating it on the pattern is the fix.
    """
    db = FakeGraphDbForProfile(responses={"`preferred`": PARTITION_ROWS})
    monkeypatch.setattr(graph_profile, "graphdb", db)

    pattern = build_profile(PARTITIONED_SCHEMA)["patterns"][0]

    assert pattern["partitioned_by"] == [
        {
            "property": "preferred",
            "distribution": [
                {"value": "yes", "count": 88},
                {"value": "no", "count": 88},
            ],
            "values_are": "categories",
            "distribution_covers": "this_pattern",
        }
    ]


def test_partitioning_costs_no_extra_query(monkeypatch):
    """Both inputs are already in hand; a second value-count pass would double
    the profile's per-property query cost for information already computed."""
    db = FakeGraphDbForProfile(responses={"`preferred`": PARTITION_ROWS})
    monkeypatch.setattr(graph_profile, "graphdb", db)
    build_profile(PARTITIONED_SCHEMA)
    assert len([q for q in db.queries if "`preferred`" in q]) == 1


def test_a_single_valued_property_is_not_a_partition(monkeypatch):
    """Negative control. Every edge sharing one value splits nothing, and
    listing it would train the agent to ignore the field where it matters."""
    db = FakeGraphDbForProfile(responses={"`preferred`": [{"value": "yes", "n": 176}]})
    monkeypatch.setattr(graph_profile, "graphdb", db)
    pattern = build_profile(PARTITIONED_SCHEMA)["patterns"][0]
    assert pattern["partitioned_by"] == []


def test_partitioned_by_is_unknown_when_the_type_was_not_profiled(monkeypatch):
    """ "Nothing partitions these edges" and "we never looked" are different
    claims; an empty list would assert the first while meaning the second."""
    monkeypatch.setattr(graph_profile, "MAX_PROFILED_ENTITIES", 0)
    monkeypatch.setattr(graph_profile, "graphdb", FakeGraphDbForProfile())
    pattern = build_profile(PARTITIONED_SCHEMA)["patterns"][0]
    assert pattern["partitioned_by"] == "unknown"


def test_partitioned_by_survives_the_pattern_budget(monkeypatch):
    """A pattern too far down the list to profile still gets its partition:
    it is derived, not queried, and matters most where degree is unknown."""
    monkeypatch.setattr(graph_profile, "MAX_PROFILED_PATTERNS", 0)
    db = FakeGraphDbForProfile(responses={"`preferred`": PARTITION_ROWS})
    monkeypatch.setattr(graph_profile, "graphdb", db)
    pattern = build_profile(PARTITIONED_SCHEMA)["patterns"][0]
    assert pattern["start_degree"] == "not_profiled"
    assert pattern["partitioned_by"][0]["property"] == "preferred"


def test_partitioned_by_is_present_on_a_relationship_type_with_no_properties(
    fake_profile_db,
):
    """A type absent from rel_props has no properties at all -- an empty
    partition list. Reading `unknown` there would have the agent disclose
    missing information that is not missing; a KeyError would take down the
    whole profile, which is the tool graphrag is told to call first."""
    schema = {
        "node_props": {},
        "rel_props": {},
        "relationships": [{"start": "A", "type": "LINKS", "end": "B"}],
    }
    pattern = build_profile(schema)["patterns"][0]
    assert pattern["partitioned_by"] == []


def test_numeric_valued_properties_are_reported_but_marked_as_numbers(monkeypatch):
    """Would catch BOTH ways of collapsing this judgement into the payload.

    `quantity` taking small integers splits the edges without naming kinds of
    edge, so treating every such property as a kind flag is how a field that
    must be obeyed becomes a field that gets skimmed. But a categorical code
    stored as digits -- `tier` 1/2/3 -- names kinds and is structurally
    identical, so DROPPING numeric-valued properties silently pools the tiers.
    Nothing in the graph tells them apart; the payload reports the shape and
    leaves the reading to the agent.
    """
    schema = {
        "node_props": {},
        "rel_props": {
            "HAS_PART": [
                {
                    "property": "quantity",
                    "type": "STRING",
                    "values": ["1", "2"],
                    "distinct_count": 2,
                }
            ]
        },
        "relationships": [{"start": "Assembly", "type": "HAS_PART", "end": "Part"}],
    }
    db = FakeGraphDbForProfile(
        responses={"`quantity`": [{"value": "1", "n": 66}, {"value": "2", "n": 22}]}
    )
    monkeypatch.setattr(graph_profile, "graphdb", db)
    entry = build_profile(schema)["patterns"][0]["partitioned_by"][0]
    assert entry["property"] == "quantity"
    assert entry["values_are"] == "numbers"


def test_pooled_distribution_says_it_is_pooled(monkeypatch):
    """Would catch: copying a type-wide distribution onto each pattern as if
    it were the pattern's own.

    `_value_counts` counts ()-[r:TYPE]->(), unscoped by end label. Where a type
    spans several label pairs those counts describe no single pattern -- the
    exact pooling error `_pattern_degree` is keyed on triples to avoid. Suppose
    every Supplier-[SUPPLIES]->Part edge is preferred and every
    Supplier-[SUPPLIES]->Service edge is not: unqualified, the profile tells the
    agent each pattern is split 88/88, and prompt rule 2 then makes it filter to
    a kind that yields zero rows.
    """
    schema = {
        "node_props": {},
        "rel_props": {
            "SUPPLIES": [
                {
                    "property": "preferred",
                    "type": "STRING",
                    "values": ["yes", "no"],
                    "distinct_count": 2,
                }
            ]
        },
        "relationships": [
            {"start": "Supplier", "type": "SUPPLIES", "end": "Part"},
            {"start": "Supplier", "type": "SUPPLIES", "end": "Service"},
        ],
    }
    db = FakeGraphDbForProfile(responses={"`preferred`": PARTITION_ROWS})
    monkeypatch.setattr(graph_profile, "graphdb", db)
    patterns = build_profile(schema)["patterns"]
    assert len(patterns) == 2
    for pattern in patterns:
        assert (
            pattern["partitioned_by"][0]["distribution_covers"]
            == "all_patterns_of_this_type"
        )


def test_a_property_the_library_only_sampled_is_reported_as_unknown(monkeypatch):
    """Would catch: dropping un-enumerable properties, yielding an empty list.

    Above EXHAUSTIVE_SEARCH_LIMIT the library samples and emits no
    distinct_count, so `value_counts` is "unknown" and the property is silently
    skipped -- publishing `partitioned_by: []`, an affirmative "nothing divides
    these edges", for exactly the large graphs where a hidden split is most
    likely and most costly. A property KNOWN to have too many distinct values
    is a different case and stays omitted.
    """
    schema = {
        "node_props": {},
        "rel_props": {
            "SUPPLIES": [
                {"property": "sampled", "type": "STRING", "values": ["a", "b"]},
                {
                    "property": "many",
                    "type": "STRING",
                    "values": ["a"],
                    "distinct_count": 900,
                },
            ]
        },
        "relationships": [{"start": "Supplier", "type": "SUPPLIES", "end": "Part"}],
    }
    monkeypatch.setattr(graph_profile, "graphdb", FakeGraphDbForProfile())
    entries = build_profile(schema)["patterns"][0]["partitioned_by"]
    assert [e["property"] for e in entries] == ["sampled"]
    assert entries[0]["distribution"] == "unknown"
    assert entries[0]["values_are"] == "unknown"
