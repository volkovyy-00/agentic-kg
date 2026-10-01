"""What each end of a relationship rule reads and matches (KG-45).

Neutral domain: an item whose 'supersededBy' column holds another item's
'itemID'.
"""

import pytest

from agentic_kg.tools.relationship_endpoints import (
    Endpoint,
    describe_end,
    is_omitted,
    relationship_endpoints,
)

RULE = {
    "from_node_label": "Item",
    "from_node_column": "itemID",
    "to_node_label": "Item",
    "to_node_column": "supersededBy",
}


@pytest.mark.parametrize("explicit", ["absent", None, ""])
def test_an_omitted_property_resolves_to_the_column(explicit):
    rule = dict(RULE)
    if explicit != "absent":
        rule["to_node_property"] = explicit
    from_end, to_end = relationship_endpoints(rule)
    assert from_end == Endpoint("from", "Item", "itemID", "itemID")
    assert to_end == Endpoint("to", "Item", "supersededBy", "supersededBy")


def test_a_text_property_resolves_to_itself():
    _, to_end = relationship_endpoints({**RULE, "to_node_property": "itemID"})
    assert to_end.column == "supersededBy"
    assert to_end.matched_property == "itemID"


@pytest.mark.parametrize("value", [0, False, [], {}, ["itemID"], " "], ids=repr)
def test_any_other_value_passes_through_unchanged(value):
    """Kept so a caller can refuse it; dropping it would hide the mistake."""
    _, to_end = relationship_endpoints({**RULE, "to_node_property": value})
    assert to_end.matched_property == value
    assert type(to_end.matched_property) is type(value)


def test_missing_keys_read_as_none_and_never_raise():
    from_end, to_end = relationship_endpoints({})
    assert from_end == Endpoint("from", None, None, None)
    assert to_end == Endpoint("to", None, None, None)


@pytest.mark.parametrize(
    "value, omitted",
    [
        (None, True),
        ("", True),
        (0, False),
        (False, False),
        ([], False),
        ({}, False),
        (" ", False),
        ("x", False),
    ],
    ids=repr,
)
def test_only_none_and_empty_text_are_omitted(value, omitted):
    assert is_omitted(value) is omitted


def test_describe_end_names_the_column_only_where_it_differs():
    assert describe_end(Endpoint("from", "Item", "itemID", "itemID")) == "Item.itemID"
    assert (
        describe_end(Endpoint("to", "Item", "supersededBy", "itemID"))
        == "Item.itemID (column 'supersededBy')"
    )
