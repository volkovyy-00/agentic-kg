"""Unit tests for the retrieval agent's partition-interpretation disclosure gate.

The gate exists because graphrag_agent_v2's instruction told the model to
silently judge whether a partitioned_by numeric property is a quantity or a
set of kinds, and to say which -- nothing enforced the "say" half (KG-5).
These tests cover the mechanism only: that a real declaration ends up in the
tool-call record. Whether the model's final answer also restates it is not
unit-testable and is verified by hand (see the plan's manual verification
task).
"""

from agentic_kg.common.tool_result import is_error, is_success
from agentic_kg.tools.graphrag_partition_tools import (
    NONE_APPLY_SENTINEL,
    PARTITION_INTERPRETATION_DECLARED_KEY,
    declare_partition_interpretation,
)


class FakeToolContext:
    """A tool context carrying .state only -- neither new tool touches .actions."""

    def __init__(self, state=None):
        self.state = dict(state or {})


def _profile(patterns):
    return {"schema": {}, "profile": {"patterns": patterns}}


def _numbers_pattern(property_name="quantity"):
    return {
        "pattern": "Part-[PART_OF]->Assembly",
        "start": "Part",
        "type": "PART_OF",
        "end": "Assembly",
        "partitioned_by": [
            {
                "property": property_name,
                "distribution": [{"value": 1, "count": 5}, {"value": 2, "count": 3}],
                "values_are": "numbers",
                "distribution_covers": "this_pattern",
            }
        ],
    }


def _categories_pattern():
    return {
        "pattern": "Part-[PART_OF]->Assembly",
        "start": "Part",
        "type": "PART_OF",
        "end": "Assembly",
        "partitioned_by": [
            {
                "property": "tier",
                "distribution": [{"value": "a", "count": 5}],
                "values_are": "categories",
                "distribution_covers": "this_pattern",
            }
        ],
    }


def test_declare_with_a_flagged_property_records_the_flag(monkeypatch):
    """Catches a declaration that reports success without writing state,
    which would leave the gated read tool refusing every declared turn."""
    monkeypatch.setattr(
        "agentic_kg.tools.graphrag_partition_tools.peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    context = FakeToolContext()
    result = declare_partition_interpretation(
        "quantity", "treating as a total", context
    )
    assert is_success(result)
    assert context.state[PARTITION_INTERPRETATION_DECLARED_KEY] is True


def test_declare_with_the_none_sentinel_also_records_the_flag(monkeypatch):
    """Catches the 'none apply' escape hatch being rejected -- it would turn
    every false-positive trigger into a dead end instead of a one-line
    reply, since the coarse gate fires whether or not a query actually
    touches the flagged property."""
    monkeypatch.setattr(
        "agentic_kg.tools.graphrag_partition_tools.peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    context = FakeToolContext()
    result = declare_partition_interpretation(
        NONE_APPLY_SENTINEL, "not touching it", context
    )
    assert is_success(result)
    assert context.state[PARTITION_INTERPRETATION_DECLARED_KEY] is True


def test_declare_with_an_unflagged_property_is_refused(monkeypatch):
    """Catches a rubber-stamped declaration: a property the profile does not
    currently flag would open the gate without the model ever having looked
    at what is actually flagged, reopening the disclosure gap this exists
    to close."""
    monkeypatch.setattr(
        "agentic_kg.tools.graphrag_partition_tools.peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    context = FakeToolContext()
    result = declare_partition_interpretation(
        "not_a_real_property", "guessing", context
    )
    assert is_error(result)
    assert PARTITION_INTERPRETATION_DECLARED_KEY not in context.state


def test_declare_refusal_names_the_properties_that_are_flagged(monkeypatch):
    """Catches a generic refusal message that leaves the model guessing what
    a valid argument would have been."""
    monkeypatch.setattr(
        "agentic_kg.tools.graphrag_partition_tools.peek_cached_profile",
        lambda: _profile([_numbers_pattern("quantity")]),
    )
    result = declare_partition_interpretation("bogus", "guessing", FakeToolContext())
    assert "quantity" in result["error_message"]
