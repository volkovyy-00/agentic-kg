"""Unit tests for the user goal tools.

The goal is the description every later agent reasons from, so an absent
field must not be storable.
"""
from agentic_kg.tools.user_goal_tools import (
    APPROVED_USER_GOAL,
    approve_perceived_user_goal,
    set_perceived_user_goal,
)


class FakeToolContext:
    """Minimal stand-in for ADK's ToolContext — these tools only use .state."""

    def __init__(self):
        self.state = {}


def test_a_goal_with_both_fields_is_stored():
    context = FakeToolContext()
    result = set_perceived_user_goal("supply chain", "Parts and who supplies them.", context)
    assert result["status"] == "success"
    assert context.state["perceived_user_goal"]["kind_of_graph"] == "supply chain"


def test_an_empty_goal_is_refused():
    """Empty strings used to be stored and approved, then quietly shaped file
    selection and schema proposal with nothing in them."""
    context = FakeToolContext()
    result = set_perceived_user_goal("", "", context)
    assert result["status"] == "error"
    for field in ("kind_of_graph", "graph_description"):
        assert field in result["error_message"]
    assert "perceived_user_goal" not in context.state


def test_a_whitespace_only_field_is_refused():
    context = FakeToolContext()
    result = set_perceived_user_goal("supply chain", "   ", context)
    assert result["status"] == "error"
    assert "graph_description" in result["error_message"]
    assert "kind_of_graph" not in result["error_message"]


def test_approval_carries_the_goal_forward():
    context = FakeToolContext()
    set_perceived_user_goal("supply chain", "Parts and suppliers.", context)
    result = approve_perceived_user_goal(context)
    assert result["status"] == "success"
    assert context.state[APPROVED_USER_GOAL]["graph_description"] == "Parts and suppliers."
