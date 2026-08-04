"""Unit tests for the retrieval agent's explicit-handoff gate.

The gate exists because `graphrag_agent_v2` used to decide for itself when the
user was finished asking questions, ejecting them back to the coordinator after
a single answer (`docs/backlog/graphrag-agent-exits-unasked.md`). These tests
cover the mechanism only -- whether the model actually stays and invites the
next question is not unit-testable and is verified by hand (see the plan's
Task 5).
"""
import inspect

from agentic_kg.common.tool_result import is_success
from agentic_kg.tools.graphrag_handoff_tools import (
    GRAPHRAG_HANDOFF_CONFIRMED_KEY,
    confirm_graphrag_handoff,
)


class FakeActions:
    def __init__(self):
        self.escalate = False
        self.transfer_to_agent = None


class FakeToolContext:
    """A tool context carrying BOTH .state and .actions.

    test_adk_tools.py's FakeToolContext has .actions only, and fakes.py
    deliberately does not centralize state-carrying fakes (see its docstring).
    The gated 'finished' reads state and then writes actions, so it needs one
    object with both -- neither existing fake will do.
    """

    def __init__(self, state=None):
        self.state = dict(state or {})
        self.actions = FakeActions()


def test_confirm_records_the_flag():
    """Catches a confirm tool that reports success without writing state,
    which would leave 'finished' refusing every confirmed handoff."""
    context = FakeToolContext()
    confirm_graphrag_handoff(context)
    assert context.state[GRAPHRAG_HANDOFF_CONFIRMED_KEY] is True


def test_confirm_returns_a_success_result():
    """Catches a bare-{} return here: unlike 'finished', this tool is not a
    transfer and follows the normal ToolResult convention, so a model reading
    the result can tell the confirmation was recorded."""
    result = confirm_graphrag_handoff(FakeToolContext())
    assert is_success(result)


def test_confirm_takes_no_arguments_beyond_context():
    """Catches a signature that asks the model to supply an argument. A
    zero-argument tool is categorically more reliable on smaller models --
    the same reasoning make_finished documents for itself."""
    parameters = list(inspect.signature(confirm_graphrag_handoff).parameters)
    assert parameters == ["tool_context"]
