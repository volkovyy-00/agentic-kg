"""Unit tests for the retrieval agent's explicit-handoff gate.

The gate exists because `graphrag_agent_v2` used to decide for itself when the
user was finished asking questions, ejecting them back to the coordinator after
a single answer (`docs/backlog/graphrag-agent-exits-unasked.md`). These tests
cover the mechanism only -- whether the model actually stays and invites the
next question is not unit-testable and is verified by hand (see the plan's
Task 5).
"""
import inspect

from agentic_kg.common.tool_result import is_error, is_success
from agentic_kg.tools.graphrag_handoff_tools import (
    GRAPHRAG_HANDOFF_CONFIRMED_KEY,
    confirm_graphrag_handoff,
)
from agentic_kg.common.agent_names import MULTI_AGENT_COORDINATOR
from agentic_kg.coordinators.multi_agent.sub_agents.graphrag_agent.variants import (
    _transfer_to_coordinator,
    finished,
    variants,
)
from agentic_kg.coordinators.multi_agent.sub_agents.graphrag_agent.agent import (
    graphrag_agent,
    reset_graphrag_handoff_confirmation,
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


def test_finished_refuses_without_confirmation():
    """Catches the gate being absent or inverted: an unconfirmed 'finished'
    must neither report success nor set a transfer, or the phase ends on the
    model's inference exactly as it did before."""
    context = FakeToolContext()
    result = finished(context)
    assert is_error(result)
    assert context.actions.transfer_to_agent is None


def test_refusal_message_scopes_the_failure_to_this_turn():
    """Catches a generic 'call confirm first' message. The likely cause of a
    refusal is a confirmation from an earlier turn cleared by the per-turn
    reset, so a message implying the user never agreed would relocate this
    ticket's ambiguity into the error string."""
    result = finished(FakeToolContext())
    assert "this turn" in result["error_message"]


def test_finished_transfers_to_the_coordinator_when_confirmed():
    """Catches a gate that refuses even when confirmed, or a transfer target
    changed while copying the construction gate -- whose wrapper deliberately
    transfers sideways to a sibling instead of up to the coordinator."""
    context = FakeToolContext({GRAPHRAG_HANDOFF_CONFIRMED_KEY: True})
    result = finished(context)
    assert result == {}
    assert context.actions.transfer_to_agent == MULTI_AGENT_COORDINATOR


def test_confirm_and_gated_finished_are_wired_into_v2():
    """Catches an instruction that tells the model to call
    'confirm_graphrag_handoff' when the tool was never added to the variant's
    tools list -- the model would then be unable to end the phase at all,
    since 'finished' refuses without it."""
    tools = variants["graphrag_agent_v2"]["tools"]
    assert confirm_graphrag_handoff in tools
    assert finished in tools


def test_v1_keeps_the_ungated_exit():
    """Catches the two variants sharing one 'finished' object again. Gating
    the module-level closure in place would leave v1 holding a gate with no
    confirm tool beside it -- unable to end its phase at all -- and would stay
    invisible until someone flips AGENT_NAME to _v1. Identity, not name: a
    same-named-but-gated function must fail this."""
    tools = variants["graphrag_agent_v1"]["tools"]
    assert any(tool is _transfer_to_coordinator for tool in tools)
    assert all(tool is not finished for tool in tools)


def test_v1_exit_still_presents_to_the_model_as_finished():
    """Catches a refactor of make_finished to a lambda or functools.partial.
    ADK derives the model-facing tool name from __name__, so such a change
    would silently rename v1's exit tool and falsify v1's own instruction line
    ('finished: signal that the user is done with the graphrag agent') without
    failing any other test. No other agent lists a renamed make_finished
    result as a tool, so nothing else covers this."""
    assert _transfer_to_coordinator.__name__ == "finished"


class FakeCallbackContext:
    """A callback context carrying state only -- callbacks never touch .actions."""

    def __init__(self, state=None):
        self.state = dict(state or {})


def test_reset_clears_a_previous_confirmation():
    """Catches a reset that only initialises a missing key. A confirmation from
    an earlier turn would otherwise let 'finished' transfer on the model's
    judgment alone -- the exact defect this ticket is about."""
    context = FakeCallbackContext({GRAPHRAG_HANDOFF_CONFIRMED_KEY: True})
    reset_graphrag_handoff_confirmation(context)
    assert context.state[GRAPHRAG_HANDOFF_CONFIRMED_KEY] is False


def test_reset_is_wired_onto_the_graphrag_agent():
    """Catches the callback being defined but never attached, which leaves the
    flag sticky for the whole session and silently disables the gate after the
    first successful handoff. Also catches an AGENT_NAME/condition mismatch
    that leaves the shipped v2 agent unwired."""
    callbacks = graphrag_agent.canonical_before_agent_callbacks
    assert reset_graphrag_handoff_confirmation in callbacks


def test_reset_parameter_is_named_callback_context():
    """Catches a rename. ADK invokes these callbacks by keyword
    (base_agent.py:385-387), so a different parameter name fails at request
    time with a TypeError rather than at import."""
    parameters = list(
        inspect.signature(reset_graphrag_handoff_confirmation).parameters
    )
    assert parameters == ["callback_context"]
