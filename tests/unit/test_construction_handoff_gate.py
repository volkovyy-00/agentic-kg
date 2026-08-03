"""Unit tests for the construction agent's explicit-handoff gate.

The gate exists because `graph_construction_agent` answers questions after
construction without graphrag_agent_v2's grounding guardrails, and used to
decide for itself when that window was over. These tests cover the mechanism
only -- whether the model actually asks before confirming is not
unit-testable and is verified by hand (see the plan's Task 5).
"""
import inspect

from agentic_kg.common.tool_result import is_error, is_success
from agentic_kg.coordinators.multi_agent.agent import full_workflow_agent
from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent.variants import (
    GRAPHRAG_AGENT_NAME,
    finished,
)
from agentic_kg.tools.construction_handoff_tools import (
    HANDOFF_CONFIRMED_KEY,
    confirm_construction_handoff,
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
    confirm_construction_handoff(context)
    assert context.state[HANDOFF_CONFIRMED_KEY] is True


def test_confirm_returns_a_success_result():
    """Catches a bare-{} return here: unlike 'finished', this tool is not a
    transfer and follows the normal ToolResult convention, so a model reading
    the result can tell the confirmation was recorded."""
    result = confirm_construction_handoff(FakeToolContext())
    assert is_success(result)


def test_confirm_takes_no_arguments_beyond_context():
    """Catches a signature that asks the model to supply an argument. A
    zero-argument tool is categorically more reliable on smaller models --
    the same reasoning make_finished documents for itself."""
    parameters = list(inspect.signature(confirm_construction_handoff).parameters)
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


def test_finished_transfers_to_retrieval_when_confirmed():
    """Catches a transfer back to the coordinator, which would strand the user
    one hop short of the retrieval agent -- the failure the ticket's fourth
    acceptance criterion names."""
    context = FakeToolContext({HANDOFF_CONFIRMED_KEY: True})
    result = finished(context)
    assert result == {}
    assert context.actions.transfer_to_agent == GRAPHRAG_AGENT_NAME


def test_the_retrieval_agent_resolves_in_the_agent_tree():
    """Catches graphrag_agent dropping out of full_workflow_agent.sub_agents in
    a future refactor while its module still defines AGENT_NAME correctly.

    It does NOT guard against a stale copy of the name -- the live import made
    that structurally impossible, since there is only one definition to read.
    Do not delete this as redundant with that: the failure it catches is a
    correct name pointing at an agent no longer in the tree, which makes
    find_agent raise inside a transfer chain with no trace span.
    """
    assert full_workflow_agent.find_agent(GRAPHRAG_AGENT_NAME) is not None
