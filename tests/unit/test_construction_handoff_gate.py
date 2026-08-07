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
from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent.agent import (
    graph_construction_agent,
    reset_construction_handoff_confirmation,
)
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


class FakeCallbackContext:
    """A callback context carrying state only -- callbacks never touch .actions."""

    def __init__(self, state=None):
        self.state = dict(state or {})


def test_reset_clears_a_previous_confirmation():
    """Catches a reset that only initialises a missing key. A confirmation from
    an earlier turn -- or from an earlier graph built in the same session --
    would otherwise let 'finished' transfer on the model's judgment alone."""
    context = FakeCallbackContext({HANDOFF_CONFIRMED_KEY: True})
    reset_construction_handoff_confirmation(context)
    assert context.state[HANDOFF_CONFIRMED_KEY] is False


def test_reset_is_wired_onto_the_construction_agent():
    """Catches the callback being defined but never attached, which leaves the
    flag sticky for the whole session and silently disables the gate after the
    first successful handoff."""
    callbacks = graph_construction_agent.canonical_before_agent_callbacks
    assert reset_construction_handoff_confirmation in callbacks


def test_reset_parameter_is_named_callback_context():
    """Catches a rename. ADK invokes these callbacks by keyword
    (base_agent.py:385-387), so a different parameter name fails at request
    time with a TypeError rather than at import."""
    parameters = list(
        inspect.signature(reset_construction_handoff_confirmation).parameters
    )
    assert parameters == ["callback_context"]


def test_confirm_tool_is_wired_into_the_construction_variant():
    """Catches an instruction that tells the model to call
    'confirm_construction_handoff' when the tool was never added to the
    variant's tools list -- the model would then be unable to end the phase at
    all, since 'finished' refuses without it."""
    from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent.variants import (
        variants,
    )
    tools = variants["graph_construction_agent_v1"]["tools"]
    assert confirm_construction_handoff in tools
    assert finished in tools


def test_finished_succeeds_on_retry_after_an_out_of_order_refusal():
    """Catches a gate that latches its refusal. ADK runs the tool calls in one
    model reply in the order the model emitted them, so a reply ordering
    'finished' before 'confirm_construction_handoff' refuses even though the
    user did agree. The confirmation is recorded by the time the model reads
    that error, so calling 'finished' again in the same turn must then
    transfer -- which is the recovery the refusal message and instruction
    step 9 promise. This matters more once Task 3 removes the injected
    transfer tool: without the retry, the user is stuck in the phase."""
    context = FakeToolContext()
    assert is_error(finished(context))
    confirm_construction_handoff(context)
    assert finished(context) == {}
    assert context.actions.transfer_to_agent == GRAPHRAG_AGENT_NAME


def test_refusal_message_names_the_same_reply_recovery():
    """Catches the refusal message losing the out-of-order recovery hint while
    instruction step 9 still promises it. The model only learns that a retry
    will work from this string, so the two must not drift apart."""
    message = finished(FakeToolContext())["error_message"]
    assert "call 'finished' once more" in message
