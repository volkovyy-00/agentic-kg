"""Unit tests for the retrieval agent's explicit-handoff gate.

The gate exists because `graphrag_agent_v2` used to decide for itself when the
user was finished asking questions, ejecting them back to the coordinator after
a single answer (see CHANGELOG.md's "Explicit retrieval handoff (#9)" entry).
These tests cover the mechanism only -- whether the model actually stays and
invites the next question is not unit-testable and is verified by hand (see
PR #9's description for the verification steps).
"""
import inspect

import asyncio

import pytest
from google.genai import types
from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from pydantic import Field

from agentic_kg.common.adk_context import drop_foreign_context
from agentic_kg.common.adk_transfer import strip_transfer_to_agent
# Imported for the side effect of building the real agent tree: this parents
# graphrag_agent under full_workflow_agent, which is what makes ADK actually
# inject transfer_to_agent into it. Without this import, graphrag_agent has no
# parent/peers at test time and every "transfer_to_agent is absent" assertion
# below would pass vacuously regardless of whether the strip callback works.
from agentic_kg.coordinators.multi_agent.agent import full_workflow_agent
from agentic_kg.coordinators.multi_agent.sub_agents.graphrag_agent import agent as graphrag_module

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


class CapturingLlm(BaseLlm):
    """Scripted LLM that also records every request it was handed.

    Copied from test_graphrag_context_filtering.py:20 rather than imported.
    tests/unit/fakes.py deliberately does not centralize fakes of this kind.
    Do not extract it.
    """
    responses: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        self.requests.append(llm_request)
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        yield self.responses[index]


def _text(text):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _call(name, args=None):
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name=name, args=args or {})),
    ]))


def _declaration_names(request):
    names = []
    for tool in (request.config.tools or []):
        for declaration in (getattr(tool, "function_declarations", None) or []):
            names.append(declaration.name)
    return names


async def _run_one_turn(agent, app_name, message="hello"):
    """Drive one real user turn through ADK and return the events it produced."""
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name, user_id="u1",
    )
    return [
        event async for event in runner.run_async(
            user_id="u1", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        )
    ]


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


def test_finished_succeeds_on_retry_after_an_out_of_order_refusal():
    """Catches a gate that latches its refusal. ADK runs the tool calls in one
    model reply in the order the model emitted them, so a reply ordering
    'finished' before 'confirm_graphrag_handoff' refuses even though the user
    did agree. The confirmation is recorded by the time the model reads that
    error, so calling 'finished' again in the same turn must then transfer --
    which is the recovery the refusal message and instruction step 9 promise."""
    context = FakeToolContext()
    assert is_error(finished(context))
    confirm_graphrag_handoff(context)
    assert finished(context) == {}
    assert context.actions.transfer_to_agent == MULTI_AGENT_COORDINATOR


def test_refusal_message_names_the_same_reply_recovery():
    """Catches the refusal message losing the out-of-order recovery hint while
    instruction step 9 still promises it. The model only learns that a retry
    will work from this string, so the two must not drift apart."""
    message = finished(FakeToolContext())["error_message"]
    assert "same reply" in message


def test_the_strip_callback_tracks_the_selected_variant():
    """Catches an unconditional attachment. Only graphrag_agent_v2 has a gate
    for the injected transfer tool to bypass; v1 is the ungated A/B baseline
    and must keep its request untouched. Written against AGENT_NAME rather
    than the literal so it survives -- and still constrains -- a future flip
    of that constant, which is precisely when an unconditional attachment
    would silently reach v1."""
    expected = graphrag_module.AGENT_NAME == "graphrag_agent_v2"
    callbacks = graphrag_agent.canonical_before_model_callbacks
    assert (strip_transfer_to_agent in callbacks) is expected


def test_both_model_callbacks_are_present_on_the_shipped_variant():
    """States the current fact the test above deliberately does not, and
    catches the strip callback replacing drop_foreign_context rather than
    joining it -- which would silently undo the #9 context filtering."""
    assert graphrag_module.AGENT_NAME == "graphrag_agent_v2"
    callbacks = graphrag_agent.canonical_before_model_callbacks
    assert drop_foreign_context in callbacks
    assert strip_transfer_to_agent in callbacks


def test_the_agent_does_not_disallow_transfers():
    """Guards the trap this design exists to avoid. Setting
    disallow_transfer_to_parent would also close the door -- and would make
    Runner._find_agent_to_run (runners.py:474-489) stop returning this agent
    for the user's SECOND message, sending every follow-up question back
    through the coordinator. See the spec's 'Why not' section."""
    assert graphrag_agent.disallow_transfer_to_parent is False
    assert graphrag_agent.disallow_transfer_to_peers is False


def test_the_agent_is_parented_so_the_absence_assertions_are_not_vacuous():
    """ADK only injects transfer_to_agent into an agent that has a parent or
    peers. If graphrag_agent were ever unparented at test time, every absence
    assertion in this file would pass without proving anything."""
    assert graphrag_agent.parent_agent is not None


def test_the_model_is_never_offered_transfer_to_agent(monkeypatch):
    """The actual guarantee: ADK injects a transfer_to_agent tool into every
    sub-agent with a parent or peers, and it does not consult this agent's
    handoff gate. Asserting on what reached the model is the only way to know
    it is gone."""
    monkeypatch.setattr(graphrag_agent, "model", CapturingLlm(
        model="scripted", responses=[_text("what would you like to know?")],
    ))
    asyncio.run(_run_one_turn(graphrag_agent, "graphrag_door_test"))

    requests = graphrag_agent.model.requests
    assert requests, "the model was never called"
    for request in requests:
        assert "transfer_to_agent" not in request.tools_dict
        assert "transfer_to_agent" not in _declaration_names(request)
        instruction = str(getattr(request.config, "system_instruction", "") or "")
        assert "transfer_to_agent" not in instruction


def test_the_agents_own_tools_survive_the_strip(monkeypatch):
    """Catches an over-broad strip that empties config.tools."""
    monkeypatch.setattr(graphrag_agent, "model", CapturingLlm(
        model="scripted", responses=[_text("what would you like to know?")],
    ))
    asyncio.run(_run_one_turn(graphrag_agent, "graphrag_tools_survive_test"))

    names = _declaration_names(graphrag_agent.model.requests[0])
    assert "finished" in names
    assert "confirm_graphrag_handoff" in names
    assert "read_neo4j_cypher" in names


def test_calling_transfer_to_agent_anyway_is_a_hard_error(monkeypatch):
    """Pins what happens if a model emits the call from memory of an earlier
    turn. The strip pops it from tools_dict, so ADK raises
    (functions.py:565-568) rather than silently transferring."""
    monkeypatch.setattr(graphrag_agent, "model", CapturingLlm(
        model="scripted",
        responses=[_call("transfer_to_agent", {"agent_name": MULTI_AGENT_COORDINATOR})],
    ))
    with pytest.raises(ValueError, match="transfer_to_agent"):
        asyncio.run(_run_one_turn(graphrag_agent, "graphrag_hard_error_test"))


def test_an_unstripped_agent_still_receives_it_negative_control():
    """Proves the assertions above are not vacuous -- that this harness does
    detect the injected tool when it is genuinely there.

    Builds a fresh, unparented v1 with no strip callback and gives it a
    throwaway parent with a peer. That is only possible because it is freshly
    constructed: base_agent.py:496-505 raises on any attempt to re-parent the
    live singletons, which is why every other test here points the Runner at
    the real tree.
    """
    spec = variants["graphrag_agent_v1"]
    child = Agent(
        name="graphrag_agent_v1",
        model=CapturingLlm(model="scripted", responses=[_text("ok")]),
        description="test",
        instruction=spec["instruction"],
        tools=spec["tools"],
    )
    peer = Agent(
        name="dummy_peer",
        model=CapturingLlm(model="scripted", responses=[_text("ok")]),
        description="a peer, so the peers branch of _get_transfer_targets runs",
    )
    Agent(
        name="fake_coordinator",
        model=CapturingLlm(model="scripted", responses=[_text("ok")]),
        description="throwaway parent",
        sub_agents=[child, peer],
    )

    asyncio.run(_run_one_turn(child, "graphrag_negative_control"))

    request = child.model.requests[0]
    assert "transfer_to_agent" in request.tools_dict
    assert "transfer_to_agent" in _declaration_names(request)
    assert "transfer_to_agent" in str(request.config.system_instruction or "")


def test_a_confirmed_handoff_still_reaches_the_coordinator(monkeypatch):
    """Closing the injected door must not disturb the sanctioned one. Runs the
    real ADK dispatch rather than calling 'finished' directly: the existing
    test_finished_transfers_to_the_coordinator_when_confirmed calls the
    closure with a FakeToolContext and would pass even if ADK's resolution
    path broke.

    Both models are scripted: the transfer runs inline in the same turn
    (base_llm_flow.py:536-542), so the coordinator's real model would
    otherwise be invoked for real.
    """
    monkeypatch.setattr(graphrag_agent, "model", CapturingLlm(
        model="scripted",
        responses=[
            _call("confirm_graphrag_handoff"),
            _call("finished"),
            _text("handing you back to the coordinator"),
        ],
    ))
    monkeypatch.setattr(full_workflow_agent, "model", CapturingLlm(
        model="scripted", responses=[_text("coordinator speaking")],
    ))

    events = asyncio.run(_run_one_turn(
        graphrag_agent, "graphrag_handoff_test",
        message="that's everything, thanks",
    ))

    authors = [event.author for event in events]
    assert MULTI_AGENT_COORDINATOR in authors, (
        f"the coordinator never took over; authors were {authors}"
    )
