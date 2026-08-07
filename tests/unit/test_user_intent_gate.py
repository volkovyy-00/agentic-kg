"""Unit tests for the user-intent phase's approval gate.

The gate exists because `user_intent_agent_v2` asked its clarifying questions
and called ADK's injected `transfer_to_agent` in the same reply, so the user's
agreement landed on the coordinator, which holds no approval tool, and
`approved_user_goal` was never written (docs/backlog/
user-goal-approval-never-recorded.md). These tests cover the mechanism only --
whether a real model reads a refusal and recovers is not unit-testable and is
verified by hand (see the plan's Task 5).

Unlike the two shipped gates, this one checks no boolean flag. It compares two
durable state keys, because this phase ends on an approval the user actively
gives rather than on a turn-scoped "yes, move on".
"""
import asyncio

import pytest
from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from pydantic import Field

from agentic_kg.common.adk_context import FOREIGN_CONTEXT_SENTINEL, drop_foreign_context
from agentic_kg.common.adk_transfer import strip_transfer_to_agent
from agentic_kg.common.agent_names import MULTI_AGENT_COORDINATOR
from agentic_kg.common.tool_result import is_error
# Imported for its import-time side effect as well as its use below: building
# the coordinator is what gives user_intent_agent a parent, and ADK only
# injects transfer_to_agent into a parented agent. Without this import every
# absence assertion in this file would pass vacuously.
from agentic_kg.coordinators.multi_agent.agent import full_workflow_agent
from agentic_kg.coordinators.multi_agent.sub_agents.user_intent_agent.agent import (
    user_intent_agent,
)
from agentic_kg.coordinators.multi_agent.sub_agents.user_intent_agent.variants import (
    _transfer_to_coordinator,
    finished,
    variants,
)
from agentic_kg.tools.user_goal_tools import (
    APPROVED_USER_GOAL,
    PERCEIVED_USER_GOAL as PERCEIVED,
)

GOAL = {"kind_of_graph": "bill of materials", "graph_description": "parts and suppliers"}
OTHER_GOAL = {"kind_of_graph": "supply chain", "graph_description": "a different graph entirely"}


class FakeActions:
    def __init__(self):
        self.escalate = False
        self.transfer_to_agent = None


class FakeToolContext:
    """A tool context carrying BOTH .state and .actions.

    test_adk_tools.py's FakeToolContext has .actions only, and fakes.py
    deliberately does not centralize state-carrying fakes (see its docstring).
    The gated 'finished' reads state and then writes actions, so it needs one
    object with both -- neither existing fake will do. Copied from
    test_construction_handoff_gate.py:45 for the same stated reason.
    """

    def __init__(self, state=None):
        self.state = dict(state or {})
        self.actions = FakeActions()


def test_finished_refuses_when_no_goal_is_recorded():
    """Catches a gate that only checks the approved key: a model that reached
    'finished' without ever calling 'set_perceived_user_goal' would otherwise
    be told to approve a goal that does not exist, and would hit
    approve_perceived_user_goal's own refusal one hop later."""
    context = FakeToolContext()
    result = finished(context)
    assert is_error(result)
    assert "no goal has been recorded" in result["error_message"]
    assert context.actions.transfer_to_agent is None


def test_finished_refuses_when_the_goal_was_never_approved():
    """The reported bug. Catches a 'finished' that transfers on the strength
    of a perceived goal alone -- exactly what the ungated closure does."""
    context = FakeToolContext({PERCEIVED: GOAL})
    result = finished(context)
    assert is_error(result)
    assert "has not been approved" in result["error_message"]
    assert context.actions.transfer_to_agent is None


def test_finished_refuses_when_the_approved_goal_is_stale():
    """The case a presence-only check would wave through: the user approved a
    goal, then revised it, and the approved key still holds the superseded
    one. Nothing else in this suite compares two state values."""
    context = FakeToolContext({PERCEIVED: OTHER_GOAL, APPROVED_USER_GOAL: GOAL})
    result = finished(context)
    assert is_error(result)
    assert "out of date" in result["error_message"]
    assert context.actions.transfer_to_agent is None


def test_the_three_refusal_messages_are_mutually_distinguishable():
    """Catches two branches sharing a message. The model has just seen
    'finished' work and then stop working; it cannot pick the right recovery
    if 'never approved' and 'stale' read the same."""
    nothing = finished(FakeToolContext())["error_message"]
    unapproved = finished(FakeToolContext({PERCEIVED: GOAL}))["error_message"]
    stale = finished(
        FakeToolContext({PERCEIVED: OTHER_GOAL, APPROVED_USER_GOAL: GOAL})
    )["error_message"]

    assert len({nothing, unapproved, stale}) == 3
    # The stale branch must not claim the goal was never approved -- it was.
    assert "has not been approved" not in stale


def test_each_refusal_message_names_the_tool_it_wants_called():
    """Recovery lives entirely in these messages (the design has no escape
    hatch). A refusal that does not name the next tool call leaves a small
    conversational model with nothing to act on."""
    nothing = finished(FakeToolContext())["error_message"]
    unapproved = finished(FakeToolContext({PERCEIVED: GOAL}))["error_message"]
    stale = finished(
        FakeToolContext({PERCEIVED: OTHER_GOAL, APPROVED_USER_GOAL: GOAL})
    )["error_message"]

    assert "set_perceived_user_goal" in nothing
    assert "approve_perceived_user_goal" in unapproved
    assert "approve_perceived_user_goal" in stale


def test_finished_transfers_to_the_coordinator_when_the_approval_is_current():
    """The sanctioned exit must still work. A gate that never opens is as
    broken as one that never closes."""
    context = FakeToolContext({PERCEIVED: GOAL, APPROVED_USER_GOAL: GOAL})
    finished(context)
    assert context.actions.transfer_to_agent == MULTI_AGENT_COORDINATOR
    assert context.actions.escalate is True


def test_finished_returns_a_bare_dict_on_success():
    """Catches a well-meaning change to tool_success(...) on the happy path.
    Every other 'finished' in this codebase returns {}; only the error paths
    speak ToolResult."""
    context = FakeToolContext({PERCEIVED: GOAL, APPROVED_USER_GOAL: GOAL})
    assert finished(context) == {}


def test_finished_succeeds_after_a_stale_goal_is_re_approved():
    """The recovery the stale message asks for must actually work. Catches a
    gate that compares against a cached or first-seen value rather than
    current state."""
    context = FakeToolContext({PERCEIVED: OTHER_GOAL, APPROVED_USER_GOAL: GOAL})
    assert is_error(finished(context))
    context.state[APPROVED_USER_GOAL] = context.state[PERCEIVED]
    assert finished(context) == {}
    assert context.actions.transfer_to_agent == MULTI_AGENT_COORDINATOR


def test_v1_holds_the_ungated_exit():
    """TRAP 1. v1 uses set_user_goal and never writes approved_user_goal, so
    the gated 'finished' in its tools list would be a lock with no key --
    invisible until someone flips AGENT_NAME to _v1."""
    v1_tools = variants["user_intent_agent_v1"]["tools"]
    assert _transfer_to_coordinator in v1_tools
    assert all(tool is not finished for tool in v1_tools)


def test_v2_holds_the_gated_finished():
    """The other half of the split: catches the swap being written but never
    applied to the variant that is actually selected."""
    v2_tools = variants["user_intent_agent_v2"]["tools"]
    assert finished in v2_tools
    assert all(tool is not _transfer_to_coordinator for tool in v2_tools)


def test_v1_exit_still_presents_to_the_model_as_finished():
    """TRAP 2. ADK derives the model-facing tool name from __name__, so a
    refactor of make_finished to a lambda or functools.partial would silently
    rename v1's exit tool and falsify v1's own instruction line ('use the
    finished tool') without failing any other test."""
    assert _transfer_to_coordinator.__name__ == "finished"


class CapturingLlm(BaseLlm):
    """Replays a fixed script and records every LlmRequest it was handed.

    Copied from test_construction_handoff_gate.py:199 rather than imported.
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
    """Drive one real user turn through ADK and return the events it produced.

    Points the Runner at an agent already wired into the real tree rather than
    re-parenting it, which base_agent.py:496-505 forbids for an agent that
    already has a parent.
    """
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


def test_the_agent_is_parented_so_the_absence_assertions_are_not_vacuous():
    """ADK only injects transfer_to_agent into an agent that has a parent or
    peers. A bare import of this agent's own module leaves parent_agent as
    None, so without the full_workflow_agent import at the top of this file
    every absence assertion below would pass without proving anything."""
    assert user_intent_agent.parent_agent is not None


def test_the_strip_callback_is_wired_onto_the_user_intent_agent():
    """The cheap statement of intent. Not sufficient on its own -- it says
    nothing about whether the strip actually worked -- which is why the
    behavioural tests below exist."""
    assert strip_transfer_to_agent in user_intent_agent.canonical_before_model_callbacks


def test_the_user_intent_agent_has_no_before_agent_callback():
    """TRAP 3. This gate compares two durable state keys and has no per-turn
    flag, so there is nothing for a reset callback to do. Catches someone
    copying graphrag_agent/agent.py's two-callback shape wholesale and
    reintroducing the flag machinery this design deliberately avoids."""
    assert user_intent_agent.before_agent_callback is None
    assert user_intent_agent.canonical_before_agent_callbacks == []


def test_both_model_callbacks_are_wired_in_order():
    """Catches either half of the pair being dropped. The strip removes the
    transfer DECLARATION; drop_foreign_context removes the worked EXAMPLE of it
    that ADK leaves in this agent's history. Removing the declaration and
    leaving the example is half a fix -- the model copies the example, the
    strip has already popped the tool from tools_dict, and ADK raises
    mid-turn. Same pairing as graph_construction_agent."""
    assert user_intent_agent.canonical_before_model_callbacks == [
        drop_foreign_context,
        strip_transfer_to_agent,
    ]


def test_the_agent_does_not_disallow_transfers():
    """Guards the trap this design exists to avoid. Setting
    disallow_transfer_to_parent would also close the door -- and would make
    Runner._find_agent_to_run (runners.py:474-489) stop returning this agent
    for the user's SECOND message, sending every mid-interview reply back
    through the coordinator to be re-arbitrated."""
    assert user_intent_agent.disallow_transfer_to_parent is False
    assert user_intent_agent.disallow_transfer_to_peers is False


def test_the_model_is_never_offered_transfer_to_agent(monkeypatch):
    """The actual guarantee, and the exact exit taken in the reported session.
    ADK injects a transfer_to_agent tool into every sub-agent with a parent or
    peers, and it does not consult this gate. Asserting on what reached the
    model is the only way to know it is gone."""
    monkeypatch.setattr(user_intent_agent, "model", CapturingLlm(
        model="scripted", responses=[_text("tell me about your data")],
    ))
    asyncio.run(_run_one_turn(user_intent_agent, "intent_door_test"))

    requests = user_intent_agent.model.requests
    assert requests, "the model was never called"
    for request in requests:
        assert "transfer_to_agent" not in request.tools_dict
        assert "transfer_to_agent" not in _declaration_names(request)
        instruction = str(getattr(request.config, "system_instruction", "") or "")
        assert "transfer_to_agent" not in instruction


def test_the_coordinators_transfer_call_never_reaches_this_agents_context(monkeypatch):
    """The behavioural half of the callback pair, and the reason it exists.

    Wiring assertions prove drop_foreign_context is attached, not that it does
    anything -- the same gap TRAP 5 guards for the strip. This agent is entered
    BY the coordinator's transfer_to_agent call, which ADK rewrites into a
    'For context: ...' turn (contents.py) that would otherwise sit in history
    for the whole interview: a worked example of the exact call the strip
    removes the declaration for. Catches drop_foreign_context being dropped, or
    being wired somewhere it never runs.
    """
    monkeypatch.setattr(full_workflow_agent, "model", CapturingLlm(
        model="scripted",
        responses=[_call("transfer_to_agent", {"agent_name": "user_intent_agent_v2"})],
    ))
    monkeypatch.setattr(user_intent_agent, "model", CapturingLlm(
        model="scripted",
        responses=[
            _text("what kind of graph did you have in mind?"),
            _text("thanks -- and what will you use it for?"),
        ],
    ))

    async def run():
        runner = InMemoryRunner(agent=full_workflow_agent, app_name="intent_context_test")
        session = await runner.session_service.create_session(
            app_name="intent_context_test", user_id="u1",
        )
        for message in ("I want a graph", "a bill of materials"):
            async for _ in runner.run_async(
                user_id="u1", session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=message)]),
            ):
                pass

    asyncio.run(run())

    requests = user_intent_agent.model.requests
    assert len(requests) >= 2, (
        f"the interview never reached a second turn; got {len(requests)} request(s)"
    )
    for request in requests:
        for content in (request.contents or []):
            for part in (getattr(content, "parts", None) or []):
                text = getattr(part, "text", None) or ""
                assert FOREIGN_CONTEXT_SENTINEL not in text, (
                    "the coordinator's transfer_to_agent call is still in this "
                    f"agent's context: {text[:200]!r}"
                )


def test_the_agents_own_tools_survive_the_strip(monkeypatch):
    """Catches an over-broad strip that empties config.tools. The agent is
    useless without its own tools, and every other assertion here is about
    absence, so nothing else would notice."""
    monkeypatch.setattr(user_intent_agent, "model", CapturingLlm(
        model="scripted", responses=[_text("tell me about your data")],
    ))
    asyncio.run(_run_one_turn(user_intent_agent, "intent_tools_survive_test"))

    names = _declaration_names(user_intent_agent.model.requests[0])
    assert "finished" in names
    assert "set_perceived_user_goal" in names
    assert "approve_perceived_user_goal" in names


def test_calling_transfer_to_agent_anyway_is_a_hard_error(monkeypatch):
    """Pins what happens if a model emits the call from memory of an earlier
    turn -- which is precisely what the reported session did. The strip pops it
    from tools_dict, so ADK raises (functions.py:565-568) rather than silently
    transferring mid-question."""
    monkeypatch.setattr(user_intent_agent, "model", CapturingLlm(
        model="scripted",
        responses=[_call("transfer_to_agent", {"agent_name": "kg_construction_agent_v1"})],
    ))
    with pytest.raises(ValueError, match="transfer_to_agent"):
        asyncio.run(_run_one_turn(user_intent_agent, "intent_hard_error_test"))
