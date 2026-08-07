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
from agentic_kg.tools.user_goal_tools import APPROVED_USER_GOAL

PERCEIVED = "perceived_user_goal"

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
