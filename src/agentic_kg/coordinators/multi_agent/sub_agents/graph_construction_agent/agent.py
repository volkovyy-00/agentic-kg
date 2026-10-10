from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from agentic_kg.common.adk_transfer import transfer_guard_callbacks
from agentic_kg.common.agent_names import GRAPH_CONSTRUCTION_AGENT
from agentic_kg.common.llm_catalog import LlmKind, get_llm
from agentic_kg.tools.construction_handoff_tools import (
    HANDOFF_CONFIRMED,
    PLAN_REVISION_CONFIRMED,
)

# variants are pairs of instructions with tools
from .variants import variants


def reset_construction_handoff_confirmation(callback_context: CallbackContext) -> None:
    """Clear the handoff confirmation at the start of every turn this agent runs.

    Fires once per BaseAgent.run_async, which for a plain Agent
    means every turn this agent is active -- not only on entry to the phase.
    There is no phase-entry-versus-turn-N distinction here and none should be
    built: trying to keep a confirmation alive across turns is the stale-flag
    bug this reset exists to prevent.

    Safe because that single call happens before _run_async_impl, where an
    LlmAgent's whole tool loop runs. 'confirm_construction_handoff' and
    'finished' both fire after the reset, inside one unbroken loop.

    The parameter name is load-bearing: ADK invokes callbacks by keyword
    (BaseAgent._handle_before_agent_callback), so renaming it fails at
    request time with a TypeError, not at import.
    """
    HANDOFF_CONFIRMED.reset(callback_context.state)


def reset_plan_revision_confirmation(callback_context: CallbackContext) -> None:
    """Clear the way-back confirmation at the start of every turn this agent runs.

    Same lifecycle as reset_construction_handoff_confirmation above, and in the
    same before_agent_callback list: a request to change the plan made in an
    earlier turn must not let 'return_to_plan' withdraw the approval later.
    The parameter name is load-bearing (ADK passes callback_context=).
    """
    PLAN_REVISION_CONFIRMED.reset(callback_context.state)


AGENT_NAME = GRAPH_CONSTRUCTION_AGENT
graph_construction_agent = Agent(
    name=AGENT_NAME,
    model=get_llm(LlmKind.reasoning),
    description="Knowledge graph construction based on approved construction rules.",
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"],
    before_agent_callback=[
        reset_construction_handoff_confirmation,
        reset_plan_revision_confirmation,
    ],
    # ADK gives this agent its own 'transfer_to_agent', which does not consult
    # the handoff gate above. transfer_guard_callbacks removes it, removes the
    # worked example of it that the coordinator's own delegating call leaves
    # in this agent's history, and answers a call made anyway. The mechanism,
    # and why disallow_transfer_to_parent is NOT used, are in
    # common/adk_transfer.py, once. Here it would cost the
    # post-construction window: every follow-up question would go back to the
    # coordinator.
    #
    # 'finished' is unaffected -- it writes actions.transfer_to_agent
    # directly, which ADK acts on after the tool returns and no
    # request-level strip touches.
    **transfer_guard_callbacks(gated=True),
)

root_agent = graph_construction_agent
