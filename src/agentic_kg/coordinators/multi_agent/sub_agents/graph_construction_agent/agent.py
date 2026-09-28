from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from agentic_kg.common.adk_transfer import transfer_guard_callbacks
from agentic_kg.common.llm_catalog import LlmKind, get_llm
from agentic_kg.tools.construction_handoff_tools import HANDOFF_CONFIRMED_KEY

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
    callback_context.state[HANDOFF_CONFIRMED_KEY] = False


AGENT_NAME = "graph_construction_agent_v1"
graph_construction_agent = Agent(
    name=AGENT_NAME,
    model=get_llm(LlmKind.reasoning),
    description="Knowledge graph construction based on approved construction rules.",
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"],
    before_agent_callback=reset_construction_handoff_confirmation,
    # transfer_guard_callbacks wires three callbacks that only work as a set
    # (common/adk_transfer.py); the same call graphrag_agent_v2 and
    # user_intent_agent_v2 make.
    #
    # ADK injects its own 'transfer_to_agent' tool, plus an instruction
    # advertising it, into any LlmAgent with a parent or peers -- and it
    # does not consult the handoff gate above. strip_transfer_to_agent
    # takes it back out of every request before the model sees it.
    #
    # drop_foreign_context closes the matching context-side hole. The
    # coordinator's own delegating call arrives here rewritten by
    # _present_other_agent_message (flows/llm_flows/_fencing.py) into a
    # "For context: ..." turn quoting "[kg_construction_agent_v1] called
    # tool `transfer_to_agent` with parameters:" and its arguments -- a
    # worked example of the exact tool name and argument shape, sitting in
    # history for every subsequent turn in this branch, while the
    # declaration itself is stripped. Removing the declaration and leaving
    # the example is half a fix.
    #
    # If the model emits the call anyway, refuse_transfer_to_agent (its
    # before_tool_callback) answers it before ADK's generic not-found
    # reply, naming 'finished' as the way out, and ends the turn with an
    # error if the model keeps calling it -- pinned by
    # test_calling_transfer_to_agent_anyway_returns_an_error_and_stays_in_phase
    # and test_a_model_that_keeps_calling_the_hidden_tool_ends_the_turn.
    #
    # Deliberately NOT disallow_transfer_to_parent: that flag would also
    # close the door, and would also stop Runner._find_agent_to_run
    # (agents/_agent_router.py find_agent_to_run) from returning this agent
    # for the user's second message, so every follow-up question in the
    # post-construction window would be re-arbitrated by the coordinator.
    # On google-adk 2.9 either flag also makes a blocked 'finished' call
    # raise ValueError, so a make_finished target must be this agent's
    # parent or a peer. See adk_transfer.py.
    #
    # 'finished' is unaffected -- it writes actions.transfer_to_agent
    # directly, which ADK acts on after the tool returns and no
    # request-level strip touches.
    **transfer_guard_callbacks(gated=True),
)

root_agent = graph_construction_agent
