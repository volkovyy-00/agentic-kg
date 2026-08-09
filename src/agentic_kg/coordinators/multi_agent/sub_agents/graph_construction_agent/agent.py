from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from agentic_kg.common.adk_context import drop_foreign_context
from agentic_kg.common.adk_transfer import strip_transfer_to_agent
from agentic_kg.common.llm_catalog import LlmKind, get_llm
from agentic_kg.tools.construction_handoff_tools import HANDOFF_CONFIRMED_KEY

# variants are pairs of instructions with tools
from .variants import variants


def reset_construction_handoff_confirmation(callback_context: CallbackContext) -> None:
    """Clear the handoff confirmation at the start of every turn this agent runs.

    Fires once per run_async (base_agent.py:218-223), which for a plain Agent
    means every turn this agent is active -- not only on entry to the phase.
    There is no phase-entry-versus-turn-N distinction here and none should be
    built: trying to keep a confirmation alive across turns is the stale-flag
    bug this reset exists to prevent.

    Safe because that single call happens before _run_async_impl, where an
    LlmAgent's whole tool loop runs. 'confirm_construction_handoff' and
    'finished' both fire after the reset, inside one unbroken loop.

    The parameter name is load-bearing: ADK invokes callbacks by keyword
    (base_agent.py:385-387), so renaming it fails at request time with a
    TypeError, not at import.
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
    # Two model callbacks, in a list -- ADK iterates
    # canonical_before_model_callbacks (base_llm_flow.py:661), so a list is
    # native here. Same shape graphrag_agent_v2 carries.
    #
    # ADK injects its own 'transfer_to_agent' tool, plus an instruction
    # advertising it, into any LlmAgent with a parent or peers -- and it
    # does not consult the handoff gate above. strip_transfer_to_agent
    # takes it back out of every request before the model sees it.
    #
    # drop_foreign_context closes the matching context-side hole. The
    # coordinator's own delegating call arrives here rewritten by
    # _convert_foreign_event (contents.py:241-245) into "For context:
    # [kg_construction_agent_v1] called tool 'transfer_to_agent' with
    # parameters: {'agent_name': 'graph_construction_agent_v1'}" -- a
    # worked example of the exact tool name and argument shape, sitting in
    # history for every subsequent turn in this branch, while the
    # declaration itself is stripped. Removing the declaration and leaving
    # the example is half a fix.
    #
    # This does NOT soften what happens if the model emits the call anyway:
    # tools_dict no longer holds it, so ADK raises (functions.py:565-568).
    # That hard error is spec-mandated and pinned by
    # test_calling_transfer_to_agent_anyway_is_a_hard_error. There is no
    # stub and no graceful path; this callback only removes the standing
    # invitation to try.
    #
    # Deliberately NOT disallow_transfer_to_parent: that flag would also
    # close the door, and would also stop Runner._find_agent_to_run
    # (runners.py:474-489) from returning this agent for the user's second
    # message, so every follow-up question in the post-construction window
    # would be re-arbitrated by the coordinator. See adk_transfer.py.
    #
    # 'finished' is unaffected -- it writes actions.transfer_to_agent
    # directly (base_llm_flow.py:536-548), which no request-level strip
    # touches.
    before_model_callback=[drop_foreign_context, strip_transfer_to_agent],
)

root_agent = graph_construction_agent
