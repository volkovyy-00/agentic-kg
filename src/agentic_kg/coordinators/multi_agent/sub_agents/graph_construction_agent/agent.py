from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from agentic_kg.common.llm_catalog import get_llm, LlmKind
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
    )

root_agent = graph_construction_agent
