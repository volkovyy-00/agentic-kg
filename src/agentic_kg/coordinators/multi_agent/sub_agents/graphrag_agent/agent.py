from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext


from agentic_kg.common.llm_catalog import get_llm, LlmKind
from agentic_kg.tools.graphrag_handoff_tools import GRAPHRAG_HANDOFF_CONFIRMED_KEY

from .variants import variants


def reset_graphrag_handoff_confirmation(callback_context: CallbackContext) -> None:
    """Clear the handoff confirmation at the start of every turn this agent runs.

    Fires once per run_async (base_agent.py:218-223), which for a plain Agent
    means every turn this agent is active -- not only on entry to the phase.
    There is no phase-entry-versus-turn-N distinction here and none should be
    built: trying to keep a confirmation alive across turns is the stale-flag
    bug this reset exists to prevent.

    Safe because that single call happens before _run_async_impl, where an
    LlmAgent's whole tool loop runs. 'confirm_graphrag_handoff' and 'finished'
    both fire after the reset, inside one unbroken loop.

    The parameter name is load-bearing: ADK invokes callbacks by keyword
    (base_agent.py:385-387), so renaming it fails at request time with a
    TypeError, not at import.
    """
    callback_context.state[GRAPHRAG_HANDOFF_CONFIRMED_KEY] = False


AGENT_NAME = "graphrag_agent_v2"
graphrag_agent = Agent(
    name=AGENT_NAME,
    # Stays on the conversational tier deliberately: the experiment is whether
    # better information alone fixes the framing errors. Changing information
    # and model together would make the result uninterpretable.
    model=get_llm(LlmKind.conversational),
    description="Information retrieval from a knowledge graph using a range of query tools.", # Crucial for delegation later
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"],
    # .get() so v1, which has no callback, stays a no-op: the field defaults to
    # None (llm_agent.py:225) and canonical_before_model_callbacks returns []
    # on falsy (390-391).
    before_model_callback=variants[AGENT_NAME].get("before_model_callback"),
    # Conditional because only v2 is gated. Attaching unconditionally would
    # write an inert graphrag_handoff_confirmed every turn under v1, read by
    # nobody -- harmless, but untrue to "v1 is untouched" and avoidable in one
    # line. Same None-default reasoning as the model callback above.
    before_agent_callback=(
        reset_graphrag_handoff_confirmation
        if AGENT_NAME == "graphrag_agent_v2"
        else None
    ),
)

root_agent = graphrag_agent
