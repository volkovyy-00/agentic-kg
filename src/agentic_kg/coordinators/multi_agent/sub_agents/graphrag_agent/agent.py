from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext


from agentic_kg.common.adk_transfer import strip_transfer_to_agent
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

# Whether this variant is the GATED one. Both callbacks below hang off this
# single fact, so it is named once rather than compared twice: flipping which
# variant is gated, or renaming the literal, is then one edit that cannot go
# half-applied.
IS_GATED_VARIANT = AGENT_NAME == "graphrag_agent_v2"

graphrag_agent = Agent(
    name=AGENT_NAME,
    # Stays on the conversational tier deliberately: the experiment is whether
    # better information alone fixes the framing errors. Changing information
    # and model together would make the result uninterpretable.
    model=get_llm(LlmKind.conversational),
    description="Information retrieval from a knowledge graph using a range of query tools.", # Crucial for delegation later
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"],
    # v2 holds two model callbacks: drop_foreign_context (from the variant
    # spec, PR #9's context filtering) and the transfer strip. ADK iterates
    # canonical_before_model_callbacks as a list (base_llm_flow.py:661), so a
    # list is native here -- the strip must JOIN drop_foreign_context, never
    # replace it.
    #
    # ADK injects its own 'transfer_to_agent' tool, plus an instruction
    # advertising it, into any LlmAgent with a parent or peers, and it does not
    # consult the handoff gate. The strip removes it from every request.
    #
    # Deliberately NOT disallow_transfer_to_parent: that flag would also close
    # the door, and would also stop Runner._find_agent_to_run
    # (runners.py:474-489) from returning this agent for the user's second
    # message, so every follow-up question would be re-arbitrated by the
    # coordinator. See adk_transfer.py.
    #
    # Conditional for the same reason as the reset callback below. v1 is the
    # ungated A/B baseline -- its 'finished' transfers unconditionally, so it
    # has no guarantee for the injected tool to bypass, and
    # test_v1_is_left_intact_for_the_acceptance_ab pins that it carries no
    # before_model_callback at all. Never attach this unconditionally.
    before_model_callback=(
        [variants[AGENT_NAME]["before_model_callback"], strip_transfer_to_agent]
        if IS_GATED_VARIANT
        else variants[AGENT_NAME].get("before_model_callback")
    ),
    # Conditional because only v2 is gated. Attaching unconditionally would
    # write an inert graphrag_handoff_confirmed every turn under v1, read by
    # nobody -- harmless, but untrue to "v1 is untouched" and avoidable in one
    # line. Same None-default reasoning as the model callback above.
    before_agent_callback=(
        reset_graphrag_handoff_confirmation if IS_GATED_VARIANT else None
    ),
)

root_agent = graphrag_agent
