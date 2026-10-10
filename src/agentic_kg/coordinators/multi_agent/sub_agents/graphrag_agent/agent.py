from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from agentic_kg.common.agent_guards import agent_guard_callbacks
from agentic_kg.common.llm_catalog import LlmKind, get_llm
from agentic_kg.tools.graphrag_handoff_tools import GRAPHRAG_HANDOFF_CONFIRMED
from agentic_kg.tools.graphrag_partition_tools import (
    PARTITION_INTERPRETATION_DECLARED,
)

from .variants import variants


def reset_graphrag_handoff_confirmation(callback_context: CallbackContext) -> None:
    """Clear the handoff confirmation at the start of every turn this agent runs.

    Fires once per BaseAgent.run_async, which for a plain Agent
    means every turn this agent is active -- not only on entry to the phase.
    There is no phase-entry-versus-turn-N distinction here and none should be
    built: trying to keep a confirmation alive across turns is the stale-flag
    bug this reset exists to prevent.

    Safe because that single call happens before _run_async_impl, where an
    LlmAgent's whole tool loop runs. 'confirm_graphrag_handoff' and 'finished'
    both fire after the reset, inside one unbroken loop.

    The parameter name is load-bearing: ADK invokes callbacks by keyword
    (BaseAgent._handle_before_agent_callback), so renaming it fails at
    request time with a TypeError, not at import.
    """
    GRAPHRAG_HANDOFF_CONFIRMED.reset(callback_context.state)


def reset_partition_interpretation_declaration(
    callback_context: CallbackContext,
) -> None:
    """Clear the partition-interpretation declaration at the start of every turn.

    Same lifecycle as reset_graphrag_handoff_confirmation above, and joined
    with it in the same before_agent_callback list: a declaration made on an
    earlier turn must not silently authorise an undeclared aggregation on a
    later one, the same stale-flag risk that reset guards against.
    """
    PARTITION_INTERPRETATION_DECLARED.reset(callback_context.state)


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
    description="Information retrieval from a knowledge graph using a range of query tools.",  # Crucial for delegation later
    instruction=variants[AGENT_NAME]["instruction"],
    tools=variants[AGENT_NAME]["tools"],
    # ADK gives this agent its own 'transfer_to_agent', which does not consult
    # the handoff gate. agent_guard_callbacks removes it and answers a call
    # made anyway, and carries drop_foreign_context (PR #9's context
    # filtering). The mechanism, and why disallow_transfer_to_parent is NOT
    # used, are in common/adk_transfer.py, once.
    #
    # Gated only for v2, for the same reason as the reset callback below. v1
    # is the ungated A/B baseline -- its 'finished' transfers unconditionally,
    # so it has no guarantee for the injected tool to bypass, and
    # test_v1_is_left_intact_for_the_acceptance_ab pins that it gets none of
    # these callbacks.
    **agent_guard_callbacks(gated=IS_GATED_VARIANT),
    # Conditional because only v2 is gated. Attaching unconditionally would
    # write inert flags every turn under v1, read by nobody -- harmless, but
    # untrue to "v1 is untouched" and avoidable in one line; v1 gets None
    # here, as it gets no transfer guard from agent_guard_callbacks above. A
    # list, not a single callback: ADK's canonical_before_agent_callbacks
    # accepts either.
    before_agent_callback=(
        [
            reset_graphrag_handoff_confirmation,
            reset_partition_interpretation_declaration,
        ]
        if IS_GATED_VARIANT
        else None
    ),
)

root_agent = graphrag_agent
