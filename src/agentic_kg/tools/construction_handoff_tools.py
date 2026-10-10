"""The explicit-handoff gate for the graph construction phase.

After construction succeeds, `graph_construction_agent` invites questions and
answers them with bare Cypher -- without the schema-profile guardrails
`graphrag_agent_v2` carries. Leaving the end of that window to the model's
reading of "the user seems satisfied" is what this module removes: the flag
below is set only by an explicit tool call, and the construction agent's own
`finished` wrapper refuses to transfer without it.

The key is spelled here, once. The agent's agent.py (which clears it every
turn) and its variants.py (which reads it) both import `HANDOFF_CONFIRMED`
rather than retyping the string, and `TurnFlag` (common/turn_flags.py) holds
the plumbing.
"""

from google.adk.tools import ToolContext

from agentic_kg.common.tool_result import ToolResult, tool_success
from agentic_kg.common.turn_flags import TurnFlag

HANDOFF_CONFIRMED_KEY = "construction_handoff_confirmed"
HANDOFF_CONFIRMED = TurnFlag(HANDOFF_CONFIRMED_KEY)


def confirm_construction_handoff(tool_context: ToolContext) -> ToolResult:
    """Record that the user has explicitly agreed to move on to the retrieval agent.

    Call this only when the user has said so in their own words in this turn --
    never on an inference that they sound finished, and never to pre-authorise a
    handoff you expect them to want. Call 'finished' in the same reply.
    """
    HANDOFF_CONFIRMED.set(tool_context.state)
    return tool_success(HANDOFF_CONFIRMED_KEY, True)


# The way back to the plan step: a second gate on the same agent, with its own
# flag, so confirming one exit never opens the other.
PLAN_REVISION_CONFIRMED_KEY = "plan_revision_confirmed"
PLAN_REVISION_CONFIRMED = TurnFlag(PLAN_REVISION_CONFIRMED_KEY)


def confirm_plan_revision(tool_context: ToolContext) -> ToolResult:
    """Record that the user has explicitly asked to change the construction plan.

    Call this only when the user has said so in their own words in this turn --
    a change to the plan's labels, relationships, keys or properties, never to
    the goal or the files, and never on an inference that they might want one.
    Call 'return_to_plan' in the same reply.
    """
    PLAN_REVISION_CONFIRMED.set(tool_context.state)
    return tool_success(PLAN_REVISION_CONFIRMED_KEY, True)
