"""The explicit-handoff gate for the graph construction phase.

After construction succeeds, `graph_construction_agent` invites questions and
answers them with bare Cypher -- without the schema-profile guardrails
`graphrag_agent_v2` carries. Leaving the end of that window to the model's
reading of "the user seems satisfied" is what this module removes: the flag
below is set only by an explicit tool call, and the construction agent's own
`finished` wrapper refuses to transfer without it.

The key is spelled here, once. The agent's agent.py (which clears it every
turn) and its variants.py (which reads it) both import this constant rather
than retyping the string.
"""
from google.adk.tools import ToolContext

from agentic_kg.common.tool_result import ToolResult, tool_success

HANDOFF_CONFIRMED_KEY = "construction_handoff_confirmed"


def confirm_construction_handoff(tool_context: ToolContext) -> ToolResult:
    """Record that the user has explicitly agreed to move on to the retrieval agent.

    Call this only when the user has said so in their own words in this turn --
    never on an inference that they sound finished, and never to pre-authorise a
    handoff you expect them to want. Call 'finished' in the same reply.
    """
    tool_context.state[HANDOFF_CONFIRMED_KEY] = True
    return tool_success(HANDOFF_CONFIRMED_KEY, True)
