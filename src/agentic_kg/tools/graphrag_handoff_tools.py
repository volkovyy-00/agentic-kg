"""The explicit-handoff gate for the retrieval phase.

`graphrag_agent_v2` answers questions over the finished graph, and used to
decide for itself when that window was over -- ejecting the user back to the
coordinator after a single answer, without ever being told they were done
(`docs/backlog/graphrag-agent-exits-unasked.md`). Leaving the end of that
window to the model's reading of "the user seems satisfied" is what this module
removes: the flag below is set only by an explicit tool call, and the retrieval
agent's own `finished` wrapper refuses to transfer without it.

This deliberately duplicates `construction_handoff_tools.py` rather than
sharing with it. The two gates' `finished` wrappers differ in transfer
topology -- construction transfers sideways to a live-imported sibling name,
retrieval transfers up to a plain constant -- so a shared factory would have to
parametrise over more than a state key. Extract at a third occurrence, not
this one.

The key is spelled here, once. The agent's agent.py (which clears it every
turn) and its variants.py (which reads it) both import this constant rather
than retyping the string. It is deliberately not named `HANDOFF_CONFIRMED_KEY`:
that name already means something else, with a different value, one module
over.
"""
from google.adk.tools import ToolContext

from agentic_kg.common.tool_result import ToolResult, tool_success

GRAPHRAG_HANDOFF_CONFIRMED_KEY = "graphrag_handoff_confirmed"


def confirm_graphrag_handoff(tool_context: ToolContext) -> ToolResult:
    """Record that the user has explicitly agreed to leave the retrieval agent.

    Call this only when the user has said so in their own words in this turn --
    never on an inference that they sound finished, and never to pre-authorise a
    handoff you expect them to want. Call 'finished' in the same reply.
    """
    tool_context.state[GRAPHRAG_HANDOFF_CONFIRMED_KEY] = True
    return tool_success(GRAPHRAG_HANDOFF_CONFIRMED_KEY, True)
