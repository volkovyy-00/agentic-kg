"""The explicit-handoff gate for the retrieval phase.

`graphrag_agent_v2` answers questions over the finished graph, and used to
decide for itself when that window was over -- ejecting the user back to the
coordinator after a single answer, without ever being told they were done
(see CHANGELOG.md's "Explicit retrieval handoff (#9)" entry). Leaving the end
of that window to the model's reading of "the user seems satisfied" is what
this module removes: the flag below is set only by an explicit tool call, and
the retrieval agent's own `finished` wrapper refuses to transfer without it.

Its confirm tool, gated 'finished' and refusal text deliberately stay its own
rather than shared with `construction_handoff_tools.py`: the transfer target
differs (construction hands the user sideways to a live-imported sibling,
retrieval hands them up to the coordinator), and so does the `finished`
docstring, which ADK shows the model as the tool's description. Only the key
and its set/read/reset plumbing are shared, through `TurnFlag`
(common/turn_flags.py).

The key is spelled here, once. The agent's agent.py (which clears it every
turn) and its variants.py (which reads it) both import
`GRAPHRAG_HANDOFF_CONFIRMED` rather than retyping the string, and `TurnFlag`
(common/turn_flags.py) holds the plumbing. It is deliberately not named
`HANDOFF_CONFIRMED_KEY`: that name already means something else, with a
different value, one module over.
"""

from google.adk.tools import ToolContext

from agentic_kg.common.tool_result import ToolResult, tool_success
from agentic_kg.common.turn_flags import TurnFlag

GRAPHRAG_HANDOFF_CONFIRMED_KEY = "graphrag_handoff_confirmed"
GRAPHRAG_HANDOFF_CONFIRMED = TurnFlag(GRAPHRAG_HANDOFF_CONFIRMED_KEY)


def confirm_graphrag_handoff(tool_context: ToolContext) -> ToolResult:
    """Record that the user has explicitly agreed to leave the retrieval agent.

    Call this only when the user has said so in their own words in this turn --
    never on an inference that they sound finished, and never to pre-authorise a
    handoff you expect them to want. Call 'finished' in the same reply.
    """
    GRAPHRAG_HANDOFF_CONFIRMED.set(tool_context.state)
    return tool_success(GRAPHRAG_HANDOFF_CONFIRMED_KEY, True)
