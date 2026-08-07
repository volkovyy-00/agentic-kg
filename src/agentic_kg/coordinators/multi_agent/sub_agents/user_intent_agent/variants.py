
"""Module for storing and retrieving agent instructions.

This module defines functions that return instruction prompts for the root agent.
These instructions guide the agent's behavior, workflow, and tool usage.
"""

from typing import Any, Dict

from google.adk.tools import ToolContext

from agentic_kg.tools.user_goal_tools import (
    set_user_goal, get_user_goal,
    set_perceived_user_goal, approve_perceived_user_goal,
    APPROVED_USER_GOAL, PERCEIVED_USER_GOAL,
)
from agentic_kg.tools.adk_tools import make_finished
from agentic_kg.common.agent_names import MULTI_AGENT_COORDINATOR
from agentic_kg.common.tool_result import tool_error

# v1's exit, ungated and unchanged in behaviour. v1 predates the
# perceived/approved split entirely -- it uses set_user_goal, which writes
# 'user_goal'/'user_goal_approved' and never touches 'approved_user_goal' --
# so gating this in place would leave v1 unable to end its phase at all.
#
# It still presents to the model as a tool named 'finished': make_finished
# returns a closure literally defined as 'def finished(...)', and ADK reads
# __name__. That is pinned by a test.
_transfer_to_coordinator = make_finished(MULTI_AGENT_COORDINATOR)


def finished(tool_context: ToolContext) -> Dict[str, Any]:
    """Finish the user-intent phase and hand control back to the coordinator.

    Refuses unless the user's goal has been approved AND that approval is
    still current -- that is, 'approve_perceived_user_goal' has run since the
    most recent 'set_perceived_user_goal'. Equality, not mere presence:
    a goal approved and then revised leaves an approved key that no longer
    describes what the user asked for, and waving that through is the same
    defect this gate exists to close, one route over.

    The three branches are ordered and non-overlapping. A missing
    'perceived_user_goal' implies a missing 'approved_user_goal', because
    approve_perceived_user_goal reads the perceived key directly
    (user_goal_tools.py:59-60) and cannot write the approved one without it.

    Returns a bare {} on success, matching every other 'finished' in this
    codebase; the error paths are the only ones that speak ToolResult.

    v2 only. v1 holds '_transfer_to_coordinator' directly.
    """
    perceived = tool_context.state.get(PERCEIVED_USER_GOAL)
    approved = tool_context.state.get(APPROVED_USER_GOAL)

    if perceived is None:
        return tool_error(
            "no goal has been recorded yet. Use 'set_perceived_user_goal' to "
            "record what the user has told you, confirm it with them, then "
            "call 'approve_perceived_user_goal' and 'finished' in the same "
            "reply."
        )
    if approved is None:
        return tool_error(
            "the user's goal has not been approved. If the user has already "
            "agreed to the goal as you described it, call "
            "'approve_perceived_user_goal' and then 'finished' in the same "
            "reply, approving first. If they have not agreed yet, ask them -- "
            "and do not leave this phase until they have."
        )
    if approved != perceived:
        return tool_error(
            "the approved goal is out of date: 'set_perceived_user_goal' has "
            "recorded a different goal since the last approval. Ask the user "
            "to confirm the goal as it now stands, then call "
            "'approve_perceived_user_goal' and 'finished' in the same reply, "
            "approving first."
        )
    return _transfer_to_coordinator(tool_context)


variants = {
    # user_intent_agent_v1
    # Benefits:
    # - simple workflow
    "user_intent_agent_v1": {
        "instruction": """
        You are an expert at knowledge graph use cases. 
        Your primary goal is to help the user come up with a knowledge graph use case.
        Knowledge graph use cases appear in all industries. Wherever there is data, there's probably a graph.
        
        If the user is unsure where to do, make some suggestions based on classic use cases like:
        - social network involving friends, family, or profressional relationships
        - logistics network with suppliers, customers, and partners
        - recommendation system with customers, products, and purchase patterns
        - fraud detection over multiple accounts with suspicious patterns of transactions
        - pop-culture graphs with movies, books, or music

        You are required to set the user goal using the set_use_goal tool.

        A user goal has two components:
        - kind_of_graph: at most 3 words describing the graph, for example "social network" or "USA freight logistics"
        - description: a few sentences about the intention of the graph, for example "A dynamic routing and delivery system for cargo." or "Analysis of product dependencies and supplier alternatives."

        Think carefully and collaborate with the user:
        1. Understand the user's goal, which is a kind_of_graph with description
        2. Ask clarifying questions as needed
        3. Verify with the user what you think the kind_of_graph and description are
        4. If the user agrees, use the 'set_user_goal' tool to set the user goal.
        5. If the user is ready to continue, use the 'finished' tool        
        """,
        "tools": [get_user_goal, set_user_goal, _transfer_to_coordinator]
    },
    "user_intent_agent_v2": {
        "instruction": """
                You are an expert at knowledge graph use cases. 
        Your primary goal is to help the user come up with a knowledge graph use case.
        Knowledge graph use cases appear in all industries. Wherever there is data, there's probably a graph.
        
        If the user is unsure where to do, make some suggestions based on classic use cases like:
        - social network involving friends, family, or profressional relationships
        - logistics network with suppliers, customers, and partners
        - recommendation system with customers, products, and purchase patterns
        - fraud detection over multiple accounts with suspicious patterns of transactions
        - pop-culture graphs with movies, books, or music

        A user goal has two components:
        - kind_of_graph: at most 3 words describing the graph, for example "social network" or "USA freight logistics"
        - description: a few sentences about the intention of the graph, for example "A dynamic routing and delivery system for cargo." or "Analysis of product dependencies and supplier alternatives."

        Think carefully and collaborate with the user:
        1. Understand the user's goal, which is a kind_of_graph with description
        2. Ask clarifying questions as needed. Stay here while you wait for the answers --
           never hand off to another agent with a question of yours outstanding.
        3. When you think you understand their goal, use the 'set_perceived_user_goal' tool to record it
        4. Verify with the user that the perceived user goal matches their expectations
        5. When the user agrees, use the 'approve_perceived_user_goal' tool. The approval IS this
           tool call. Saying that the goal is approved approves nothing, and no other agent can
           record it for you.
        6. Finally, use the 'finished' tool to signal completion of the user intent agent.
           It will refuse until an approval has been recorded. If the user revises their goal
           after approving it, call 'approve_perceived_user_goal' again before 'finished'.
        """,
        "tools": [set_perceived_user_goal, approve_perceived_user_goal, finished]
    }
}
