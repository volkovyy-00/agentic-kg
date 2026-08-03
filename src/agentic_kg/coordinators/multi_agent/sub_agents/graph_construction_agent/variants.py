
"""Module for storing and retrieving agent instructions.

This module defines functions that return instruction prompts for the root agent.
These instructions guide the agent's behavior, workflow, and tool usage.
"""

from agentic_kg.tools.user_goal_tools import (
    get_user_goal,  get_approved_user_goal, 
)
from agentic_kg.tools.construction_plan_tools import (
    get_approved_construction_plan, 
)
from agentic_kg.tools.cypher_tools import (
    read_neo4j_cypher, write_neo4j_cypher, create_uniqueness_constraint, 
    get_physical_schema,
)
from agentic_kg.tools.file_tools import get_approved_files
from agentic_kg.tools.kg_construction_tools import build_graph_from_construction_rules
from typing import Any, Dict

from google.adk.tools import ToolContext

from agentic_kg.common.tool_result import tool_error
from agentic_kg.tools.adk_tools import make_finished
from agentic_kg.tools.construction_handoff_tools import (
    HANDOFF_CONFIRMED_KEY, confirm_construction_handoff,
)
# Imported live rather than copied: only the selected variant is built into an
# Agent and registered in the tree, and this project already runs two A/B
# sub-agents on different generations (cypher_agent on v1, graphrag_agent on
# v2). A duplicated name that went stale would make find_agent raise inside the
# transfer chain -- no trace span, indistinguishable from a hang. One
# definition, imported. This is the first sub-agent -> sub-agent import in the
# tree; graphrag_agent/agent.py imports nothing that leads back here.
from ..graphrag_agent.agent import AGENT_NAME as GRAPHRAG_AGENT_NAME

# Construction hands the user straight to retrieval, not back to the
# coordinator: the coordinator has no new information at that point, and
# routing the user's just-confirmed decision through another model is the
# inference this gate exists to remove.
_transfer_to_retrieval = make_finished(GRAPHRAG_AGENT_NAME)


def finished(tool_context: ToolContext) -> Dict[str, Any]:
    """Finish construction and hand the user to the retrieval agent.

    Refuses unless 'confirm_construction_handoff' recorded the user's explicit
    agreement in this same turn. Returns a bare {} on success, matching every
    other 'finished' in this codebase; the error path is the only one that
    speaks ToolResult.
    """
    if not tool_context.state.get(HANDOFF_CONFIRMED_KEY):
        return tool_error(
            "no confirmation recorded this turn -- if the user already agreed, "
            "ask them to confirm once more, then call "
            "'confirm_construction_handoff' and 'finished' in the same reply."
        )
    return _transfer_to_retrieval(tool_context)

variants = {
    "graph_construction_agent_v1": {
        "instruction": """
        You are an expert at knowledge graph construction. Construct a graph using
        the available tools, according to the approved schema and construction rules.

        Before beginning construction, make sure you know the user goal, 
        approved files, approved schema and construction rules.
        - Use the get_approved_user_goal to check the user goal
        - Use the get_approved_files to check the approved files
        - Use the get_approved_construction_plan to check the approved construction rules

        Follow these steps to construct a knowledge graph:
        1. check that the construction rules are valid by comparing the construction plan with the approved files and schema
        2. create appropriate constraints for every node construction using the 'create_uniqueness_constraint' tool
        3. use the 'build_graph_from_construction_rules' tool to build the graph
        4. verify that the graph has been built by comparing the physical schema with the approved schema using the 'read_neo4j_cypher' tool
        5. verify that the graph is reasonable by proposing a hypothetical question that reflects the user goal. try to answer it using the 'read_neo4j_cypher' tool
        6. summarize the state of the graph and your post-construction analysis to the user.
           If the 'build_graph_from_construction_rules' result includes a 'warnings' list, report every
           warning to the user verbatim: a relationship that matched far fewer rows than were read is a
           sign that its join columns do not line up, even though construction reported success.
           When reporting counts, never call 'rows' or 'rows_matched' a number of nodes or relationships.
           Those are CSV rows processed; several rows sharing a key merge into one node or relationship,
           so the graph usually holds fewer. Report node counts from 'nodes_in_graph' and relationship
           counts from 'relationships_in_graph'. If either is absent, say how many rows were processed
           and count the label or type yourself with 'read_neo4j_cypher' before quoting a number.
        7. invite the user to try some questions that you'll answer using the 'read_neo4j_cypher' tool
        8. when the user is satisfied, use the 'finished' tool to signal that this phase of graph construction is complete

        """,
        "tools": [
            get_approved_user_goal, get_approved_files, get_approved_construction_plan,
            create_uniqueness_constraint, build_graph_from_construction_rules,
            get_physical_schema, read_neo4j_cypher, 
            finished
        ]
    },
}
