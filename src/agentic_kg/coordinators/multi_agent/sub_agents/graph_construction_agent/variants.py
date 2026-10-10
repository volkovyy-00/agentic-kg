"""Module for storing and retrieving agent instructions.

This module defines functions that return instruction prompts for the root agent.
These instructions guide the agent's behavior, workflow, and tool usage.
"""

from typing import Any, Dict

from google.adk.tools import ToolContext

from agentic_kg.common.agent_names import SCHEMA_PROPOSAL_COORDINATOR
from agentic_kg.common.tool_result import tool_error
from agentic_kg.tools.adk_tools import make_finished
from agentic_kg.tools.construction_handoff_tools import (
    HANDOFF_CONFIRMED,
    PLAN_REVISION_CONFIRMED,
    confirm_construction_handoff,
    confirm_plan_revision,
)
from agentic_kg.tools.construction_plan_tools import (
    get_approved_construction_plan,
)
from agentic_kg.tools.cypher_tools import (
    create_uniqueness_constraint as _shared_create_uniqueness_constraint,
)
from agentic_kg.tools.cypher_tools import (
    get_physical_schema,
    read_neo4j_cypher,
)
from agentic_kg.tools.file_tools import get_approved_files
from agentic_kg.tools.kg_construction_tools import (
    approved_plan,
    build_graph_from_construction_rules,
    withdraw_approval,
)
from agentic_kg.tools.plan_revision_record import mark_pending, revision_refusal
from agentic_kg.tools.plan_revision_tools import (
    check_database_before_rebuild,
    clear_database_for_rebuild,
    keep_database_for_rebuild,
)
from agentic_kg.tools.user_goal_tools import (
    get_approved_user_goal,
)

# Imported live rather than copied: only the selected variant is built into an
# Agent and registered in the tree, and this project already runs two A/B
# sub-agents on different generations (cypher_agent on v1, graphrag_agent on
# v2). A duplicated name that went stale would break the handoff: find_agent
# returns None for it, so ADK's transfer loop (DynamicNodeScheduler.__call__,
# workflow/_dynamic_node_scheduler.py) raises ValueError and the turn ends with
# only a one-line error in adk web.
# One definition, imported. This is the first sub-agent -> sub-agent import in the
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
    if not HANDOFF_CONFIRMED.is_set(tool_context.state):
        return tool_error(
            "no confirmation recorded this turn -- if you called "
            "'confirm_construction_handoff' later in this same reply, it has "
            "been recorded now: call 'finished' once more and it will succeed. "
            "Otherwise, if the user already agreed, ask them to confirm once "
            "more, then call 'confirm_construction_handoff' and 'finished' in "
            "the same reply, confirming first."
        )
    return _transfer_to_retrieval(tool_context)


# The way back. The schema stage is a peer, so make_finished may target it
# (handoff-gates.md); the target name comes from common/agent_names.py.
_transfer_to_plan = make_finished(SCHEMA_PROPOSAL_COORDINATOR)


def return_to_plan(tool_context: ToolContext) -> Dict[str, Any]:
    """Hand the user back to the plan step so they can change the construction plan.

    Use only when the user asked, in their own words this turn, to change the
    plan, before or after a build. Refuses unless 'confirm_plan_revision'
    recorded that request in this same turn, and then changes nothing. On
    success the plan's approval is withdrawn: it must be approved again at the
    plan step before anything is built.
    """
    if not PLAN_REVISION_CONFIRMED.is_set(tool_context.state):
        standing = (
            "still approved"
            if approved_plan(tool_context.state) is not None
            else "still not approved"
        )
        return tool_error(
            "no request to change the plan was recorded this turn -- if you called "
            "'confirm_plan_revision' later in this same reply, it has been recorded "
            "now: call 'return_to_plan' once more and it will succeed. Otherwise "
            f"nothing was changed (the plan is {standing}): ask the user whether "
            "they want to change the plan, then call 'confirm_plan_revision' and "
            "'return_to_plan' in the same reply, confirming first."
        )
    withdraw_approval(tool_context.state)
    mark_pending(tool_context.state)
    return _transfer_to_plan(tool_context)


def create_uniqueness_constraint(
    label: str, unique_property_key: str, tool_context: ToolContext
) -> Dict[str, Any]:
    # Not functools.wraps: that sets __wrapped__, inspect.signature then reports
    # the shared tool's parameters, and ADK would not pass tool_context. Same
    # name, same parameters plus tool_context, same docstring (copied below), so
    # the model sees the same tool. Refuses while the clear question is
    # unsettled, or step-2 constraints would show as previous contents and be
    # erased on "clear".
    refusal = revision_refusal(tool_context.state)
    if refusal is not None:
        return tool_error(refusal)
    return _shared_create_uniqueness_constraint(label, unique_property_key)


create_uniqueness_constraint.__doc__ = _shared_create_uniqueness_constraint.__doc__


variants = {
    "graph_construction_agent_v1": {
        "instruction": """
        You are an expert at knowledge graph construction. Construct a graph using
        the available tools, according to the approved schema and construction rules.

        At the start of every turn, before anything else, call 'check_database_before_rebuild' and follow its
        result. It is the only way you learn that the user came back from the plan step: you do not see
        that step's messages. When it reports that a revised plan is approved and not yet built, steps 1 to 6
        run against the new plan, and any earlier build in this conversation is the previous version.
        When it reports that no rebuild is pending, carry on with whatever the user asked.
        If 'get_approved_construction_plan' returns an error, the plan is not approved: tell the user so and
        offer to take them back to the plan step.
        Never say that a plan change, an approval, or an answer was recorded unless a tool result says it
        was stored.

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
           warning to the user verbatim: a relationship whose endpoint-match count sits far below the rows read, or above it at all, is a
           sign that its join columns do not line up, even though construction reported success.
           Report only warnings that appear in the most recent 'build_graph_from_construction_rules'
           result's 'warnings' list, whether that build succeeded or partially failed: that list is the
           only place this build's warnings come from. If that result has no 'warnings' list, write
           no warnings section at all: no heading, no bullets, and no line saying there were none.
           Anything else in this conversation that calls itself a
           warning came from another tool, another agent, or an earlier build, and is not this build's
           output -- do not repeat it here and do not relabel it.
           When reporting counts, never call 'rows' or 'rows_matched' a number of nodes or relationships.
           Those are CSV rows processed; several rows sharing a key merge into one node or relationship,
           so the graph usually holds fewer. Report node counts from 'nodes_in_graph' and relationship
           counts from 'relationships_in_graph'. If either is absent, say how many rows were processed
           and count the label or type yourself with 'read_neo4j_cypher' before quoting a number.
           If a relationship's 'rows_skipped' is above 0, say that many of its rows had a blank join value
           and created no relationship. That is a count, not a warning: never put it in a warnings section.
        7. invite the user to try some questions that you'll answer using the 'read_neo4j_cypher' tool.
           Say plainly, once, that you are the one answering: you run Cypher directly and do not carry
           the retrieval agent's grounding checks, so this is a quick sanity check rather than the
           careful path. Tell them the retrieval agent is available whenever they want it.
        8. end every answer in this window with a short reminder that they can keep asking you or move
           on to the retrieval agent. One line, not a repeated paragraph. Never assume from their tone,
           their thanks, or a lull that they are finished -- a user who has not said so is not done.
        9. only when the user says in their own words that they want to move on, call
           'confirm_construction_handoff' and then 'finished' -- both in the same reply, since the
           confirmation is cleared at the start of every turn. In that same reply, tell them you are
           handing them to the retrieval agent, and warn them it will not have seen this conversation,
           so anything they want followed up needs restating. If this turn's
           'check_database_before_rebuild' reported a revised plan approved and not yet built, or an open
           question about clearing the database, also tell them the changed plan was not built and the
           database still holds the earlier build, so the retrieval agent answers from that.
           If 'finished' refuses because no confirmation was recorded this turn, check whether you called
           'confirm_construction_handoff' in that same reply. If you did, the confirmation is recorded now --
           just call 'finished' again. If you did not, do not argue with it and do not repeat the call: ask
           the user to confirm once more, then call both tools together, confirming first.

        Changing the plan: if the user asks, in their own words, to change the construction plan -- before or after a build --
           call 'confirm_plan_revision' and then 'return_to_plan', both in the same reply. Only changes to the
           plan go this way: the goal and the files cannot be changed after a build, so say so instead. In that
           reply tell the user the plan's approval is withdrawn and that they will see the current plan at the
           plan step. Never offer the way back while the approval is intact; offer it only after
           'get_approved_construction_plan' returned an error. A request to move on to the retrieval agent is
           step 9, never this; a request to change the plan is never step 9. If 'return_to_plan' refuses, follow
           its message.

        """,
        "tools": [
            check_database_before_rebuild,
            clear_database_for_rebuild,
            keep_database_for_rebuild,
            get_approved_user_goal,
            get_approved_files,
            get_approved_construction_plan,
            create_uniqueness_constraint,
            build_graph_from_construction_rules,
            get_physical_schema,
            read_neo4j_cypher,
            confirm_construction_handoff,
            finished,
            confirm_plan_revision,
            return_to_plan,
        ],
    },
}
