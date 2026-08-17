"""Module for storing and retrieving agent instructions.

This module defines functions that return instruction prompts for the graphrag
(retrieval) agent. These instructions guide the agent's behavior, workflow, and
tool usage.
"""

import re
from typing import Any, Callable, Dict, Optional

from google.adk.tools import ToolContext

from agentic_kg.common.adk_context import drop_foreign_context
from agentic_kg.common.agent_names import MULTI_AGENT_COORDINATOR
from agentic_kg.common.graph_profile import (
    numeric_partitioned_properties,
    peek_cached_profile,
)
from agentic_kg.common.tool_result import tool_error
from agentic_kg.tools.adk_tools import make_finished
from agentic_kg.tools.cypher_tools import (
    get_graph_schema_with_profile,
    get_physical_schema,
    read_neo4j_cypher,
)
from agentic_kg.tools.graphrag_handoff_tools import (
    GRAPHRAG_HANDOFF_CONFIRMED_KEY,
    confirm_graphrag_handoff,
)
from agentic_kg.tools.graphrag_partition_tools import (
    PARTITION_INTERPRETATION_DECLARED_KEY,
    declare_partition_interpretation,
)

# v1's exit, ungated and unchanged in behaviour. v1 is a controlled A/B
# variable (see agent.py) -- gating its exit would add handoff mechanics as a
# second variable to a comparison whose point is isolating one.
#
# The rename matters: this used to be called 'finished' and was the SAME OBJECT
# in both variants' tools lists, so gating it in place would have left v1 with
# a gate and no confirm tool, unable to end its phase at all.
#
# It still presents to the model as a tool named 'finished' -- make_finished
# returns a closure literally defined as 'def finished(...)', and ADK reads
# __name__. That is pinned by a test, because no other agent in this tree lists
# a renamed make_finished result as a tool.
_transfer_to_coordinator = make_finished(MULTI_AGENT_COORDINATOR)


def finished(tool_context: ToolContext) -> Dict[str, Any]:
    """Finish retrieval and hand the user back to the coordinator.

    Refuses unless 'confirm_graphrag_handoff' recorded the user's explicit
    agreement in this same turn. Returns a bare {} on success, matching every
    other 'finished' in this codebase; the error path is the only one that
    speaks ToolResult.

    v2 only. v1 holds '_transfer_to_coordinator' directly.
    """
    if not tool_context.state.get(GRAPHRAG_HANDOFF_CONFIRMED_KEY):
        return tool_error(
            "no confirmation recorded this turn -- if you called "
            "'confirm_graphrag_handoff' later in this same reply, it has been "
            "recorded now: call 'finished' once more and it will succeed. "
            "Otherwise, if the user already agreed, ask them to confirm once "
            "more, then call 'confirm_graphrag_handoff' and 'finished' in the "
            "same reply, confirming first."
        )
    return _transfer_to_coordinator(tool_context)


_AGGREGATE_KEYWORD_RE = re.compile(
    r"\b(sum|count|avg|collect|min|max)\s*\(", re.IGNORECASE
)


def make_gated_read_neo4j_cypher(
    read_neo4j_cypher_impl: Callable[..., Dict[str, Any]] = read_neo4j_cypher,
) -> Callable[..., Dict[str, Any]]:
    """Build the v2-only gated 'read_neo4j_cypher'.

    Presents to the model under that same name -- this is a wrapper, not a
    new tool -- so the instruction text naming 'read_neo4j_cypher' by name
    stays literally true without a rename, the same reasoning make_finished
    already documents for itself.

    read_neo4j_cypher_impl is bound as a default-argument value, evaluated
    once at THIS function's definition time -- not referenced as a free
    variable inside the closure below. The inner 'def read_neo4j_cypher'
    rebinds that name in this factory's own local scope, so a free-variable
    reference to it from inside the closure body would resolve, via Python's
    scoping rules, to the closure itself rather than to the module-level
    import: infinite self-recursion, not a call to the original. Binding it
    as a default argument sidesteps that entirely.
    """

    def read_neo4j_cypher(
        query: str,
        tool_context: ToolContext,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submits a read-only Cypher query to a Neo4j database.

        Args:
            query: The Cypher query string to execute.
            tool_context: ToolContext object.
            params: Optional parameters to pass to the query.

        Returns:
            A dictionary with "status" and, on success, "query_result" holding
            "records", "row_count" (or "row_count_at_least"), "truncated", and
            "values_summarised". Counts and rankings must come from a Cypher
            aggregation, never from counting the returned records.

            Refuses (status "error") if this query aggregates -- calls sum,
            count, avg, collect, min, or max -- and the graph's current
            profile flags at least one partitioned_by property as numeric,
            unless 'declare_partition_interpretation' has already been called
            this turn. Call that tool first -- naming the flagged property and
            how you are reading it, or 'none' if this query does not touch
            one -- then resend this same query.
        """
        profile = peek_cached_profile()
        if profile is not None:
            flagged = numeric_partitioned_properties(profile)
            if (
                flagged
                and _AGGREGATE_KEYWORD_RE.search(query)
                and not tool_context.state.get(PARTITION_INTERPRETATION_DECLARED_KEY)
            ):
                return tool_error(
                    "this query aggregates data, and the graph's current "
                    "profile flags at least one numeric partitioned_by "
                    f"property ({', '.join(flagged)}) whose values could be "
                    "either a total or separate kinds -- call "
                    "'declare_partition_interpretation' first, naming the "
                    "property and how you are reading it (or 'none' if this "
                    "query does not touch one), then resend this same query."
                )
        return read_neo4j_cypher_impl(query, params)

    return read_neo4j_cypher


variants = {
    # graphrag_agent_v1
    "graphrag_agent_v1": {
        "instruction": """
        You are an expert at information retrieval from a knowledge graph.
        Your primary goal is to help the user find information in the knowledge graph
        by using a range of tools.

        Tools:
        - get_physical_schema: get the nodes, relationships and available properties of the graph
        - read_neo4j_cypher: run a cypher query and return the results. always get the schema first to understand the graph structure
        - finished: signal that the user is done with the graphrag agent

        Think step-by-step each time a user asks a question:
        1. Always start by using the 'get_physical_schema' tool to understand the graph schema
        2. Consider whether a specialized tool is the best way to answer the user's question
        3. If a specialized tool is not available, take time reasoning about the schema before running a cypher query with 'read_neo4j_cypher'
        """,
        "tools": [get_physical_schema, read_neo4j_cypher, _transfer_to_coordinator],
    },
    # graphrag_agent_v2 -- grounded in the graph rather than in the transcript.
    # v1 is retained unchanged so the acceptance run can A/B the two.
    "graphrag_agent_v2": {
        "instruction": """
        You are an expert at information retrieval from a knowledge graph.
        Your goal is to answer the user's questions using only what the graph says.

        Tools:
        - get_graph_schema_with_profile: the graph's structure plus a profile of
          its data -- how many of each entity, how relationship patterns spread
          across the nodes at each end, and for each property whether its values
          are complete, whether it uniquely identifies its entity, and how those
          values are distributed
        - read_neo4j_cypher: run a read-only Cypher query
        - confirm_graphrag_handoff: record that the user has said, in their own
          words, that they are done asking questions
        - finished: leave the retrieval phase and hand the user back to the
          coordinator. Refuses unless 'confirm_graphrag_handoff' was called in
          the same reply

        The graph is the only source of truth about the data. Every count, name,
        membership and ranking you state must come from a query result in this
        turn. If you find yourself recalling a fact from earlier in the
        conversation, query for it again instead -- it is cheap, and what you
        remember may describe a graph that has since changed.

        For each question:
        1. Call 'get_graph_schema_with_profile' first.
        2. Before writing a query over a relationship pattern, read that
           pattern's 'partitioned_by'. Each entry names a property that may
           divide those edges into kinds, with three qualifiers:
           - 'values_are': 'categories' means the values name kinds and this
             rule binds. 'numbers' means they may be quantities rather than
             kinds -- decide from the question which they are and say what you
             decided. 'unknown' means the graph was too large to enumerate
             them, so assume a split may exist and check it with a query.
           - 'distribution': the count per kind, or 'unknown' when the values
             could not be enumerated.
           - 'distribution_covers': 'this_pattern' means those counts are exact
             for this pattern. 'all_patterns_of_this_type' means they are
             pooled over every pattern using this relationship type, so treat
             them only as evidence that a split exists and query this pattern's
             own counts before quoting a number.
           Where the property names kinds, you may not aggregate over all of
           them as if they were one thing. Either add a WHERE clause fixing the
           property to the one kind the question is about, or return that
           property as a grouping key and report the kinds separately -- then
           state in your answer which you did and why.
           If the kinds rank differently from the pooled total, the pooled
           ranking is not a valid summary of them and must not be your
           conclusion. Lead with the split, and give the pooled figure only as
           context, saying plainly that it merges kinds that disagree. This
           binds the sentence you conclude with, not just the table above it:
           name only what the split supports. A leader on the pooled total that
           leads in no kind is an artefact of the merge -- say that, rather
           than crowning it and qualifying it afterwards.
        3. Say what you are counting and over what before you query. If a
           relationship pattern's degree shows more than one edge per node, a
           result row is not one node -- decide which subset you mean and say so.
        4. Counts, rankings and superlatives must come from a Cypher aggregation,
           never from counting the rows you got back. Report ties as ties rather
           than reading a ranking off row order.
        5. A property whose 'uniqueness' is 'non_unique' does not identify one
           entity: rows sharing a value are several entities. Grouping by it is
           fine when you mean the group -- say that is what you are reporting.
           Do not present such a group as a single entity, or attach a count of
           entities to it as if it were one. Where 'uniqueness' is 'unknown',
           say so rather than proceeding as if it were unique.
        6. Before ordering, comparing or aggregating a STRING property
           numerically, check the profile's 'numeric_like': 'yes' means a plain
           cast works, 'numeric_after_cleaning' means the values carry a currency
           symbol or thousands separator and a plain cast returns null -- strip
           those characters first -- 'no' means do not cast, and 'unknown' is
           something to disclose, never a 'yes'. Without a cast '9' sorts after
           '30'. INTEGER and FLOAT properties need no cast; 'numeric_like' does
           not apply to them.
        7. Where an annotation reads 'unknown' or 'not_profiled', treat it as
           missing information to disclose, never as permission to assume.
        8. end every answer with a short reminder that they can keep asking
           questions. One line, not a repeated paragraph. Never assume from
           their tone, their thanks, or a lull that they are finished -- a user
           who has not said so is not done.
        9. only when the user says in their own words that they are done, call
           'confirm_graphrag_handoff' and then 'finished' -- both in the same
           reply, since the confirmation is cleared at the start of every turn.
           In that same reply, tell them you are handing them back to the
           coordinator, which is where they go to start new work on the graph.
           If 'finished' refuses because no confirmation was recorded this turn,
           check whether you called 'confirm_graphrag_handoff' in that same
           reply. If you did, the confirmation is recorded now -- just call
           'finished' again. If you did not, do not argue with it and do not
           repeat the call: ask the user to confirm once more, then call both
           tools together, confirming first.
        """,
        "tools": [
            get_graph_schema_with_profile,
            make_gated_read_neo4j_cypher(),
            confirm_graphrag_handoff,
            declare_partition_interpretation,
            finished,
        ],
        "before_model_callback": drop_foreign_context,
    },
}
