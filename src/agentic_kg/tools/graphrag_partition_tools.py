"""The partition-interpretation disclosure gate for the retrieval phase.

graphrag_agent_v2's instruction told the model to silently judge whether a
partitioned_by property's numeric values are a quantity (sum them) or a set
of kinds (split them), and to say which -- nothing enforced the "say" half,
and the judgment quietly stopped being stated by a session's third
aggregating question (KG-5). This module makes the judgment a real,
reviewable fact in the tool-call record instead of trusting it to survive in
prose: 'declare_partition_interpretation' is the only way to set the flag
that 'sub_agents/graphrag_agent/variants.py's gated read tool checks before
it will run an aggregating query over a graph that has such a property.

Deliberately not shared with 'graphrag_handoff_tools.py' despite the
identical flag/reset/gate shape -- this is the third instance of that
pattern, and per the construction-handoff design's own Rule-of-Three
reasoning, extraction is the copy-#3 decision, made when a fourth instance
is needed, not before.
"""

from google.adk.tools import ToolContext

from agentic_kg.common.graph_profile import (
    numeric_partitioned_properties,
    peek_cached_profile,
)
from agentic_kg.common.tool_result import ToolResult, tool_error, tool_success

PARTITION_INTERPRETATION_DECLARED_KEY = "partition_interpretation_declared"

# The escape hatch for the gate's coarse trigger: it fires on any aggregating
# query while ANY numeric-flagged property exists anywhere in the graph, not
# only queries that actually touch one. This sentinel lets the model clear
# the gate in one line when that is a false alarm, rather than being stuck.
NONE_APPLY_SENTINEL = "none"


def declare_partition_interpretation(
    property: str, reading: str, tool_context: ToolContext
) -> ToolResult:
    """Record how a numeric partitioned_by property's values are being read this turn.

    Call this before running an aggregating query (sum, count, avg, collect,
    min, or max) over a relationship pattern whose partitioned_by lists a
    property with values_are 'numbers'. Name that property and state whether
    you are treating its values as a total to sum or as separate kinds to
    split -- or pass 'none' for property if the query you are about to run
    does not touch any currently flagged property.

    'property' must name one of the properties the graph's current profile
    actually flags as numeric-partitioned, or be 'none'; anything else is
    refused, so this cannot be satisfied by naming something that was never
    actually flagged.
    """
    flagged = numeric_partitioned_properties(peek_cached_profile())
    normalized = property.strip()
    if normalized.lower() != NONE_APPLY_SENTINEL and normalized not in flagged:
        return tool_error(
            f"'{property}' is not a currently numeric-flagged partitioned_by "
            "property. The properties flagged this way right now are: "
            f"{', '.join(flagged) if flagged else '(none)'}. Pass one of "
            "those, or 'none' if this query does not touch any of them."
        )
    tool_context.state[PARTITION_INTERPRETATION_DECLARED_KEY] = True
    return tool_success(PARTITION_INTERPRETATION_DECLARED_KEY, True)
