"""Turns the enriched Neo4j schema into annotated, honestly-labelled facts.

neo4j_graphrag's get_structured_schema(is_enhanced=True) reports what values a
property holds, but says nothing about whether that report is complete. Its
exhaustive branch emits `values` plus a true `distinct_count`; its sampled
branch (anything above EXHAUSTIVE_SEARCH_LIMIT = 10000 entities) emits `values`
from five rows with no `distinct_count` at all. Handed to a model unlabelled,
five sampled values read exactly like the whole truth.

Every annotation here is therefore tri-state and ALWAYS present. An absent key
would read as "fine", and the entities we cannot annotate are precisely the
large, unfamiliar ones where a confident wrong answer is most likely.

A third library branch needs no special case. Where a property has a RANGE
index whose own statistics report <= DISTINCT_VALUE_LIMIT distinct values, the
library reads the complete distinct set from the index instead of sampling rows
(neo4j_graphrag/schema.py:546-564) and emits `distinct_count` despite not being
row-exhaustive. That path always yields len(values) == distinct_count, so the
comparison below classifies it "complete", which is correct by construction. It
cannot arise here today -- the only index creation is
create_uniqueness_constraint, on ID properties -- but a reader diffing this file
against schema.py will find that branch and wonder why it is unhandled.
"""
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Mirrors neo4j_graphrag.schema.DISTINCT_VALUE_LIMIT. Above it the library
# truncates its own `values` list, so per-value counts would be partial
# regardless -- exactly the misleading half-complete output these annotations
# exist to prevent.
VALUE_COUNT_MAX_DISTINCT = 10

_NUMERIC_TYPES = {"INTEGER", "FLOAT"}

# Spots values that *look* numeric while being stored as text -- which is where
# silent lexicographic ordering comes from ('9' sorts after '30').
#
# The currency class is explicit rather than "any non-digit". A wildcard prefix
# matches ordinary identifier shapes -- 'a1', 'x9', 'Q3', '#5' all pass -- and
# flagging an identifier column as numeric would tell the agent (prompt rule 5)
# to cast a key to a number.
_NUMERIC_LIKE = re.compile(
    r"^\s*[$€£¥₹]?\s*[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s*$")


def _is_numeric_like(values) -> bool:
    if not values:
        return False
    return all(isinstance(v, str) and _NUMERIC_LIKE.match(v) for v in values)


def annotate_property(prop: Dict[str, Any], entity_count: Optional[int]) -> Dict[str, Any]:
    """Annotate one property dict from the enriched schema.

    Works identically for node and relationship properties -- the library
    returns the same shape under "node_props" and "rel_props". `entity_count`
    is the node count for a label or the edge count for a relationship type;
    pass None when it is unavailable, which yields "unknown" rather than a
    guess.

    Returns a new dict; the input is never mutated.
    """
    out = dict(prop)
    values = prop.get("values")
    distinct_count = prop.get("distinct_count")

    if distinct_count is None:
        # Sampled branch: five arbitrary rows, completeness unknowable. Drop
        # the values rather than present a sample as if it were the set.
        out.pop("values", None)
        out["completeness"] = "unknown"
    elif values is not None and len(values) < distinct_count:
        out["completeness"] = "partial"
    else:
        out["completeness"] = "complete"

    if distinct_count is None or entity_count is None:
        out["uniqueness"] = "unknown"
    elif distinct_count >= entity_count:
        out["uniqueness"] = "unique"
    else:
        out["uniqueness"] = "non_unique"

    # Tri-state, for the same reason as the other two. An earlier draft
    # computed this from `values` even on the sampled branch -- i.e. from the
    # five rows we had just discarded as untrustworthy -- so a property could
    # carry completeness "unknown" alongside numeric_like True. That is the one
    # place the design contradicted its own rule, and it fed prompt rule 5,
    # which tells the agent to cast.
    # All three annotations are strings from the same tri-state vocabulary.
    # A bool/str mix would put a TRUTHY "unknown" next to False, so any future
    # `if prop["numeric_like"]:` would read "I don't know" as "yes, cast it" --
    # the original bug wearing a new coat.
    prop_type = prop.get("type")
    if prop_type in _NUMERIC_TYPES:
        out["numeric_like"] = "no"           # already numeric; nothing to infer
    elif out["completeness"] == "unknown":
        out["numeric_like"] = "unknown"      # sampled values are not evidence
    else:
        out["numeric_like"] = "yes" if _is_numeric_like(values) else "no"

    return out
