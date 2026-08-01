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

from .neo4j_for_adk import get_graphdb
from .tool_result import is_success

logger = logging.getLogger(__name__)

graphdb = get_graphdb()

# Mirrors neo4j_graphrag.schema.DISTINCT_VALUE_LIMIT. Above it the library
# truncates its own `values` list, so per-value counts would be partial
# regardless -- exactly the misleading half-complete output these annotations
# exist to prevent.
VALUE_COUNT_MAX_DISTINCT = 10

# Cold profile cost is N + M + 2P + Q + 2 queries: one library scan per label
# and per relationship type, TWO degree queries per (start, type, end) pattern,
# one per qualifying property, plus the two entity-count queries.
# get_cached_profile adds one fingerprint query per call on top. On a large
# ingested corpus that is hundreds, in one synchronous tool call --
# indistinguishable from a hang in `adk web`. Profile the largest entities and
# mark the rest.
MAX_PROFILED_ENTITIES = 25

# Separate cap for the degree loop. Patterns and entities scale independently:
# a multi-label graph inflates patterns much faster than labels, because one
# relationship type spans every (start, end) label combination it touches.
MAX_PROFILED_PATTERNS = 25

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


def quote(name: str) -> str:
    """Backtick-quote an identifier that came from the database.

    Deliberately NOT common.cypher_identifiers.checked(): that guards against
    injection from model-supplied names and rejects anything which is not a
    bare identifier. These names are read out of the graph, so `Legal Entity`
    and `10-K` are perfectly legal and must survive. Escaping doubles any
    embedded backtick, which is Cypher's own convention.
    """
    return "`" + name.replace("`", "``") + "`"


class ProfileQueryError(RuntimeError):
    """A profile query failed. Caught per entity by build_profile."""


def _records(result) -> list:
    """Lenient accessor: a failure yields no rows.

    Only for queries whose absence is survivable -- entity counts and the
    fingerprint, where "unknown" is a valid answer.
    """
    if not is_success(result):
        return []
    return result.get("query_result", {}).get("records", [])


def _records_or_raise(result) -> list:
    """Strict accessor: a failure raises so per-entity isolation can act.

    Using the lenient accessor everywhere would make build_profile's
    "profile_error" branch unreachable -- a failed query would silently look
    like an empty result, which is the opposite of the requirement that one
    failing entity degrade only itself.
    """
    if not is_success(result):
        raise ProfileQueryError(result.get("error_message", "profile query failed"))
    return result.get("query_result", {}).get("records", [])


def _entity_counts(labels, rel_types) -> Dict[str, Optional[int]]:
    """One grouped query per direction rather than one query per entity."""
    counts: Dict[str, Optional[int]] = {name: None for name in [*labels, *rel_types]}

    for row in _records(graphdb.send_read_query(
        "MATCH (n) UNWIND labels(n) AS label "
        "RETURN label AS name, count(*) AS n", max_rows=None)):
        counts[row["name"]] = row["n"]

    for row in _records(graphdb.send_read_query(
        "MATCH ()-[r]->() RETURN type(r) AS name, count(r) AS n", max_rows=None)):
        counts[row["name"]] = row["n"]

    return counts


def _pattern_degree(start: str, rel_type: str, end: str) -> Dict[str, Any]:
    """Degree statistics for ONE (start, type, end) pattern.

    Keyed per pattern, never per relationship type. On a graph where one type
    spans several label pairs, pooled statistics describe no actual pattern,
    and min == max can hold across the pool while being false for every
    pattern in it -- reintroducing the exact grain error this profile exists
    to prevent.
    """
    # Two queries, one grouped at each end. Grouping by the start node yields
    # the start-degree stats *and*, as a by-product, the edge total (the sum of
    # per-node degrees) and the distinct start count -- so no separate
    # cardinality query is needed. Every figure below is scoped to this one
    # (start, type, end) pattern; pooling patterns of the same type is the bug
    # this whole keying decision exists to avoid.
    start_rows = _records_or_raise(graphdb.send_read_query(
        f"MATCH (a:{quote(start)})-[r:{quote(rel_type)}]->(:{quote(end)}) "
        "WITH a, count(r) AS d "
        "RETURN count(a) AS distinct_nodes, sum(d) AS edges, "
        "       min(d) AS lo, max(d) AS hi, avg(d) AS avg",
        max_rows=None))
    end_rows = _records_or_raise(graphdb.send_read_query(
        f"MATCH (:{quote(start)})-[r:{quote(rel_type)}]->(b:{quote(end)}) "
        "WITH b, count(r) AS d "
        "RETURN count(b) AS distinct_nodes, "
        "       min(d) AS lo, max(d) AS hi, avg(d) AS avg",
        max_rows=None))

    if not start_rows or start_rows[0].get("edges") in (None, 0):
        return {"edges": 0, "start_degree": "unknown", "end_degree": "unknown"}

    def _degree(row):
        if not row or row.get("lo") is None:
            return "unknown"
        return {"min": row["lo"], "max": row["hi"], "mean": round(row["avg"], 2)}

    start_row = start_rows[0]
    end_row = end_rows[0] if end_rows else {}

    return {
        "edges": start_row["edges"],
        "distinct_start": start_row["distinct_nodes"],
        "distinct_end": end_row.get("distinct_nodes"),
        "start_degree": _degree(start_row),
        "end_degree": _degree(end_row),
    }


def _value_counts(entity: str, prop_name: str, is_relationship: bool) -> Any:
    if is_relationship:
        query = (
            f"MATCH ()-[r:{quote(entity)}]->() "
            f"WITH r.{quote(prop_name)} AS value, count(*) AS n "
            "WHERE value IS NOT NULL RETURN value, n ORDER BY n DESC"
        )
    else:
        query = (
            f"MATCH (n:{quote(entity)}) "
            f"WITH n.{quote(prop_name)} AS value, count(*) AS n "
            "WHERE value IS NOT NULL RETURN value, n ORDER BY n DESC"
        )
    rows = _records_or_raise(graphdb.send_read_query(query, max_rows=None))
    if not rows:
        return "unknown"
    return {str(r["value"]): r["n"] for r in rows}


def _profile_entity(entity, props, entity_count, is_relationship):
    annotated = []
    for prop in props:
        out = annotate_property(prop, entity_count)
        distinct_count = prop.get("distinct_count")
        if distinct_count is not None and distinct_count <= VALUE_COUNT_MAX_DISTINCT:
            out["value_counts"] = _value_counts(
                entity, prop["property"], is_relationship)
        else:
            out["value_counts"] = "unknown"
        annotated.append(out)
    return annotated


def build_profile(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Compute entity counts, per-pattern degree, and annotated properties.

    Never raises. A failure profiling one entity marks that entry
    "profile_error" and leaves the rest intact, matching what the library
    already does per entity (neo4j_graphrag/schema.py:858-859).
    """
    node_props = schema.get("node_props", {}) or {}
    rel_props = schema.get("rel_props", {}) or {}
    relationships = schema.get("relationships", []) or []

    counts = _entity_counts(list(node_props), list(rel_props))

    entities = [(name, props, False) for name, props in node_props.items()]
    entities += [(name, props, True) for name, props in rel_props.items()]
    # Largest first, so a budget cut drops the entities we could say least
    # about anyway rather than an arbitrary slice.
    entities.sort(key=lambda e: counts.get(e[0]) or 0, reverse=True)

    properties: Dict[str, Any] = {}
    profiled = 0
    for name, props, is_rel in entities:
        if profiled >= MAX_PROFILED_ENTITIES:
            properties[name] = "not_profiled"
            continue
        try:
            properties[name] = _profile_entity(name, props, counts.get(name), is_rel)
        except Exception:
            # Includes ProfileQueryError. Deliberately broad: a profile is a
            # convenience, and no failure inside it may take down the schema
            # tool that graphrag is instructed to call first.
            logger.exception("Profiling failed for entity %s; continuing", name)
            properties[name] = "profile_error"
        profiled += 1

    # The budget covers patterns too. Gating only the property loop would leave
    # the 2P degree queries unbounded, so "cold-start cost is bounded" would
    # hold for Q alone -- which is not what the spec claims. Largest patterns
    # first, by the edge count of their relationship type.
    ranked = [r for r in relationships
              if r.get("start") and r.get("type") and r.get("end")]
    ranked.sort(key=lambda r: counts.get(r["type"]) or 0, reverse=True)

    patterns = []
    for index, rel in enumerate(ranked):
        start, rel_type, end = rel["start"], rel["type"], rel["end"]
        entry = {"pattern": f"{start}-[{rel_type}]->{end}",
                 "start": start, "type": rel_type, "end": end}
        if index >= MAX_PROFILED_PATTERNS:
            entry["start_degree"] = "not_profiled"
            entry["end_degree"] = "not_profiled"
            patterns.append(entry)
            continue
        try:
            entry.update(_pattern_degree(start, rel_type, end))
        except Exception:
            logger.exception("Degree profiling failed for %s; continuing", entry["pattern"])
            entry["start_degree"] = "profile_error"
            entry["end_degree"] = "profile_error"
        patterns.append(entry)

    return {
        "entity_counts": counts,
        "patterns": patterns,
        "properties": properties,
        "budget": {
            "entities_profiled": min(profiled, MAX_PROFILED_ENTITIES),
            "entities_skipped": max(0, len(entities) - MAX_PROFILED_ENTITIES),
            "entity_limit": MAX_PROFILED_ENTITIES,
            "patterns_profiled": min(len(ranked), MAX_PROFILED_PATTERNS),
            "patterns_skipped": max(0, len(ranked) - MAX_PROFILED_PATTERNS),
            "pattern_limit": MAX_PROFILED_PATTERNS,
        },
    }
