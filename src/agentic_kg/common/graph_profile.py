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


def _payload_or_raise(result) -> Dict[str, Any]:
    """Strict accessor returning the WHOLE payload, not just its rows.

    send_read_query reports more than records -- notably `values_summarised`,
    which says whether an oversized list value was replaced by a placeholder.
    Anything that shows real property VALUES to the model has to look at that
    flag; unwrapping straight to `records` throws it away and reintroduces
    exactly the silent-omission problem the flag exists to signal.
    """
    if not is_success(result):
        raise ProfileQueryError(result.get("error_message", "profile query failed"))
    return result.get("query_result", {})


def _records_or_raise(result) -> list:
    """Strict accessor: a failure raises so per-entity isolation can act.

    Using the lenient accessor everywhere would make build_profile's
    "profile_error" branch unreachable -- a failed query would silently look
    like an empty result, which is the opposite of the requirement that one
    failing entity degrade only itself.
    """
    return _payload_or_raise(result).get("records", [])


def _entity_counts(labels, rel_types) -> Dict[tuple, Optional[int]]:
    """One grouped query per direction rather than one query per entity.

    Keyed on (is_relationship, name), NOT on bare name. Neo4j keeps node
    labels and relationship types in separate namespaces, so a graph may hold
    both a `FOLLOWS` label and a `FOLLOWS` relationship type. Keyed on name
    alone the relationship pass -- which runs second -- would overwrite the
    label's node count, and annotate_property would then be handed an EDGE
    count as the entity count for a LABEL's properties, silently producing a
    wrong `uniqueness` verdict that prompt rule 4 acts on.
    """
    counts: Dict[tuple, Optional[int]] = {(False, name): None for name in labels}
    counts.update({(True, name): None for name in rel_types})

    for row in _records(graphdb.send_read_query(
        "MATCH (n) UNWIND labels(n) AS label "
        "RETURN label AS name, count(*) AS n", max_rows=None)):
        counts[(False, row["name"])] = row["n"]

    for row in _records(graphdb.send_read_query(
        "MATCH ()-[r]->() RETURN type(r) AS name, count(r) AS n", max_rows=None)):
        counts[(True, row["name"])] = row["n"]

    return counts


def _display_keys(counts: Dict[tuple, Optional[int]]) -> Dict[tuple, str]:
    """Map each (is_relationship, name) to the key used in the output.

    Bare `name` when it is unambiguous -- which is every real graph seen so far,
    and what every consumer and test expects. Only when the SAME name exists in
    both namespaces are both sides suffixed, so a collision is visible rather
    than silently dropping one entity's profile.
    """
    colliding = {
        name for (_, name) in counts
        if (False, name) in counts and (True, name) in counts
    }
    return {
        (is_rel, name): (
            f"{name} ({'relationship' if is_rel else 'node'})"
            if name in colliding else name
        )
        for (is_rel, name) in counts
    }


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
    payload = _payload_or_raise(graphdb.send_read_query(query, max_rows=None))
    rows = payload.get("records", [])
    # A LIST of {value, count}, not a dict keyed by str(value). Neo4j allows
    # heterogeneous types on one property key, so integer 1 and string "1"
    # are two distinct values that str() collapses onto one key -- the second
    # silently overwriting the first, leaving a distribution whose counts do
    # not sum to the entity count and whose missing value is invisible.
    #
    # An empty list is a real answer ("no non-null values"), not "unknown": the
    # query succeeded. A query that FAILS raises out of the strict accessor
    # above and degrades the whole entity to "profile_error", so absence of
    # information is reported there rather than mislabelled here.
    counts = [{"value": r["value"], "count": r["n"]} for r in rows]

    # A property value can itself be a list long enough that send_read_query
    # replaced it with a placeholder string. Say so rather than presenting the
    # placeholder as the value: "complete" here means every value shown is the
    # real one, not a summary of it.
    complete = "no" if payload.get("values_summarised") else "yes"
    return counts, complete


def _profile_entity(entity, props, entity_count, is_relationship):
    annotated = []
    for prop in props:
        out = annotate_property(prop, entity_count)
        distinct_count = prop.get("distinct_count")
        if distinct_count is not None and distinct_count <= VALUE_COUNT_MAX_DISTINCT:
            out["value_counts"], out["value_counts_complete"] = _value_counts(
                entity, prop["property"], is_relationship)
        else:
            out["value_counts"] = "unknown"
            # Same tri-state vocabulary as the other annotations: no counts
            # were computed, so whether they would have been complete is
            # unknowable -- never silently "yes".
            out["value_counts_complete"] = "unknown"
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

    # Keyed on (is_relationship, name) so a label and a relationship type
    # sharing a name cannot overwrite one another. `display` maps those keys
    # back to the output keys, which stay bare names unless they actually
    # collide.
    counts = _entity_counts(list(node_props), list(rel_props))
    display = _display_keys(counts)

    entities = [(name, props, False) for name, props in node_props.items()]
    entities += [(name, props, True) for name, props in rel_props.items()]
    # Largest first, so a budget cut spends what it has on the entities most
    # likely to dominate an answer rather than on an arbitrary slice.
    #
    # Note the trade-off this makes: the entities dropped are the SMALL ones,
    # which are also the ones the library scanned exhaustively and can
    # therefore say the most about (only they carry a distinct_count, so only
    # they generate value-count queries at all). Largest-first is a judgement
    # about consequence, not about how much is knowable.
    entities.sort(key=lambda e: counts.get((e[2], e[0])) or 0, reverse=True)

    properties: Dict[str, Any] = {}
    profiled = 0
    for name, props, is_rel in entities:
        key = display[(is_rel, name)]
        if profiled >= MAX_PROFILED_ENTITIES:
            properties[key] = "not_profiled"
            continue
        try:
            properties[key] = _profile_entity(
                name, props, counts.get((is_rel, name)), is_rel)
        except Exception:
            # Includes ProfileQueryError. Deliberately broad: a profile is a
            # convenience, and no failure inside it may take down the schema
            # tool that graphrag is instructed to call first.
            logger.exception("Profiling failed for entity %s; continuing", key)
            properties[key] = "profile_error"
        profiled += 1

    # The budget covers patterns too. Gating only the property loop would leave
    # the 2P degree queries unbounded, so "cold-start cost is bounded" would
    # hold for Q alone -- which is not what the spec claims. Largest patterns
    # first, by the edge count of their relationship type.
    ranked = [r for r in relationships
              if r.get("start") and r.get("type") and r.get("end")]
    ranked.sort(key=lambda r: counts.get((True, r["type"])) or 0, reverse=True)

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
        # Re-keyed from (is_relationship, name) back to the display key, so the
        # payload keeps bare names for every non-colliding graph.
        "entity_counts": {display[k]: v for k, v in counts.items()},
        "patterns": patterns,
        "properties": properties,
        "budget": {
            # No min() needed: the loop above stops incrementing `profiled` the
            # moment it reaches the cap, so it cannot overshoot. (`patterns_profiled`
            # below DOES need one -- len(ranked) is not bounded by the cap.)
            "entities_profiled": profiled,
            "entities_skipped": max(0, len(entities) - MAX_PROFILED_ENTITIES),
            "entity_limit": MAX_PROFILED_ENTITIES,
            "patterns_profiled": min(len(ranked), MAX_PROFILED_PATTERNS),
            "patterns_skipped": max(0, len(ranked) - MAX_PROFILED_PATTERNS),
            "pattern_limit": MAX_PROFILED_PATTERNS,
        },
    }


# Module-level, deliberately not session state: one adk web process serves
# every session, and what is cached is a property of the database, not of a
# conversation. Per-session caches would disagree with each other the moment
# one session rebuilt the graph.
_cache: Dict[str, Any] = {}

_FINGERPRINT_QUERY = (
    "MATCH (n) WITH count(n) AS nodes "
    "OPTIONAL MATCH ()-[r]->() "
    "RETURN nodes, count(r) AS rels"
)


def reset_cache() -> None:
    """Discard the cached profile. For tests and for explicit invalidation."""
    _cache.clear()


def _fingerprint():
    """Node and relationship totals, or None when they cannot be read.

    Catches writes the counter structurally cannot see -- anything done to the
    database from outside this process. Blind to edits that change property
    values without changing counts; the shape and cardinality we cache do not
    move under those.
    """
    rows = _records(graphdb.send_read_query(_FINGERPRINT_QUERY, max_rows=None))
    if not rows:
        return None
    return (rows[0].get("nodes"), rows[0].get("rels"))


def get_cached_profile(schema_loader) -> Dict[str, Any]:
    """Return {"schema": ..., "profile": ...}, recomputing only when stale.

    schema_loader is a zero-argument callable returning the enriched schema.
    It is only invoked on a miss, so the expensive enriched pass is skipped
    entirely on a hit.
    """
    write_count = getattr(graphdb, "write_count", None)
    fingerprint = _fingerprint()

    if (
        _cache
        and _cache.get("write_count") == write_count
        and _cache.get("fingerprint") == fingerprint
        and fingerprint is not None
    ):
        return _cache["value"]

    schema = schema_loader()
    value = {"schema": schema, "profile": build_profile(schema)}

    if fingerprint is None:
        # The fingerprint could not be read, so this result's freshness is
        # unverifiable and must not be cached. Critically, we also leave any
        # EXISTING entry alone rather than overwriting it with None: storing
        # None would guarantee a miss on the next call too (a real fingerprint
        # never equals None), turning one transient blip into two full cold
        # rebuilds of hundreds of queries for a graph that never changed.
        # Leaving the old entry means the next successful fingerprint can still
        # match it and serve a hit.
        return value

    _cache.clear()
    _cache.update(
        {"write_count": write_count, "fingerprint": fingerprint, "value": value}
    )
    return value
