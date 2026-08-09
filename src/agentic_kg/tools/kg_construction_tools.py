"""Build the domain graph from approved construction rules.

Rows are read in Python and sent as parameterised UNWIND batches rather than
asking Neo4j to read files itself. Aura forbids LOAD CSV FROM "file:///", and
client-side reading works identically against a local instance.

Labels and relationship types are interpolated into the query text after
checked() validation (common/cypher_identifiers.py) rather than passed as
Cypher dynamic labels: dynamic labels plan as Merge instead of
MergeUniqueNode, so they cannot use the uniqueness index and every row
triggers an all-nodes scan.
"""

import logging
from itertools import chain
from typing import Any, Dict, List, Optional

from google.adk.tools import ToolContext

from agentic_kg.common.csv_reader import read_csv_batches, read_csv_header
from agentic_kg.common.cypher_identifiers import InvalidIdentifier
from agentic_kg.common.cypher_identifiers import checked as _checked
from agentic_kg.common.neo4j_for_adk import get_graphdb
from agentic_kg.common.tool_result import tool_error, tool_success
from agentic_kg.common.value_types import (
    BLANK,
    CONVERTED,
    MAJORITY_SHARE,
    UNCONVERTIBLE,
    coerce,
)
from agentic_kg.tools.cypher_tools import create_uniqueness_constraint

logger = logging.getLogger(__name__)

graphdb = get_graphdb()

APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"

# Marks a typed value that must be REMOVED from the node/relationship rather
# than written. Cypher cannot distinguish "this row never had this column" from
# "this value failed to parse" -- both read as row[p] IS NULL -- and the two
# need opposite treatment: the first must leave an earlier row's value alone,
# the second must clear a stale one. This sentinel is the signal an absent key
# cannot produce by accident. It cannot collide with source data: a typed
# property's row value is only ever a converted number/bool, this sentinel, or
# an omitted key -- a raw source string never survives coercion.
CLEAR_SENTINEL = "\x00__agentic_kg_clear__"

# Above this share of a batch's present, non-blank values failing to convert,
# the column is the wrong type rather than dirty, and continuing would clear
# real values row by row. Deliberately the same number classify() uses to
# suggest a type, imported rather than restated so the two cannot drift.
# Matching numbers do NOT make a suggested column safe from this gate, though:
# classify() weighs a whole column, this weighs one batch AND the batches read
# so far (see _type_failure), so a file whose dirty rows are clustered can pass
# classification and still abort part-way through the load. That is the intended
# loud failure, not a contradiction. A column exactly half unconvertible passes
# both arms, which is the same strict majority classify() applies -- the
# remaining case where wrong values are cleared with only a warning.
# Still NOT inherited from import_relationships' join under-match warning
# below, which is a different failure at a different severity and keeps its own
# literal.
TYPE_FAILURE_LIMIT = MAJORITY_SHARE

# Below this many present, non-blank values, a failure is not evidence of a
# wrong type; it is one row. The column still counts toward the warning.
#
# Counted ACROSS batches as well as within one. The exemption was originally
# per-batch only, which made it permanent for a sparse column: one present value
# per batch never reaches two, so a column whose every value failed was cleared
# in full and the load reported success. The evidence bar is the same, the
# window it is measured over is the whole rule -- so the exemption now expires
# once the file has shown two present values, wherever they fell.
#
# This narrows the exemption on purpose. A column whose first two present values
# both fail is refused before any later good value is read. That is not a new
# behaviour so much as a consistent one: the batch arm already refused exactly
# that when the two landed in the same batch, so the outcome no longer depends
# on where the batch boundaries happen to fall -- or on DEFAULT_BATCH_SIZE,
# which anyone may tune.
TYPE_FAILURE_MIN_SAMPLE = 2

# Re-exported for backward compatibility: this module used to define its own
# InvalidIdentifier/_checked; both now live in common/cypher_identifiers so
# cypher_tools.create_uniqueness_constraint can share the same validation.
__all__ = [
    "InvalidIdentifier",
    "load_nodes_from_csv",
    "import_nodes",
    "import_relationships",
    "construct_domain_graph",
    "build_graph_from_construction_rules",
]


def _split_properties(properties, property_types):
    """Separate untyped from typed property names.

    A name must appear in exactly one of the two lists: $properties keeps the
    original ragged-row guard, and $typed_properties gets the write/clear pair.
    A name in both would be written twice per row.
    Also returns the declared types NARROWED to the names that are actually in
    'properties'. A type declared for a name the plan never lists is refused at
    approval time, but the loader must not depend on that: coercing off the raw
    map would let such a name be converted, tallied, and able to trip the gate
    and abort the whole rule -- for a property no FOREACH would ever have
    written. Worse, if that name were the unique_column_name, the row's key
    would be replaced by CLEAR_SENTINEL and MERGE would key the node on it.
    """
    property_types = property_types or {}
    untyped = [name for name in properties if name not in property_types]
    typed = [name for name in properties if name in property_types]
    return untyped, typed, {name: property_types[name] for name in typed}


def _new_tally():
    """A zeroed per-column tally. One definition, since _coerce_batch starts one
    per column and _merge_tallies has to start an identical one for a column it
    has not seen before."""
    return {CONVERTED: 0, BLANK: 0, UNCONVERTIBLE: 0, "examples": []}


def _missing_typed_columns_error(typed_properties, header, source_file, loaded_into):
    """The refusal for a typed property with no such column, or None.

    Shared by both loaders rather than written twice: a misspelled TYPED
    property would clear that property on every row of the file, so the wording
    has to stay the same on both paths. (A misspelled UNTYPED property stays as
    silent as it is today -- that asymmetry is deliberate.)
    """
    missing_typed = [name for name in typed_properties if name not in header]
    if not missing_typed:
        return None
    return (
        f"{source_file} has no column "
        f"{' or '.join(repr(name) for name in missing_typed)} to load as a "
        f"typed property of {loaded_into}, so nothing was loaded. "
        f"Available columns: {header}"
    )


def _batches_and_header(source_file):
    """Return (batches, header) with the header available before any row loads.

    Both loaders used to check their columns inside the batch loop, on the first
    batch. read_csv_batches yields nothing at all for a header-only file -- a
    valid empty export -- so that check never ran for one, and a rule naming a
    column the file does not have was reported as a clean zero-row success
    rather than the promised missing-column error.

    The header comes from the first batch when there is one, so an ordinary load
    still reads the source exactly once; only the no-batch case pays for a
    second open, and it has nothing else to read. The consumed first batch is put
    back in front of the iterator, so the caller still streams.
    """
    batches = read_csv_batches(source_file)
    first = next(batches, None)
    if first is None:
        return iter(()), read_csv_header(source_file)
    return chain([first], batches), first[0]


def _coerce_batch(batch, property_types):
    """Convert every typed value in a batch, returning new rows and per-column tallies.

    A key the row does not carry stays absent (see CLEAR_SENTINEL). Anything
    else becomes either the converted value or the sentinel.
    """
    # A rule with no declared types is the pre-existing path -- every plan
    # written before this feature. Hand its rows straight back rather than
    # copying each one only to change nothing in it.
    if not property_types:
        return batch, {}

    tallies = {name: _new_tally() for name in property_types}
    rows = []
    for row in batch:
        coerced = dict(row)
        for name, declared in property_types.items():
            if name not in row:
                continue
            value, outcome = coerce(row[name], declared)
            tally = tallies[name]
            tally[outcome] += 1
            if outcome == CONVERTED:
                coerced[name] = value
            else:
                coerced[name] = CLEAR_SENTINEL
                if outcome == UNCONVERTIBLE and len(tally["examples"]) < 3:
                    tally["examples"].append(row[name])
        rows.append(coerced)
    return rows, tallies


def _merge_tallies(totals, tallies):
    for name, tally in tallies.items():
        running = totals.setdefault(name, _new_tally())
        for field in (CONVERTED, BLANK, UNCONVERTIBLE):
            running[field] += tally[field]
        for example in tally["examples"]:
            if len(running["examples"]) < 3:
                running["examples"].append(example)
    return totals


def _type_failure(tallies, totals, property_types, source_file, rows_committed):
    """The gate: an error message when a column is the wrong type, else None.

    Two arms, and both are needed. 'totals' must already include this batch.

    The BATCH arm catches dirt that is clustered: a file whose bad rows sit
    together can pass whole-column classification and still be wrong where it
    lands, and aborting there is the intended loud failure.

    The CUMULATIVE arm closes a bypass the batch arm cannot see. A column sparse
    enough that each batch holds a single present value never reaches
    TYPE_FAILURE_MIN_SAMPLE, so the exemption below never expires: every value
    in the file could fail, every one of them be cleared, and the load still
    report success with only a warning. Judged across batches, five failures out
    of five present values is not one row -- it is the strongest evidence of a
    wrong type there is.

    Neither arm subsumes the other. On the first batch they are the same test,
    because the totals are the batch. From the second batch on, the cumulative
    arm can be diluted by earlier clean batches (which is what makes clustered
    dirt need its own arm) and the batch arm can be starved by sparsity (which
    is what makes the sparse case need its own).

    Blanks stay out of both denominators on purpose -- they are absence, not a
    wrong type, and counting them would abort a correct load of any sparse
    optional column. A column that is exactly half unconvertible passes both
    arms, matching classify()'s strict majority so the two cannot disagree about
    a column it suggested a type for.
    """
    for name, tally in tallies.items():
        running = totals[name]
        batch_present = tally[CONVERTED] + tally[UNCONVERTIBLE]
        total_present = running[CONVERTED] + running[UNCONVERTIBLE]

        by_batch = (
            batch_present >= TYPE_FAILURE_MIN_SAMPLE
            and tally[UNCONVERTIBLE] > batch_present * TYPE_FAILURE_LIMIT
        )
        by_total = (
            total_present >= TYPE_FAILURE_MIN_SAMPLE
            and running[UNCONVERTIBLE] > total_present * TYPE_FAILURE_LIMIT
        )
        if not (by_batch or by_total):
            continue

        examples = ", ".join(repr(example) for example in running["examples"])
        if by_batch:
            counted = (
                f"{tally[UNCONVERTIBLE]} of {batch_present} non-blank "
                f"values in this batch"
            )
        else:
            # "read so far", not "in this file": the rest of the file is unread.
            counted = (
                f"{running[UNCONVERTIBLE]} of {total_present} non-blank "
                f"values read so far ({tally[UNCONVERTIBLE]} of "
                f"{batch_present} in the batch just read)"
            )
        return (
            f"{source_file}: '{name}' is declared {property_types[name]} but "
            f"{counted} could not be read as one (e.g. {examples}). Load stopped "
            f"after {rows_committed} rows committed (this batch was not sent; the "
            f"committed rows stay in the graph, with '{name}' already cleared on "
            f"every row where it could not be converted). Correct or remove the "
            f"declared type for '{name}' in the construction plan, then run the "
            f"build again."
        )
    return None


def _type_warning(totals, property_types, source_file):
    """One sentence per column that had unconvertible values, joined into one string.

    Blank counts are deliberately absent here: a sparse column is not a problem
    to report. They stay visible in the per-column breakdown.
    """
    parts = []
    for name in sorted(totals):
        tally = totals[name]
        if not tally[UNCONVERTIBLE]:
            continue
        examples = ", ".join(repr(example) for example in tally["examples"])
        parts.append(
            f"{source_file}: {tally[UNCONVERTIBLE]} value(s) of '{name}' could not "
            f"be read as {property_types[name]} and were not stored (e.g. {examples})"
        )
    return "; ".join(parts)


def load_nodes_from_csv(
    source_file: str,
    label: str,
    unique_column_name: str,
    properties: List[str],
    # Optional[...] rather than Dict[...] = None, the shape ADK's
    # FunctionDeclaration builder rejects (isinstance(None, dict) is False).
    # Not model-visible today -- only build_graph_from_construction_rules is
    # wired as a tool -- but wiring this one later should not be a trap.
    property_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Load nodes from a source CSV in batches."""
    try:
        label = _checked("label", label)
        unique_column_name = _checked("column name", unique_column_name)
    except InvalidIdentifier as exc:
        return tool_error(str(exc))

    property_types = property_types or {}
    untyped_properties, typed_properties, typed_types = _split_properties(
        properties, property_types
    )

    # Only set properties the row actually carries. read_csv_batches omits the
    # key for a row shorter than the header, and SET n[k] = null *removes* the
    # property rather than skipping it -- so a ragged row, or a re-run against a
    # file that lost a column, silently erased values an earlier row had loaded,
    # with the result depending on which row came last.
    #
    # Typed properties need the opposite behaviour for a value that is present
    # but unreadable: leaving the old string behind would produce one property
    # holding numbers on most nodes and stale text on a few. Hence the second
    # pair of passes, keyed on CLEAR_SENTINEL, which an absent key cannot
    # produce.
    query = f"""UNWIND $rows AS row
    MERGE (n:{label} {{ {unique_column_name} : row[$unique_column_name] }})
    FOREACH (k IN [p IN $properties WHERE row[p] IS NOT NULL] | SET n[k] = row[k])
    FOREACH (k IN [p IN $typed_properties
                   WHERE row[p] IS NOT NULL AND row[p] <> $clear] | SET n[k] = row[k])
    FOREACH (k IN [p IN $typed_properties WHERE row[p] = $clear] | SET n[k] = null)
    """

    # The database does catch a key column missing from the header: MERGE on a
    # null property raises Neo.ClientError.Statement.SemanticError and commits
    # nothing. But it does so only after the batch has crossed the wire, and it
    # names the property alone -- not the file, and not the columns that file
    # actually has, which is what tells an agent how to correct the plan.
    # (The relationship loader below is the case that genuinely fails silently.)
    rows_committed = 0
    totals: Dict[str, Any] = {}
    try:
        batches, header = _batches_and_header(source_file)
        if unique_column_name not in header:
            return tool_error(
                f"{source_file} has no column '{unique_column_name}' to key {label} "
                f"nodes by, so nothing was loaded. Available columns: {header}"
            )
        missing_typed = _missing_typed_columns_error(
            typed_properties, header, source_file, label
        )
        if missing_typed is not None:
            return tool_error(missing_typed)

        for _batch_header, batch in batches:
            rows, tallies = _coerce_batch(batch, typed_types)
            # Merged BEFORE the gate: the cumulative arm has to see this batch,
            # and on a refusal the totals are discarded with the error anyway.
            _merge_tallies(totals, tallies)
            failure = _type_failure(
                tallies, totals, typed_types, source_file, rows_committed
            )
            if failure is not None:
                return tool_error(failure)

            result = graphdb.send_query(
                query,
                {
                    "rows": rows,
                    "unique_column_name": unique_column_name,
                    "properties": untyped_properties,
                    "typed_properties": typed_properties,
                    "clear": CLEAR_SENTINEL,
                },
            )
            if result["status"] == "error":
                return tool_error(
                    f"{source_file}: load failed after {rows_committed} rows committed "
                    f"(the failing batch was rolled back): {result['error_message']}"
                )
            rows_committed += len(batch)
    except FileNotFoundError:
        return tool_error(f"{source_file}: no such source file")
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        # Not only SourceError: a non-UTF-8 CSV raises UnicodeDecodeError out of
        # read_csv_batches, and clevercsv raises parse errors of its own.
        # Log it: returning the text alone leaves a genuine bug in this module
        # showing up as an LLM politely reporting "TypeError", traceback gone.
        logger.exception("%s: read failed", source_file)
        return tool_error(f"{source_file}: {type(exc).__name__}: {exc}")

    loaded = {"source_file": source_file, "rows": rows_committed}
    if totals:
        loaded["type_conversion"] = totals
        warning = _type_warning(totals, typed_types, source_file)
        if warning:
            loaded["warning"] = warning
            logger.warning(warning)

    # Rows processed is not the number of nodes that exist: MERGE collapses
    # every row sharing a key value into one node, which is by design for a
    # file with one row per component of an assembly. Reporting rows alone let
    # an agent tell a user "64 Assembly nodes loaded" when there were 10. Count
    # the label itself instead of deriving it from the file.
    counted = _count_in_graph(f"MATCH (n:{label}) RETURN count(n) AS count")
    if counted is not None:
        loaded["nodes_in_graph"] = counted

    return tool_success("rows_loaded", loaded)


def _count_in_graph(query: str) -> int | None:
    """Run a MATCH...count() query, or return None if it failed.

    This counts everything the query matches in the graph right now, not just
    what the load just wrote — it is a label/type-wide count, so pre-existing
    data or another rule targeting the same label is included too. The rows
    being counted are already committed by this point, so a failed count must
    not turn a successful load into an error — it just leaves the count
    unreported.
    """
    result = graphdb.send_query(query)
    if result["status"] == "error":
        logger.warning("post-load count failed: %s", result["error_message"])
        return None
    for record in result.get("records") or []:
        return record.get("count")
    return None


def import_nodes(node_construction: dict) -> Dict[str, Any]:
    """Import nodes as defined by a node construction rule."""
    try:
        _checked("label", node_construction["label"])
        _checked("column name", node_construction["unique_column_name"])
    except InvalidIdentifier as exc:
        return tool_error(str(exc))

    uniqueness_result = create_uniqueness_constraint(
        node_construction["label"],
        node_construction["unique_column_name"],
    )
    if uniqueness_result["status"] == "error":
        return uniqueness_result

    return load_nodes_from_csv(
        node_construction["source_file"],
        node_construction["label"],
        node_construction["unique_column_name"],
        node_construction["properties"],
        node_construction.get("property_types", {}),
    )


def import_relationships(relationship_construction: dict) -> Dict[str, Any]:
    """Import relationships as defined by a relationship construction rule."""
    try:
        relationship_type = _checked(
            "relationship type", relationship_construction["relationship_type"]
        )
        from_label = _checked("label", relationship_construction["from_node_label"])
        to_label = _checked("label", relationship_construction["to_node_label"])
        from_column = _checked(
            "column name", relationship_construction["from_node_column"]
        )
        to_column = _checked("column name", relationship_construction["to_node_column"])
    except InvalidIdentifier as exc:
        return tool_error(str(exc))

    source_file = relationship_construction["source_file"]
    properties = relationship_construction["properties"]
    # A rule proposed before types existed carries no such key at all.
    property_types = relationship_construction.get("property_types") or {}
    untyped_properties, typed_properties, typed_types = _split_properties(
        properties, property_types
    )

    # The join columns are NOT coerced: they are matched against whatever the
    # node loader stored, which is raw CSV text for identifiers. That is why a
    # typed join column is refused at approval time -- see
    # check_construction_plan_consistency. Coercion touches typed_properties
    # only, which by that rule can never include a join column.
    query = f"""UNWIND $rows AS row
    MATCH (from_node:{from_label} {{ {from_column} : row[$from_node_column] }}),
          (to_node:{to_label} {{ {to_column} : row[$to_node_column] }})
    MERGE (from_node)-[r:{relationship_type}]->(to_node)
    FOREACH (k IN [p IN $properties WHERE row[p] IS NOT NULL] | SET r[k] = row[k])
    FOREACH (k IN [p IN $typed_properties
                   WHERE row[p] IS NOT NULL AND row[p] <> $clear] | SET r[k] = row[k])
    FOREACH (k IN [p IN $typed_properties WHERE row[p] = $clear] | SET r[k] = null)
    RETURN count(r) AS rows_matched
    """

    # count(r) counts rows that matched *both* endpoints, not edges newly
    # created: MERGE binds r on every matched row whether it created the
    # relationship or found an existing one. That is deliberately what we want
    # to report — a correct re-run then reports the same count instead of zero.
    #
    # A join column missing from the header makes row[$..._node_column] null,
    # which matches no node and silently produces zero relationships rather
    # than an error. Check the header before sending anything.
    rows_committed = 0
    rows_matched = 0
    totals: Dict[str, Any] = {}
    try:
        batches, header = _batches_and_header(source_file)
        missing = [
            column for column in (from_column, to_column) if column not in header
        ]
        if missing:
            return tool_error(
                f"{source_file} has no column {' or '.join(repr(c) for c in missing)} "
                f"to join {relationship_type} on, so nothing was loaded. "
                f"Available columns: {header}"
            )
        missing_typed = _missing_typed_columns_error(
            typed_properties, header, source_file, relationship_type
        )
        if missing_typed is not None:
            return tool_error(missing_typed)

        for _batch_header, batch in batches:
            rows, tallies = _coerce_batch(batch, typed_types)
            # Merged BEFORE the gate: the cumulative arm has to see this batch,
            # and on a refusal the totals are discarded with the error anyway.
            _merge_tallies(totals, tallies)
            failure = _type_failure(
                tallies, totals, typed_types, source_file, rows_committed
            )
            if failure is not None:
                return tool_error(failure)

            result = graphdb.send_query(
                query,
                {
                    "rows": rows,
                    "from_node_column": from_column,
                    "to_node_column": to_column,
                    "properties": untyped_properties,
                    "typed_properties": typed_properties,
                    "clear": CLEAR_SENTINEL,
                },
            )
            if result["status"] == "error":
                return tool_error(
                    f"{source_file}: load failed after {rows_committed} rows committed "
                    f"(the failing batch was rolled back): {result['error_message']}"
                )
            for record in result.get("records") or []:
                rows_matched += record.get("rows_matched", 0) or 0
            rows_committed += len(batch)
    except FileNotFoundError:
        return tool_error(f"{source_file}: no such source file")
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        # Not only SourceError: a non-UTF-8 CSV raises UnicodeDecodeError out of
        # read_csv_batches, and clevercsv raises parse errors of its own.
        # Log it: returning the text alone leaves a genuine bug in this module
        # showing up as an LLM politely reporting "TypeError", traceback gone.
        logger.exception("%s: read failed", source_file)
        return tool_error(f"{source_file}: {type(exc).__name__}: {exc}")

    loaded = {
        "source_file": source_file,
        "rows": rows_committed,
        "rows_matched": rows_matched,
    }
    if totals:
        loaded["type_conversion"] = totals

    # rows_matched collapses the same way node loading does — several rows
    # naming the same endpoint pair MERGE into one relationship — so it is not
    # an edge count either. Added alongside rather than replacing it, so the
    # idempotent-re-run property above is kept. count() over a relationship
    # type is a count-store lookup, not a scan, so this is cheap per type.
    counted = _count_in_graph(
        f"MATCH ()-[r:{relationship_type}]->() RETURN count(r) AS count"
    )
    if counted is not None:
        loaded["relationships_in_graph"] = counted

    # Two independent conditions can now warn about the same load, and 'warning'
    # is a single string that construct_domain_graph lifts verbatim. Collect and
    # join rather than assigning twice, or the second silently erases the first.
    warnings = []
    type_warning = _type_warning(totals, typed_types, source_file)
    if type_warning:
        warnings.append(type_warning)

    # A row whose join columns match nothing produces no relationship, and the
    # MERGE reports no error for it. Half the rows failing to match is already
    # a design smell worth a human look; zero matches is almost certainly a
    # wrong join key, so both are surfaced rather than silently succeeding.
    if rows_committed and rows_matched < rows_committed / 2:
        warnings.append(
            f"{source_file}: only {rows_matched} of {rows_committed} rows matched both "
            f"endpoints ({from_label}.{from_column} -> "
            f"{to_label}.{to_column}) — check whether the join columns actually match. "
            "A join column that is a per-row property collapsed during node loading "
            "will match few or no rows."
        )

    if warnings:
        loaded["warning"] = "; ".join(warnings)
        logger.warning(loaded["warning"])

    return tool_success("rows_loaded", loaded)


def _loaded_summary(key: str, loaded: dict) -> str:
    """Describe one loaded rule, preferring what is in the graph over rows read.

    This string is what a partial-failure message hands an LLM agent, so it is
    the number most likely to be repeated verbatim to the user — it must not
    say "rows" where the user will hear "nodes".
    """
    for field, noun in (
        ("nodes_in_graph", "nodes"),
        ("relationships_in_graph", "relationships"),
    ):
        if loaded.get(field) is not None:
            # "from N rows" would claim this load produced that many nodes,
            # but the count is label-wide (see _count_in_graph) and can
            # include nodes a re-run's own rows had nothing to do with.
            return f"{key} ({loaded[field]} {noun} now in graph, {loaded['rows']} rows read)"
    return f"{key} ({loaded['rows']} rows)"


def construct_domain_graph(construction_plan: dict) -> Dict[str, Any]:
    """Construct a domain graph according to a construction plan.

    Nodes are loaded before relationships, because the relationship query
    matches nodes that must already exist.
    """
    logger.debug("Building domain graph from plan: %s", construction_plan)

    outcomes = {}
    failures = []

    # construction_plan is LLM-produced, so a rule can be missing keys this
    # module otherwise indexes directly (import_nodes/import_relationships).
    # .get() here, and catching KeyError per rule below, keeps a malformed
    # rule from raising an unhandled KeyError into ADK instead of reporting
    # a tool_error.
    node_rules = [
        rule
        for rule in construction_plan.values()
        if rule.get("construction_type") == "node"
    ]
    for rule in node_rules:
        key = rule.get("label", rule.get("source_file", "?"))
        try:
            result = import_nodes(rule)
        except KeyError as exc:
            result = tool_error(
                f"{key}: node construction rule is missing required key {exc}"
            )
        outcomes[key] = result
        if result["status"] == "error":
            failures.append(f"{key}: {result['error_message']}")

    relationship_rules = [
        rule
        for rule in construction_plan.values()
        if rule.get("construction_type") == "relationship"
    ]
    for rule in relationship_rules:
        key = rule.get("relationship_type", rule.get("source_file", "?"))
        try:
            result = import_relationships(rule)
        except KeyError as exc:
            result = tool_error(
                f"{key}: relationship construction rule is missing required key {exc}"
            )
        outcomes[key] = result
        if result["status"] == "error":
            failures.append(f"{key}: {result['error_message']}")

    # Warnings (e.g. a relationship join that matched almost nothing) are not
    # failures, but the agent has no reason to dig into per-rule payloads on a
    # successful build, so they are lifted to the top level.
    warnings = [
        result["rows_loaded"]["warning"]
        for result in outcomes.values()
        if result.get("status") == "success"
        and isinstance(result.get("rows_loaded"), dict)
        and result["rows_loaded"].get("warning")
    ]

    if failures:
        # Fold in what did load: on partial failure the caller (an LLM agent)
        # needs to know which rules already committed, both to report
        # accurately and to avoid re-running already-loaded rules on retry.
        successes = [
            _loaded_summary(key, result["rows_loaded"])
            for key, result in outcomes.items()
            if result.get("status") == "success" and "rows_loaded" in result
        ]
        message_parts = []
        if successes:
            message_parts.append("loaded: " + ", ".join(successes))
        if warnings:
            message_parts.append("warnings: " + "; ".join(warnings))
        message_parts.append("failed: " + "; ".join(failures))
        return tool_error("; ".join(message_parts))

    success = tool_success("domain_graph_constructed", outcomes)
    if warnings:
        success["warnings"] = warnings
    return success


def build_graph_from_construction_rules(tool_context: ToolContext) -> Dict[str, Any]:
    """Build a graph from the approved construction rules."""
    if APPROVED_CONSTRUCTION_PLAN not in tool_context.state:
        return tool_error(f"{APPROVED_CONSTRUCTION_PLAN} not set.")

    return construct_domain_graph(tool_context.state[APPROVED_CONSTRUCTION_PLAN])
