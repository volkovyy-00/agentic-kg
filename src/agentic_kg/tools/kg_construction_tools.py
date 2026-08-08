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

from google.adk.tools import ToolContext
from typing import Any, Dict, List

from agentic_kg.common.csv_reader import read_csv_batches
from agentic_kg.common.cypher_identifiers import InvalidIdentifier, checked as _checked
from agentic_kg.common.neo4j_for_adk import get_graphdb
from agentic_kg.common.tool_result import tool_error, tool_success
from agentic_kg.common.value_types import BLANK, CONVERTED, UNCONVERTIBLE, coerce
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
# real values row by row. Chosen for this gate, not inherited from
# import_relationships' join under-match warning, which is a different failure
# at a different severity.
TYPE_FAILURE_LIMIT = 0.5

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
    """
    property_types = property_types or {}
    untyped = [name for name in properties if name not in property_types]
    typed = [name for name in properties if name in property_types]
    return untyped, typed


def _coerce_batch(batch, property_types):
    """Convert every typed value in a batch, returning new rows and per-column tallies.

    A key the row does not carry stays absent (see CLEAR_SENTINEL). Anything
    else becomes either the converted value or the sentinel.
    """
    tallies = {name: {"converted": 0, "blank": 0, "unconvertible": 0, "examples": []}
               for name in property_types}
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
        running = totals.setdefault(
            name, {"converted": 0, "blank": 0, "unconvertible": 0, "examples": []})
        for field in ("converted", "blank", "unconvertible"):
            running[field] += tally[field]
        for example in tally["examples"]:
            if len(running["examples"]) < 3:
                running["examples"].append(example)
    return totals


def _type_failure(tallies, property_types, source_file, rows_committed):
    """The gate: an error message when a column is the wrong type, else None.

    Blanks are excluded from the denominator on purpose -- they are absence, not
    a wrong type, and counting them would abort a correct load of any sparse
    optional column.
    """
    for name, tally in tallies.items():
        present = tally["converted"] + tally["unconvertible"]
        # A single present value failing is not evidence of a wrong type -- it
        # takes at least two data points for "more than half failed" to mean
        # anything. Without this, one bad value in an otherwise-unseen column
        # would trip the gate on its very first row.
        if present > 1 and tally["unconvertible"] > present * TYPE_FAILURE_LIMIT:
            examples = ", ".join(repr(example) for example in tally["examples"])
            return (
                f"{source_file}: '{name}' is declared {property_types[name]} but "
                f"{tally['unconvertible']} of {present} non-blank values in this "
                f"batch could not be read as one (e.g. {examples}). Load stopped "
                f"after {rows_committed} rows committed (this batch was not sent). "
                f"Correct or remove the declared type for '{name}' in the "
                f"construction plan, then run the build again."
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
        if not tally["unconvertible"]:
            continue
        examples = ", ".join(repr(example) for example in tally["examples"])
        parts.append(
            f"{source_file}: {tally['unconvertible']} value(s) of '{name}' could not "
            f"be read as {property_types[name]} and were not stored (e.g. {examples})")
    return "; ".join(parts)


def load_nodes_from_csv(
    source_file: str,
    label: str,
    unique_column_name: str,
    properties: List[str],
    property_types: Dict[str, str] = None,
) -> Dict[str, Any]:
    """Load nodes from a source CSV in batches."""
    try:
        label = _checked("label", label)
        unique_column_name = _checked("column name", unique_column_name)
    except InvalidIdentifier as exc:
        return tool_error(str(exc))

    property_types = property_types or {}
    untyped_properties, typed_properties = _split_properties(properties, property_types)

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
    header_checked = False
    totals: Dict[str, Any] = {}
    try:
        for header, batch in read_csv_batches(source_file):
            if not header_checked:
                if unique_column_name not in header:
                    return tool_error(
                        f"{source_file} has no column '{unique_column_name}' to key {label} "
                        f"nodes by, so nothing was loaded. Available columns: {header}"
                    )
                # Typed properties only. A misspelled untyped property stays as
                # silent as it is today; a misspelled typed one would clear that
                # property on every row of the file instead of merely not setting it.
                missing_typed = [name for name in typed_properties if name not in header]
                if missing_typed:
                    return tool_error(
                        f"{source_file} has no column "
                        f"{' or '.join(repr(name) for name in missing_typed)} to load as a "
                        f"typed property of {label}, so nothing was loaded. "
                        f"Available columns: {header}"
                    )
                header_checked = True

            rows, tallies = _coerce_batch(batch, property_types)
            failure = _type_failure(tallies, property_types, source_file, rows_committed)
            if failure is not None:
                return tool_error(failure)
            _merge_tallies(totals, tallies)

            result = graphdb.send_query(query, {
                "rows": rows,
                "unique_column_name": unique_column_name,
                "properties": untyped_properties,
                "typed_properties": typed_properties,
                "clear": CLEAR_SENTINEL,
            })
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
        warning = _type_warning(totals, property_types, source_file)
        if warning:
            loaded["warning"] = warning

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
            "relationship type", relationship_construction["relationship_type"])
        from_label = _checked("label", relationship_construction["from_node_label"])
        to_label = _checked("label", relationship_construction["to_node_label"])
        from_column = _checked("column name", relationship_construction["from_node_column"])
        to_column = _checked("column name", relationship_construction["to_node_column"])
    except InvalidIdentifier as exc:
        return tool_error(str(exc))

    source_file = relationship_construction["source_file"]
    properties = relationship_construction["properties"]
    # A rule proposed before types existed carries no such key at all.
    property_types = relationship_construction.get("property_types") or {}
    untyped_properties, typed_properties = _split_properties(properties, property_types)

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
    header_checked = False
    totals: Dict[str, Any] = {}
    try:
        for header, batch in read_csv_batches(source_file):
            if not header_checked:
                missing = [
                    column for column in (from_column, to_column) if column not in header
                ]
                if missing:
                    return tool_error(
                        f"{source_file} has no column {' or '.join(repr(c) for c in missing)} "
                        f"to join {relationship_type} on, so nothing was loaded. "
                        f"Available columns: {header}"
                    )
                # Typed properties only. A misspelled untyped property stays as
                # silent as it is today; a misspelled typed one would clear that
                # property on every row of the file instead of merely not setting it.
                missing_typed = [name for name in typed_properties if name not in header]
                if missing_typed:
                    return tool_error(
                        f"{source_file} has no column "
                        f"{' or '.join(repr(name) for name in missing_typed)} to load as a "
                        f"typed property of {relationship_type}, so nothing was loaded. "
                        f"Available columns: {header}"
                    )
                header_checked = True

            rows, tallies = _coerce_batch(batch, property_types)
            failure = _type_failure(tallies, property_types, source_file, rows_committed)
            if failure is not None:
                return tool_error(failure)
            _merge_tallies(totals, tallies)

            result = graphdb.send_query(query, {
                "rows": rows,
                "from_node_column": from_column,
                "to_node_column": to_column,
                "properties": untyped_properties,
                "typed_properties": typed_properties,
                "clear": CLEAR_SENTINEL,
            })
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
        f"MATCH ()-[r:{relationship_type}]->() RETURN count(r) AS count")
    if counted is not None:
        loaded["relationships_in_graph"] = counted

    # Two independent conditions can now warn about the same load, and 'warning'
    # is a single string that construct_domain_graph lifts verbatim. Collect and
    # join rather than assigning twice, or the second silently erases the first.
    warnings = []
    type_warning = _type_warning(totals, property_types, source_file)
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
    for field, noun in (("nodes_in_graph", "nodes"),
                        ("relationships_in_graph", "relationships")):
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
    node_rules = [rule for rule in construction_plan.values()
                  if rule.get("construction_type") == "node"]
    for rule in node_rules:
        key = rule.get("label", rule.get("source_file", "?"))
        try:
            result = import_nodes(rule)
        except KeyError as exc:
            result = tool_error(f"{key}: node construction rule is missing required key {exc}")
        outcomes[key] = result
        if result["status"] == "error":
            failures.append(f"{key}: {result['error_message']}")

    relationship_rules = [rule for rule in construction_plan.values()
                          if rule.get("construction_type") == "relationship"]
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
