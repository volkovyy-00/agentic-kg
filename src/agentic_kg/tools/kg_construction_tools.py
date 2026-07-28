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
from agentic_kg.tools.cypher_tools import create_uniqueness_constraint

logger = logging.getLogger(__name__)

graphdb = get_graphdb()

APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"

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


def load_nodes_from_csv(
    source_file: str,
    label: str,
    unique_column_name: str,
    properties: List[str],
) -> Dict[str, Any]:
    """Load nodes from a source CSV in batches."""
    try:
        label = _checked("label", label)
        unique_column_name = _checked("column name", unique_column_name)
    except InvalidIdentifier as exc:
        return tool_error(str(exc))

    # Only set properties the row actually carries. read_csv_batches omits the
    # key for a row shorter than the header, and SET n[k] = null *removes* the
    # property rather than skipping it -- so a ragged row, or a re-run against a
    # file that lost a column, silently erased values an earlier row had loaded,
    # with the result depending on which row came last.
    query = f"""UNWIND $rows AS row
    MERGE (n:{label} {{ {unique_column_name} : row[$unique_column_name] }})
    FOREACH (k IN [p IN $properties WHERE row[p] IS NOT NULL] | SET n[k] = row[k])
    """

    # The database does catch a key column missing from the header: MERGE on a
    # null property raises Neo.ClientError.Statement.SemanticError and commits
    # nothing. But it does so only after the batch has crossed the wire, and it
    # names the property alone -- not the file, and not the columns that file
    # actually has, which is what tells an agent how to correct the plan.
    # (The relationship loader below is the case that genuinely fails silently.)
    rows_committed = 0
    header_checked = False
    try:
        for header, batch in read_csv_batches(source_file):
            if not header_checked:
                if unique_column_name not in header:
                    return tool_error(
                        f"{source_file} has no column '{unique_column_name}' to key {label} "
                        f"nodes by, so nothing was loaded. Available columns: {header}"
                    )
                header_checked = True
            result = graphdb.send_query(query, {
                "rows": batch,
                "unique_column_name": unique_column_name,
                "properties": properties,
            })
            if result["status"] == "error":
                return tool_error(
                    f"{source_file}: load failed after {rows_committed} rows committed "
                    f"(the failing batch was rolled back): {result['error_message']}"
                )
            rows_committed += len(batch)
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        # Not just SourceError/FileNotFoundError: a source CSV that is not UTF-8
        # raises UnicodeDecodeError out of read_csv_batches, and clevercsv can
        # raise its own parse errors. Escaping into ADK breaks the contract that
        # every tool returns a ToolResult, and the agent sees a crashed run
        # rather than a file it could ask the user about.
        return tool_error(f"{source_file}: {type(exc).__name__}: {exc}")

    return tool_success("rows_loaded", {"source_file": source_file, "rows": rows_committed})


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

    query = f"""UNWIND $rows AS row
    MATCH (from_node:{from_label} {{ {from_column} : row[$from_node_column] }}),
          (to_node:{to_label} {{ {to_column} : row[$to_node_column] }})
    MERGE (from_node)-[r:{relationship_type}]->(to_node)
    FOREACH (k IN [p IN $properties WHERE row[p] IS NOT NULL] | SET r[k] = row[k])
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
                header_checked = True
            result = graphdb.send_query(query, {
                "rows": batch,
                "from_node_column": from_column,
                "to_node_column": to_column,
                "properties": properties,
            })
            if result["status"] == "error":
                return tool_error(
                    f"{source_file}: load failed after {rows_committed} rows committed "
                    f"(the failing batch was rolled back): {result['error_message']}"
                )
            for record in result.get("records") or []:
                rows_matched += record.get("rows_matched", 0) or 0
            rows_committed += len(batch)
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        # Not just SourceError/FileNotFoundError: a source CSV that is not UTF-8
        # raises UnicodeDecodeError out of read_csv_batches, and clevercsv can
        # raise its own parse errors. Escaping into ADK breaks the contract that
        # every tool returns a ToolResult, and the agent sees a crashed run
        # rather than a file it could ask the user about.
        return tool_error(f"{source_file}: {type(exc).__name__}: {exc}")

    loaded = {
        "source_file": source_file,
        "rows": rows_committed,
        "rows_matched": rows_matched,
    }

    # A row whose join columns match nothing produces no relationship, and the
    # MERGE reports no error for it. Half the rows failing to match is already
    # a design smell worth a human look; zero matches is almost certainly a
    # wrong join key, so both are surfaced rather than silently succeeding.
    if rows_committed and rows_matched < rows_committed / 2:
        warning = (
            f"{source_file}: only {rows_matched} of {rows_committed} rows matched both "
            f"endpoints ({from_label}.{from_column} -> "
            f"{to_label}.{to_column}) — check whether the join columns actually match. "
            "A join column that is a per-row property collapsed during node loading "
            "will match few or no rows."
        )
        loaded["warning"] = warning
        logger.warning(warning)

    return tool_success("rows_loaded", loaded)


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
            f"{key} ({result['rows_loaded']['rows']} rows)"
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
