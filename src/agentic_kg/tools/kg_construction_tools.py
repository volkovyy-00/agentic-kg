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
from agentic_kg.common.file_source import SourceError
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

    query = f"""UNWIND $rows AS row
    MERGE (n:{label} {{ {unique_column_name} : row[$unique_column_name] }})
    FOREACH (k IN $properties | SET n[k] = row[k])
    """

    rows_committed = 0
    try:
        for _header, batch in read_csv_batches(source_file):
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
    except (SourceError, FileNotFoundError) as exc:
        return tool_error(f"{source_file}: {exc}")

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
    FOREACH (k IN $properties | SET r[k] = row[k])
    """

    rows_committed = 0
    try:
        for _header, batch in read_csv_batches(source_file):
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
            rows_committed += len(batch)
    except (SourceError, FileNotFoundError) as exc:
        return tool_error(f"{source_file}: {exc}")

    return tool_success("rows_loaded", {"source_file": source_file, "rows": rows_committed})


def construct_domain_graph(construction_plan: dict) -> Dict[str, Any]:
    """Construct a domain graph according to a construction plan.

    Nodes are loaded before relationships, because the relationship query
    matches nodes that must already exist.
    """
    logger.debug("Building domain graph from plan: %s", construction_plan)

    outcomes = {}
    failures = []

    node_rules = [rule for rule in construction_plan.values()
                  if rule["construction_type"] == "node"]
    for rule in node_rules:
        result = import_nodes(rule)
        key = rule.get("label", rule.get("source_file", "?"))
        outcomes[key] = result
        if result["status"] == "error":
            failures.append(f"{key}: {result['error_message']}")

    relationship_rules = [rule for rule in construction_plan.values()
                          if rule["construction_type"] == "relationship"]
    for rule in relationship_rules:
        result = import_relationships(rule)
        key = rule.get("relationship_type", rule.get("source_file", "?"))
        outcomes[key] = result
        if result["status"] == "error":
            failures.append(f"{key}: {result['error_message']}")

    if failures:
        return tool_error("Graph construction had failures:\n" + "\n".join(failures))

    return tool_success("domain_graph_constructed", outcomes)


def build_graph_from_construction_rules(tool_context: ToolContext) -> Dict[str, Any]:
    """Build a graph from the approved construction rules."""
    if APPROVED_CONSTRUCTION_PLAN not in tool_context.state:
        return tool_error(f"{APPROVED_CONSTRUCTION_PLAN} not set.")

    return construct_domain_graph(tool_context.state[APPROVED_CONSTRUCTION_PLAN])
