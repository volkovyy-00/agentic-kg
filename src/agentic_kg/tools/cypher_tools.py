from typing import Any, Dict, Optional

from google.adk.tools import ToolContext
from neo4j_graphrag.schema import get_structured_schema

from agentic_kg.common.cypher_identifiers import (
    InvalidIdentifier,
    checked,
    checked_field,
    quote,
)
from agentic_kg.common.graph_profile import get_cached_profile
from agentic_kg.common.neo4j_for_adk import (
    QUERY_TIMEOUT_SECONDS,
    close_graphdb,
    get_graphdb,
)
from agentic_kg.common.tool_result import is_error, tool_error, tool_success

graphdb = get_graphdb()


def neo4j_is_ready():
    """Tool to check that the Neo4j database is ready.
    Replies with either a positive message about the database being ready or an error message.
    """
    results = graphdb.send_query("RETURN 'Neo4j is Ready!' as message")

    if results["status"] == "error":
        close_graphdb()

    return results


def _physical_schema(include_data_profile: bool) -> Dict[str, Any]:
    """Internal implementation. NOT bound as a tool -- see the two wrappers.

    The flag must not appear in any tool's signature. ADK builds a tool's
    declaration from the callable, so a public
    `get_physical_schema(include_data_profile=False)` would advertise the flag
    to the model (google-adk 2.9 declares it optional with its default, but
    the model can still set it). All four
    consumers -- the coordinator, graph_construction_agent, graphrag and
    single_agent's cypher_agent -- would be handed a knob they know nothing
    about, and a model that guessed True would silently trigger a full scan per
    label on a latency-tuned agent. Two zero-argument wrappers keep the choice
    in code where it belongs.
    """
    try:
        # Inside the try: a driver or config failure must return a structured
        # error, not raise out of a tool call.
        driver = graphdb.get_driver()
        database_name = graphdb.get_config().database

        if not include_data_profile:
            return tool_success(
                "schema", get_structured_schema(driver, database=database_name)
            )

        def load_enriched_schema():
            return get_structured_schema(
                driver,
                is_enhanced=True,
                database=database_name,
                timeout=QUERY_TIMEOUT_SECONDS,
                sanitize=True,
            )

        cached = get_cached_profile(load_enriched_schema)
        # Project down to the profile rather than passing the library's schema
        # through beside it. The raw node_props/rel_props describe every
        # property the profile also describes, and on the library's sampled
        # branch (any label above its EXHAUSTIVE_SEARCH_LIMIT) the raw copy
        # lists five arbitrary sample values while the profile says
        # completeness "unknown" and withholds them -- so the payload asserts
        # exactly what the profile exists to deny, with the raw copy appearing
        # first. `metadata` (constraints, indexes) goes too: it describes
        # write-time guarantees, not anything a retrieval agent can ask about.
        #
        # `relationships` stays because it is the only exhaustive list of
        # patterns: profile["patterns"] carries the same triples but is what
        # the degree budget acts on. Property names for an entity past the
        # entity budget are NOT recovered here -- the profile marks that entity
        # "not_profiled", which prompt rule 7 tells the agent to disclose.
        schema = {
            "profile": cached["profile"],
            "relationships": cached["schema"].get("relationships", []),
        }
        return tool_success("schema", schema)
    except Exception as e:
        return tool_error(str(e))


def get_physical_schema() -> Dict[str, Any]:
    """Tool to get the physical schema of a Neo4j graph database.

    Returns:
        A dictionary containing:
        - "status": "success" or "error"
        - "schema": the schema as a JSON object if "success"
        - "error_message": the error message if "error"
    """
    return _physical_schema(include_data_profile=False)


def get_graph_schema_with_profile() -> Dict[str, Any]:
    """Get the graph schema together with a profile of the data it holds.

    Returns the node labels, relationship types and properties, plus for each
    property whether its reported values are complete, whether it uniquely
    identifies its entity, and how its values are distributed; and for each
    relationship pattern how many edges it has, how they spread across the
    nodes at each end, and whether a property divides those edges into kinds
    that must not be counted together. Use this before writing any query: it
    tells you the grain of a pattern, which determines whether counting rows
    is meaningful.
    """
    return _physical_schema(include_data_profile=True)


def read_neo4j_cypher(
    query: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Submits a read-only Cypher query to a Neo4j database.

    Args:
        query: The Cypher query string to execute.
        params: Optional parameters to pass to the query.

    Returns:
        A dictionary with "status" and, on success, "query_result" holding:
        - "records": the rows, capped in number
        - "row_count" (or "row_count_at_least" for very large results)
        - "truncated": whether ROWS were dropped
        - "values_summarised": whether an oversized list value inside a row was
          replaced by a summary of its shape. Independent of "truncated": a
          result can return every row while still withholding part of one.
        - "note": guidance on how to proceed, present when either of those is
          true. Not a discriminator -- the two flags above are the record of
          what happened, and when both are true the note carries the
          truncation guidance alone.

        Counts and rankings must come from a Cypher aggregation, never from
        counting the returned records.
    """
    return graphdb.send_read_query(query, params)


def write_neo4j_cypher(
    query: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Submits a Cypher query to write to a Neo4j database.
    Make sure you have permission to write before calling this.

    Args:
        query: The Cypher query string to execute.
        params: Optional parameters to pass to the query.

    Returns:
        A list of dictionaries containing the results of the query.
        Returns an empty list "[]" if no results are found.
    """
    results = graphdb.send_query(query, params)
    return results


def reset_neo4j_data() -> Dict[str, Any]:
    """Resets the neo4j graph database by removing all data,
    indexes and constraints.
    Use with caution! Confirm with the user
    that they know this will completely reset the database.

    Returns:
        Success or an error.
    """
    # First, remove all nodes and relationships in batches
    data_removed = graphdb.send_query(
        """MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS"""
    )
    if is_error(data_removed):
        return data_removed

    # Constraint and index names are interpolated, not parameterised: Cypher
    # does not accept a parameter in a DDL name position, so `DROP CONSTRAINT
    # $constraint_name` is rejected by the server every time -- this function
    # previously dropped nothing at all. The names come from SHOW
    # CONSTRAINTS/INDEXES, i.e. from the database rather than from a model, so
    # they are backtick-quoted with cypher_identifiers.quote() rather than
    # passed through checked(), which rejects legal generated names.
    #
    # The status checks below compare result["status"], not the result dict
    # itself; `result == "error"` compares a dict to a string and is never true,
    # so a failed listing used to fall through into a TypeError on ["records"].

    # remove all constraints
    list_constraints = graphdb.send_query("""SHOW CONSTRAINTS YIELD name""")
    if is_error(list_constraints):
        return list_constraints
    constraint_names = [row["name"] for row in list_constraints["records"]]
    for constraint_name in constraint_names:
        dropped_constraint = graphdb.send_query(
            f"""DROP CONSTRAINT {quote(constraint_name)}"""
        )
        if is_error(dropped_constraint):
            return dropped_constraint

    # remove all indexes
    list_indexes = graphdb.send_query("""SHOW INDEXES YIELD name""")
    if is_error(list_indexes):
        return list_indexes
    index_names = [row["name"] for row in list_indexes["records"]]
    for index_name in index_names:
        dropped_index = graphdb.send_query(f"""DROP INDEX {quote(index_name)}""")
        if is_error(dropped_index):
            return dropped_index

    return tool_success("message", "Neo4j database has been reset.")


# KG-52. The constraint carries no name of ours. IF NOT EXISTS matches on the
# NAME when one is given, and `{label}_{key}_constraint` is ambiguous: label A_b
# with key c and label A with key b_c both gave A_b_c_constraint, so the second
# create was skipped silently and its nodes loaded with no constraint and no
# index. Unnamed, IF NOT EXISTS matches on label and key, so it is skipped
# exactly when a single-key uniqueness constraint on that pair already exists,
# whatever it is called (constraints a 0.8.x build named still match; so do ones
# made by hand). Neo4j generates the name, the same one for the same pair
# (constraint_58c5f77e). Old constraints are neither renamed nor dropped, so a
# graph can hold both kinds.
#
# The check below stays even though names can no longer collide: "success means
# the constraint exists" must hold for any outcome the create does not report.
# On Neo4j 5.26 a constraint squatting the generated name, or a single-property
# node key constraint on the same pair, already makes the create itself fail;
# the case left is a role that may create constraints but not list them. It is
# sent through send_query (write access mode), not send_read_query, because on a
# cluster a read session could land on a follower that has not yet applied the
# create. SHOW UNIQUENESS CONSTRAINTS is used rather than SHOW CONSTRAINTS
# filtered on `type` because the keyword does not depend on the `type` strings,
# which are unverified for newer versions. entityType keeps a relationship
# constraint on a same-named type out; `properties = [$key]` keeps a composite
# constraint that contains the key out. Parameters, so no name is interpolated.
_NODE_UNIQUENESS_FOUND = """SHOW UNIQUENESS CONSTRAINTS
    YIELD entityType, labelsOrTypes, properties
    WHERE entityType = 'NODE' AND labelsOrTypes = [$label] AND properties = [$key]
    RETURN count(*) AS found"""
_SHOWN_CHARACTERS = 80


def create_uniqueness_constraint(
    label: str,
    unique_property_key: str,
) -> Dict[str, Any]:
    """Creates a uniqueness constraint for a node label and property key.
    A uniqueness constraint ensures that no two nodes with the same label and property key have the same value.
    This improves the performance and integrity of data import and later queries.
    After creating it, the tool checks that the database lists a uniqueness
    constraint on exactly this label and property key, and returns an error if not.

    Args:
        label: The label of the node to create a constraint for.
        unique_property_key: The property key that should have a unique value.

    Returns:
        A dictionary with a status key ('success' or 'error').
        On error, includes an 'error_message' key (also when no such constraint exists after the create).
    """
    # Validate input, then quote. checked() refuses anything but a plain
    # identifier for the label, so a label never carries newlines, parens or
    # braces. A key is a column or property of the user's file, so
    # checked_field() refuses only empty text, NUL and over-long names (Aura's
    # limit, kept for every target). quote() then
    # writes each name so that nothing inside it can end its backticks or start
    # a backslash-u escape, and lets a plain identifier that is also a Cypher
    # keyword (Order, END) through as a name.
    try:
        label = checked("label", label)
        unique_property_key = checked_field("property key", unique_property_key)
    except InvalidIdentifier as exc:
        return tool_error(str(exc))

    # Use string formatting since Neo4j doesn't support parameterization of labels and property keys when creating a constraint.
    # No name: see the comment above _NODE_UNIQUENESS_FOUND.
    query = f"""CREATE CONSTRAINT IF NOT EXISTS
    FOR (n:{quote(label)})
    REQUIRE n.{quote(unique_property_key)} IS UNIQUE"""
    results = graphdb.send_query(query)
    if is_error(results):
        return results

    listed = graphdb.send_query(
        _NODE_UNIQUENESS_FOUND, {"label": label, "key": unique_property_key}
    )
    if is_error(listed):
        return listed
    records = listed.get("records") or []
    found = records[0].get("found") if records else None
    if not isinstance(found, int) or found < 1:
        shown = unique_property_key
        if len(shown) > _SHOWN_CHARACTERS:
            shown = shown[:_SHOWN_CHARACTERS] + "..."
        return tool_error(
            f"No single-key uniqueness constraint on {label}/{shown} was found "
            "after the create."
        )
    return results


def merge_node_into_graph(
    label_name: str,
    id_property_name: str,
    properties: Dict[str, Any],
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """Merges a node into the graph. The label_name/id_property_name pair will
    be used for the MERGE pattern to ensure uniqueness.
    The properties dictionary will be used in a SET to set all properties of the node.

    Args:
        label_name: the label of the node to create
        id_property_name: the name of the property that will be used to set the id of the node
        properties: a dictionary of properties to set on the node
        tool_context: ToolContext object.

    Returns:
        dict: A dictionary indicating success or failure.
              Includes a 'status' key ('success' or 'error').
              If 'error', includes an 'error_message' key.
    """
    query = "MERGE (t:$($label_name) {id: $props[$id_property_name]}) SET t += $props"
    properties = {
        "label_name": label_name,
        "id_property_name": id_property_name,
        "props": properties,
    }
    return write_neo4j_cypher(query, properties)


def merge_singleton_node_into_graph(
    label_name: str, properties: Dict[str, Any], tool_context: ToolContext
) -> Dict[str, Any]:
    """Merges a singleton node into the graph. The label_name will be used for the MERGE pattern,
    ensuring a singleton by having no either qualifying properties.
    The properties dictionary will be used in a SET to set all properties of the node.

    Args:
        label_name: the label of the node to create
        id_property_name: the name of the property that will be used to set the id of the node
        properties: a dictionary of properties to set on the node
        tool_context: ToolContext object.

    Returns:
        dict: A dictionary indicating success or failure.
              Includes a 'status' key ('success' or 'error').
              If 'error', includes an 'error_message' key.
    """
    query = "MERGE (t:$($label_name)) SET t += $props"
    properties = {"label_name": label_name, "props": properties}
    return write_neo4j_cypher(query, properties)
