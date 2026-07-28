from google.adk.tools import ToolContext
from typing import Dict, Any

from agentic_kg.common.neo4j_for_adk import get_graphdb
from agentic_kg.common.tool_result import tool_success, tool_error

graphdb = get_graphdb()

from .file_tools import search_file

PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"
APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"

#  Tool: Propose Node Construction

NODE_CONSTRUCTION = "node_construction"

def propose_node_construction(approved_file: str, proposed_label: str, unique_column_name: str, proposed_properties: list[str], tool_context:ToolContext) -> dict:
    """Propose a node construction for an approved file that supports the user goal.

    The construction will be added to the proposed construction plan dictionary under using proposed_label as the key.

    The construction entry will be a dictionary with the following keys:
    - construction_type: "node"
    - source_file: the approved file to propose a node construction for
    - label: the proposed label of the node
    - unique_column_name: the name of the column that will be used to uniquely identify constructed nodes
    - properties: A list of property names for the node, derived from column names in the approved file

    Args:
        approved_file: The approved file to propose a node construction for
        proposed_label: The proposed label for constructed nodes (used as key in the construction plan)
        unique_column_name: The name of the column that will be used to uniquely identify constructed nodes

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a "node_construction" key with the construction plan for the node
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    # quick sanity check -- does the approved file have the unique column?
    search_results = search_file(approved_file, unique_column_name)
    if search_results["status"] == "error":
        return search_results # return the error
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have the column {unique_column_name}. Check the file content and try again.")

    # get the current construction plan, or an empty one if none exists
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    node_construction_rule = {
        "construction_type": "node",
        "source_file": approved_file,
        "label": proposed_label,
        "unique_column_name": unique_column_name,
        "properties": proposed_properties
    }   
    construction_plan[proposed_label] = node_construction_rule
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success(NODE_CONSTRUCTION, node_construction_rule)

def propose_node_constructions(node_constructions: list[dict], tool_context:ToolContext) -> dict:
    """Propose several node constructions at once, instead of one call per node.

    Each entry is proposed with the same rules and the same validation as
    'propose_node_construction'. Proposing stops at the first entry that fails, so
    earlier entries stay in the plan and the error names the entry to correct.

    Args:
        node_constructions: a list of dictionaries, each with the keys
            'approved_file', 'proposed_label', 'unique_column_name' and 'proposed_properties',
            matching the arguments of 'propose_node_construction'

    Returns:
        dict: Includes a 'status' key ('success' or 'error').
                If 'success', includes a "node_construction" key with the list of construction rules.
                If 'error', includes an 'error_message' key naming the entry that failed.
    """
    proposed = []
    for index, node_construction in enumerate(node_constructions):
        result = propose_node_construction(
            node_construction.get("approved_file"),
            node_construction.get("proposed_label"),
            node_construction.get("unique_column_name"),
            node_construction.get("proposed_properties", []),
            tool_context,
        )
        if result["status"] == "error":
            return tool_error(
                f"node construction {index} ({node_construction.get('proposed_label')}) failed: "
                f"{result['error_message']}"
            )
        proposed.append(result[NODE_CONSTRUCTION])
    return tool_success(NODE_CONSTRUCTION, proposed)

# Tool: Remove Node Construction
def remove_node_construction(node_label: str, tool_context:ToolContext) -> dict:
    """Remove a node construction from the proposed construction plan based on label.

    Args:
        node_label: The label of the node construction to remove
        tool_context: The tool context

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a 'node_construction_removed' key with either the label of the
                    removed node construction, or a message indicating no removal was needed
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    if node_label not in construction_plan:
        return tool_success("node_construction_removed", "node construction rule not found. removal not needed.")

    del construction_plan[node_label]

    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success("node_construction_removed", node_label)

#  Tool: Propose Relationship Construction

RELATIONSHIP_CONSTRUCTION = "relationship_construction"

def propose_relationship_construction(approved_file: str, proposed_relationship_type: str, 
    from_node_label: str,from_node_column: str, to_node_label:str, to_node_column: str, 
    proposed_properties: list[str], 
    tool_context:ToolContext) -> dict:
    """Propose a relationship construction for an approved file that supports the user goal.

    The construction will be added to the proposed construction plan dictionary under using proposed_relationship_type as the key.

    Args:
        approved_file: The approved file to propose a node construction for
        proposed_relationship_type: The proposed label for constructed relationships
        from_node_label: The label of the source node
        from_node_column: The name of the column within the approved file that will be used to uniquely identify source nodes
        to_node_label: The label of the target node
        to_node_column: The name of the column within the approved file that will be used to uniquely identify target nodes
        unique_column_name: The name of the column that will be used to uniquely identify target nodes

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a "relationship_construction" key with the construction plan for the node
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    # quick sanity check -- does the approved file have the from_node_column?
    search_results = search_file(approved_file, from_node_column)
    if search_results["status"] == "error": 
      return search_results  # return the error if there is one
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have the from node column {from_node_column}. Check the content of the file and reconsider the relationship.")

    # quick sanity check -- does the approved file have the to_node_column?
    search_results = search_file(approved_file, to_node_column)
    if search_results["status"] == "error" or search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have the to node column {to_node_column}. Check the content of the file and reconsider the relationship.")

    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    relationship_construction_rule = {
        "construction_type": "relationship",
        "source_file": approved_file,
        "relationship_type": proposed_relationship_type,
        "from_node_label": from_node_label,
        "from_node_column": from_node_column,
        "to_node_label": to_node_label,
        "to_node_column": to_node_column,
        "properties": proposed_properties
    }   
    construction_plan[proposed_relationship_type] = relationship_construction_rule
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success(RELATIONSHIP_CONSTRUCTION, relationship_construction_rule)

def propose_relationship_constructions(relationship_constructions: list[dict], tool_context:ToolContext) -> dict:
    """Propose several relationship constructions at once, instead of one call per relationship.

    Each entry is proposed with the same rules and the same validation as
    'propose_relationship_construction'. Proposing stops at the first entry that fails, so
    earlier entries stay in the plan and the error names the entry to correct.

    Args:
        relationship_constructions: a list of dictionaries, each with the keys
            'approved_file', 'proposed_relationship_type', 'from_node_label', 'from_node_column',
            'to_node_label', 'to_node_column' and 'proposed_properties', matching the arguments
            of 'propose_relationship_construction'

    Returns:
        dict: Includes a 'status' key ('success' or 'error').
                If 'success', includes a "relationship_construction" key with the list of construction rules.
                If 'error', includes an 'error_message' key naming the entry that failed.
    """
    proposed = []
    for index, relationship_construction in enumerate(relationship_constructions):
        result = propose_relationship_construction(
            relationship_construction.get("approved_file"),
            relationship_construction.get("proposed_relationship_type"),
            relationship_construction.get("from_node_label"),
            relationship_construction.get("from_node_column"),
            relationship_construction.get("to_node_label"),
            relationship_construction.get("to_node_column"),
            relationship_construction.get("proposed_properties", []),
            tool_context,
        )
        if result["status"] == "error":
            return tool_error(
                f"relationship construction {index} "
                f"({relationship_construction.get('proposed_relationship_type')}) failed: "
                f"{result['error_message']}"
            )
        proposed.append(result[RELATIONSHIP_CONSTRUCTION])
    return tool_success(RELATIONSHIP_CONSTRUCTION, proposed)

# Tool: Remove Relationship Construction
def remove_relationship_construction(relationship_type: str, tool_context:ToolContext) -> dict:
    """Remove a relationship construction from the proposed construction plan based on type.

    Args:
        relationship_type: The type of the relationship construction to remove
        tool_context: The tool context

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a 'relationship_construction_removed' key with the type of the removed relationship construction
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})

    if relationship_type not in construction_plan:
        return tool_success("relationship_construction_removed", "relationship construction rule not found. removal not needed.")
    
    construction_plan.pop(relationship_type)
    
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success("relationship_construction_removed", relationship_type) 


def check_construction_plan_consistency(construction_plan: dict) -> list[str]:
    """Find internal inconsistencies between relationship joins and node constructions.

    Purely structural: it compares the plan against itself, with no file or database
    access. It exists because a relationship can only ever match nodes if its join
    column is a value the referenced node actually carries — its unique identifier,
    or one of its stored properties. A join on any other column silently produces
    zero relationships at build time (and the relationship type never appears in the
    database at all), which is indistinguishable from success in the tool output.

    This is also the mechanical check that catches a revision drifting out of sync:
    if a node's unique identifier is changed (or reverted) without the relationships
    that join on it being updated to match, the plan becomes inconsistent here.

    Args:
        construction_plan: the construction plan dictionary, keyed by label/type

    Returns:
        list[str]: a problem description per inconsistency; empty if the plan is consistent
    """
    if not isinstance(construction_plan, dict):
        return []

    nodes = {
        key: rule for key, rule in construction_plan.items()
        if isinstance(rule, dict) and rule.get("construction_type") == "node"
    }
    problems = []

    def check_endpoint(rel_key, side, label, column):
        node_rule = nodes.get(label)
        if node_rule is None:
            problems.append(
                f"{rel_key}: {side} node label '{label}' has no node construction in the plan."
            )
            return
        unique_column = node_rule.get("unique_column_name")
        known_columns = {unique_column, *(node_rule.get("properties") or [])}
        if column not in known_columns:
            problems.append(
                f"{rel_key}: {side} join column '{column}' is not a column of the "
                f"'{label}' node, which is keyed by '{unique_column}' with properties "
                f"{sorted(c for c in known_columns if c and c != unique_column)}. "
                f"This join would match zero rows. Either key '{label}' by '{column}' "
                f"or join on '{unique_column}'."
            )

    for key, rule in construction_plan.items():
        if not isinstance(rule, dict) or rule.get("construction_type") != "relationship":
            continue
        check_endpoint(key, "from", rule.get("from_node_label"), rule.get("from_node_column"))
        check_endpoint(key, "to", rule.get("to_node_label"), rule.get("to_node_column"))

    return problems


# Tool: Approve the proposed construction plan
def approve_proposed_construction_plan(tool_context:ToolContext) -> dict:
    """Approve the proposed construction plan, if it is internally consistent.

    Approval is refused when a relationship construction joins on a column the
    referenced node does not carry, or names an endpoint label that has no node
    construction in the plan, since such a plan cannot build the graph that was
    described to the user no matter what was said in conversation.
    """
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN)
    if not construction_plan:
        return tool_error(
            "There is no proposed construction plan to approve. "
            "Produce one first, then present it to the user."
        )

    problems = check_construction_plan_consistency(construction_plan)
    if problems:
        return tool_error(
            "The proposed construction plan is internally inconsistent and was NOT approved:\n- "
            + "\n- ".join(problems)
            + "\nFix the plan, then show the user the corrected plan returned by "
            "'get_proposed_construction_plan' and ask them to approve again. Do not "
            "describe the plan as fixed until this tool reports success."
        )

    tool_context.state[APPROVED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success(APPROVED_CONSTRUCTION_PLAN, tool_context.state[APPROVED_CONSTRUCTION_PLAN])

# Tool: Get Proposed construction Plan

def get_proposed_construction_plan(tool_context:ToolContext) -> dict:
    """Get the proposed construction plan."""
    return tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, [])


def get_approved_construction_plan(tool_context:ToolContext) -> dict:
    """Get the approved construction plan."""
    return tool_context.state.get(APPROVED_CONSTRUCTION_PLAN, [])