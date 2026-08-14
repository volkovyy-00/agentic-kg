import json
from typing import Optional

from google.adk.tools import ToolContext

from agentic_kg.common.neo4j_for_adk import get_graphdb
from agentic_kg.common.tool_result import tool_error, tool_success
from agentic_kg.common.value_types import ALLOWED_TYPES

graphdb = get_graphdb()

from .file_tools import APPROVED_FILES, search_file
from .reference_reachability import check_reference_columns_are_reachable

PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"
APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"

#  Tool: Propose Node Construction

NODE_CONSTRUCTION = "node_construction"


def _missing_values(**fields) -> list[str]:
    """Names of the given fields that are absent or empty.

    A construction is assembled from an LLM-produced argument list, so a field
    can simply not arrive. Without this, an absent label is stored as a plan
    entry keyed None with "label": None, which check_construction_plan_consistency
    accepts and which only fails much later at import time.
    """
    return [name for name, value in fields.items() if not value]


def propose_node_construction(
    approved_file: str,
    proposed_label: str,
    unique_column_name: str,
    proposed_properties: list[str],
    tool_context: ToolContext,
    proposed_property_types: Optional[dict] = None,
) -> dict:
    """Propose a node construction for an approved file that supports the user goal.

    The construction will be added to the proposed construction plan dictionary under using proposed_label as the key.

    The construction entry will be a dictionary with the following keys:
    - construction_type: "node"
    - source_file: the approved file to propose a node construction for
    - label: the proposed label of the node
    - unique_column_name: the name of the column that will be used to uniquely identify constructed nodes
    - properties: A list of property names for the node, derived from column names in the approved file
    - property_types: An optional map of property name to declared type, one of
      "integer", "float" or "boolean". A property absent from this map is stored
      as text. Never declare a type for the unique_column_name, or for any column
      a relationship joins on -- both are compared as raw text and typing them
      makes the join match nothing.

    Args:
        approved_file: The approved file to propose a node construction for
        proposed_label: The proposed label for constructed nodes (used as key in the construction plan)
        unique_column_name: The name of the column that will be used to uniquely identify constructed nodes
        proposed_properties: The columns of the approved file to store on each
            constructed node
        proposed_property_types: Optional map of property name to "integer",
            "float" or "boolean". Omit or pass {} to store every property as text.

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a "node_construction" key with the construction plan for the node
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    missing = _missing_values(
        approved_file=approved_file,
        proposed_label=proposed_label,
        unique_column_name=unique_column_name,
    )
    if missing:
        return tool_error(
            f"missing required values: {', '.join(missing)}. "
            "Supply every one of them and propose the node again."
        )

    # quick sanity check -- does the approved file have the unique column?
    search_results = search_file(approved_file, unique_column_name)
    if search_results["status"] == "error":
        return search_results  # return the error
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(
            f"{approved_file} does not have the column {unique_column_name}. Check the file content and try again."
        )

    # get the current construction plan, or an empty one if none exists
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    node_construction_rule = {
        "construction_type": "node",
        "source_file": approved_file,
        "label": proposed_label,
        "unique_column_name": unique_column_name,
        # A model may send a JSON null here rather than omitting the field. That
        # reaches Cypher as FOREACH (k IN null | ...), which is a silent no-op:
        # the nodes load with no properties at all and nothing reports a problem.
        "properties": proposed_properties or [],
        # Same defence, same reason: a null here would reach the loader as a key
        # that reads as "typed" and fail on .items(). Absent means text.
        "property_types": proposed_property_types or {},
    }
    construction_plan[proposed_label] = node_construction_rule
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success(NODE_CONSTRUCTION, node_construction_rule)


def propose_node_constructions(
    node_constructions: list[dict], tool_context: ToolContext
) -> dict:
    """Propose several node constructions at once, instead of one call per node.

    Each entry is proposed with the same rules and the same validation as
    'propose_node_construction'. Proposing stops at the first entry that fails, so
    earlier entries stay in the plan and the error names the entry to correct.

    Args:
        node_constructions: a list of dictionaries, each with the keys
            'approved_file', 'proposed_label', 'unique_column_name',
            'proposed_properties' and the optional 'proposed_property_types',
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
            node_construction.get("proposed_property_types", {}),
        )
        if result["status"] == "error":
            return tool_error(
                f"node construction {index} ({node_construction.get('proposed_label')}) failed: "
                f"{result['error_message']}"
            )
        proposed.append(result[NODE_CONSTRUCTION])
    return tool_success(NODE_CONSTRUCTION, proposed)


# Tool: Remove Node Construction
def remove_node_construction(node_label: str, tool_context: ToolContext) -> dict:
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
        return tool_success(
            "node_construction_removed",
            "node construction rule not found. removal not needed.",
        )

    del construction_plan[node_label]

    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success("node_construction_removed", node_label)


#  Tool: Propose Relationship Construction

RELATIONSHIP_CONSTRUCTION = "relationship_construction"


def propose_relationship_construction(
    approved_file: str,
    proposed_relationship_type: str,
    from_node_label: str,
    from_node_column: str,
    to_node_label: str,
    to_node_column: str,
    proposed_properties: list[str],
    tool_context: ToolContext,
    proposed_property_types: Optional[dict] = None,
) -> dict:
    """Propose a relationship construction for an approved file that supports the user goal.

    The construction will be added to the proposed construction plan dictionary under using proposed_relationship_type as the key.

    The construction entry will be a dictionary with the following keys:
    - property_types: An optional map of property name to declared type, one of
      "integer", "float" or "boolean". A property absent from this map is stored
      as text. Never declare a type for from_node_column or to_node_column: they
      are compared against the stored node property as raw text, so typing them
      makes the relationship match nothing.

    Args:
        approved_file: The approved file to propose a relationship construction for
        proposed_relationship_type: The proposed label for constructed relationships
        from_node_label: The label of the source node
        from_node_column: The name of the column within the approved file that will be used to uniquely identify source nodes
        to_node_label: The label of the target node
        to_node_column: The name of the column within the approved file that will be used to uniquely identify target nodes
        proposed_properties: The columns of the approved file to store on each
            constructed relationship
        proposed_property_types: Optional map of property name to "integer",
            "float" or "boolean". Omit or pass {} to store every property as text.

    Returns:
        dict: A dictionary containing metadata about the content.
                Includes a 'status' key ('success' or 'error').
                If 'success', includes a "relationship_construction" key with the construction plan for the node
                If 'error', includes an 'error_message' key.
                The 'error_message' may have instructions about how to handle the error.
    """
    missing = _missing_values(
        approved_file=approved_file,
        proposed_relationship_type=proposed_relationship_type,
        from_node_label=from_node_label,
        from_node_column=from_node_column,
        to_node_label=to_node_label,
        to_node_column=to_node_column,
    )
    if missing:
        return tool_error(
            f"missing required values: {', '.join(missing)}. "
            "Supply every one of them and propose the relationship again."
        )

    # quick sanity check -- does the approved file have the from_node_column?
    search_results = search_file(approved_file, from_node_column)
    if search_results["status"] == "error":
        return search_results  # return the error if there is one
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(
            f"{approved_file} does not have the from node column {from_node_column}. Check the content of the file and reconsider the relationship."
        )

    # quick sanity check -- does the approved file have the to_node_column?
    search_results = search_file(approved_file, to_node_column)
    if (
        search_results["status"] == "error"
        or search_results["search_results"]["metadata"]["lines_found"] == 0
    ):
        return tool_error(
            f"{approved_file} does not have the to node column {to_node_column}. Check the content of the file and reconsider the relationship."
        )

    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    relationship_construction_rule = {
        "construction_type": "relationship",
        "source_file": approved_file,
        "relationship_type": proposed_relationship_type,
        "from_node_label": from_node_label,
        "from_node_column": from_node_column,
        "to_node_label": to_node_label,
        "to_node_column": to_node_column,
        # See propose_node_construction: a null reaches Cypher as a silent no-op.
        "properties": proposed_properties or [],
        # Same defence, same reason: a null here would reach the loader as a key
        # that reads as "typed" and fail on .items(). Absent means text.
        "property_types": proposed_property_types or {},
    }
    construction_plan[proposed_relationship_type] = relationship_construction_rule
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success(RELATIONSHIP_CONSTRUCTION, relationship_construction_rule)


def propose_relationship_constructions(
    relationship_constructions: list[dict], tool_context: ToolContext
) -> dict:
    """Propose several relationship constructions at once, instead of one call per relationship.

    Each entry is proposed with the same rules and the same validation as
    'propose_relationship_construction'. Proposing stops at the first entry that fails, so
    earlier entries stay in the plan and the error names the entry to correct.

    Args:
        relationship_constructions: a list of dictionaries, each with the keys
            'approved_file', 'proposed_relationship_type', 'from_node_label',
            'from_node_column', 'to_node_label', 'to_node_column',
            'proposed_properties' and the optional 'proposed_property_types',
            matching the arguments of 'propose_relationship_construction'

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
            relationship_construction.get("proposed_property_types", {}),
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
def remove_relationship_construction(
    relationship_type: str, tool_context: ToolContext
) -> dict:
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
        return tool_success(
            "relationship_construction_removed",
            "relationship construction rule not found. removal not needed.",
        )

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
        key: rule
        for key, rule in construction_plan.items()
        if isinstance(rule, dict) and rule.get("construction_type") == "node"
    }
    problems = []

    # Every column a relationship joins on, so a property can be checked against
    # the whole plan rather than only its own construction. This is what makes a
    # type retroactively invalid when a later relationship joins on it.
    joined_columns = {}
    for key, rule in construction_plan.items():
        if (
            not isinstance(rule, dict)
            or rule.get("construction_type") != "relationship"
        ):
            continue
        for label, column in (
            (rule.get("from_node_label"), rule.get("from_node_column")),
            (rule.get("to_node_label"), rule.get("to_node_column")),
        ):
            joined_columns.setdefault((label, column), []).append(key)

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
        if (
            not isinstance(rule, dict)
            or rule.get("construction_type") != "relationship"
        ):
            continue
        check_endpoint(
            key, "from", rule.get("from_node_label"), rule.get("from_node_column")
        )
        check_endpoint(key, "to", rule.get("to_node_label"), rule.get("to_node_column"))

    # Declared property types. Three rules, all refusing at approval time rather
    # than failing much later at import time.
    for key, rule in construction_plan.items():
        if not isinstance(rule, dict):
            continue
        property_types = rule.get("property_types") or {}
        if not isinstance(property_types, dict):
            problems.append(
                f"{key}: 'property_types' must be a map of property name to type, "
                f"got {type(property_types).__name__}. Supply a map or omit it."
            )
            continue

        properties = rule.get("properties") or []
        unique_column = rule.get("unique_column_name")

        for name, declared in property_types.items():
            if declared not in ALLOWED_TYPES:
                problems.append(
                    f"{key}: property '{name}' declares unknown type '{declared}'. "
                    f"Use one of {', '.join(ALLOWED_TYPES)}, or drop the type to "
                    f"store it as text."
                )

            if name not in properties:
                problems.append(
                    f"{key}: '{name}' has a declared type but is not in the "
                    f"properties list {sorted(properties)}, so nothing would load "
                    f"it. Either add '{name}' to properties or drop its type."
                )

            if unique_column is not None and name == unique_column:
                problems.append(
                    f"{key}: '{name}' is this node's unique identifier and must "
                    f"stay text — identifiers are matched as raw CSV values, so a "
                    f"typed identifier matches nothing. Drop the type for '{name}'."
                )

            if rule.get("construction_type") == "relationship":
                joining = (
                    [key]
                    if name
                    in (rule.get("from_node_column"), rule.get("to_node_column"))
                    else []
                )
                own_node_label = (
                    rule.get("from_node_label")
                    if name == rule.get("from_node_column")
                    else rule.get("to_node_label")
                )
                own_node_rule = nodes.get(own_node_label)
                join_target = (
                    own_node_rule.get("unique_column_name") if own_node_rule else None
                )
            else:
                label = rule.get("label", key)
                joining = joined_columns.get((label, name), [])
                own_node_label = label
                join_target = unique_column
            if joining:
                if join_target is not None:
                    second_exit = (
                        f"join {', '.join(sorted(joining))} on '{join_target}' instead"
                    )
                else:
                    second_exit = (
                        f"'{own_node_label}' has no node construction in this plan, "
                        f"so there is no identifier to join on instead"
                    )
                problems.append(
                    f"{key}: '{name}' carries a declared type but "
                    f"{', '.join(sorted(joining))} joins on it. Join columns are "
                    f"compared as raw CSV text, so a typed column matches zero "
                    f"rows with no error. Either drop the type for '{name}', or "
                    f"{second_exit}."
                )

    return problems


NO_PROPOSED_PLAN_MESSAGE = (
    "There is no proposed construction plan to approve. "
    "Produce one first, then present it to the user."
)


def _read_plan_for_approval(
    tool_context: ToolContext,
) -> tuple[dict | None, list[str], list[str]]:
    """Read the proposed plan and whatever would make approval refuse it.

    Approval and its dry run both read their preconditions through here, so the
    dry run's verdict cannot drift from what approval actually does: a second
    precondition added here binds both at once. Copying the preconditions into
    each tool instead would make that equivalence hold only by convention, and
    a dry run that reports success where approval refuses is the exact bug the
    dry run exists to prevent.

    The reachability check reads the approved files, unlike the structural one.
    It fails open: a source it cannot read produces a note in the third return
    value, never a problem, so an unreachable disk cannot make a plan unshowable.

    Callers phrase their own refusals -- only the facts are shared. Returns
    (None, [], []) when there is no plan to read.
    """
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN)
    if not construction_plan:
        return None, [], []

    problems = check_construction_plan_consistency(construction_plan)
    reachability_problems, unverified = check_reference_columns_are_reachable(
        construction_plan, tool_context.state.get(APPROVED_FILES) or []
    )
    return construction_plan, problems + reachability_problems, unverified


# Tool: Approve the proposed construction plan
def approve_proposed_construction_plan(tool_context: ToolContext) -> dict:
    """Approve the proposed construction plan, if it is internally consistent.

    Approval is refused when a relationship construction joins on a column the
    referenced node does not carry, or names an endpoint label that has no node
    construction in the plan, since such a plan cannot build the graph that was
    described to the user no matter what was said in conversation.
    """
    construction_plan, problems, unverified = _read_plan_for_approval(tool_context)
    if construction_plan is None:
        return tool_error(NO_PROPOSED_PLAN_MESSAGE)

    notes = "\n\nNot verified:\n- " + "\n- ".join(unverified) if unverified else ""

    if problems:
        return tool_error(
            "The proposed construction plan was NOT approved. It is inconsistent, "
            "or it leaves an approved file's reference column unreachable:\n- "
            + "\n- ".join(problems)
            + "\nFix the plan, then show the user the corrected plan returned by "
            "'get_proposed_construction_plan_with_approval_check' and ask them to "
            "approve again. Do not describe the plan as fixed until "
            "'get_proposed_construction_plan_with_approval_check' reports success."
            + notes
        )

    tool_context.state[APPROVED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success(
        "result",
        {"approved_construction_plan": construction_plan, "not_verified": unverified},
    )


def get_proposed_construction_plan_with_approval_check(
    tool_context: ToolContext,
) -> dict:
    """Get the proposed construction plan, and whether it can be approved right now.

    Use this whenever you are about to show the user a construction plan. It runs
    the same consistency check 'approve_proposed_construction_plan' runs, without
    approving anything, so its answer is exactly what approval will do.

    An error result means approval would be refused, and the message lists the
    problems that would cause it. A success result means nothing is blocking
    approval, and carries both the plan and what to do next.
    """
    construction_plan, problems, unverified = _read_plan_for_approval(tool_context)
    if construction_plan is None:
        return tool_error(NO_PROPOSED_PLAN_MESSAGE)

    notes = "\n\nNot verified:\n- " + "\n- ".join(unverified) if unverified else ""

    if problems:
        return tool_error(
            # States the fact and stops. It deliberately does NOT say "run
            # schema_refinement_loop with these as feedback": that is right
            # mid-turn but contradicts the instruction outright on the second
            # 'retry' and on 'stopped:', where the standing rule is to stop
            # calling the loop. Only the instruction knows which branch it is
            # in, and it already says what to do in each.
            "This plan cannot be approved as it stands. "
            "'approve_proposed_construction_plan' will refuse it for:\n- "
            + "\n- ".join(problems)
            # Carries the plan even though approval would refuse it. The
            # coordinator's instruction forbids describing a plan from memory
            # and requires reproducing this tool's returned fields, and this is
            # its only plan-reading tool -- so an error that withheld the plan
            # would leave it nothing to show on exactly the branch where the
            # user most needs to see what is wrong. Refusing approval is not
            # the same as refusing to show.
            + "\n\nThe plan as it currently stands is:\n"
            + json.dumps(construction_plan, indent=2)
            + "\nShow the user this plan and these problems if your instruction "
            "has you presenting it, but do not present it for approval until "
            "this tool reports success." + notes
        )

    return tool_success(
        "result",
        {
            "proposed_construction_plan": construction_plan,
            "not_verified": unverified,
            "message": (
                # Does NOT claim the critic's remaining objections are advisory
                # or that no schema change will clear them: this tool cannot see
                # which branch the coordinator is in, and on a first 'retry' the
                # instruction mandates another schema_refinement_loop pass on an
                # objection this consistency check has no way to observe (it
                # only knows joins, endpoint labels, and typed join columns). An
                # unconditional claim here would be true on 'stopped:' and the
                # second 'retry' but false on the first, contradicting the
                # instruction on exactly the branch where refinement is still
                # required.
                "This plan can be approved right now: "
                "'approve_proposed_construction_plan' will accept it as it stands. "
                "That is all this tool knows: it checks joins, endpoint labels, "
                "typed columns, and whether every approved file's reference "
                "columns can still be reached, not whether the plan is the right "
                "one. When your instruction has you presenting this plan, show it "
                "to the user together with any outstanding critic objections, ask "
                "them to approve it as it stands or ask for a change, and leave "
                "that decision to them -- when you present it, do not tell them "
                "the plan is not ready for approval."
            ),
        },
    )


# Tool: Get Proposed construction Plan


def get_proposed_construction_plan(tool_context: ToolContext) -> dict:
    """Get the proposed construction plan."""
    return tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, [])


def get_approved_construction_plan(tool_context: ToolContext) -> dict:
    """Get the approved construction plan."""
    return tool_context.state.get(APPROVED_CONSTRUCTION_PLAN, [])
