import json
from typing import Any, Optional, Protocol

from google.adk.tools import ToolContext

from agentic_kg.common.cypher_identifiers import InvalidIdentifier
from agentic_kg.common.neo4j_for_adk import get_graphdb
from agentic_kg.common.tool_result import tool_error, tool_success
from agentic_kg.common.value_types import ALLOWED_TYPES

graphdb = get_graphdb()

from .file_tools import APPROVED_FILES, check_columns_in_header
from .join_property_check import check_joined_properties_hold_one_value
from .kg_construction_tools import (
    APPROVED_CONSTRUCTION_PLAN,
    NOT_APPROVED_MESSAGE,
    approved_plan,
    matched_property_name_problem,
    node_rule_name_problem,
    required_relationship_name_problem,
)
from .node_key_check import node_key_refusal, summarize_node_key
from .reference_reachability import (
    check_reference_columns_are_reachable,
    declared_properties,
)
from .relationship_endpoints import is_omitted, relationship_endpoints

PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"

# Added after the build's own refusal text (KG-44), but only where a way forward
# exists: a label or relationship type can be renamed to follow its rule. A
# key, join column or matched property is the file's own header (KG-51), so a
# refusal of one (empty, NUL, too long) says what it needs and no more.
_NAME_HINT = "A label or relationship type can be renamed to follow this rule."
# A relationship end can also name the node property it is matched on (KG-45);
# only a refusal of that property says this.
_MATCHED_PROPERTY_HINT = (
    "A matched node property must be spelled exactly as the node stores it; omit it "
    "when the node stores the value under the column's own name."
)


def _refusal(problem: InvalidIdentifier, *, matched: bool = False) -> str:
    """The build's own text, then the way forward that fits the name that failed."""
    if problem.renamable:
        return f"{problem} {_NAME_HINT}"
    if matched:
        return f"{problem} {_MATCHED_PROPERTY_HINT}"
    return str(problem)


# One name, one rule (KG-39). The plan is a single map filed by a node's label or a
# relationship's type, so a node and a relationship spelled alike would share a
# key and the later proposal would replace the earlier rule without a word. The
# propose tools therefore refuse a name the other kind holds, and the remove tools
# leave it alone. Neo4j itself allows a label and a relationship type with the
# same name; a plan that wants both cannot be built, and the refusal asks for a
# different name instead. Refusing, rather than filing by (kind, name), keeps
# every reader of the plan sound: they resolve a node by its label as the key.


_OTHER_KIND = {"node": "relationship", "relationship": "node"}


def _rule_kind(plan: dict, name: str) -> Optional[str]:
    """The `construction_type` of the rule filed under `name`; None for an entry
    that is not a rule, which only a hand-edited state can hold."""
    rule = plan.get(name)
    return rule.get("construction_type") if isinstance(rule, dict) else None


def _other_kind_holding(plan: dict, name: str, kind: str) -> Optional[str]:
    """The kind opposite `kind` ("node" or "relationship") when it holds `name`,
    else None: the name is free, or `kind` itself holds it."""
    other = _OTHER_KIND[kind]
    return other if _rule_kind(plan, name) == other else None


def _collision_refusal(plan: dict, name: str, proposing: str) -> Optional[str]:
    """Why proposing a `proposing` rule under `name` must be refused, or None.
    A same-kind re-proposal is not refused: it replaces the rule, as it always has."""
    held = _other_kind_holding(plan, name, proposing)
    if held is None:
        return None
    shared = (
        f"The plan already has a {held} rule named '{name}'. A node label and a "
        "relationship type share one name in the plan, so the existing rule is "
        "unchanged."
    )
    if proposing == "relationship":
        return f"{shared} Propose this relationship under a different type."
    # Renaming a node can leave relationships pointing at its old label, which
    # approval then reports as a missing node: say which way out is cheaper.
    return (
        f"{shared} Either rename this label (only if no relationship refers to it "
        "yet), or remove that relationship rule with remove_relationship_construction "
        "and re-propose it under a different type."
    )


def _wrong_kind_removal_refusal(plan: dict, name: str, removing: str) -> Optional[str]:
    """Why removing a `removing` rule by `name` must be refused because the other
    kind holds the name, or None."""
    held = _other_kind_holding(plan, name, removing)
    if held is None:
        return None
    return (
        f"'{name}' is a {held} rule, not a {removing} rule, so nothing was removed. "
        f"Use remove_{held}_construction to remove it."
    )


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
      "integer", "float", "boolean", "date", "datetime" or "localdatetime". A
      property absent from this map is stored as text. "date" is a day
      ("2025-03-04"); "datetime" is a timestamp with an offset or Z;
      "localdatetime" is one with neither. Declare the type column_type_hint
      reports. Never declare a type for the unique_column_name, or for any column
      a relationship joins on -- both are compared as raw text and typing them
      makes the join match nothing.

    The label must be a letter or underscore followed by letters, digits or
    underscores. Cypher keywords such as Order or END are fine. The unique column
    may be any header of the approved file, spelled as the file spells it (such as
    'Order ID' or 'customer-id'): do not rename or reformat it.

    The plan holds one rule per name. If the label is already the type of a
    relationship rule in the plan, the proposal is refused and that rule is left
    as it is: choose another label. Proposing a node under a label another node
    already has still replaces that node.

    The unique column must be exactly one of the approved file's header names,
    letter for letter and in the same case. If it is not, the proposal is refused
    and the error lists the file's headers: choose one of them rather than trying
    another spelling.

    The unique column must have a value in every row. It may repeat across rows
    only if the rows sharing a value agree on every property you list for the node,
    because the build keeps one node per value and one row's values on it. If it
    does not, the proposal is refused: the error gives the row, distinct and blank
    counts and names the properties that disagree. Choose another key, model the
    file as a relationship, or leave those properties off the node.

    Args:
        approved_file: The approved file to propose a node construction for
        proposed_label: The proposed label for constructed nodes (used as key in the construction plan)
        unique_column_name: The name of the column that will be used to uniquely identify constructed nodes
        proposed_properties: The columns of the approved file to store on each
            constructed node
        proposed_property_types: Optional map of property name to "integer",
            "float", "boolean", "date", "datetime" or "localdatetime". Omit or
            pass {} to store every property as text.

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

    # Names first, before the file is read (KG-44): a name the build would refuse
    # is refused now, with the build's own text, whatever the file holds.
    problem = node_rule_name_problem(node_construction_rule)
    if problem is not None:
        return tool_error(_refusal(problem))

    # A name the other kind holds (KG-39): found from the plan alone, so before
    # any file is read.
    collision = _collision_refusal(
        tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {}), proposed_label, "node"
    )
    if collision is not None:
        return tool_error(collision)

    # Exact header match, as the build does (KG-50).
    column_error = check_columns_in_header(approved_file, [unique_column_name])
    if column_error is not None:
        return column_error

    # Then the key's own values, and the properties of a repeating key (KG-48).
    # After the names and the header, so a bad name never costs a data read.
    key_summary, key_error = summarize_node_key(
        approved_file, unique_column_name, declared_properties(node_construction_rule)
    )
    if key_error is not None:
        return key_error
    assert key_summary is not None
    key_refusal = node_key_refusal(
        proposed_label, approved_file, unique_column_name, key_summary
    )
    if key_refusal is not None:
        return tool_error(key_refusal)

    # get the current construction plan, or an empty one if none exists
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
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
            node_construction.get("approved_file", ""),
            node_construction.get("proposed_label", ""),
            node_construction.get("unique_column_name", ""),
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

    If the label belongs to a relationship rule, nothing is removed and the error
    names 'remove_relationship_construction'.

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
    wrong_kind = _wrong_kind_removal_refusal(construction_plan, node_label, "node")
    if wrong_kind is not None:
        return tool_error(wrong_kind)
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
    from_node_property: Optional[str] = None,
    to_node_property: Optional[str] = None,
) -> dict:
    """Propose a relationship construction for an approved file that supports the user goal.

    The construction will be added to the proposed construction plan dictionary under using proposed_relationship_type as the key.

    Each end reads one column of the approved file and matches that value against
    one property of the end's node. By default the property has the column's own
    name. When the column holds the node's key (or another of its properties)
    under a different name -- including a column holding the key of another node
    with the same label -- pass that property as from_node_property or
    to_node_property; the column itself stays the file's column.

    The construction entry will be a dictionary with the following keys:
    - property_types: An optional map of property name to declared type, one of
      "integer", "float", "boolean", "date", "datetime" or "localdatetime". A
      property absent from this map is stored as text. "date" is a day
      ("2025-03-04"); "datetime" is a timestamp with an offset or Z;
      "localdatetime" is one with neither. Declare the type column_type_hint
      reports. Never declare a type for from_node_column or to_node_column: they
      are compared against the stored node property as raw text, so typing them
      makes the relationship match nothing. The same holds for the node property
      an end is matched on: declare no type for it on its node either.

    The relationship type and both node labels must each be a letter or underscore
    followed by letters, digits or underscores. Cypher keywords such as Order or
    END are fine. A join column, and a matched node property, may be any text the
    file or node holds, spelled exactly (such as 'Order ID' or 'customer-id'): do
    not rename or reformat it.

    The plan holds one rule per name. If the type is already the label of a node
    rule in the plan, the proposal is refused and that rule is left as it is:
    choose another type. Proposing a relationship under a type another
    relationship already has still replaces that relationship.

    Both join columns must be exactly header names of the approved file, letter
    for letter and in the same case. If either is not, the proposal is refused and
    the error names each missing column and lists the file's headers: choose from
    that list rather than trying another spelling.

    Args:
        approved_file: The approved file to propose a relationship construction for
        proposed_relationship_type: The proposed label for constructed relationships
        from_node_label: The label of the source node
        from_node_column: The column of the approved file whose value identifies
            the from node of each row
        to_node_label: The label of the target node
        to_node_column: The column of the approved file whose value identifies
            the to node of each row
        proposed_properties: The columns of the approved file to store on each
            constructed relationship
        proposed_property_types: Optional map of property name to "integer",
            "float", "boolean", "date", "datetime" or "localdatetime". Omit or
            pass {} to store every property as text.
        from_node_property: Optional. The property of the from node that the
            from_node_column value is matched against. Omit it when the from node
            stores that value under the column's own name.
        to_node_property: Optional. The property of the to node that the
            to_node_column value is matched against. Omit it when the to node
            stores that value under the column's own name.

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
    # KG-45: stored only when given, so a plan that never uses the field stays
    # byte-identical. "" counts as omitted (is_omitted); anything else is kept
    # so the name check below refuses it rather than dropping it.
    for side, value in (("from", from_node_property), ("to", to_node_property)):
        if not is_omitted(value):
            relationship_construction_rule[f"{side}_node_property"] = value

    # Names first, before the file is read (KG-44): see propose_node_construction.
    # The same two checks, in the build's order, as relationship_rule_name_problem.
    # The sentence on spelling a matched property comes only with that check's
    # refusal: for a type, label or column it would point at a field that is fine.
    problem = required_relationship_name_problem(relationship_construction_rule)
    matched = False
    if problem is None:
        problem = matched_property_name_problem(relationship_construction_rule)
        matched = True
    if problem is not None:
        return tool_error(_refusal(problem, matched=matched))

    # A name the other kind holds (KG-39): see propose_node_construction.
    collision = _collision_refusal(
        tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {}),
        proposed_relationship_type,
        "relationship",
    )
    if collision is not None:
        return tool_error(collision)

    # Exact header match, as the build does (KG-50), for both join columns at once.
    column_error = check_columns_in_header(
        approved_file, [from_node_column, to_node_column]
    )
    if column_error is not None:
        return column_error

    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
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
            'from_node_property' and 'to_node_property', matching the arguments
            of 'propose_relationship_construction'

    Returns:
        dict: Includes a 'status' key ('success' or 'error').
                If 'success', includes a "relationship_construction" key with the list of construction rules.
                If 'error', includes an 'error_message' key naming the entry that failed.
    """
    proposed = []
    for index, relationship_construction in enumerate(relationship_constructions):
        result = propose_relationship_construction(
            approved_file=relationship_construction.get("approved_file", ""),
            proposed_relationship_type=relationship_construction.get(
                "proposed_relationship_type", ""
            ),
            from_node_label=relationship_construction.get("from_node_label", ""),
            from_node_column=relationship_construction.get("from_node_column", ""),
            to_node_label=relationship_construction.get("to_node_label", ""),
            to_node_column=relationship_construction.get("to_node_column", ""),
            proposed_properties=relationship_construction.get(
                "proposed_properties", []
            ),
            tool_context=tool_context,
            proposed_property_types=relationship_construction.get(
                "proposed_property_types", {}
            ),
            from_node_property=relationship_construction.get("from_node_property"),
            to_node_property=relationship_construction.get("to_node_property"),
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

    If the type belongs to a node rule, nothing is removed and the error names
    'remove_node_construction'.

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

    wrong_kind = _wrong_kind_removal_refusal(
        construction_plan, relationship_type, "relationship"
    )
    if wrong_kind is not None:
        return tool_error(wrong_kind)
    if relationship_type not in construction_plan:
        return tool_success(
            "relationship_construction_removed",
            "relationship construction rule not found. removal not needed.",
        )

    construction_plan.pop(relationship_type)

    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success("relationship_construction_removed", relationship_type)


# The problem text is shown to the user and fed back to the agent, so an echoed
# value is capped: a large map as 'properties' would otherwise become one
# problem string of tens of kilobytes.
MAX_ECHOED_VALUE_LENGTH = 200


def _bounded_repr(value: Any) -> str:
    """repr(value), cut to MAX_ECHOED_VALUE_LENGTH characters."""
    text = repr(value)
    if len(text) <= MAX_ECHOED_VALUE_LENGTH:
        return text
    return f"{text[:MAX_ECHOED_VALUE_LENGTH]}... ({len(text)} characters)"


def _keeps_ends_apart(end, other, **change) -> bool:
    """Whether `end`, rewritten with `change`, still differs from `other`.

    A refusal's suggested fix that gave a rule two identical ends would only be
    refused in turn (_identical_ends), so each fix is offered only
    where this holds.
    """
    return not end._replace(**change).same_as(other)


def _can_join_on_key(end, other, key) -> bool:
    """Whether 'join on the key' is a fix for this end: matched on its column's
    own name, it reads the key column instead, and stays apart from the other end."""
    return (
        end.column == end.matched_property
        and end.column != key
        and _keeps_ends_apart(end, other, column=key, matched_property=key)
    )


def _either(options: list[str]) -> str:
    """'Either a, b, or c.', or 'A.' when one option is left."""
    if len(options) == 1:
        return f"{options[0][0].upper()}{options[0][1:]}."
    last = options[-1]
    joint = ", or, " if last.startswith("if ") else ", or "
    return f"Either {', '.join(options[:-1])}{joint}{last}."


def _identical_ends(
    key, from_end, to_end, source_file, nodes, unreadable
) -> str | None:
    """Both ends read one column and match it on one property of one label (KG-45).

    Each row then links every node holding its value to each node holding it,
    itself included: on a key, one self-loop per row. On a property holding one
    value per node no join warning fires at the build.
    Compared after resolving, so a field omitted and a field set to the column
    are the same rule. Silent when the node rule is missing or unreadable, and
    when the node does not carry the property: those are reported already.
    """
    if not from_end.same_as(to_end):
        return None
    label, prop = from_end.label, from_end.matched_property
    if not isinstance(label, str) or not isinstance(prop, str):
        return None
    node_rule = nodes.get(label)
    if node_rule is None or label in unreadable:
        return None
    node_key = node_rule.get("unique_column_name")
    if prop not in {node_key, *(node_rule.get("properties") or [])}:
        return None
    problem = (
        f"{key}: both ends read '{from_end.column}' and match it on "
        f"'{label}.{prop}', so each row would link every '{label}' holding the "
        f"row's value to each one holding it, itself included."
    )
    # As in check_endpoint: with no key there is nothing to name, so the fix
    # is keying the node first.
    if not isinstance(node_key, str) or node_key == "":
        return f"{problem} Key '{label}' by the column that identifies it first."
    return (
        f"{problem} If another column of '{source_file}' holds the {node_key} of "
        f"the related {label}, use that column on that end and set that end's "
        f"'from_node_property' or 'to_node_property' to '{node_key}'."
    )


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

    # A properties value that is not a list of text cannot be read as one: a
    # string would contribute its characters, a dict its keys, a falsy value
    # nothing at all, and anything unhashable or unordered would crash the
    # checks below. Report it once and give no verdict that reads the value --
    # a node's joins and declared types, a relationship's declared types -- until
    # it is fixed, since any verdict would be about the misreading, not the plan.
    # Missing or null declares no properties.
    unreadable = set()
    for key, rule in construction_plan.items():
        if not isinstance(rule, dict) or declared_properties(rule) is not None:
            continue
        unreadable.add(key)
        skipped = (
            "Joins onto it and its declared types"
            if rule.get("construction_type") == "node"
            else "Its declared types"
        )
        problems.append(
            f"{key}: 'properties' must be a list of property names, got "
            f"{_bounded_repr(rule.get('properties'))}. Supply a list of column "
            f"names (or an empty list). {skipped} are not checked until this is "
            f"fixed."
        )

    # Every node property a relationship matches on (KG-45: the resolved
    # matched property, never the file column), so a property can be checked
    # against the whole plan rather than only its own construction. This is
    # what makes a type retroactively invalid when a later relationship joins
    # on it. A non-text value cannot be hashed; it is reported below instead.
    # Each entry keeps the end and its partner, so a refusal can tell which
    # fixes would leave the two ends identical.
    joined_columns = {}
    for key, rule in construction_plan.items():
        if (
            not isinstance(rule, dict)
            or rule.get("construction_type") != "relationship"
        ):
            continue
        from_end, to_end = relationship_endpoints(rule)
        for end, other in ((from_end, to_end), (to_end, from_end)):
            if isinstance(end.matched_property, str):
                joined_columns.setdefault((end.label, end.matched_property), []).append(
                    (key, end, other)
                )

    def check_endpoint(rel_key, end, other):
        label, prop = end.label, end.matched_property
        node_rule = nodes.get(label)
        if node_rule is None:
            problems.append(
                f"{rel_key}: {end.side} node label '{label}' has no node construction in the plan."
            )
            return
        if label in unreadable:
            return
        unique_column = node_rule.get("unique_column_name")
        known_columns = {unique_column, *(node_rule.get("properties") or [])}
        if prop in known_columns:
            return
        others = sorted(c for c in known_columns if c and c != unique_column)
        field = f"{end.side}_node_property"
        # Each fix is offered only where following it leaves the two ends
        # different. Keying the node by this end's property cannot do that when
        # both ends are already the same: they would stay the same. A node
        # rule with no key gets no key-based fix, which could only name 'None';
        # keying it is then the fix.
        has_key = isinstance(unique_column, str) and unique_column != ""
        rekey = [] if has_key and end.same_as(other) else [f"key '{label}' by '{prop}'"]
        match_on_key = has_key and _keeps_ends_apart(
            end, other, matched_property=unique_column
        )
        if prop == end.column:
            options = list(rekey)
            if has_key and _can_join_on_key(end, other, unique_column):
                options.append(f"join on '{unique_column}'")
            if match_on_key:
                options.append(
                    f"if '{prop}' holds the '{unique_column}' of a '{label}' under "
                    f"another name, keep the column and set '{field}' to "
                    f"'{unique_column}'"
                )
            problems.append(
                f"{rel_key}: {end.side} join column '{prop}' is not a column of the "
                f"'{label}' node, which is keyed by '{unique_column}' with properties "
                f"{others}. This join would match zero rows. {_either(options)}"
            )
        else:
            target = (
                f"'{unique_column}' or one of those properties"
                if match_on_key
                else "one of those properties"
            )
            set_property = f"set '{field}' to {target}"
            problems.append(
                f"{rel_key}: the {end.side} end matches its column '{end.column}' on "
                f"'{label}.{prop}', but '{prop}' is not a property of the '{label}' "
                f"node, which is keyed by '{unique_column}' with properties {others}. "
                f"This join would match zero rows. {_either([set_property, *rekey])}"
            )

    for key, rule in construction_plan.items():
        if (
            not isinstance(rule, dict)
            or rule.get("construction_type") != "relationship"
        ):
            continue
        from_end, to_end = relationship_endpoints(rule)
        for end, other in ((from_end, to_end), (to_end, from_end)):
            property_given = not is_omitted(rule.get(f"{end.side}_node_property"))
            if property_given and not isinstance(end.column, str):
                # The checks below read the given property, never the column,
                # but the build's name check refuses a non-text column too.
                problems.append(
                    f"{key}: '{end.side}_node_column' must be a column name, got "
                    f"{_bounded_repr(end.column)}. The {end.side} end reads "
                    f"nothing until it is one."
                )
            if not isinstance(end.matched_property, str):
                # Name the field that supplied the value: with the property
                # omitted, it came from the column.
                field = (
                    f"{end.side}_node_property"
                    if property_given
                    else f"{end.side}_node_column"
                )
                problems.append(
                    f"{key}: '{field}' must be a property name, got "
                    f"{_bounded_repr(end.matched_property)}. The {end.side} end "
                    f"matches nothing until it is one."
                )
                continue
            check_endpoint(key, end, other)
        identical = _identical_ends(
            key, from_end, to_end, rule.get("source_file"), nodes, unreadable
        )
        if identical is not None:
            problems.append(identical)

    # Declared property types. Three rules, all refusing at approval time rather
    # than failing much later at import time.
    for key, rule in construction_plan.items():
        if not isinstance(rule, dict) or key in unreadable:
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
                from_end, to_end = relationship_endpoints(rule)
                joining = [key] if name in (from_end.column, to_end.column) else []
                own_end, other_end = (
                    (from_end, to_end)
                    if name == from_end.column
                    else (to_end, from_end)
                )
                own_node_label = own_end.label
                own_node_rule = nodes.get(own_node_label)
                join_target = (
                    own_node_rule.get("unique_column_name") if own_node_rule else None
                )
                ends = [(own_end, other_end)]
                # The typed value is the file column itself, so matching it on
                # another property would still compare a converted value.
                can_rematch = False
            else:
                label = rule.get("label", key)
                joins = joined_columns.get((label, name), [])
                joining = [rel_key for rel_key, _end, _other in joins]
                ends = [(end, other) for _rel_key, end, other in joins]
                own_node_label = label
                join_target = unique_column
                can_rematch = True
            if joining:
                # A rule whose two ends both match this property joins on it
                # once, not twice.
                rels = ", ".join(sorted(set(joining)))
                # Each fix is offered only where following it, on every end
                # that joins here, leaves that rule's two ends different. A
                # typed key is already what both key fixes would match on.
                options = [f"drop the type for '{name}'"]
                if join_target is None:
                    options.append(
                        f"'{own_node_label}' has no node construction in this plan, "
                        f"so there is no identifier to join on instead"
                    )
                elif name != join_target:
                    if all(
                        _can_join_on_key(end, other, join_target) for end, other in ends
                    ):
                        options.append(f"join {rels} on '{join_target}' instead")
                    if can_rematch and all(
                        _keeps_ends_apart(end, other, matched_property=join_target)
                        for end, other in ends
                    ):
                        options.append(
                            f"if the column {rels} reads holds the '{join_target}' "
                            f"of a '{own_node_label}' under another name, set that "
                            f"end's 'from_node_property' or 'to_node_property' to "
                            f"'{join_target}'"
                        )
                problems.append(
                    f"{key}: '{name}' carries a declared type but {rels} joins on "
                    f"it. Join columns are compared as raw CSV text, so a typed "
                    f"column matches zero rows with no error. {_either(options)}"
                )

    return problems


NO_PROPOSED_PLAN_MESSAGE = (
    "There is no proposed construction plan to approve. "
    "Produce one first, then present it to the user."
)


def _format_unverified_notes(unverified: list[str]) -> str:
    """Render find_plan_problems' unverified notes for a caller's message.

    Shared for the same reason the preconditions themselves are: both callers
    append this block, and two copies of the format would drift the moment one
    is reworded.
    """
    if not unverified:
        return ""
    return "\n\nNot verified:\n- " + "\n- ".join(unverified)


def format_problem_bullets(problems: list[str]) -> str:
    """Render find_plan_problems' output as the bullet list every caller shows.

    Same argument as _format_unverified_notes above, one list over: approval,
    its dry run, and the refinement loop's retry composite all render the same
    problem strings, so a third spelling of the format would drift the moment
    one is reworded -- and the loop's text would stop matching what approval
    shows the user for the identical problem.

    Public, unlike its sibling, because the composer lives in the schema
    proposal agent's module rather than here.
    """
    return "\n".join(f"- {problem}" for problem in problems)


class StateLike(Protocol):
    """Anything the plan checks can read session state from.

    ADK's State is a plain class exposing .get(key, default=None) -- not a
    Mapping -- while the refinement loop's stop-check holds a plain dict. The
    parameters are positional-only (`, /`) so that dict's own overloaded .get
    satisfies this protocol: without the slash, pyright rejects
    dict[str, Any] with "No overloaded function matches type
    (key: str, default: Any = None) -> Any" and the CI gate fails at the
    stop-check's call site.
    """

    def get(self, key: str, default: Any = None, /) -> Any: ...


def find_plan_problems(state: StateLike) -> tuple[list[str], list[str]]:
    """Everything that would make approval refuse the proposed plan.

    Returns (problems, unverified). Structural problems come first, then the
    joined-property check's, then reachability's; `unverified` holds those two
    checks' notes about sources they could not read.

    Both the approval path and the refinement loop's stop-check call this, so
    what the loop catches cannot drift from what approval refuses. Copying the
    calls into each caller instead would make that equivalence hold only by
    convention -- the drift _read_plan_for_approval was created to prevent.

    A falsy plan returns nothing at all. The reachability check reports every
    shared unique column as stranded when there are no node rules to carry
    them, so a caller that lost its own emptiness guard would otherwise loop on
    a plan that does not exist yet.

    The checks run here -- at approval and in the loop's stop-check -- and
    never as a tool the critic calls: a critic-side tool would run only when
    the model chose to call it.

    IT DELIBERATELY DOES NOT CATCH. A guard here would be inherited by
    approve_proposed_construction_plan, which today fails closed because an
    exception propagates before the approved plan is written -- a swallowed
    raise would approve a plan whose checks never ran. The loop, which only
    needs this as an optimisation approval already backstops, guards its own
    call instead.
    """
    construction_plan = state.get(PROPOSED_CONSTRUCTION_PLAN)
    if not construction_plan:
        return [], []

    problems = check_construction_plan_consistency(construction_plan)
    joined_problems, joined_unverified = check_joined_properties_hold_one_value(
        construction_plan
    )
    reachability_problems, reachability_unverified = (
        check_reference_columns_are_reachable(
            construction_plan, state.get(APPROVED_FILES) or []
        )
    )
    return (
        problems + joined_problems + reachability_problems,
        joined_unverified + reachability_unverified,
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

    The joined-property and reachability checks read files, unlike the structural
    one. Both fail open: a source they cannot read produces a note in the third
    return value, never a problem, so an unreachable disk cannot make a plan
    unshowable.

    The checks themselves live in find_plan_problems, which the refinement
    loop's stop-check also calls. The plan is still read here as well, because
    only this path distinguishes "no plan at all" (NO_PROPOSED_PLAN_MESSAGE)
    from a plan with nothing wrong.

    Callers phrase their own refusals -- only the facts are shared. Returns
    (None, [], []) when there is no plan to read.
    """
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN)
    if not construction_plan:
        return None, [], []

    problems, unverified = find_plan_problems(tool_context.state)
    return construction_plan, problems, unverified


# Tool: Approve the proposed construction plan
def approve_proposed_construction_plan(tool_context: ToolContext) -> dict:
    """Approve the proposed construction plan, if it can actually be built.

    Approval is refused when a relationship construction matches an end on a
    property the referenced node does not carry, or reads one column at both
    ends and matches it on the same node property (linking every node to itself),
    or names an endpoint label that has no node construction in the plan, or
    joins on a node property that holds more than one value per node, or when it
    leaves an approved file's reference column with no node in the plan that
    carries it reachably -- in every one of these cases the plan cannot build
    the graph that was described to the user no matter what was said in
    conversation.
    """
    construction_plan, problems, unverified = _read_plan_for_approval(tool_context)
    if construction_plan is None:
        return tool_error(NO_PROPOSED_PLAN_MESSAGE)

    notes = _format_unverified_notes(unverified)

    if problems:
        return tool_error(
            "The proposed construction plan was NOT approved. It is inconsistent, "
            "joins on a node property that holds several values per node, or "
            "leaves an approved file's reference column unreachable:\n"
            + format_problem_bullets(problems)
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
    the same checks 'approve_proposed_construction_plan' runs, without
    approving anything, so its answer is exactly what approval will do.

    An error result means approval would be refused, and the message lists the
    problems that would cause it. A success result means nothing is blocking
    approval, and carries both the plan and what to do next.
    """
    construction_plan, problems, unverified = _read_plan_for_approval(tool_context)
    if construction_plan is None:
        return tool_error(NO_PROPOSED_PLAN_MESSAGE)

    notes = _format_unverified_notes(unverified)

    if problems:
        return tool_error(
            # States the fact and stops. It deliberately does NOT say "run
            # schema_refinement_loop with these as feedback": that is right
            # mid-turn but contradicts the instruction outright on the second
            # 'retry' and on 'stopped:', where the standing rule is to stop
            # calling the loop. Only the instruction knows which branch it is
            # in, and it already says what to do in each.
            "This plan cannot be approved as it stands. "
            "'approve_proposed_construction_plan' will refuse it for:\n"
            + format_problem_bullets(problems)
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
                # objection these checks have no way to observe (they only know
                # joins, endpoint labels, typed join columns, multi-valued joined
                # properties, and reference-column reachability). An
                # unconditional claim here would be true on 'stopped:' and the
                # second 'retry' but false on the first, contradicting the
                # instruction on exactly the branch where refinement is still
                # required.
                "This plan can be approved right now: "
                "'approve_proposed_construction_plan' will accept it as it stands. "
                "That is all this tool knows: it checks joins, endpoint labels, "
                "typed columns, relationships whose two ends are the same node, "
                "whether each joined node property holds one value "
                "per node, and whether every approved file's reference "
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


# Not a leftover: the refinement loop's proposer (schema_proposal_agent_v1) and
# critic (schema_critic_agent_v1) read the draft plan through this tool, while
# the approval-facing schema_proposal_agent_coordinator reads it through
# get_proposed_construction_plan_with_approval_check. The docstring below is
# the tool's model-visible description, so this note stays a comment.
def get_proposed_construction_plan(tool_context: ToolContext) -> dict:
    """Get the proposed construction plan."""
    return tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, [])


def get_approved_construction_plan(tool_context: ToolContext) -> dict:
    """Get the approved construction plan, or an error saying none is approved."""
    plan = approved_plan(tool_context.state)
    if plan is None:
        return tool_error(NOT_APPROVED_MESSAGE)
    return plan
