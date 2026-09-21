"""Refuse a relationship that joins on a node property holding several values.

A node rule MERGEs one node per distinct key value and then overwrites each
retained property from every row, so if the rows sharing a key disagree about a
property, only one arbitrary value survives on the node. A relationship joining on
that property then matches only the rows that happen to carry the surviving value,
and nothing at build time says so.

What is refused is that: rows sharing a key that disagree about the joined
property. What is NOT refused is several nodes sharing one value, which a join may
legitimately match. That is why this reads only `conflict_count` from
`summarize_key_groups` and does not reuse reachability's `_property_failure`, which
also refuses shared values.

Same contract as `check_reference_columns_are_reachable`, and for the same reason:
`find_plan_problems` runs it on every plan presentation and does not catch, so it
returns `(problems, unverified)`, both lists, and never raises. It does that
without a blanket `except` (a swallowed raise at approval would approve a plan
whose checks never ran): every field is type-guarded before it is used as a dict
key or a path, and `summarize_key_groups` already returns a read failure as an
error result. A refusal rests on evidence; anything unverifiable is a note.
"""

from typing import Dict, List, Tuple

from .file_tools import summarize_key_groups
from .reference_reachability import declared_properties, node_description, quoted_list


def _joins_by_node_property(
    construction_plan: dict,
) -> Dict[Tuple[str, str], List[str]]:
    """Each (node plan key, column) a relationship joins on, with its relationships.

    Plan order, `from` endpoint before `to`, each relationship once per pair. The
    node is named by its plan key because that is how
    check_construction_plan_consistency resolves an endpoint's label
    (`nodes.get(label)`), so the two checks cannot disagree about which rule a
    join lands on. An endpoint whose label or column is not text cannot be looked
    up (a list is unhashable) and names nothing, so it is skipped.
    """
    joins: Dict[Tuple[str, str], List[str]] = {}
    for relationship_key, rule in construction_plan.items():
        if (
            not isinstance(rule, dict)
            or rule.get("construction_type") != "relationship"
        ):
            continue
        endpoints = (
            (rule.get("from_node_label"), rule.get("from_node_column")),
            (rule.get("to_node_label"), rule.get("to_node_column")),
        )
        for label, column in endpoints:
            if not isinstance(label, str) or not isinstance(column, str):
                continue
            relationships = joins.setdefault((label, column), [])
            name = str(relationship_key)
            if name not in relationships:
                relationships.append(name)
    return joins


def _unusable_field(rule: dict) -> str | None:
    """The first field a read needs that is not usable, or None if both are.

    An empty key is as unusable as a non-text one, as in `_property_failure`:
    read anyway, it would come back as an error blamed on the file."""
    if not isinstance(rule.get("source_file"), str):
        return "source_file"
    key = rule.get("unique_column_name")
    if not isinstance(key, str) or not key:
        return "unique_column_name"
    return None


def _refusal(
    relationships: List[str], column: str, rule: dict, conflicts: int, groups: int
) -> str:
    """The refusal, built only from the plan's own names and two counts.

    It says the relationship cannot be trusted, never that a percentage is low
    (the join preview reports coverage), and never suggests dropping the
    relationship. The one fix it offers is always available at plan level: a node
    keyed by the joined column, built from the node's own source, under a label of
    its own. The join then targets a key, which this check skips.

    The counts, not "each node": one conflicting node in ten thousand still
    refuses, and "each keeps one arbitrary value" would then be false. A blank
    cell counts as a value because the loader writes over an earlier one with it.
    """
    named = quoted_list(relationships)
    verb = "joins" if len(relationships) == 1 else "join"
    return (
        f"{named} {verb} on '{column}' of {node_description(rule)}, but that "
        f"property holds more than one value per node (a blank cell counts as a "
        f"value): in {conflicts} of {groups} nodes, rows sharing a key disagree "
        f"about it, so the affected nodes keep one arbitrary value each, and the "
        f"join matches only the rows carrying the value kept. Fix it by adding a "
        f"node construction from '{rule['source_file']}' keyed by '{column}', "
        f"under a label of its own, alongside the existing one, and joining "
        f"{named} on that label's '{column}' instead."
    )


def check_joined_properties_hold_one_value(
    construction_plan: dict,
) -> Tuple[List[str], List[str]]:
    """Report relationships that join on a property their node holds several values of.

    Returns (problems, unverified). Both are always lists and this never raises.

    One line per node and property, listing every relationship that joins on it:
    the fix acts on the node and property, so a line per relationship would repeat
    it and invite duplicate nodes. A join on the node's key is sound and is not
    read; a column that is not a declared property, a node with no rule, and a node
    whose `properties` cannot be read are left to the structural check, which
    already reports them. A source that cannot be read, or a rule missing a field
    the read needs, is a note: one per unreadable file and one per unusable rule,
    each naming every join it left unchecked.
    """
    if not isinstance(construction_plan, dict):
        return [], []

    nodes = {
        key: rule
        for key, rule in construction_plan.items()
        if isinstance(rule, dict) and rule.get("construction_type") == "node"
    }
    problems: List[str] = []
    unusable_rules: Dict[Tuple[str, str], List[str]] = {}
    unreadable: Dict[Tuple[str, str], List[str]] = {}

    joins = _joins_by_node_property(construction_plan)
    for (label, column), relationships in joins.items():
        rule = nodes.get(label)
        if rule is None:
            continue
        properties = declared_properties(rule)
        if (
            properties is None
            or column == rule.get("unique_column_name")
            or column not in properties
        ):
            continue
        unusable = _unusable_field(rule)
        if unusable is not None:
            unusable_rules.setdefault((label, unusable), []).append(f"{label}.{column}")
            continue
        summary, error = summarize_key_groups(
            rule["source_file"], rule["unique_column_name"], column
        )
        if error is not None:
            key = (rule["source_file"], error["error_message"])
            unreadable.setdefault(key, []).append(f"{label}.{column}")
            continue
        assert summary is not None  # summarize_key_groups: error is None => summary set
        if summary.conflict_count:
            problems.append(
                _refusal(
                    relationships,
                    column,
                    rule,
                    summary.conflict_count,
                    summary.group_count,
                )
            )

    shape_notes = [
        f"{quoted_list(unchecked)} could not be checked for one value per node: "
        f"the rule has no usable '{field}'"
        for (_, field), unchecked in unusable_rules.items()
    ]
    file_notes = [
        f"{quoted_list(unchecked)} could not be checked for one value per node: "
        f"'{path}' could not be read ({message})"
        for (path, message), unchecked in unreadable.items()
    ]
    return problems, shape_notes + file_notes
