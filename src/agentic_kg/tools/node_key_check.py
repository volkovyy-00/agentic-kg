"""Refuse a node proposed on a key whose rows would collapse.

The node loader MERGEs one node per distinct key value and then overwrites each
property from every row, so rows sharing a key keep one arbitrary row's values and
the node count falls to the distinct count. A key cell with no usable value is
worse: an empty one merges every such row into one node, and a row with no cell at
all (a blank line at the end of the file reads as one) sends a null key, which
makes the database reject the whole batch.

What is refused: a key with any blank value, and a repeating key whose rows
disagree on a proposed property. What is NOT refused is a repeating key whose rows
agree on every proposed property (customer ids taken from an orders file), because
nothing is lost. Agreement is judged by `summarize_key_groups`, the reader behind
`collapse_check`, so the two cannot disagree: a blank cell is a value, a missing
cell is not.

Cells are compared as written, before the loader types them, so `42` and `42.0`
in a column declared integer count as disagreeing although the build would store
one value. That over-refuses and never under-refuses, it is what `collapse_check`
does too, and the refusal quotes both values so the agent can see the cause.
Typing the cells first would need the declared types passed into the shared reader
and would make the two disagree; it was left out on purpose (KG-48).

This runs when a node is proposed and not at approval; why it is not a plan check
is in `.claude/rules/construction-plan.md`. Its refusal has no build-time twin, so
KG-44's rule that a proposal refuses in the build's own words does not apply.

Cost: the key column is read in full, where a proposal used to read only the
header. A unique key stops there; a repeating key reads once more per property.
"""

from typing import List, NamedTuple, Optional

from agentic_kg.common.csv_reader import read_csv_header
from agentic_kg.common.tool_result import tool_error

from .file_tools import summarize_column, summarize_key_groups
from .reference_reachability import quoted_list, quoted_value


class KeyExample(NamedTuple):
    """One property on which rows sharing a key disagree, shown on one such key."""

    property: str
    node_key: str
    first_value: str
    second_value: str


class NodeKeySummary(NamedTuple):
    """What a refusal needs to know about a node's key column."""

    row_count: int
    distinct_count: int
    blank_count: int
    """Rows whose key is empty, whitespace-only or missing altogether."""
    conflicts: List[KeyExample]
    """One entry for every proposed property on which rows sharing a key disagree,
    in the order proposed, each on its earliest conflicting key. Empty when the key
    is unique, blank (the properties are never read) or repeats harmlessly."""


def summarize_node_key(file_path: str, key: str, properties: Optional[List[str]]):
    """Read a node's key column and, only if it repeats, judge its properties.

    Returns (summary, error) and never raises: a missing file, a key the header
    lacks and a failed read all come back as the tool_error result of the reader
    that met them. `properties` is None when the rule's own list could not be read
    (`declared_properties`); the agreement check is then skipped and the key's own
    values are still judged.

    A property the header lacks is skipped, not refused: the loader writes nothing
    for it, so it cannot lose data. The key itself and a repeated name are not read
    as properties.
    """
    column, error = summarize_column(file_path, key)
    if error is not None:
        return None, error
    assert column is not None

    summary = NodeKeySummary(
        column.row_count, len(column.distinct), column.empty_count, []
    )
    unique = column.is_unique
    # The distinct set is as large as the key column can be; only its size is
    # needed from here, and the property passes below each hold state of their own.
    del column
    if summary.blank_count or unique or not properties:
        return summary, None

    try:
        header = set(read_csv_header(file_path))
    except Exception as exc:  # noqa: BLE001 - report read failures to the agent
        return None, tool_error(f"Error reading CSV file {file_path}: {exc}")

    conflicts: List[KeyExample] = []
    for name in dict.fromkeys(properties):
        if name == key or name not in header:
            continue
        groups, error = summarize_key_groups(file_path, key, name, keep_examples=1)
        if error is not None:
            return None, error
        assert groups is not None
        if groups.conflict_count:
            held = groups.examples[0]
            first, second = held["values"][:2]
            conflicts.append(KeyExample(name, held["node_key"], first, second))
    return summary._replace(conflicts=conflicts), None


def node_key_refusal(
    label: str, source_file: str, key: str, summary: NodeKeySummary
) -> Optional[str]:
    """The refusal for a key that would collapse its rows, or None if it would not.

    A blank key is judged first and its text claims no one outcome, because the
    build does a different thing for each kind of blank. The way forward is always
    another key or a relationship; dropping a property is offered only where a
    property is what disagrees, and not prescribed, because it changes what the
    node holds.
    """
    counts = (
        f"{summary.row_count} rows, {summary.distinct_count} distinct values, "
        f"{summary.blank_count} blank"
    )
    head = f"Cannot key {label} nodes on '{key}' in {source_file}: {counts}."
    if summary.blank_count:
        return (
            f"{head} A blank is an empty, whitespace-only or missing key (a blank "
            f"line at the end of the file counts). Such rows can merge into one "
            f"node, make the build fail, or load as meaningless keys. Choose "
            f"another key, or model the file as a relationship; if '{key}' is the "
            f"right key, the blank rows have to be fixed in the file first."
        )
    if summary.conflicts:
        example = summary.conflicts[0]
        properties = quoted_list([conflict.property for conflict in summary.conflicts])
        return (
            f"{head} Rows sharing a '{key}' disagree on {properties} (for example "
            f"'{key}' {quoted_value(example.node_key)} has "
            f"'{example.property}' values {quoted_value(example.first_value)} and "
            f"{quoted_value(example.second_value)}), so the build would merge them "
            f"into one node and keep one row's values. Choose another key, model "
            f"the file as a relationship, or leave those properties off the node."
        )
    return None
