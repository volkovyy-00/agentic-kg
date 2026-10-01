"""The two ends of a relationship rule, read one way by every reader (KG-45).

Each end reads one column of the relationship's file (`<side>_node_column`) and
matches its value against one property of that end's node
(`<side>_node_property`). When the property is omitted -- absent, None or "" --
the end matches on the column's own name, as every rule did before KG-45. The
build's name check, the loader and the approval checks all read the ends
through here, so they cannot disagree about what an end matches.

Nothing here type-checks: a value that is not text passes through unchanged,
and each caller decides what that means (the name check refuses it, the
approval checks report it). Reads use .get only, so this never raises on a
dict. This module imports nothing from tools/, so every tools module can use it
without an import cycle.
"""

from typing import Any, NamedTuple, Tuple


class Endpoint(NamedTuple):
    side: str
    label: Any
    column: Any
    matched_property: Any

    def same_as(self, other: "Endpoint") -> bool:
        """Same label, column and matched property: the side aside, one end.

        A rule whose two ends are the same links each row's node to itself.
        """
        return (self.label, self.column, self.matched_property) == (
            other.label,
            other.column,
            other.matched_property,
        )


def is_omitted(value: Any) -> bool:
    """None or "" only. A model often fills an unused optional argument with "".

    Any other value, falsy or not (0, [], {}), is kept so it can be refused
    rather than silently dropped.
    """
    return value is None or (isinstance(value, str) and value == "")


def _endpoint(rule: dict, side: str) -> Endpoint:
    column = rule.get(f"{side}_node_column")
    explicit = rule.get(f"{side}_node_property")
    return Endpoint(
        side,
        rule.get(f"{side}_node_label"),
        column,
        column if is_omitted(explicit) else explicit,
    )


def relationship_endpoints(rule: dict) -> Tuple[Endpoint, Endpoint]:
    """The from end, then the to end."""
    return _endpoint(rule, "from"), _endpoint(rule, "to")


def describe_end(end: Endpoint) -> str:
    """'Label.property', naming the file column only where it differs."""
    text = f"{end.label}.{end.matched_property}"
    if end.column != end.matched_property:
        text += f" (column '{end.column}')"
    return text
