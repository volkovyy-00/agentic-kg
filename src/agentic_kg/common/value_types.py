"""Decide what a CSV string means, and convert it.

Shared by the schema-proposal evidence tool (tools/file_tools.column_type_hint)
and by both loaders (tools/kg_construction_tools). Using one converter on both
paths is deliberate: what the model is told a column supports and what the
loader actually does to that column cannot then drift apart.

No ToolContext, no database, no I/O -- strings in, answers out. Same shape and
the same reason as common/cypher_identifiers.py.

The numeric character classes below deliberately duplicate
common/graph_profile.py's _BARE_NUMERIC / _NUMERIC_LIKE rather than importing
them. The profile is retrieval-side and out of scope for this change, and the
ticket's dependency reasoning -- that a genuinely typed property takes the
profile's type branch and never reaches its regex path -- holds only while that
file stays untouched.
"""
import re
from typing import Any, Iterable, Optional, Tuple

# The closed set of types a construction plan may declare. Lowercase to match
# the plan's other values ("node", "relationship"), not Neo4j's schema spelling.
INTEGER = "integer"
FLOAT = "float"
BOOLEAN = "boolean"
ALLOWED_TYPES = (INTEGER, FLOAT, BOOLEAN)

# What a column's values look like. Reported to the model as evidence; never
# used to decide a type at write time.
BARE_NUMERIC = "bare_numeric"
NUMERIC_AFTER_CLEANING = "numeric_after_cleaning"
BOOLEAN_LIKE = "boolean_like"
TEXT = "text"

# What happened to one value. Three outcomes rather than success/failure:
# blank clears a stale value exactly as unconvertible does, but it is not
# evidence of a wrong type and must never count toward the loader's gate.
CONVERTED = "converted"
BLANK = "blank"
UNCONVERTIBLE = "unconvertible"

_BARE_NUMERIC = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*$")
_NUMERIC_LIKE = re.compile(
    r"^\s*[$€£¥₹]?\s*[-+]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s*$")

# Applied only after _NUMERIC_LIKE has matched, so this cannot turn "1,2,3"
# into 123 -- that string never reaches here.
_STRIP_FROM_NUMBERS = str.maketrans("", "", "$€£¥₹, ")

_TRUE_VALUES = frozenset({"yes", "true", "y", "1"})
_FALSE_VALUES = frozenset({"no", "false", "n", "0"})
_BOOLEAN_VALUES = _TRUE_VALUES | _FALSE_VALUES

# The share of non-blank values that has to fit a shape for classify() to report
# it. Defined here rather than written into classify() as a bare comparison
# because the loader's refusal gate (kg_construction_tools.TYPE_FAILURE_LIMIT)
# imports this exact number: a column this module suggests a type for must not
# then trip that gate. Two independent 0.5 literals could drift apart silently,
# and the only thing that would notice is a build refusing a column the model
# was told was fine.
MAJORITY_SHARE = 0.5


def is_blank(value: Any) -> bool:
    """True for a value that carries nothing: None, empty, or only whitespace."""
    return value is None or str(value).strip() == ""


def classify(values: Iterable[Any]) -> str:
    """Report the most specific shape a majority of non-blank values fit.

    Majority rather than all: one bad row in four hundred must not collapse a
    numeric column to text, because the loader itself tolerates up to half a
    batch failing. Both places read MAJORITY_SHARE on purpose -- a column this
    suggests a type for cannot then trip the loader's refusal gate.

    Order is load-bearing. Boolean is checked FIRST because _BARE_NUMERIC
    matches "1" and "0": a genuine 0/1 flag would otherwise match the numeric
    pattern at 100% and never reach the boolean check. Values outside the
    boolean vocabulary (2, 3, 5 ...) still fall through to the numeric shapes.
    """
    non_blank = [str(value) for value in values if not is_blank(value)]
    if not non_blank:
        return TEXT

    def majority(matches) -> bool:
        fitting = sum(1 for value in non_blank if matches(value))
        return fitting > len(non_blank) * MAJORITY_SHARE

    if majority(lambda value: value.strip().lower() in _BOOLEAN_VALUES):
        return BOOLEAN_LIKE
    if majority(_BARE_NUMERIC.match):
        return BARE_NUMERIC
    if majority(_NUMERIC_LIKE.match):
        return NUMERIC_AFTER_CLEANING
    return TEXT


def coerce(value: Any, declared_type: str) -> Tuple[Optional[Any], str]:
    """Convert one source value to the type the construction plan declared.

    Returns (converted_value, outcome). The value is None for every outcome
    other than CONVERTED; callers distinguish BLANK from UNCONVERTIBLE, never
    by testing the value.

    A value with a real fractional part declared integer is UNCONVERTIBLE, not
    rounded: "42.0" -> 42, "42.7" -> failure. Silently truncating is a wrong
    answer nobody sees; a counted failure is visible.
    """
    if is_blank(value):
        return None, BLANK

    text = str(value).strip()

    if declared_type == BOOLEAN:
        lowered = text.lower()
        if lowered in _TRUE_VALUES:
            return True, CONVERTED
        if lowered in _FALSE_VALUES:
            return False, CONVERTED
        return None, UNCONVERTIBLE

    if declared_type in (INTEGER, FLOAT):
        if not _NUMERIC_LIKE.match(text):
            return None, UNCONVERTIBLE
        try:
            number = float(text.translate(_STRIP_FROM_NUMBERS))
        except ValueError:  # pragma: no cover - guarded by the pattern above
            return None, UNCONVERTIBLE
        if declared_type == FLOAT:
            return number, CONVERTED
        if number != int(number):
            return None, UNCONVERTIBLE
        return int(number), CONVERTED

    # An unknown declared type reaches here only if the plan carried one that
    # check_construction_plan_consistency should have refused. Fail closed.
    return None, UNCONVERTIBLE
