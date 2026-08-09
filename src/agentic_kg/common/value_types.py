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
from math import isfinite
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

_CURRENCY = r"[$€£¥₹]"
_DIGITS = r"(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?"
# The sign may sit on either side of the currency symbol. Both orders are real:
# "$-42.00" is what a bare float formatter emits, "-$42.00" is what Excel and
# most ERP exports emit. Accepting only one of them is worse than accepting
# neither -- a money column's positive rows convert while its refunds and
# credits become unconvertible and get CLEARED, so a later sum() is wrong in a
# way nothing in the graph shows. Two alternatives rather than two optional
# signs, so "+-42" still fails to match.
_NUMERIC_LIKE = re.compile(
    rf"^\s*(?:[-+]?\s*{_CURRENCY}?|{_CURRENCY}\s*[-+]?)\s*{_DIGITS}\s*$")

# Accounting notation for a negative: (42.00), ($42.00). Handled separately
# because the minus sign is implied by the brackets rather than written, so it
# survives neither the pattern above nor _STRIP_FROM_NUMBERS.
_PARENTHESISED_NEGATIVE = re.compile(
    rf"^\s*\(\s*{_CURRENCY}?\s*{_DIGITS}\s*\)\s*$")

# Applied only after a numeric pattern has matched, so this cannot turn "1,2,3"
# into 123 -- that string never reaches here.
_STRIP_FROM_NUMBERS = str.maketrans("", "", "$€£¥₹,() ")

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

# Neo4j's INTEGER is a signed 64-bit value, and the driver packs it as one.
# Python's int is unbounded, so a wider value converts happily here and then
# raises OverflowError inside the driver, mid-batch -- an opaque failure with
# rows already committed and nothing naming the column responsible. Refusing it
# here instead routes it down the path this module already has for a value that
# cannot be stored: counted, cleared, and named by the loader's gate.
INT64_MIN = -(2 ** 63)
INT64_MAX = 2 ** 63 - 1


def is_blank(value: Any) -> bool:
    """True for a value that carries nothing: None, empty, or only whitespace."""
    return value is None or str(value).strip() == ""


def _is_really_a_count(non_blank) -> bool:
    """True for a column that only LOOKS boolean because most of its values are
    0 or 1.

    A quantity that is overwhelmingly 0 or 1 -- a backorder quantity, a defect
    count, a discount count -- wins the boolean majority on those rows while
    carrying real 2s and 3s. Calling it boolean would make every value above 1
    unconvertible, and the loader clears an unconvertible value: the large
    numbers, the only ones that change an answer, are exactly the ones that
    would disappear, and at a minority share the refusal gate never fires.

    Narrow on purpose, so TRAP 5 still holds: this fires only when the values
    carrying the boolean majority are all drawn from the NUMERIC half of the
    vocabulary. A genuine 0/1 flag has nothing else in the column and stays
    boolean_like; a yes/no column with a stray "5" stays boolean_like too.
    """
    numeric_vocabulary = {"0", "1"}
    fitting = [value.strip().lower() for value in non_blank
               if value.strip().lower() in _BOOLEAN_VALUES]
    if not all(value in numeric_vocabulary for value in fitting):
        return False
    return any(_BARE_NUMERIC.match(value)
               and value.strip().lower() not in _BOOLEAN_VALUES
               for value in non_blank)


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

    def in_boolean_vocabulary(value) -> bool:
        return value.strip().lower() in _BOOLEAN_VALUES

    if majority(in_boolean_vocabulary) and not _is_really_a_count(non_blank):
        return BOOLEAN_LIKE
    if majority(_BARE_NUMERIC.match):
        return BARE_NUMERIC

    def numeric_after_cleaning(value) -> bool:
        # Both forms coerce() accepts, or this reports text for a column the
        # loader would have converted -- the exact evidence/behaviour drift
        # sharing this module is meant to prevent. A column written entirely in
        # accounting parentheses would otherwise get no type suggestion at all.
        return bool(_NUMERIC_LIKE.match(value)
                    or _PARENTHESISED_NEGATIVE.match(value))

    if majority(numeric_after_cleaning):
        return NUMERIC_AFTER_CLEANING
    return TEXT


def _clean_number(text: str) -> Optional[str]:
    """Strip currency, thousands separators and accounting brackets.

    Returns None when the value is not numeric at all. Shared by coerce and
    has_fractional_part so the two cannot disagree about what counts as a
    number or how it is cleaned.
    """
    negated_by_brackets = bool(_PARENTHESISED_NEGATIVE.match(text))
    if not (negated_by_brackets or _NUMERIC_LIKE.match(text)):
        return None
    cleaned = text.translate(_STRIP_FROM_NUMBERS)
    return f"-{cleaned}" if negated_by_brackets else cleaned


def has_fractional_part(value: Any) -> bool:
    """True for a numeric value carrying a non-zero fractional part.

    Exists so a caller can tell INTEGER's two refusals apart. coerce refuses
    "42.7" because it is fractional and "9223372036854775809" because Neo4j's
    INTEGER cannot hold it, and inferring "fractional" from a failed integer
    coercion conflates them: the overflowing value would be read as evidence
    the column is fractional and typed float, which stores a rounded, wrong
    number and reports a clean conversion.

    False for a blank or a non-numeric value -- neither is evidence of anything.
    """
    if is_blank(value):
        return False
    cleaned = _clean_number(str(value).strip())
    if cleaned is None:
        return False
    _whole, _, fraction = cleaned.partition(".")
    return bool(fraction.strip("0"))


def _as_storable_integer(number: int) -> Tuple[Optional[int], str]:
    """Accept an integer only if Neo4j can actually hold it. See INT64_MAX."""
    if INT64_MIN <= number <= INT64_MAX:
        return number, CONVERTED
    return None, UNCONVERTIBLE


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
        cleaned = _clean_number(text)
        if cleaned is None:
            return None, UNCONVERTIBLE

        if declared_type == INTEGER:
            # Read from the digits, never through float. float() rounds
            # anything past 2**53 to the nearest representable value, and a
            # fractional check applied afterwards cannot see damage done before
            # it ran: "9007199254740993" and "9007199254740993.0" both come back
            # as ...992 and report CONVERTED. Neo4j's INTEGER is a full 64-bit
            # signed, so the wrong number is stored without a murmur. Splitting
            # the string keeps every digit the source actually wrote.
            whole, _, fraction = cleaned.partition(".")
            if fraction.strip("0"):
                return None, UNCONVERTIBLE
            try:
                return _as_storable_integer(int(whole))
            except ValueError:  # pragma: no cover - guarded by the patterns above
                return None, UNCONVERTIBLE

        try:
            number = float(cleaned)
        except ValueError:  # pragma: no cover - guarded by the patterns above
            return None, UNCONVERTIBLE
        if not isfinite(number):
            # A literal with more digits than a double can hold becomes inf,
            # which the driver packs happily as a Neo4j FLOAT -- the graph would
            # store Infinity and the load would report success.
            return None, UNCONVERTIBLE
        return number, CONVERTED

    # An unknown declared type reaches here only if the plan carried one that
    # check_construction_plan_consistency should have refused. Fail closed.
    return None, UNCONVERTIBLE
