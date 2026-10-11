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

The temporal types build neo4j.time objects, never datetime.datetime and never
through fromisoformat. A Python datetime stops at microseconds, so nine
fractional digits would be cut with no error; fromisoformat also accepts
"20250304" and week dates, which this module must refuse. neo4j.time is a pure
value library (no connection), so the "no I/O" rule above still holds.
"""

import re
from datetime import timedelta, timezone
from math import isfinite
from typing import Any, Dict, Iterable, NamedTuple, Optional, Tuple

from neo4j import time as neo4j_time

# The closed set of types a construction plan may declare. Lowercase to match
# the plan's other values ("node", "relationship"), not Neo4j's schema spelling.
# The temporal names mirror Cypher's date(), datetime() and localdatetime().
INTEGER = "integer"
FLOAT = "float"
BOOLEAN = "boolean"
DATE = "date"
DATETIME = "datetime"  # zoned: carries an offset or Z
LOCALDATETIME = "localdatetime"  # no offset
TEMPORAL_TYPES = (DATE, DATETIME, LOCALDATETIME)
ALLOWED_TYPES = (INTEGER, FLOAT, BOOLEAN, *TEMPORAL_TYPES)

# What a column's values look like. Reported to the model as evidence; never
# used to decide a type at write time.
BARE_NUMERIC = "bare_numeric"
NUMERIC_AFTER_CLEANING = "numeric_after_cleaning"
BOOLEAN_LIKE = "boolean_like"
TEXT = "text"
DATE_LIKE = "date_like"
DATETIME_LIKE = "datetime_like"
LOCALDATETIME_LIKE = "localdatetime_like"

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
    rf"^\s*(?:[-+]?\s*{_CURRENCY}?|{_CURRENCY}\s*[-+]?)\s*{_DIGITS}\s*$"
)

# Accounting notation for a negative: (42.00), ($42.00). Handled separately
# because the minus sign is implied by the brackets rather than written, so it
# survives neither the pattern above nor _STRIP_FROM_NUMBERS.
_PARENTHESISED_NEGATIVE = re.compile(rf"^\s*\(\s*{_CURRENCY}?\s*{_DIGITS}\s*\)\s*$")

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

# A temporal shape names its own type: unlike the numeric shapes there is nothing
# to split within it. classify() reports the shape and ColumnTally.suggested_type
# maps it to the type through this one table.
TEMPORAL_SHAPE_TYPES = {
    DATE_LIKE: DATE,
    DATETIME_LIKE: DATETIME,
    LOCALDATETIME_LIKE: LOCALDATETIME,
}

# Neo4j's INTEGER is a signed 64-bit value, and the driver packs it as one.
# Python's int is unbounded, so a wider value converts happily here and then
# raises OverflowError inside the driver, mid-batch -- an opaque failure with
# rows already committed and nothing naming the column responsible. Refusing it
# here instead routes it down the path this module already has for a value that
# cannot be stored: counted, cleared, and named by the loader's gate.
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


# One strict pattern for all three temporal shapes. [0-9] and fullmatch on
# purpose: \d matches Arabic-Indic digits (and int() converts them), and "$"
# matches before a trailing newline. A fraction needs seconds; the offset is
# only legal after a time.
_TEMPORAL = re.compile(
    r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"(?:[T ](?P<hour>[0-9]{2}):(?P<minute>[0-9]{2})"
    r"(?::(?P<second>[0-9]{2})(?:\.(?P<fraction>[0-9]{1,9}))?)?"
    r"(?P<offset>Z|[+-][0-9]{2}:[0-9]{2})?)?"
)

# Neo4j accepts offsets up to +-18:00. Python's constructors accept up to just
# under 24:00 and the server then rejects the value mid-batch, dropping the
# connection -- the same shape as INT64_MAX above.
MAX_OFFSET_MINUTES = 18 * 60


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
    return ColumnTally.of(values).shape


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


def _has_fraction(cleaned: str) -> bool:
    """True when a cleaned number's digits after the point are not all zeros.

    The one statement of the fraction rule. has_fractional_part, the integer
    converter's refusal and ColumnTally's any_fraction flag all ask it, so what
    counts as "fractional" cannot differ between the hint and the loader.
    """
    _whole, _, fraction = cleaned.partition(".")
    return bool(fraction.strip("0"))


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
    return _has_fraction(cleaned)


def _as_storable_integer(number: int) -> Tuple[Optional[int], str]:
    """Accept an integer only if Neo4j can actually hold it. See INT64_MAX."""
    if INT64_MIN <= number <= INT64_MAX:
        return number, CONVERTED
    return None, UNCONVERTIBLE


def _fixed_offset(text: str) -> Optional[timezone]:
    """The timezone for 'Z' or '+-HH:MM', or None when Neo4j cannot store it."""
    if text == "Z":
        return timezone.utc
    hours, minutes = int(text[1:3]), int(text[4:6])
    if minutes > 59 or hours * 60 + minutes > MAX_OFFSET_MINUTES:
        return None
    sign = -1 if text[0] == "-" else 1
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def parse_temporal(text: str) -> Optional[Tuple[str, Any]]:
    """Read one already-stripped string as a date, a zoned or a local datetime.

    Returns (kind, neo4j.time value), kind being DATE, DATETIME or LOCALDATETIME,
    or None when the text is not exactly one of those shapes. The one function
    both classify() and coerce() use, so the shape the hint reports and the value
    the loader stores cannot disagree -- a column of well-formed but impossible
    dates ("2025-02-30") is never suggested, because it does not parse.

    Validity is established by constructing the value: neo4j.time raises
    ValueError for an impossible date, an hour of 24 or a second of 60, which
    become None. A zoned value must also exist in UTC: 9999-12-31T23:59:59-01:00
    builds fine, then the driver raises converting it to UTC while packing, and
    one such cell would fail the whole batch. So the value is converted to UTC
    here, with the same astimezone() call the driver's packer makes, and a
    refusal there is a refusal here.
    """
    match = _TEMPORAL.fullmatch(text)
    if match is None:
        return None
    parts = match.groupdict()
    year, month, day = int(parts["year"]), int(parts["month"]), int(parts["day"])
    # Year 0 is outside the driver's range, but neo4j.time does not always raise
    # for it: 0000-00-00 (MySQL's "no value") builds a ZeroDate, which would be
    # stored as a wrong date, or as a DateTime the driver refuses mid-batch.
    if year == 0:
        return None
    try:
        if parts["hour"] is None:
            return DATE, neo4j_time.Date(year, month, day)
        hour, minute = int(parts["hour"]), int(parts["minute"])
        second = int(parts["second"] or 0)
        # Padded on the right, so ".5" is 500000000 ns and every digit is kept.
        nanosecond = int((parts["fraction"] or "").ljust(9, "0"))
        if parts["offset"] is None:
            return LOCALDATETIME, neo4j_time.DateTime(
                year, month, day, hour, minute, second, nanosecond
            )
        zone = _fixed_offset(parts["offset"])
        if zone is None:
            return None
        value = neo4j_time.DateTime(
            year, month, day, hour, minute, second, nanosecond, tzinfo=zone
        )
        value.astimezone(timezone.utc)
        return DATETIME, value
    except (ValueError, OverflowError):
        return None


def _to_boolean(lowered: str) -> Tuple[Optional[bool], str]:
    """The boolean rule: the stripped, lowered text is in the vocabulary."""
    if lowered in _TRUE_VALUES:
        return True, CONVERTED
    if lowered in _FALSE_VALUES:
        return False, CONVERTED
    return None, UNCONVERTIBLE


def _to_integer(cleaned: Optional[str]) -> Tuple[Optional[int], str]:
    """The integer rule, given _clean_number's result (None: not a number)."""
    if cleaned is None:
        return None, UNCONVERTIBLE
    # Read from the digits, never through float. float() rounds
    # anything past 2**53 to the nearest representable value, and a
    # fractional check applied afterwards cannot see damage done before
    # it ran: "9007199254740993" and "9007199254740993.0" both come back
    # as ...992 and report CONVERTED. Neo4j's INTEGER is a full 64-bit
    # signed, so the wrong number is stored without a murmur. Splitting
    # the string keeps every digit the source actually wrote.
    if _has_fraction(cleaned):
        return None, UNCONVERTIBLE
    whole = cleaned.partition(".")[0]
    try:
        return _as_storable_integer(int(whole))
    except ValueError:  # pragma: no cover - guarded by the patterns above
        return None, UNCONVERTIBLE


def _to_float(cleaned: Optional[str]) -> Tuple[Optional[float], str]:
    """The float rule, given _clean_number's result (None: not a number)."""
    if cleaned is None:
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


def _to_temporal(
    parsed: Optional[Tuple[str, Any]], declared_type: str
) -> Tuple[Optional[Any], str]:
    """The temporal rule, given parse_temporal's result."""
    # Each type takes only its own shape: a bare date is not a datetime, and
    # a zoned value is not a local one. Never trimmed, never filled in.
    if parsed is not None and parsed[0] == declared_type:
        return parsed[1], CONVERTED
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

    # Only the piece this one type needs: the loaders call coerce on every
    # typed cell, so it must not take _read's all-types reading.
    text = str(value).strip()
    if declared_type == BOOLEAN:
        return _to_boolean(text.lower())
    if declared_type == INTEGER:
        return _to_integer(_clean_number(text))
    if declared_type == FLOAT:
        return _to_float(_clean_number(text))
    if declared_type in TEMPORAL_TYPES:
        return _to_temporal(parse_temporal(text), declared_type)

    # An unknown declared type reaches here only if the plan carried one that
    # check_construction_plan_consistency should have refused. Fail closed.
    return None, UNCONVERTIBLE


class _Reading(NamedTuple):
    """One non-blank value, read once for every type at the same time."""

    raw: Any
    text: str
    lowered: str
    cleaned: Optional[str]
    temporal: Optional[Tuple[str, Any]]


def _read(value: Any) -> _Reading:
    """Strip, lower, clean and parse a non-blank value once.

    What coerce_all and ColumnTally share, so trying six types costs one parse,
    not six. coerce deliberately does not use it: see coerce.
    """
    text = str(value).strip()
    return _Reading(
        value, text, text.lower(), _clean_number(text), parse_temporal(text)
    )


def _convert_all(reading: _Reading) -> Dict[str, Tuple[Optional[Any], str]]:
    """Every converter on one reading, keyed in ALLOWED_TYPES order."""
    return {
        INTEGER: _to_integer(reading.cleaned),
        FLOAT: _to_float(reading.cleaned),
        BOOLEAN: _to_boolean(reading.lowered),
        **{kind: _to_temporal(reading.temporal, kind) for kind in TEMPORAL_TYPES},
    }


def coerce_all(value: Any) -> Dict[str, Tuple[Optional[Any], str]]:
    """coerce(value, t) for every type in ALLOWED_TYPES, from one reading.

    The same converters coerce dispatches to, so only the dispatch is written
    twice -- never a rule. test_value_types_invariants pins the two together over
    the adversarial corpus.
    """
    if is_blank(value):
        return {declared_type: (None, BLANK) for declared_type in ALLOWED_TYPES}
    return _convert_all(_read(value))


EXAMPLE_LIMIT = 3
"""Unconvertible values kept per type: the hint reports the first three."""

_NUMERIC_BOOLEANS = frozenset({"0", "1"})


class ColumnTally:
    """What a column's values add up to, kept as counts rather than values.

    Fixed per column: counts, three flags and at most EXAMPLE_LIMIT examples per
    type, however many rows are added. This is what lets the hint tools read a
    source of any length (KG-11); classify is this tally's shape over a list.

    Every count comes from one _read of each value and the same converters
    coerce uses, so the shape, the suggestion and the counts the hint reports
    cannot disagree with what the loader will do.
    """

    __slots__ = (
        "blank",
        "non_blank",
        "bare_numeric",
        "numeric_after_cleaning",
        "word_boolean_seen",
        "number_outside_vocabulary_seen",
        "any_fraction",
        "converted",
        "unconvertible",
        "examples",
    )

    def __init__(self) -> None:
        self.blank = 0
        self.non_blank = 0
        self.bare_numeric = 0
        self.numeric_after_cleaning = 0
        self.word_boolean_seen = False
        self.number_outside_vocabulary_seen = False
        self.any_fraction = False
        self.converted: Dict[str, int] = dict.fromkeys(ALLOWED_TYPES, 0)
        self.unconvertible: Dict[str, int] = dict.fromkeys(ALLOWED_TYPES, 0)
        self.examples: Dict[str, list] = {t: [] for t in ALLOWED_TYPES}

    @classmethod
    def of(cls, values: Iterable[Any]) -> "ColumnTally":
        tally = cls()
        for value in values:
            tally.add(value)
        return tally

    def add(self, value: Any) -> None:
        """Count one value. None and whitespace are blank, as everywhere here."""
        if is_blank(value):
            self.blank += 1
            return
        self.non_blank += 1
        reading = _read(value)
        self._count_shapes(reading)
        self._count_conversions(reading)

    def _count_shapes(self, reading: _Reading) -> None:
        """The shape counts and flags the converters cannot give."""
        # Unstripped, as classify always matched it; the pattern allows the
        # surrounding whitespace itself.
        bare = _BARE_NUMERIC.match(str(reading.raw)) is not None
        in_vocabulary = reading.lowered in _BOOLEAN_VALUES
        if bare:
            self.bare_numeric += 1
            if not in_vocabulary:
                self.number_outside_vocabulary_seen = True
        if in_vocabulary and reading.lowered not in _NUMERIC_BOOLEANS:
            self.word_boolean_seen = True
        if reading.cleaned is not None:
            self.numeric_after_cleaning += 1
            if _has_fraction(reading.cleaned):
                self.any_fraction = True

    def _count_conversions(self, reading: _Reading) -> None:
        """Each type's outcome, and its first EXAMPLE_LIMIT failures as given."""
        for declared_type, (_value, outcome) in _convert_all(reading).items():
            if outcome == CONVERTED:
                self.converted[declared_type] += 1
                continue
            self.unconvertible[declared_type] += 1
            examples = self.examples[declared_type]
            if len(examples) < EXAMPLE_LIMIT:
                examples.append(reading.raw)

    def _majority(self, count: int) -> bool:
        return count > self.non_blank * MAJORITY_SHARE

    @property
    def shape(self) -> str:
        """The most specific shape a strict majority of non-blank values fit.

        The rules and their order are classify's; see its docstring for why
        boolean is checked first. The boolean and temporal counts are the
        converters' own converted counts -- the same test, not a second one.
        """
        if not self.non_blank:
            return TEXT
        if self._majority(self.converted[BOOLEAN]) and not self._is_really_a_count():
            return BOOLEAN_LIKE
        if self._majority(self.bare_numeric):
            return BARE_NUMERIC
        if self._majority(self.numeric_after_cleaning):
            return NUMERIC_AFTER_CLEANING
        # A kind has to be a strict majority ALONE: a 45% zoned / 45% local
        # column gets no suggestion, because either type would clear the other
        # half.
        for shape, kind in TEMPORAL_SHAPE_TYPES.items():
            if self._majority(self.converted[kind]):
                return shape
        return TEXT

    def _is_really_a_count(self) -> bool:
        """True for a column that only LOOKS boolean because most of its values are
        0 or 1.

        A quantity that is overwhelmingly 0 or 1 -- a backorder quantity, a defect
        count, a discount count -- wins the boolean majority on those rows while
        carrying real 2s and 3s. Calling it boolean would make every value above 1
        unconvertible, and the loader clears an unconvertible value: the large
        numbers, the only ones that change an answer, are exactly the ones that
        would disappear, and at a minority share the refusal gate never fires.

        Narrow on purpose, so TRAP 5 still holds: this fires only when no value in
        the column is a word from the vocabulary (yes, true, y ...), so the boolean
        majority is carried by 0 and 1 alone, and some bare number outside the
        vocabulary is present. A genuine 0/1 flag has nothing else in the column
        and stays boolean_like; a yes/no column with a stray "5" stays
        boolean_like too.
        """
        return self.number_outside_vocabulary_seen and not self.word_boolean_seen

    @property
    def suggested_type(self) -> Optional[str]:
        """The type to suggest for this column's shape.

        The shape decides, and for a column needing cleaning it decides alone: every
        price in the bundled products.csv is a round dollar amount, so a
        whole-number test would suggest integer for a currency column and then
        refuse the first fractional price the data ever gains. Needing a currency
        symbol or thousands separator stripped is itself the evidence it is money.

        Wholeness is consulted only to split integer from float WITHIN the
        bare_numeric shape, where there is nothing else to go on -- and only for
        values that are numbers at all, so one "N/A" in a column of 400 integers
        cannot make it look fractional.

        It reads any_fraction (has_fractional_part's rule, _has_fraction) rather
        than inferring the fraction from a failed integer coercion. coerce refuses
        a value for two different reasons, and treating them alike types a column
        float on the strength of a whole number too large for Neo4j's INTEGER --
        which then loads as a rounded, wrong number and reports a clean
        conversion. An overflowing value leaves the suggestion at integer, so the
        loader counts it, reports it as an example, and clears it; one unstorable
        outlier does not cost the whole column its type, the same tolerance
        classify() already applies.
        """
        shape = self.shape
        if shape in TEMPORAL_SHAPE_TYPES:
            return TEMPORAL_SHAPE_TYPES[shape]
        if shape == BOOLEAN_LIKE:
            return BOOLEAN
        if shape == NUMERIC_AFTER_CLEANING:
            return FLOAT
        if shape == BARE_NUMERIC:
            return FLOAT if self.any_fraction else INTEGER
        return None
