"""Unit tests for the shared type classifier and converter.

This module is where correctness lives for KG-7: every other piece is wiring.
It needs no database and no source files.
"""

from datetime import timedelta, timezone

import pytest
from neo4j import time as neo4j_time

from agentic_kg.common.value_types import (
    BARE_NUMERIC,
    BLANK,
    BOOLEAN,
    BOOLEAN_LIKE,
    CONVERTED,
    DATE,
    DATE_LIKE,
    DATETIME,
    DATETIME_LIKE,
    FLOAT,
    INTEGER,
    LOCALDATETIME,
    LOCALDATETIME_LIKE,
    NUMERIC_AFTER_CLEANING,
    TEXT,
    UNCONVERTIBLE,
    classify,
    coerce,
    has_fractional_part,
    is_blank,
    parse_temporal,
)


def test_classify_survives_one_bad_row_among_many():
    """A classifier that required every value to match (the shape of
    graph_profile._numeric_like_state) would call this column text, and the model
    would never be told to type a column that is 99% numeric -- one the loader's
    own 50% tolerance would happily accept."""
    values = [str(n) for n in range(400)] + ["N/A"]
    assert classify(values) == BARE_NUMERIC


def test_classify_prefers_boolean_over_numeric_for_zero_one():
    """bare_numeric matches '1' and '0', so a classifier that checked numeric
    first would return bare_numeric here and suggest integer for a genuine flag.
    Nothing in data/bom would catch it: preferred_supplier is yes/no."""
    assert classify(["1", "0", "1", "1", "0"]) == BOOLEAN_LIKE


def test_classify_plain_counts_are_bare_numeric():
    """Values outside the boolean vocabulary must fall through the boolean check
    rather than being captured by it."""
    assert classify(["2", "3", "5", "10", "12"]) == BARE_NUMERIC


def test_classify_currency_is_numeric_after_cleaning():
    """A currency column must be distinguishable from a plain count, because
    that distinction alone decides integer vs float in the hint tool."""
    assert classify(["$246", "$489", "$1289"]) == NUMERIC_AFTER_CLEANING


def test_classify_text_column():
    assert classify(["Nordic Wood", "Shanghai Metal"]) == TEXT


def test_classify_ignores_blanks_when_deciding():
    """Blanks are absence, not evidence. Counting them in the denominator would
    make a sparse numeric column classify as text."""
    assert classify(["", "  ", "12", "13", "14"]) == BARE_NUMERIC


def test_classify_of_all_blanks_is_text():
    assert classify(["", "  "]) == TEXT


def test_coerce_strips_currency_and_thousands_separator():
    """No column in data/bom carries a thousands separator, so nothing built from
    the bundled data exercises the comma-stripping path at all -- even though the
    ticket names it as a motivating case."""
    assert coerce("$1,234.50", FLOAT) == (1234.50, CONVERTED)


def test_coerce_whole_valued_decimal_as_integer():
    assert coerce("42.0", INTEGER) == (42, CONVERTED)


def test_coerce_fractional_as_integer_refuses_rather_than_rounding():
    """Truncating 42.7 to 42 is a wrong answer nobody sees. A counted failure is
    visible."""
    converted, outcome = coerce("42.7", INTEGER)
    assert outcome == UNCONVERTIBLE
    assert converted is None


def test_coerce_fractional_as_float():
    assert coerce("42.7", FLOAT) == (42.7, CONVERTED)


def test_coerce_boolean_vocabulary_is_case_insensitive():
    for value in ("yes", "YES", "true", "True", "y", "Y", "1"):
        assert coerce(value, BOOLEAN) == (True, CONVERTED), value
    for value in ("no", "NO", "false", "False", "n", "N", "0"):
        assert coerce(value, BOOLEAN) == (False, CONVERTED), value


def test_coerce_outside_boolean_vocabulary_is_unconvertible():
    """The vocabulary is closed. 'maybe' must not become True by truthiness."""
    assert coerce("maybe", BOOLEAN) == (None, UNCONVERTIBLE)


def test_coerce_blank_is_its_own_outcome():
    """Blank must not be reported as unconvertible: the loader's refusal gate
    counts unconvertible values only, and a sparse-but-correct column would
    otherwise abort its own load."""
    for value in ("", "   ", None):
        assert coerce(value, FLOAT) == (None, BLANK), repr(value)


def test_coerce_tolerates_surrounding_whitespace():
    assert coerce("  42  ", INTEGER) == (42, CONVERTED)
    assert coerce(" yes ", BOOLEAN) == (True, CONVERTED)


def test_coerce_rejects_text_for_a_numeric_type():
    assert coerce("N/A", FLOAT) == (None, UNCONVERTIBLE)


def test_is_blank():
    assert is_blank(None)
    assert is_blank("")
    assert is_blank("   ")
    assert not is_blank("0")


# --- defects found by the pre-merge audit ------------------------------------


def test_coerce_accepts_a_negative_with_the_sign_before_the_currency_symbol():
    """'-$42.00' is what Excel and most ERP exports emit; '$-42.00' is what a
    bare float formatter emits. Accepting only the second is worse than
    accepting neither: a money column's positives convert while its refunds and
    credits become unconvertible and get CLEARED, so a later sum() is wrong with
    nothing in the graph showing why."""
    assert coerce("-$42.00", FLOAT) == (-42.0, CONVERTED)
    assert coerce("$-42.00", FLOAT) == (-42.0, CONVERTED)
    assert coerce("-$1,234.50", FLOAT) == (-1234.50, CONVERTED)


def test_coerce_reads_accounting_parentheses_as_negative():
    """(42.00) is a negative in every accounting export. Left unhandled it is
    unconvertible, and an unconvertible value is cleared -- so the credits
    vanish from a column whose positives all loaded."""
    assert coerce("(42.00)", FLOAT) == (-42.0, CONVERTED)
    assert coerce("($1,234.50)", FLOAT) == (-1234.50, CONVERTED)
    assert coerce("(42)", INTEGER) == (-42, CONVERTED)


def test_coerce_still_refuses_a_doubled_sign():
    """Widening the pattern to take a sign on either side of the symbol must not
    start accepting both at once."""
    assert coerce("+-42", FLOAT) == (None, UNCONVERTIBLE)


def test_classify_sees_a_negative_currency_column_as_numeric():
    values = ["$1,000.00", "$2,500.00", "-$42.00", "($300.00)"]
    assert classify(values) == NUMERIC_AFTER_CLEANING


def test_coerce_keeps_an_integer_too_large_for_a_float_exact():
    """Parsing through float rounds anything past 2**53 to the nearest
    representable value and still reports CONVERTED, because the fractional
    check cannot see damage done before it ran. Neo4j's INTEGER is a full 64-bit
    signed, so the wrong number would be stored without a murmur."""
    assert coerce("9007199254740993", INTEGER) == (9007199254740993, CONVERTED)
    assert coerce("9223372036854775807", INTEGER) == (9223372036854775807, CONVERTED)
    assert coerce("-9223372036854775808", INTEGER) == (-9223372036854775808, CONVERTED)


def test_coerce_refuses_an_integer_neo4j_cannot_hold():
    """Python's int is unbounded; Neo4j's INTEGER is signed 64-bit and the
    driver raises OverflowError packing anything wider. Converting it here would
    move the failure into the middle of a batch write, with rows already
    committed and nothing naming the column -- the opaque failure this whole
    change exists to replace. Refused here, it is counted and cleared like any
    other value that cannot be stored.

    The float fallback below the exact parse needs the same bound: a
    whole-valued float past the range ("1e19") reaches int() by a different
    route and would otherwise slip through."""
    assert coerce("9223372036854775808", INTEGER) == (None, UNCONVERTIBLE)
    assert coerce("9223372036854775808.0", INTEGER) == (None, UNCONVERTIBLE)
    assert coerce("-9223372036854775809", INTEGER) == (None, UNCONVERTIBLE)
    assert coerce("12345678901234567890", INTEGER) == (None, UNCONVERTIBLE)
    assert coerce("$99,223,372,036,854,775,808.00", INTEGER) == (None, UNCONVERTIBLE)


def test_has_fractional_part_separates_a_fraction_from_an_overflow():
    """coerce refuses "42.7" for being fractional and "9223372036854775809" for
    being too large for Neo4j's INTEGER. A caller inferring "fractional" from a
    failed integer coercion conflates the two and reads an overflowing whole
    number as evidence the column is fractional -- typing it float, which stores
    a rounded, wrong number and reports a clean conversion."""
    assert has_fractional_part("42.7") is True
    assert has_fractional_part("42.0") is False
    assert has_fractional_part("9223372036854775809") is False
    # Not evidence of anything: neither blanks nor dirt make a column fractional.
    assert has_fractional_part("N/A") is False
    assert has_fractional_part("") is False
    assert has_fractional_part(None) is False
    # Cleaned the same way coerce cleans, so the two cannot disagree.
    assert has_fractional_part("($42.50)") is True
    assert has_fractional_part("-$1,000.00") is False


def test_classify_counts_the_accounting_negatives_coerce_accepts():
    """classify and coerce must agree on what a column supports -- that is the
    whole reason both live in this module. coerce converts "($10)", so a column
    written entirely in accounting parentheses (a credits or adjustments column)
    must not come back as text: it would get no type suggestion and stay stored
    as strings, the defect this change exists to remove."""
    assert classify(["($10)", "($20)", "($30)"]) == NUMERIC_AFTER_CLEANING
    for value in ["($10)", "($20)", "($30)"]:
        assert coerce(value, FLOAT)[1] == CONVERTED


def test_coerce_keeps_a_float_formatted_integer_exact():
    """The trailing ".0" is the form a spreadsheet or a pandas round-trip emits
    for a whole number, so it is the likely shape of a large id in a real CSV.
    Reading it through float() rounded it to the nearest representable double
    and still reported CONVERTED -- the same silent corruption as the bare-digit
    form, surviving one string suffix away from it."""
    assert coerce("9007199254740993.0", INTEGER) == (9007199254740993, CONVERTED)
    assert coerce("9007199254740993.000", INTEGER) == (9007199254740993, CONVERTED)
    # The fractional refusal still holds; only trailing zeros are whole.
    assert coerce("42.7", INTEGER) == (None, UNCONVERTIBLE)
    assert coerce("42.0", INTEGER) == (42, CONVERTED)


def test_coerce_refuses_a_float_too_large_for_a_double():
    """A literal with more digits than a double can hold becomes inf, and the
    driver packs inf happily as a Neo4j FLOAT -- the graph would store Infinity
    and the load would report success."""
    too_many_digits = "1" + "0" * 400
    assert coerce(too_many_digits, FLOAT) == (None, UNCONVERTIBLE)


def test_classify_does_not_call_a_mostly_zero_one_count_a_flag():
    """A backorder quantity that is overwhelmingly 0 or 1 wins the boolean
    majority on those rows while carrying real 2s and 3s. Called boolean, every
    value above 1 is unconvertible and the loader CLEARS it -- the large numbers,
    the only ones that change an answer, are exactly the ones that disappear,
    and at a minority share the refusal gate never fires."""
    assert classify(["1", "0", "1", "2", "3"]) == BARE_NUMERIC


def test_classify_still_prefers_boolean_for_a_genuine_flag():
    """The count guard above must stay narrow enough to leave TRAP 5 intact: a
    real 0/1 flag has nothing else in the column."""
    assert classify(["1", "0", "1", "1", "0"]) == BOOLEAN_LIKE
    assert classify(["yes", "no", "yes", "5"]) == BOOLEAN_LIKE


# --- temporal types (KG-31) ---------------------------------------------------

ACCEPTED = [
    ("2025-03-04", DATE),
    ("9999-12-31", DATE),
    ("0001-01-01", DATE),
    ("2025-03-04T10:11", LOCALDATETIME),
    ("2025-03-04 10:11:12", LOCALDATETIME),
    ("2025-03-04T10:11:12.1", LOCALDATETIME),
    ("2025-03-04T10:11:12.123456789", LOCALDATETIME),
    ("2025-03-04T10:11:12Z", DATETIME),
    ("2025-03-04T10:11:12.123456789+01:00", DATETIME),
    ("2025-03-04T10:11-05:30", DATETIME),
    ("2025-03-04T10:11:12+18:00", DATETIME),
    ("2025-03-04T10:11:12-00:00", DATETIME),
    ("9999-12-31T23:59:59+01:00", DATETIME),
    ("0001-01-01T00:00:00-01:00", DATETIME),
    ("9999-12-31T23:59:59Z", DATETIME),
]


@pytest.mark.parametrize("text, kind", ACCEPTED)
def test_parse_temporal_reads_each_shape_as_its_own_kind(text, kind):
    parsed = parse_temporal(text)
    assert parsed is not None, text
    assert parsed[0] == kind


REFUSED = [
    # other date spellings: guessing which one is meant stores a wrong date
    "03/04/2025",
    "20250304",
    "2025-W10-2",
    "2025-063",
    "20250304T101112",
    "25-03-04",
    "2025-3-4",
    # separators and case: only 'T' or one space, only upper-case 'Z'
    "2025-03-04t10:11",
    "2025-03-04T10:11:12z",
    "2025-03-04  10:11",
    "2025-03-04T10:11:12,5",
    "2025-03-04T10:11:12+0100",
    # ranges the grammar cannot express
    "2025-03-04T24:00:00",
    "2025-03-04T23:59:60",
    "2025-03-04T10:11:12.1234567890",
    "2025-03-04T10:11.5",
    "2025-03-04T10",
    "2025-03-04Z",
    "2025-03-04T10:11:12+05:75",
    # offsets the server refuses
    "2025-03-04T10:11:12+18:01",
    "2025-03-04T10:11:12+23:59",
    # not real dates
    "2025-02-30",
    "0000-01-01",
    # Unicode digits: \d would match these, int() would convert them
    "٢٠٢٥-٠٣-٠٤",
    # a time with no date
    "15:15:00",
    "",
    "2025-03-04\n",
]


@pytest.mark.parametrize("text", REFUSED)
def test_parse_temporal_refuses_everything_outside_the_grammar(text):
    assert parse_temporal(text) is None


def test_the_temporal_type_names_mirror_cypher():
    assert (DATE, DATETIME, LOCALDATETIME) == ("date", "datetime", "localdatetime")


def test_coerce_stores_a_date_as_a_neo4j_date():
    assert coerce("2025-03-04", DATE) == (neo4j_time.Date(2025, 3, 4), CONVERTED)


@pytest.mark.parametrize(
    "digits",
    ["1", "12", "123", "1234", "12345", "123456", "1234567", "12345678", "123456789"],
)
@pytest.mark.parametrize(
    "suffix, declared", [("", LOCALDATETIME), ("Z", DATETIME), ("+01:00", DATETIME)]
)
def test_every_fractional_digit_is_kept(digits, suffix, declared):
    """fromisoformat truncates .123456789 to six digits with no error; the whole
    point of the strict pattern is that it never does."""
    value, outcome = coerce(f"2025-03-04T10:11:12.{digits}{suffix}", declared)
    assert outcome == CONVERTED
    assert value.nanosecond == int(digits.ljust(9, "0"))


def test_a_zoned_value_keeps_its_offset_as_written():
    expected = {
        "2025-03-04T10:11:12+01:00": timedelta(hours=1),
        "2025-03-04T10:11:12-05:30": -timedelta(hours=5, minutes=30),
        "2025-03-04T10:11:12Z": timedelta(0),
    }
    for text, offset in expected.items():
        value, outcome = coerce(text, DATETIME)
        assert outcome == CONVERTED
        assert value.utcoffset() == offset, text


def _fields(value):
    """Every date and clock field plus the offset, as a tuple.

    DateTime.__eq__ compares two zoned values by UTC instant when their offsets
    differ, so 10:11-05:30 equals 15:41Z. Comparing fields and offset as well
    pins what was written, not just when it was.
    """
    return (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.nanosecond,
        value.utcoffset(),
    )


def test_a_zoned_value_keeps_every_field_it_was_written_with():
    """Distinct month/day and hour/minute/second, so a swapped group fails."""
    value, outcome = coerce("2025-03-04T10:11:12.123456789-05:30", DATETIME)
    assert outcome == CONVERTED
    zone = timezone(-timedelta(hours=5, minutes=30))
    assert value == neo4j_time.DateTime(2025, 3, 4, 10, 11, 12, 123456789, tzinfo=zone)
    assert _fields(value) == (
        2025,
        3,
        4,
        10,
        11,
        12,
        123456789,
        -timedelta(hours=5, minutes=30),
    )


def test_a_local_value_keeps_every_field_it_was_written_with():
    value, outcome = coerce("2025-03-04 10:11:12.5", LOCALDATETIME)
    assert outcome == CONVERTED
    assert value == neo4j_time.DateTime(2025, 3, 4, 10, 11, 12, 500000000)
    assert _fields(value) == (2025, 3, 4, 10, 11, 12, 500000000, None)


@pytest.mark.parametrize(
    "text, declared",
    [
        ("2025-03-04", DATETIME),
        ("2025-03-04", LOCALDATETIME),
        ("2025-03-04T10:11:12", DATE),
        ("2025-03-04T10:11:12Z", DATE),
        ("2025-03-04T10:11:12Z", LOCALDATETIME),
        ("2025-03-04T10:11:12", DATETIME),
    ],
)
def test_each_type_takes_only_its_own_shape(text, declared):
    assert coerce(text, declared) == (None, UNCONVERTIBLE)


@pytest.mark.parametrize(
    "text", ["9999-12-31T23:59:59-01:00", "0001-01-01T00:00:00+01:00"]
)
def test_a_zoned_value_whose_utc_instant_does_not_exist_is_unconvertible(text):
    """These construct fine in Python. The driver then converts to UTC while
    packing and raises, which fails the whole batch 'after N rows committed'.
    9999-12-31 is a common 'no end date' value in real exports."""
    assert parse_temporal(text) is None
    assert coerce(text, DATETIME) == (None, UNCONVERTIBLE)


def test_a_temporal_value_tolerates_surrounding_whitespace_but_not_inner_space():
    assert coerce("  2025-03-04  ", DATE)[1] == CONVERTED
    assert coerce(" 2025-03-04T10:11:12Z ", DATETIME)[1] == CONVERTED
    assert coerce("2025-03-04  10:11:12", LOCALDATETIME) == (None, UNCONVERTIBLE)


def test_a_blank_temporal_cell_is_blank_not_unconvertible():
    for declared in (DATE, DATETIME, LOCALDATETIME):
        for value in ("", "   ", None):
            assert coerce(value, declared) == (None, BLANK), (declared, value)


def test_classify_reports_each_temporal_shape():
    assert classify(["2025-03-04", "2025-03-05"]) == DATE_LIKE
    assert (
        classify(["2025-03-04T10:11:12Z", "2025-03-05T10:11:12+01:00"]) == DATETIME_LIKE
    )
    assert classify(["2025-03-04T10:11:12", "2025-03-05 10:11"]) == LOCALDATETIME_LIKE


def test_classify_needs_one_temporal_kind_to_be_a_strict_majority():
    mixed = ["2025-03-04T10:00:00Z"] * 45 + ["2025-03-04T10:00:00"] * 45 + ["n/a"] * 10
    assert classify(mixed) == TEXT
    # day/month ambiguous and a bare time carry no ISO shape at all
    assert classify(["03/04/2025", "04/05/2025", "13/05/2025"]) == TEXT
    assert classify(["15:15:00", "09:30:00"]) == TEXT
    # exactly half is not a majority, the same strictness as the numeric shapes
    assert classify(["2025-03-04", "2025-03-05", "x", "y"]) == TEXT


def test_classify_does_not_call_impossible_dates_a_date():
    """A suggestion for a column the loader would then clear in full is the exact
    drift sharing one parser exists to prevent."""
    assert classify(["2025-02-30"] * 5) == TEXT


def test_classify_strips_padded_values_like_coerce_does():
    """coerce strips before parsing; if classify did not, a padded date column
    would classify as text while every value converted."""
    assert classify([" 2025-03-04", "2025-03-05 ", "\t2025-03-06"]) == DATE_LIKE
