"""Unit tests for the shared type classifier and converter.

This module is where correctness lives for KG-7: every other piece is wiring.
It needs no database and no source files.
"""
from agentic_kg.common.value_types import (
    BARE_NUMERIC, BLANK, BOOLEAN, BOOLEAN_LIKE, CONVERTED, FLOAT, INTEGER,
    NUMERIC_AFTER_CLEANING, TEXT, UNCONVERTIBLE, classify, coerce, is_blank,
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
    assert is_blank(None) and is_blank("") and is_blank("   ")
    assert not is_blank("0")
