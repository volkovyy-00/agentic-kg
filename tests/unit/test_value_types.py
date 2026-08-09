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
