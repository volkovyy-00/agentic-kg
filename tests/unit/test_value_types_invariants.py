"""Cross-component invariants for the typed-property pipeline.

Every defect found in review on this branch lived in a SEAM rather than in a
component: classify and coerce disagreeing about accounting parentheses,
coerce and the driver disagreeing about how large an integer may be,
_suggested_type and coerce disagreeing about what an integer refusal means.
Each piece was individually defensible; the pair was not. Two of the three were
introduced by the fix for the previous one.

Example-based tests cannot catch that class, because the example that breaks is
by definition the one nobody thought of. These tests state the relationships
themselves and run one adversarial corpus through them, so a future change that
makes two components disagree fails here rather than in a graph.

Kept separate from test_value_types.py: those pin decisions ("42.7 must not
round"), these pin relationships ("whatever coerce accepts, the driver stores").
"""

from decimal import Decimal
from math import isfinite

import pytest

# Deliberately the driver's own packer rather than our belief about its limits:
# the seam being pinned is coerce-versus-driver, and asserting against a
# reimplementation of the rule would pin the belief instead. A neo4j upgrade
# that moves this import should fail loudly here -- that is the signal to
# re-verify the bounds, not a reason to guard the import.
from neo4j._codec.packstream.v1 import Packer

from agentic_kg.common.value_types import (
    BARE_NUMERIC,
    BLANK,
    BOOLEAN,
    BOOLEAN_LIKE,
    CONVERTED,
    FLOAT,
    INTEGER,
    MAJORITY_SHARE,
    NUMERIC_AFTER_CLEANING,
    TEXT,
    _clean_number,
    classify,
    coerce,
    has_fractional_part,
    is_blank,
)
from agentic_kg.tools.file_tools import _suggested_type

INT64_LIMIT = 2**63

# One value per way a CSV cell has ever surprised this pipeline.
ADVERSARIAL_VALUES = [
    # plain
    "0",
    "1",
    "42",
    "-7",
    "+7",
    "  42  ",
    # float precision boundaries
    "9007199254740992",
    "9007199254740993",
    "9007199254740993.0",
    # int64 boundaries and just past them, both signs, both spellings
    str(INT64_LIMIT - 1),
    str(INT64_LIMIT),
    str(INT64_LIMIT + 1),
    str(-INT64_LIMIT),
    str(-INT64_LIMIT - 1),
    f"{INT64_LIMIT}.0",
    # fractional, and whole numbers wearing a fractional coat
    "42.7",
    "42.0",
    "42.000",
    "-0.0",
    "0.5",
    # money, in every shape the exporters emit
    "$42",
    "$42.00",
    "-$42.00",
    "$-42.00",
    "(42.00)",
    "($1,234.50)",
    "$1,234,567.89",
    "€10",
    "£10",
    "¥10",
    "₹10",
    # booleans, whole vocabulary and casing
    "yes",
    "YES",
    "no",
    "N",
    "true",
    "False",
    "y",
    "1",
    "0",
    # blanks
    "",
    "   ",
    None,
    # dirt that must never become a number
    "N/A",
    "-",
    "--",
    "+-42",
    "1,2,3",
    "1e5",
    "abc",
    "4 2",
    "$",
    "()",
    # too many digits for a double
    "1" + "0" * 400,
]

# Columns, not values: the majority rules only mean something over a column.
ADVERSARIAL_COLUMNS = [
    ["1", "2", "3"],
    ["1", "0", "1", "1"],
    ["1", "0", "1", "2", "3"],
    ["yes", "no", "yes"],
    ["1.5", "2.5", "3.0"],
    ["42.0", "7.0"],
    ["$10", "$20", "$30"],
    ["($10)", "($20)", "($30)"],
    ["$1,000.00", "-$42.00", "($300.00)"],
    [str(n) for n in range(50)] + ["N/A"],
    ["1", "2", str(INT64_LIMIT + 1)],
    ["1", "2", "9007199254740993"],
    ["", "  ", "12", "13"],
    ["Nordic Wood", "Shanghai Metal"],
    ["1", "", "2", None, "3"],
]

ALL_TYPES = (INTEGER, FLOAT, BOOLEAN)


def _packs(value) -> bool:
    """True when the driver can actually put this on the wire."""
    import io

    try:
        Packer(io.BytesIO()).pack(value)
        return True
    except Exception:  # noqa: BLE001 - any refusal is a refusal
        return False


@pytest.mark.parametrize("value", ADVERSARIAL_VALUES)
@pytest.mark.parametrize("declared_type", ALL_TYPES)
def test_every_converted_value_is_one_the_driver_can_store(value, declared_type):
    """coerce reporting CONVERTED is a promise that the write will succeed.

    Broken twice on this branch: an integer past int64 raised OverflowError
    inside the driver mid-batch, and a 400-digit literal became inf and packed
    happily as Infinity. Both were reported as clean conversions, so the failure
    surfaced with rows already committed and nothing naming the column.
    """
    converted, outcome = coerce(value, declared_type)
    if outcome != CONVERTED:
        return
    assert _packs(converted), (
        f"coerce({value!r}, {declared_type}) reported CONVERTED as {converted!r}, "
        f"which the driver cannot pack"
    )
    # Packing is necessary but not sufficient: inf packs perfectly happily, and
    # the graph then holds Infinity for a value the source wrote as digits.
    if isinstance(converted, float):
        assert isfinite(converted), (
            f"coerce({value!r}, {declared_type}) reported CONVERTED as "
            f"{converted!r}, which stores as Infinity"
        )


@pytest.mark.parametrize("value", ADVERSARIAL_VALUES)
@pytest.mark.parametrize("declared_type", ALL_TYPES)
def test_a_non_converted_outcome_never_carries_a_value(value, declared_type):
    """Callers distinguish BLANK from UNCONVERTIBLE by the outcome, never by
    testing the value -- so a non-CONVERTED outcome must not smuggle one."""
    converted, outcome = coerce(value, declared_type)
    if outcome != CONVERTED:
        assert converted is None
    assert (outcome == BLANK) == is_blank(value)


@pytest.mark.parametrize("column", ADVERSARIAL_COLUMNS)
def test_a_suggested_type_never_trips_the_loaders_refusal_gate(column):
    """The hint tool and the loader must not contradict each other.

    value_types' docstring states this directly: a column this module suggests a
    type for must not then trip the loader's gate. If it can, the model is told
    a column is fine and the build then refuses it -- with the user holding an
    approved plan that cannot load. Broken once already, by classify not
    counting the accounting parentheses coerce accepts.
    """
    suggested = _suggested_type(classify(column), column)
    if suggested is None:
        return
    non_blank = [value for value in column if not is_blank(value)]
    unconvertible = sum(
        1 for value in non_blank if coerce(value, suggested)[1] != CONVERTED
    )
    assert unconvertible <= len(non_blank) * MAJORITY_SHARE, (
        f"column {column!r} was suggested {suggested}, which fails on "
        f"{unconvertible} of {len(non_blank)} non-blank values -- the loader "
        f"would refuse a column the model was told was fine"
    )


@pytest.mark.parametrize("value", ADVERSARIAL_VALUES)
def test_has_fractional_part_agrees_with_coerce_about_what_a_fraction_is(value):
    """The two must not drift, because _suggested_type uses one to explain the
    other's refusals. A fractional value can never be a valid integer, and a
    value that converts as an integer can never be fractional."""
    integer_outcome = coerce(value, INTEGER)[1]
    if has_fractional_part(value):
        assert integer_outcome != CONVERTED, (
            f"{value!r} is fractional yet coerced to an integer"
        )
    if integer_outcome == CONVERTED:
        assert not has_fractional_part(value)


@pytest.mark.parametrize("value", ADVERSARIAL_VALUES)
def test_anything_an_integer_accepts_a_float_accepts_too(value):
    """FLOAT is the wider type; a value storable as an integer must be storable
    as a float. If this ever fails, _suggested_type's fallback from integer to
    float would silently drop values rather than widening the column."""
    if coerce(value, INTEGER)[1] == CONVERTED:
        assert coerce(value, FLOAT)[1] == CONVERTED, (
            f"{value!r} converts as INTEGER but not as FLOAT"
        )


@pytest.mark.parametrize("column", ADVERSARIAL_COLUMNS)
def test_a_column_the_converter_accepts_is_never_classified_text(column):
    """The under-claim direction, which the gate invariant above cannot see.

    If classify says text, the column gets no suggestion, so the loader never
    runs and no gate can fire -- the failure is silent by construction and the
    column stays a string, which is the whole defect this branch exists to fix.
    That is exactly how the accounting-parenthesis drift hid: coerce converted
    every value of "($10), ($20), ($30)" while classify called it text.
    """
    non_blank = [value for value in column if not is_blank(value)]
    if not non_blank:
        return
    for declared_type in ALL_TYPES:
        convertible = sum(
            1 for value in non_blank if coerce(value, declared_type)[1] == CONVERTED
        )
        if convertible > len(non_blank) * MAJORITY_SHARE:
            assert classify(column) != TEXT, (
                f"column {column!r} was classified text, but {convertible} of "
                f"{len(non_blank)} non-blank values convert as {declared_type} "
                f"-- the model is never told it could be typed"
            )
            return


@pytest.mark.parametrize("value", ADVERSARIAL_VALUES)
def test_an_integer_is_never_stored_as_a_different_integer(value):
    """A declared integer must arrive exactly or not at all.

    Losing precision is inherent to a declared FLOAT and is the cost of asking
    for one, so this binds INTEGER only -- where it was broken twice, by reading
    a value past 2**53 through float and by reporting an out-of-range value as
    converted. Both stored a number the source never wrote.
    """
    if is_blank(value) or _clean_number(str(value).strip()) is None:
        return
    converted, outcome = coerce(value, INTEGER)
    if outcome != CONVERTED:
        return
    assert Decimal(converted) == Decimal(_clean_number(str(value).strip())), (
        f"{value!r} declared integer was stored as {converted!r}, "
        f"a different number from the one the source wrote"
    )


@pytest.mark.parametrize("column", ADVERSARIAL_COLUMNS)
def test_a_column_of_whole_bare_numbers_is_never_suggested_float(column):
    """Where wholeness is the ONLY thing separating integer from float, a column
    without a single fractional value must not be typed float.

    Scoped to the bare_numeric shape on purpose: a currency column is money and
    is suggested float even when every amount is round, which is deliberate and
    documented. Within bare_numeric there is nothing else to go on, so a float
    suggestion here means something was mistaken for a fraction -- which is
    exactly what an out-of-range integer became once coerce gained a second
    reason to refuse one, storing 9223372036854775809 as 9.223372036854776e+18.
    """
    if classify(column) != BARE_NUMERIC:
        return
    if any(has_fractional_part(value) for value in column):
        return
    assert _suggested_type(BARE_NUMERIC, column) != FLOAT, (
        f"column {column!r} holds no fractional value yet was suggested float, "
        f"which cannot store every one of its values exactly"
    )


@pytest.mark.parametrize("column", ADVERSARIAL_COLUMNS)
def test_a_numeric_shape_is_never_reported_for_a_column_of_text(column):
    """classify's shapes are evidence shown to the model. A numeric shape is a
    claim that a majority of the column reads as a number, so it must be true
    under the converter that will actually run."""
    shape = classify(column)
    if shape not in (BARE_NUMERIC, NUMERIC_AFTER_CLEANING, BOOLEAN_LIKE):
        return
    declared = BOOLEAN if shape == BOOLEAN_LIKE else FLOAT
    non_blank = [value for value in column if not is_blank(value)]
    convertible = sum(
        1 for value in non_blank if coerce(value, declared)[1] == CONVERTED
    )
    assert convertible > len(non_blank) * MAJORITY_SHARE, (
        f"column {column!r} was classified {shape}, but only {convertible} of "
        f"{len(non_blank)} non-blank values convert as {declared}"
    )
