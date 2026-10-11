"""The text of graph_profile.NUMERIC_CAST and NUMERIC_CAST_GAP_CHECK.

What the cast does in Neo4j is pinned by tests/integration/test_numeric_cast_integration.py; these pin
that it is generated from the converter's constants and is safe to put in a prompt.
"""

import pytest

from agentic_kg.common.graph_profile import (
    _NEO4J_TRIM_MISSES,
    NUMERIC_CAST,
    NUMERIC_CAST_GAP_CHECK,
    _cypher_char,
)
from agentic_kg.common.value_types import STRIP_FROM_NUMBERS


def test_the_cast_lists_every_stripped_character_but_the_opening_bracket():
    for char in STRIP_FROM_NUMBERS:
        if char != "(":
            assert f"'{char}'" in NUMERIC_CAST, char
    assert NUMERIC_CAST.startswith("toFloat(trim(reduce(text = replace(v, '(', '-'), ")


def test_the_cast_writes_whitespace_as_escapes_never_as_raw_characters():
    for char in _NEO4J_TRIM_MISSES:
        assert char not in NUMERIC_CAST
        assert f"'\\u{ord(char):04X}'" in NUMERIC_CAST


def test_the_cast_and_the_gap_check_carry_no_brace():
    for text in (NUMERIC_CAST, NUMERIC_CAST_GAP_CHECK):
        assert "{" not in text
        assert "}" not in text


def test_the_gap_check_counts_non_blank_values_the_cast_nulls():
    assert NUMERIC_CAST_GAP_CHECK == (
        f"WHERE trim(v) <> '' AND {NUMERIC_CAST} IS NULL "
        "RETURN count(*) AS uncast, collect(v)[..5] AS examples"
    )


@pytest.mark.parametrize("char", ["'", "\\"])
def test_a_quote_or_backslash_cannot_enter_the_list(char):
    with pytest.raises(ValueError):
        _cypher_char(char)


def test_a_visible_character_and_the_space_are_written_as_themselves():
    assert _cypher_char("€") == "'€'"
    assert _cypher_char(" ") == "' '"
    assert _cypher_char("\xa0") == "'\\u00A0'"
