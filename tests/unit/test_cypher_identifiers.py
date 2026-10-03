"""checked() enforces the character rule for labels and types, checked_field()
only what Neo4j itself refuses for columns and properties, and quote() keeps any
name inside its backticks (KG-44, KG-51).

Neo4j accepts keywords such as Order, END or null as names, and the build
backtick-quotes every name it writes into Cypher, so a keyword is an ordinary
name. checked() still refuses anything but a plain identifier.
"""

import re

import pytest

from agentic_kg.common.cypher_identifiers import (
    InvalidIdentifier,
    checked,
    checked_field,
    quote,
)

CHARACTER_RULE = (
    "It must be a letter or underscore followed by letters, digits or underscores."
)


@pytest.mark.parametrize("name", ["Order", "END", "null", "SET", "_id", "Order2"])
def test_checked_accepts_keywords_and_plain_identifiers(name):
    assert checked("label", name) == name


@pytest.mark.parametrize(
    "name", ["1id", "Order ID", "Or`der", "Order)", "Order{", "Order\nX", ""]
)
def test_checked_refuses_anything_but_a_plain_identifier(name):
    with pytest.raises(InvalidIdentifier) as exc:
        checked("label", name)
    assert "keyword" not in str(exc.value)


def test_checked_message_names_the_kind_and_the_character_rule():
    with pytest.raises(InvalidIdentifier) as exc:
        checked("column name", "1id")
    assert str(exc.value) == f"Invalid column name: '1id'. {CHARACTER_RULE}"


def test_checked_refuses_a_non_string():
    with pytest.raises(InvalidIdentifier) as exc:
        checked("label", 123)  # type: ignore[arg-type]
    assert str(exc.value) == f"Invalid label: '123'. {CHARACTER_RULE}"


def test_quote_backticks_a_name_and_doubles_embedded_backticks():
    assert quote("Order") == "`Order`"
    assert quote("we`ird") == "`we``ird`"


INJECTION_KEY = "x\\u0060: 1}) SET n.pwned = true //"


@pytest.mark.parametrize(
    "name, quoted",
    [
        ("Order", "`Order`"),
        ("we`ird", "`we``ird`"),
        ("C:\\users", "`C:\\u005Cusers`"),
        ("a\\", "`a\\u005C`"),
        ("a\\u0041b", "`a\\u005Cu0041b`"),
        (INJECTION_KEY, "`x\\u005Cu0060: 1}) SET n.pwned = true //`"),
    ],
    ids=[
        "plain",
        "backtick",
        "windows-path",
        "trailing-backslash",
        "escape",
        "injection",
    ],
)
def test_quote_pins_its_exact_output(name, quoted):
    """Neo4j decodes a backslash-u escape inside backticks, so a bare backslash can
    rename a key (a\\u0041b is stored as aAb) or close the name and run Cypher
    after it. Each backslash is written as its own escape instead."""
    assert quote(name) == quoted


_ONLY_ITS_OWN_ESCAPES = re.compile(r"`(?:[^\\`]|``|\\u005C)*`")


@pytest.mark.parametrize(
    "name",
    ["Order ID", "Straße", "a\nb", " ", "C:\\users", "a\\", "a\\\\b", INJECTION_KEY],
)
def test_quote_leaves_only_backslashes_it_wrote_itself_and_doubled_backticks(name):
    assert _ONLY_ITS_OWN_ESCAPES.fullmatch(quote(name))


FIELD_RULE = "It must be 1 to 16,383 characters of text, with no NUL."


@pytest.mark.parametrize(
    "name",
    [
        "Order ID",
        "customer-id",
        "Straße",
        "we`ird",
        "C:\\users",
        "a\nb",
        " ",
        "\t",
        "1id",
        "k" * 16383,
        "é" * 16383,
    ],
    ids=lambda name: repr(name)[:24],
)
def test_checked_field_accepts_any_non_empty_text(name):
    assert checked_field("column name", name) == name


@pytest.mark.parametrize(
    "value, shown",
    [
        ("", ""),
        ("a\x00b", "a\x00b"),
        ("k" * 16384, "k" * 80 + "..."),
        (None, "None"),
        (5, "5"),
        ([], "[]"),
        (["a"], "['a']"),
        ({}, "{}"),
    ],
    ids=["empty", "nul", "too-long", "none", "int", "empty-list", "list", "dict"],
)
def test_checked_field_refuses_what_neo4j_cannot_take(value, shown):
    with pytest.raises(InvalidIdentifier) as exc:
        checked_field("column name", value)  # type: ignore[arg-type]
    assert str(exc.value) == f"Invalid column name: '{shown}'. {FIELD_RULE}"


def test_only_a_refusal_by_checked_can_be_fixed_by_renaming():
    with pytest.raises(InvalidIdentifier) as field:
        checked_field("column name", "")
    with pytest.raises(InvalidIdentifier) as strict:
        checked("label", "Not A Label")
    assert field.value.renamable is False
    assert strict.value.renamable is True
