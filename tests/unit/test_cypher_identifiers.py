"""checked() enforces the character rule, not a keyword list (KG-44).

Neo4j accepts keywords such as Order, END or null as labels, types and keys,
and the build backtick-quotes every name it writes into Cypher, so a keyword
is an ordinary name. checked() still refuses anything but a plain identifier.
"""

import pytest

from agentic_kg.common.cypher_identifiers import InvalidIdentifier, checked, quote

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
