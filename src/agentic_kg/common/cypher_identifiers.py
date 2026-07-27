"""Shared validator for identifiers destined for interpolation into Cypher.

Labels, relationship types and property/column names cannot be parameterised
in Cypher's structural positions (label position, relationship-type position,
property keys in `CREATE CONSTRAINT ... FOR (n:Label) REQUIRE n.prop ...`), so
callers that need them there interpolate validated strings into query text
instead. `is_symbol()` (`common/neo4j_for_adk.py`) is not a safe basis for
that by itself: it rejects only strings containing a literal space and
strings exactly equal to one of its ~50 keywords, so newlines, tabs,
parentheses, braces, a leading digit or a backtick all pass it and can escape
the identifier position. Requiring a bare identifier via regex, in addition
to the keyword check, is what makes the interpolation safe.

Every caller that interpolates a label, relationship type or property/column
name into Cypher must validate it with `checked()` first.
"""
import re

from agentic_kg.common.neo4j_for_adk import is_symbol

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class InvalidIdentifier(ValueError):
    """A label, relationship type or column/property name failed validation."""


def checked(kind: str, value: str) -> str:
    """Validate an identifier destined for interpolation into Cypher.

    is_symbol() alone is not sufficient: it rejects only literal spaces and
    exact keyword matches, so newlines, tabs, parentheses and braces pass it
    and can escape the identifier position. Requiring a bare identifier is
    what makes the interpolation safe. is_symbol() is still called, since it
    usefully rejects Cypher keywords the regex alone would allow.

    Raises:
        InvalidIdentifier: if the value is not a safe bare identifier.

    Returns:
        The validated value, unchanged, for convenient chaining.
    """
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or not is_symbol(value):
        raise InvalidIdentifier(
            f"Invalid {kind}: '{value}'. It must be a letter or underscore followed by "
            f"letters, digits or underscores, and cannot be a Cypher keyword."
        )
    return value
