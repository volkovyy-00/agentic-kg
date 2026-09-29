"""Checking and quoting identifiers that go into Cypher query text.

Labels, relationship types and property keys cannot be parameterised in
Cypher's structural positions (label position, relationship-type position,
map keys in a MERGE/MATCH pattern, property keys in
`CREATE CONSTRAINT ... FOR (n:Label) REQUIRE n.prop ...`), so the build writes
them into the query text itself. Which helper applies depends on where the
name came from, not on what it looks like:

- A name the model supplies (a plan's label, relationship type, key or join
  column) is checked with `checked()`, then written into the query with
  `quote()`. `checked()` refuses anything but a plain identifier -- a letter
  or underscore, then letters, digits or underscores -- so a newline,
  parenthesis, brace or backtick never reaches the query. It does not refuse
  Cypher keywords: `Order`, `END` or `null` are ordinary names once quoted,
  and Neo4j accepts them as labels, types and keys.
- A name read back out of the database is only quoted. Neo4j accepts labels
  `checked()` would refuse (`Legal Entity`, `10-K`), so checking them would
  fail on data the graph legitimately holds. The distinction is provenance,
  not syntax: `checked()` guards against injection from an untrusted source,
  and the database is not one.
"""

import re

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class InvalidIdentifier(ValueError):
    """A label, relationship type or column/property name failed validation."""


def checked(kind: str, value: str) -> str:
    """Refuse a model-supplied name that is not a plain identifier.

    Raises:
        InvalidIdentifier: if the value is not a letter or underscore followed
            by letters, digits or underscores.

    Returns:
        The value, unchanged. Write it into query text with quote().
    """
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise InvalidIdentifier(
            f"Invalid {kind}: '{value}'. It must be a letter or underscore followed by "
            f"letters, digits or underscores."
        )
    return value


def quote(name: str) -> str:
    """Backtick-quote a name for Cypher query text.

    Escaping doubles any embedded backtick, which is Cypher's own convention.
    """
    return "`" + name.replace("`", "``") + "`"
