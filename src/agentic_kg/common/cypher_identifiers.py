"""Checking and quoting identifiers that go into Cypher query text.

Labels, relationship types and property keys cannot be parameterised in
Cypher's structural positions (label position, relationship-type position,
map keys in a MERGE/MATCH pattern, property keys in
`CREATE CONSTRAINT ... FOR (n:Label) REQUIRE n.prop ...`), so the build writes
them into the query text itself. Which helper applies depends on where the
name came from, not on what it looks like:

- A label or relationship type the model supplies is checked with `checked()`,
  then written into the query with `quote()`. `checked()` refuses anything but
  a plain identifier -- a letter or underscore, then letters, digits or
  underscores. It does not refuse Cypher keywords: `Order`, `END` or `null` are
  ordinary names once quoted, and Neo4j accepts them as labels and types.
- A node key, join column, matched property or constraint key is a column or
  property of the user's own file, so the file decides how it is spelled: an
  exact header is already required, and nothing can rename a column. It is
  checked with `checked_field()`, which refuses only what Neo4j itself cannot
  take as a name -- empty text, a NUL character, more than 16,383 characters --
  and then written with `quote()`. `Order ID`, `customer-id` and `Straße` are
  ordinary names.
- A name read back out of the database is only quoted. Neo4j accepts labels
  `checked()` would refuse (`Legal Entity`, `10-K`), so checking them would
  fail on data the graph legitimately holds.

`quote()` is what keeps a name inside its identifier position, for any text.
Inside backticks two things still act: a backtick, which ends the name, and a
backslash, because Neo4j decodes a \\uXXXX escape there. `quote()` doubles every
backtick and writes every backslash as its own escape, so the server reads back
exactly the characters it was given. For labels and relationship types the
character rule in `checked()` is a policy on which names the build accepts and
a second guard; for columns and properties `quote()` is the only guard, which is
why its output is pinned in tests/unit/test_cypher_identifiers.py.
"""

import re

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Neo4j's own limit on a name's length, in characters (TokenLengthError above it).
MAX_NAME_LENGTH = 16383
_SHOWN_CHARACTERS = 80


class InvalidIdentifier(ValueError):
    """A label, relationship type or column/property name failed validation.

    `renamable` says whether renaming the name can fix it: true for a label or
    relationship type, which the model chooses, false for a column or property,
    which the file decides.
    """

    def __init__(self, message: str, *, renamable: bool = True):
        super().__init__(message)
        self.renamable = renamable


def checked(kind: str, value: str) -> str:
    """Refuse a model-supplied label or relationship type that is not a plain identifier.

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


def checked_field(kind: str, value: str) -> str:
    """Refuse a column or property name that Neo4j itself cannot take.

    Any text of 1 to MAX_NAME_LENGTH characters without a NUL passes, whitespace
    only included: the file's header is the authority on its own spelling.

    Raises:
        InvalidIdentifier: (renamable=False) if the value is not such text.

    Returns:
        The value, unchanged. Write it into query text with quote().
    """
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_NAME_LENGTH
        or "\x00" in value
    ):
        shown = str(value)
        if len(shown) > _SHOWN_CHARACTERS:
            shown = shown[:_SHOWN_CHARACTERS] + "..."
        raise InvalidIdentifier(
            f"Invalid {kind}: '{shown}'. It must be 1 to {MAX_NAME_LENGTH:,} "
            f"characters of text, with no NUL.",
            renamable=False,
        )
    return value


def quote(name: str) -> str:
    """Backtick-quote a name for Cypher query text, for any text the name holds.

    Two things still act inside backticks: a backtick, which ends the name, and a
    backslash, because Neo4j decodes a \\uXXXX escape there. Each backtick is
    doubled, which is Cypher's own convention. Each backslash is written as its
    own escape, \\u005C, so the server reads back exactly the characters it was
    given: a key `a\\u0041b` stays that text rather than becoming `aAb`, and a key
    holding \\u0060 cannot close the name and run Cypher after it. The two
    replacements are independent: neither adds the other's character.
    """
    return "`" + name.replace("\\", "\\u005C").replace("`", "``") + "`"
