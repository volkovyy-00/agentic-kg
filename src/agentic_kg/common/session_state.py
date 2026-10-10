"""What this package needs from session state: read with a default, and write.

ADK's State is a plain class exposing .get(key, default=None) -- not a
Mapping -- while tests and the refinement loop's stop-check hold a plain dict.
The parameters are positional-only (`/`) so that dict's own overloaded .get
satisfies these protocols: without the slash, pyright rejects dict[str, Any]
with "No overloaded function matches type (key: str, default: Any = None) ->
Any" and the CI gate fails at the stop-check's call site.
"""

from typing import Any, Protocol


class SessionStateReader(Protocol):
    """Anything session state can be read from; the plan checks only read."""

    def get(self, key: str, default: Any = None, /) -> Any: ...


class SessionState(SessionStateReader, Protocol):
    def __setitem__(self, key: str, value: Any, /) -> None: ...
