"""What this package needs from session state: read with a default, and write.

ADK's State is a plain class, not a Mapping, and tests and the refinement
loop's stop-check hold a plain dict. Both parameters are positional-only (`/`)
so that dict's overloaded .get satisfies the protocol under pyright -- the
same reason construction_plan_tools.StateLike is written that way.
"""

from typing import Any, Protocol


class SessionState(Protocol):
    def get(self, key: str, default: Any = None, /) -> Any: ...

    def __setitem__(self, key: str, value: Any, /) -> None: ...
