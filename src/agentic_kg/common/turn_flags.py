"""A turn-scoped confirmation flag: the key and its plumbing, shared by every gate.

A gate of this shape (construction exit, way back to the plan, retrieval exit,
partition disclosure) has a confirm tool that sets a flag, a before-agent
callback that clears it at the start of every turn, and a gated tool that
refuses unless it is set. Only the key and those three operations are shared
here. Each gate keeps its own hand-written confirm tool, gated tool, reset
function, docstrings and refusal text: ADK shows a tool's docstring to the
model as its description, and each gate's says something different.

set() and reset() write the literal True and False; tests assert `is False`.
"""

from dataclasses import dataclass

from agentic_kg.common.session_state import SessionState


@dataclass(frozen=True)
class TurnFlag:
    key: str

    def set(self, state: SessionState) -> None:
        state[self.key] = True

    def is_set(self, state: SessionState) -> bool:
        return bool(state.get(self.key))

    def reset(self, state: SessionState) -> None:
        state[self.key] = False
