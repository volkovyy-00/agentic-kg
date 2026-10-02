"""Wiring rules every agent in the tree must follow, checked on the built agents.

These used to be guarded only by prose in CLAUDE.md, which a session loses
when it never reads that paragraph or compacts it away (KG-55).

- No agent sets output_key. On google-adk 2.x, output_key also stores the
  text an agent writes alongside tool calls, accumulated over its run (KG-25),
  so a value code parses must be written by a callback instead, as
  record_critic_verdict does for the schema critic. There is no allowlist: a
  display-only use would be a deliberate edit here.
- The transfer-guard callbacks come as a set. transfer_guard_callbacks(gated=True)
  is the source of truth: an agent carries every one of them, each in its own
  slot and in the helper's order, once, or none. Order matters because
  end_turn_past_hidden_transfer_cap must run first among the model callbacks:
  when it answers, ADK skips the ones after it. Callbacks that are not guards
  are ignored, so a helper extended with an agent's own callback still passes.
- Each known gated agent carries the set exactly while its gated variant is
  selected, as its module's IS_GATED_VARIANT says, and every other agent
  carries none. Above all the coordinator: its transfer_to_agent is how the
  workflow advances, so stripping it would stop the workflow. A fourth gated
  agent therefore fails here until it is added to _gated_cases -- the
  deliberate edit that wiring a new gate should be.
"""

from dataclasses import dataclass
from typing import Any

from agent_tree import all_llm_agents

from agentic_kg.common.adk_transfer import (
    end_turn_past_hidden_transfer_cap,
    transfer_guard_callbacks,
)
from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent import (
    agent as construction,
)
from agentic_kg.coordinators.multi_agent.sub_agents.graphrag_agent import (
    agent as graphrag,
)
from agentic_kg.coordinators.multi_agent.sub_agents.user_intent_agent import (
    agent as user_intent,
)

_SLOTS = (
    "before_model_callback",
    "after_model_callback",
    "before_tool_callback",
    "after_tool_callback",
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _ids(callbacks: list[Any]) -> list[int]:
    """Identities in order, so a reordered or repeated guard does not match."""
    return [id(callback) for callback in callbacks]


def _names(callbacks: list[Any]) -> list[str]:
    return [getattr(callback, "__name__", repr(callback)) for callback in callbacks]


@dataclass(frozen=True)
class _GuardReport:
    """Whether an agent carries any guard callback, and which slots are wrong.

    Full set: carries_any and no wrong slots. None: not carries_any.
    """

    carries_any: bool
    wrong_slots: list[str]

    @property
    def full(self) -> bool:
        return self.carries_any and not self.wrong_slots


def _guard_report(agent: Any) -> _GuardReport:
    expected = {
        slot: _as_list(callbacks)
        for slot, callbacks in transfer_guard_callbacks(gated=True).items()
    }
    guard_ids = set(_ids([cb for callbacks in expected.values() for cb in callbacks]))
    present = {
        slot: [cb for cb in _as_list(getattr(agent, slot)) if id(cb) in guard_ids]
        for slot in _SLOTS
    }
    wrong = [
        f"{slot} has {_names(present[slot])}, expected {_names(expected.get(slot, []))}"
        for slot in _SLOTS
        if _ids(present[slot]) != _ids(expected.get(slot, []))
    ]
    return _GuardReport(carries_any=any(present.values()), wrong_slots=wrong)


_WIRING_FIX = (
    "Fix: spread **transfer_guard_callbacks(gated=...) into the Agent(...) call "
    "and wire none of its callbacks by hand; see this module's docstring and "
    ".claude/rules/handoff-gates.md."
)


def test_the_helper_fills_only_slots_this_file_inspects():
    """Guards _guard_report: a fifth slot would go unchecked."""
    assert set(transfer_guard_callbacks(gated=True)) <= set(_SLOTS)


def test_the_turn_end_runs_first_among_the_model_callbacks():
    """The agents are compared against the helper, so the helper's own order
    is pinned here: end_turn_past_hidden_transfer_cap answers in place of the
    model call, and ADK then skips the callbacks after it."""
    first = transfer_guard_callbacks(gated=True)["before_model_callback"][0]
    assert first is end_turn_past_hidden_transfer_cap, (
        "end_turn_past_hidden_transfer_cap must stay first in the helper's "
        "before_model_callback list; see its docstring."
    )


def test_no_agent_sets_output_key():
    offenders = [
        f"{agent.name}: output_key={agent.output_key!r}"
        for agent in all_llm_agents()
        if agent.output_key is not None
    ]
    assert not offenders, "\n".join(
        [
            *offenders,
            "Fix: write a value code reads from a callback (as record_critic_verdict "
            "does): on google-adk 2.x output_key also stores text written beside "
            "tool calls. See this module's docstring.",
        ]
    )


def test_every_agent_carries_the_whole_guard_set_or_none_of_it():
    partial = [
        f"{agent.name}: " + "; ".join(report.wrong_slots)
        for agent in all_llm_agents()
        if (report := _guard_report(agent)).carries_any and report.wrong_slots
    ]
    assert not partial, "\n".join(
        [
            *partial,
            _WIRING_FIX,
            "drop_foreign_context belongs to the set; an agent that needs it alone "
            "needs a deliberate new option in transfer_guard_callbacks.",
        ]
    )


def _gated_cases() -> list[tuple[Any, bool]]:
    """Each phase agent that can be gated, and whether its variant is gated now."""
    return [
        (construction.graph_construction_agent, True),
        (graphrag.graphrag_agent, graphrag.IS_GATED_VARIANT),
        (user_intent.user_intent_agent, user_intent.IS_GATED_VARIANT),
    ]


def test_each_gated_agent_is_guarded_exactly_while_its_gated_variant_is_selected():
    in_tree = {id(agent) for agent in all_llm_agents()}
    wrong = []
    for agent, gated in _gated_cases():
        if id(agent) not in in_tree:
            wrong.append(f"{agent.name}: not reached by the agent-tree walk")
        report = _guard_report(agent)
        if gated and not report.full:
            wrong.append(f"{agent.name}: gated variant selected but not fully guarded")
        if not gated and report.carries_any:
            wrong.append(f"{agent.name}: ungated variant selected but carries guards")
    assert not wrong, "\n".join([*wrong, _WIRING_FIX])


def test_every_agent_not_listed_as_gated_carries_no_guard():
    gated = {id(agent) for agent, is_gated in _gated_cases() if is_gated}
    wrong = [
        f"{agent.name}: carries guard callbacks"
        for agent in all_llm_agents()
        if id(agent) not in gated and _guard_report(agent).carries_any
    ]
    assert not wrong, "\n".join(
        [
            *wrong,
            "Fix: a new gated agent goes into _gated_cases() above (read "
            ".claude/rules/handoff-gates.md first); the coordinator never takes "
            "the set, because its transfer_to_agent is how the workflow advances.",
        ]
    )
