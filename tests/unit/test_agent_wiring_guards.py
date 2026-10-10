"""Wiring rules every agent in the tree must follow, checked on the built agents.

These used to be guarded only by prose in CLAUDE.md, which a session loses
when it never reads that paragraph or compacts it away (KG-55).

- No agent sets output_key. On google-adk 2.x, output_key also stores the
  text an agent writes alongside tool calls, accumulated over its run (KG-25),
  so a value code parses must be written by a callback instead, as
  record_critic_verdict does for the schema critic. There is no allowlist: a
  display-only use would be a deliberate edit here.
- Every agent carries the rejected-call cap, from agent_guard_callbacks:
  end_turn_past_cap FIRST among its before-model callbacks (when it answers,
  ADK skips the rest, which keeps a capped critic's stop text out of
  record_critic_verdict), mark_tool_outcome as an after-tool callback and
  mark_unknown_tool as an on-tool-error callback, each exactly once.
- The transfer guard -- what agent_guard_callbacks(gated=True) adds over
  gated=False -- comes as a set, in the helper's order, on each known gated
  agent exactly while its gated variant is selected, and on no other agent.
  Above all not on the coordinator: its transfer_to_agent is how the workflow
  advances. A fourth gated agent fails here until it is added to
  _gated_cases -- the deliberate edit that wiring a new gate should be.
- The critic keeps its own after-model callback, record_critic_verdict.
"""

from dataclasses import dataclass
from typing import Any

from agent_tree import all_llm_agents

from agentic_kg.common.agent_guards import agent_guard_callbacks
from agentic_kg.common.rejected_call_cap import (
    end_turn_past_cap,
    mark_tool_outcome,
    mark_unknown_tool,
)
from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent import (
    agent as construction,
)
from agentic_kg.coordinators.multi_agent.sub_agents.graphrag_agent import (
    agent as graphrag,
)
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    record_critic_verdict,
    schema_critic_agent,
)
from agentic_kg.coordinators.multi_agent.sub_agents.user_intent_agent import (
    agent as user_intent,
)

_SLOTS = (
    "before_model_callback",
    "after_model_callback",
    "before_tool_callback",
    "after_tool_callback",
    "on_tool_error_callback",
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


_CAP = {
    "before_model_callback": [end_turn_past_cap],
    "after_tool_callback": [mark_tool_outcome],
    "on_tool_error_callback": [mark_unknown_tool],
}
_CAP_IDS = {id(callback) for callbacks in _CAP.values() for callback in callbacks}

_WIRING_FIX = (
    "Fix: spread **agent_guard_callbacks(gated=...) into the Agent(...) call "
    "and wire none of its callbacks by hand; see this module's docstring, "
    ".claude/rules/rejected-call-cap.md and .claude/rules/handoff-gates.md."
)


def _transfer_guard() -> dict[str, list[Any]]:
    """What the gated helper adds over the cap, per slot."""
    return {
        slot: [cb for cb in _as_list(callbacks) if id(cb) not in _CAP_IDS]
        for slot, callbacks in agent_guard_callbacks(gated=True).items()
    }


def _guard_report(agent: Any) -> _GuardReport:
    expected = _transfer_guard()
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


def _cap_problems(agent: Any) -> list[str]:
    problems = []
    before_model = _as_list(agent.before_model_callback)
    if not before_model or before_model[0] is not end_turn_past_cap:
        problems.append(
            f"before_model_callback starts {_names(before_model[:1])}, expected end_turn_past_cap"
        )
    for slot in _SLOTS:
        callbacks = _as_list(getattr(agent, slot))
        for callback in [cb for callbacks in _CAP.values() for cb in callbacks]:
            count = _ids(callbacks).count(id(callback))
            expected = 1 if callback in _CAP.get(slot, []) else 0
            if count != expected:
                problems.append(
                    f"{slot} has {callback.__name__} {count} times, expected {expected}"
                )
    return problems


def test_the_helper_fills_only_slots_this_file_inspects():
    for gated in (False, True):
        assert set(agent_guard_callbacks(gated=gated)) <= set(_SLOTS)


def test_the_helper_gives_the_cap_first_with_or_without_the_transfer_guard():
    for gated in (False, True):
        wired = agent_guard_callbacks(gated=gated)
        assert wired["before_model_callback"][0] is end_turn_past_cap
        assert _as_list(wired["after_tool_callback"]) == [mark_tool_outcome]
        assert _as_list(wired["on_tool_error_callback"]) == [mark_unknown_tool]


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


def test_every_agent_carries_the_cap():
    wrong = [
        f"{agent.name}: " + "; ".join(problems)
        for agent in all_llm_agents()
        if (problems := _cap_problems(agent))
    ]
    assert not wrong, "\n".join([*wrong, _WIRING_FIX])


def test_every_agent_carries_the_whole_transfer_guard_or_none_of_it():
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
            "needs a deliberate new option in agent_guard_callbacks.",
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
            wrong.append(
                f"{agent.name}: ungated variant selected but carries the transfer guard"
            )
    assert not wrong, "\n".join([*wrong, _WIRING_FIX])


def test_every_agent_not_listed_as_gated_carries_no_transfer_guard():
    gated = {id(agent) for agent, is_gated in _gated_cases() if is_gated}
    wrong = [
        f"{agent.name}: carries the transfer guard"
        for agent in all_llm_agents()
        if id(agent) not in gated and _guard_report(agent).carries_any
    ]
    assert not wrong, "\n".join(
        [
            *wrong,
            "Fix: a new gated agent goes into _gated_cases() above (read "
            ".claude/rules/handoff-gates.md first); the coordinator never takes "
            "the transfer guard, because its transfer_to_agent is how the workflow "
            "advances.",
        ]
    )


def test_the_critic_keeps_its_own_verdict_recorder():
    assert id(schema_critic_agent) in {id(agent) for agent in all_llm_agents()}
    assert record_critic_verdict in _as_list(schema_critic_agent.after_model_callback)
