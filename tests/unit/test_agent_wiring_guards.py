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
  selected, as its module's IS_GATED_VARIANT says.

LIMIT: a fourth gated agent that nobody adds to the last test passes as
"none". CLAUDE.md's pointer to .claude/rules/handoff-gates.md and that rule
guard that case, not this file.
"""

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
    return [callback.__name__ for callback in callbacks]


def _guarding(agent: Any) -> str:
    """'full', 'none', or which slots hold the wrong part of the set."""
    expected = {
        slot: _as_list(callbacks)
        for slot, callbacks in transfer_guard_callbacks(gated=True).items()
    }
    guard_ids = set(_ids([cb for callbacks in expected.values() for cb in callbacks]))
    present = {
        slot: [cb for cb in _as_list(getattr(agent, slot)) if id(cb) in guard_ids]
        for slot in _SLOTS
    }
    if not any(present.values()):
        return "none"
    wrong = [
        slot for slot in _SLOTS if _ids(present[slot]) != _ids(expected.get(slot, []))
    ]
    if not wrong:
        return "full"
    return "partial: " + "; ".join(
        f"{slot} has {_names(present[slot])}, expected {_names(expected.get(slot, []))}"
        for slot in wrong
    )


def test_the_helper_fills_only_slots_this_file_inspects():
    """Guards _guarding: a fifth slot would go unchecked."""
    assert set(transfer_guard_callbacks(gated=True)) <= set(_SLOTS)


def test_the_turn_end_runs_first_among_the_model_callbacks():
    """The agents are compared against the helper, so the helper's own order
    is pinned here: end_turn_past_hidden_transfer_cap answers in place of the
    model call, and ADK then skips the callbacks after it."""
    first = transfer_guard_callbacks(gated=True)["before_model_callback"][0]
    assert first is end_turn_past_hidden_transfer_cap


def test_no_agent_sets_output_key():
    offenders = [
        f"{agent.name}: output_key={agent.output_key!r}"
        for agent in all_llm_agents()
        if agent.output_key is not None
    ]
    assert not offenders, "\n".join(offenders)


def test_every_agent_carries_the_whole_guard_set_or_none_of_it():
    partial = [
        f"{agent.name}: {state}"
        for agent in all_llm_agents()
        if (state := _guarding(agent)) not in ("full", "none")
    ]
    assert not partial, "\n".join(partial)


def test_each_gated_agent_is_guarded_exactly_while_its_gated_variant_is_selected():
    cases = [
        (construction.graph_construction_agent, True),
        (graphrag.graphrag_agent, graphrag.IS_GATED_VARIANT),
        (user_intent.user_intent_agent, user_intent.IS_GATED_VARIANT),
    ]
    in_tree = {id(agent) for agent in all_llm_agents()}
    wrong = []
    for agent, gated in cases:
        if id(agent) not in in_tree:
            wrong.append(f"{agent.name}: not reached by the agent-tree walk")
        expected = "full" if gated else "none"
        if (state := _guarding(agent)) != expected:
            wrong.append(f"{agent.name}: {state}, expected {expected}")
    assert not wrong, "\n".join(wrong)
