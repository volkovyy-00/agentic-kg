"""prepare_refinement_loop_invocation must reset 'feedback' exactly once per
schema_refinement_loop invocation (never mid-loop), and must enforce at most
one such invocation per user turn.

Regression test for a bug found via live smoke-testing on 2026-07-29: the
reset used to live on schema_proposal_agent's own before_agent_callback,
which ADK's LoopAgent re-invokes on every iteration -- so the loop's retry
round always saw an empty <feedback> block and silently re-derived the
schema from scratch instead of acting on the critic's actual objections.
See docs/superpowers/specs/2026-07-29-schema-refinement-loop-latency-design.md.
"""

from types import SimpleNamespace

from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    prepare_refinement_loop_invocation,
    refinement_loop,
    reset_schema_refinement_turn_budget,
    root_agent,
    schema_proposal_agent,
)


def _ctx(state):
    return SimpleNamespace(state=state)


def test_first_invocation_this_turn_resets_feedback_and_proceeds():
    state = {
        "schema_refinement_calls_this_turn": 0,
        "feedback": "stale text from a previous turn",
    }
    result = prepare_refinement_loop_invocation(_ctx(state))
    assert result is None
    assert state["feedback"] == ""
    assert state["schema_refinement_calls_this_turn"] == 1


def test_second_invocation_this_turn_short_circuits_without_touching_feedback():
    state = {
        "schema_refinement_calls_this_turn": 1,
        "feedback": "retry\n- bad join key",
    }
    result = prepare_refinement_loop_invocation(_ctx(state))
    assert result is not None
    # Untouched: the short-circuit message below quotes this value verbatim.
    assert state["feedback"] == "retry\n- bad join key"
    assert state["schema_refinement_calls_this_turn"] == 2


def test_short_circuit_message_quotes_feedback_and_cannot_be_misrouted():
    state = {
        "schema_refinement_calls_this_turn": 1,
        "feedback": "retry\n- bad join key",
    }
    result = prepare_refinement_loop_invocation(_ctx(state))
    text = "\n".join(p.text for p in result.parts if p.text)
    assert "retry\n- bad join key" in text
    # The coordinator routes on whether the tool result BEGINS WITH 'retry';
    # this message must not collide with that check.
    assert not text.strip().lower().startswith("retry")
    assert text.strip().lower().startswith("stopped:")


def test_schema_proposal_agent_no_longer_resets_feedback_itself():
    """The exact bug: schema_proposal_agent used to carry its own
    before_agent_callback, which LoopAgent fires on every iteration. It must
    carry none now -- the reset lives one level up, on refinement_loop,
    which fires only once per invocation."""
    assert schema_proposal_agent.before_agent_callback is None


def test_refinement_loop_carries_the_new_callback():
    assert refinement_loop.before_agent_callback is prepare_refinement_loop_invocation


def test_reset_schema_refinement_turn_budget_zeroes_the_counter():
    state = {"schema_refinement_calls_this_turn": 5}
    reset_schema_refinement_turn_budget(_ctx(state))
    assert state["schema_refinement_calls_this_turn"] == 0


def test_coordinator_carries_the_reset_callback():
    assert root_agent.before_agent_callback is reset_schema_refinement_turn_budget
