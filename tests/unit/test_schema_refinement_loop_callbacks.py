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

import pytest

from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    FEEDBACK_KIND_KEY,
    VerdictKind,
    clear_verdict_before_critic,
    prepare_refinement_loop_invocation,
    refinement_loop,
    reset_schema_refinement_turn_budget,
    root_agent,
    schema_critic_agent,
    schema_proposal_agent,
)


def _ctx(state):
    return SimpleNamespace(state=state)


def test_first_invocation_this_turn_resets_feedback_and_proceeds():
    state = {
        "schema_refinement_calls_this_turn": 0,
        "feedback": "stale text from a previous turn",
        "feedback_kind": "mechanical",
    }
    result = prepare_refinement_loop_invocation(_ctx(state))
    assert result is None
    assert state["feedback"] == ""
    assert state[FEEDBACK_KIND_KEY] == VerdictKind.NONE.value
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


def _stopped(feedback, kind=None):
    state = {"schema_refinement_calls_this_turn": 1, "feedback": feedback}
    if kind is not None:
        state[FEEDBACK_KIND_KEY] = kind
    result = prepare_refinement_loop_invocation(_ctx(state))
    return "\n".join(p.text for p in result.parts if p.text)


@pytest.mark.parametrize(
    "feedback,kind",
    [
        ("retry: checks found\n- bad join key", VerdictKind.MECHANICAL.value),
        ("retry\n- bad join key", VerdictKind.CRITIC.value),
        ("", VerdictKind.NONE.value),
        ("", None),
    ],
)
def test_short_circuit_message_cannot_be_misrouted(feedback, kind):
    # The coordinator routes on whether the tool result BEGINS WITH 'retry';
    # this message must not collide with that check.
    text = _stopped(feedback, kind)
    assert not text.strip().lower().startswith("retry")
    assert text.strip().lower().startswith("stopped:")


def test_a_mechanical_verdict_is_quoted_as_one_approval_refuses():
    """KG-30 AC1/AC2: a mechanical finding is named as such, never offered as
    something the user may approve as it stands."""
    text = _stopped("retry: checks found\n- bad join key", VerdictKind.MECHANICAL.value)
    assert "retry: checks found\n- bad join key" in text
    assert "mechanical check finding" in text
    assert "approval will refuse" in text
    assert "let the user decide" not in text


def test_a_critic_verdict_is_quoted_as_the_critics_opinion():
    """KG-30 AC4: a critic objection stays the user's call, as before."""
    text = _stopped("retry\n- bad join key", VerdictKind.CRITIC.value)
    assert "retry\n- bad join key" in text
    assert "the critic's opinion" in text
    assert "let the user decide" in text


@pytest.mark.parametrize("kind", [VerdictKind.NONE.value, None])
def test_no_verdict_is_said_plainly_and_nothing_is_quoted(kind):
    """None has its own wording; an absent key (a session from before KG-30)
    reads as none. Whatever the slot holds, a none kind quotes nothing."""
    text = _stopped("leftover text", kind)
    assert "It recorded no verdict." in text
    assert "leftover text" not in text


def test_schema_proposal_agent_no_longer_resets_feedback_itself():
    """The exact bug: schema_proposal_agent used to carry its own
    before_agent_callback, which LoopAgent fires on every iteration. It must
    carry none now -- the reset lives one level up, on refinement_loop,
    which fires only once per invocation."""
    assert schema_proposal_agent.before_agent_callback is None


def test_the_critic_step_starts_every_round_with_an_empty_slot():
    """KG-29: ADK writes the critic's output_key only when its final response
    carries text, so a silent critic would otherwise inherit the previous
    round's verdict. Cleared here, a silent round is structurally no verdict."""
    state = {
        "feedback": "retry\n- the previous round's objection",
        FEEDBACK_KIND_KEY: VerdictKind.CRITIC.value,
    }
    assert clear_verdict_before_critic(_ctx(state)) is None
    assert state["feedback"] == ""
    assert state[FEEDBACK_KIND_KEY] == VerdictKind.NONE.value


def test_the_per_round_reset_sits_on_the_critic():
    """On the critic, which runs AFTER the proposal step has read {feedback}
    -- never on schema_proposal_agent, the 2026-07-29 regression above."""
    assert schema_critic_agent.before_agent_callback is clear_verdict_before_critic


def test_refinement_loop_carries_the_new_callback():
    assert refinement_loop.before_agent_callback is prepare_refinement_loop_invocation


def test_reset_schema_refinement_turn_budget_zeroes_the_counter():
    state = {"schema_refinement_calls_this_turn": 5}
    reset_schema_refinement_turn_budget(_ctx(state))
    assert state["schema_refinement_calls_this_turn"] == 0


def test_coordinator_carries_the_reset_callback():
    assert root_agent.before_agent_callback is reset_schema_refinement_turn_budget
