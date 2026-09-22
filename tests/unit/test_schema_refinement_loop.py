"""The schema_refinement_loop's StopChecker must produce a non-empty summary.

The coordinator invokes the loop through ADK's AgentTool, which returns the
text of the *last* event of the wrapped agent's run — and StopChecker always
runs last. When its event carried no content, every schema_refinement_loop
call returned "" to the coordinator; observed in a live session, the
coordinator model read that as "the tool returned no results", told the user
the column statistics "could not be retrieved" (they had in fact succeeded
inside the loop), and fell back to a worse schema. The event must therefore
always carry the critic's verdict as text, and its escalate flag must route on
the first word of the feedback when no mechanical plan problem was found --
see tests/unit/test_schema_refinement_loop_plan_checks.py for the case where
one was.
"""

import asyncio
from types import SimpleNamespace

import pytest

from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    FEEDBACK_KIND_KEY,
    CheckStatusAndEscalate,
    VerdictKind,
)


def _run(feedback):
    checker = CheckStatusAndEscalate(name="StopChecker")
    ctx = SimpleNamespace(session=SimpleNamespace(state={"feedback": feedback}))

    async def collect():
        return [event async for event in checker._run_async_impl(ctx)]

    return asyncio.run(collect())


def _event_text(event):
    if not event.content or not event.content.parts:
        return ""
    return "\n".join(p.text for p in event.content.parts if p.text)


@pytest.mark.parametrize(
    "feedback,should_escalate",
    [
        ("valid", True),
        ("valid\nWarnings:\n- partial join coverage", True),
        ("Valid.", True),
        ("retry\n- Component identifier is not unique", False),
        ("Validation failed: bad join key", False),
    ],
)
def test_escalate_routes_on_the_verdicts_first_word_when_no_problems_are_found(
    feedback, should_escalate
):
    """The verdict's first word decides the route only when the mechanical
    checks found nothing. A plan problem overrides it -- see
    tests/unit/test_schema_refinement_loop_plan_checks.py. These cases carry no
    plan in state, so the checks return nothing and this is the pure router."""
    events = _run(feedback)
    assert len(events) == 1
    assert events[0].actions.escalate is should_escalate


def test_event_carries_the_critic_verdict_as_text():
    """AgentTool returns this event's text; empty content meant an empty
    tool result, which coordinator models treated as a tool failure."""
    feedback = "retry\n- join column collapses"
    events = _run(feedback)
    text = _event_text(events[0])
    assert text, "StopChecker event must not be contentless"
    assert feedback in text


@pytest.mark.parametrize(
    "feedback",
    [
        "retry\n- Component identifier is not unique",
        "Validation failed: bad join key",
        "valid",
    ],
)
def test_the_verdict_is_returned_verbatim(feedback):
    """The coordinator is instructed to re-run the loop when the result begins
    with 'retry'. Any preamble ahead of the verdict makes that test false for
    every result the loop can return, so a retry reads as a finished plan."""
    assert _event_text(_run(feedback)[0]) == feedback


def test_a_missing_verdict_is_reported_as_no_verdict():
    """The text must not say valid, and must send the coordinator to the plan
    rather than inventing one from memory. Nor may it say retry (KG-29): the
    coordinator re-runs the loop on a result beginning 'retry', so a plan that
    passed both checks would go back into the loop this same text tells it not
    to re-run."""
    text = _event_text(_run("")[0])
    assert text.lower().startswith("no verdict:")
    assert not text.lower().startswith(("retry", "valid"))
    assert "get_proposed_construction_plan_with_approval_check" in text


def test_an_absent_slot_is_no_verdict_not_valid():
    """The resets always set the key, but an absent one must still read as no
    verdict rather than manufacture a 'valid' (KG-29 review)."""
    checker = CheckStatusAndEscalate(name="StopChecker")
    ctx = SimpleNamespace(session=SimpleNamespace(state={}))

    async def collect():
        return [event async for event in checker._run_async_impl(ctx)]

    event = asyncio.run(collect())[0]
    assert _event_text(event).startswith("no verdict:")
    assert event.actions.state_delta == {FEEDBACK_KIND_KEY: VerdictKind.NONE.value}


def test_a_missing_verdict_still_escalates():
    """A missing verdict must stop the *internal* loop immediately, not spend
    another schema_proposal_agent/schema_critic_agent iteration first: the
    returned text already tells the coordinator to inspect the plan itself
    rather than wait for the loop to try again on the same no-answer input.
    escalate=False here would keep looping despite the text saying not to."""
    events = _run("")
    assert events[0].actions.escalate is True


@pytest.mark.parametrize(
    "feedback,kind",
    [
        ("valid", VerdictKind.CRITIC),
        ("valid\nWarnings:\n- partial join coverage", VerdictKind.CRITIC),
        ("retry\n- Component identifier is not unique", VerdictKind.CRITIC),
        ("", VerdictKind.NONE),
    ],
)
def test_a_pass_through_tags_the_slot_by_whether_the_critic_spoke(feedback, kind):
    """KG-30: with no mechanical problem, the kind is the critic's whenever it
    said anything, and none when it said nothing -- chosen from the branch,
    written as a delta so it leaves the loop's child session."""
    delta = _run(feedback)[0].actions.state_delta
    assert delta[FEEDBACK_KIND_KEY] == kind.value
