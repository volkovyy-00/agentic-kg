"""The schema_refinement_loop's StopChecker must produce a non-empty summary.

The coordinator invokes the loop through ADK's AgentTool, which returns the
text of the *last* event of the wrapped agent's run — and StopChecker always
runs last. When its event carried no content, every schema_refinement_loop
call returned "" to the coordinator; observed in a live session, the
coordinator model read that as "the tool returned no results", told the user
the column statistics "could not be retrieved" (they had in fact succeeded
inside the loop), and fell back to a worse schema. The event must therefore
always carry the critic's verdict as text, and its escalate flag must still
route on the first word of the feedback exactly as before.
"""
import asyncio
from types import SimpleNamespace

import pytest

from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    CheckStatusAndEscalate,
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
def test_escalate_routes_on_first_word_only(feedback, should_escalate):
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


@pytest.mark.parametrize("feedback", [
    "retry\n- Component identifier is not unique",
    "Validation failed: bad join key",
    "valid",
])
def test_the_verdict_is_returned_verbatim(feedback):
    """The coordinator is instructed to re-run the loop when the result begins
    with 'retry'. Any preamble ahead of the verdict makes that test false for
    every result the loop can return, so a retry reads as a finished plan."""
    assert _event_text(_run(feedback)[0]) == feedback


def test_a_missing_verdict_reads_as_retry_not_as_valid():
    """should_stop is False in that case, so the text must not say otherwise --
    and it must send the coordinator to the plan, not into another blind run."""
    text = _event_text(_run("")[0])
    assert text.lower().startswith("retry")
    assert "get_proposed_construction_plan" in text
