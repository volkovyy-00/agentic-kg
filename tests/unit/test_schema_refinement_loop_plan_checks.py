"""The refinement loop's own verdict when a mechanical check finds problems.

Approval already refuses these plans; catching them here buys back the user
turn that the one-call-per-turn cap would otherwise cost. See
docs/superpowers/specs/2026-09-20-loop-side-plan-checks-design.md.
"""

import asyncio
import logging
from types import SimpleNamespace

import fsspec
import pytest

from agentic_kg.common.config import reset_settings
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    CRITIC_PREAMBLE,
    EMPTY_VERDICT_SUMMARY,
    PLAN_PROBLEM_HEADER,
    CheckStatusAndEscalate,
    _compose_feedback,
    _is_loop_authored,
    _normalized,
)

PROBLEM = "Plot: to node label 'Site' has no node construction in the plan."
OTHER_PROBLEM = "'plot_id' identifies rows in 'plots.csv' but no node carries it."


def test_the_composite_begins_with_the_retry_the_coordinator_routes_on():
    """The coordinator branches on the result *beginning* with 'retry'
    (agent.py's instruction), so no preamble may precede it -- and the word is
    written here rather than inherited, because a critic verdict that routes as
    a retry ('Validation failed: ...') need not begin with it."""
    composite = _compose_feedback("Validation failed: bad join key", [PROBLEM])
    assert composite.startswith("retry")


def test_problem_strings_appear_verbatim_as_bullets():
    """Approval renders the same strings the same way. Reusing them verbatim is
    what makes loop/approval parity true by construction rather than by
    wording discipline."""
    composite = _compose_feedback("valid", [PROBLEM, OTHER_PROBLEM])
    assert f"- {PROBLEM}" in composite
    assert f"- {OTHER_PROBLEM}" in composite


def test_a_bare_valid_is_dropped_but_its_warnings_block_survives():
    """A bare 'valid' under a retry header reads as a contradiction. The
    Warnings block is data-quality notes meant for the user, and would be lost
    altogether if the loop's iterations ran out."""
    composite = _compose_feedback(
        "valid\nWarnings:\n- partial join coverage", [PROBLEM]
    )
    assert "Warnings:\n- partial join coverage" in composite
    assert "valid" not in composite.replace(PROBLEM, "")


def test_a_critic_retry_survives_verbatim():
    """The whole reason problems are added alongside the verdict rather than
    replacing it: on a second retry the coordinator must still show the
    critic's own objections."""
    verdict = "retry\n- Component identifier is not unique"
    composite = _compose_feedback(verdict, [PROBLEM])
    assert "- Component identifier is not unique" in composite


def test_an_empty_verdict_gives_header_and_bullets_only():
    """The empty-verdict fallback tells the coordinator to judge the plan
    itself rather than re-run the loop on no feedback. With problems in hand
    there IS feedback to act on, so that sentence would contradict the repair
    request."""
    composite = _compose_feedback("", [PROBLEM])
    assert composite == f"{PLAN_PROBLEM_HEADER}\n- {PROBLEM}"
    assert "judge the plan yourself" not in composite


def test_loop_authored_text_is_recognised_and_never_recomposed():
    """ADK writes output_key only when the critic's final response has a text
    part, so a critic that ends a round silently leaves the previous round's
    composite in the slot. Recognising it is what makes nesting impossible."""
    composite = _compose_feedback("valid", [PROBLEM])
    assert _is_loop_authored(composite)
    assert not _is_loop_authored("retry\n- a genuine critic objection")
    assert not _is_loop_authored("valid")
    assert not _is_loop_authored("")


def test_the_loop_authored_wording_carries_no_approval_framing():
    """PR #20's rule: approval guidance has no business in the critic's
    context, and this text is read by the critic and proposal steps on the next
    iteration. Asserted against the loop's OWN wording with a neutral problem
    string -- a substring scan over rendered output would fail on a column
    legitimately named 'approved_by'."""
    neutral = "Widget: to node label 'Gadget' has no node construction in the plan."
    authored = _compose_feedback("", [neutral]).replace(neutral, "")
    assert "approv" not in authored.lower()
    assert "ready" not in authored.lower()
    # EMPTY_VERDICT_SUMMARY is deliberately NOT asserted here: it names
    # get_proposed_construction_plan_with_approval_check, which contains
    # "approv". That is correct -- it is emitted only with escalate=True, i.e.
    # as the loop's final result to the COORDINATOR, which is the one reader
    # PR #20's rule does not cover. Keep this assertion strict rather than
    # widening it to accommodate that string.


def test_one_normalisation_serves_the_router_and_the_drop_rule():
    """The router compares the first token and the drop rule the first line.
    Two copies of the same rule would drift the moment one is reworded."""
    assert _normalized("Valid.") == "valid"
    assert _normalized("  valid:  ") == "valid"
    assert _normalized("retry,") == "retry"


@pytest.fixture
def stranding_plan_state(monkeypatch):
    """A plan whose node key leaves readings.csv's plot_id reference
    unreachable -- the same shape test_construction_plan_tools.py pins for the
    approval path, so the two can be compared directly."""
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    with fs.open("/src/plots.csv", "w") as handle:
        handle.write("plot_label,plot_id\nridge,PL-1\nridge,PL-2\nhollow,PL-3\n")
    with fs.open("/src/readings.csv", "w") as handle:
        handle.write("reading_id,plot_id\nR-1,PL-1\nR-2,PL-3\n")
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield {
        "proposed_construction_plan": {
            "Plot": {
                "construction_type": "node",
                "source_file": "plots.csv",
                "label": "Plot",
                "unique_column_name": "plot_label",
                "properties": [],
            }
        },
        "approved_file_list": ["plots.csv", "readings.csv"],
    }
    fs.store.clear()
    fs.pseudo_dirs.clear()


def _run(state):
    checker = CheckStatusAndEscalate(name="StopChecker")
    ctx = SimpleNamespace(session=SimpleNamespace(state=state))

    async def collect():
        return [event async for event in checker._run_async_impl(ctx)]

    return asyncio.run(collect())


def _text(event):
    if not event.content or not event.content.parts:
        return ""
    return "\n".join(part.text for part in event.content.parts if part.text)


def test_problems_are_published_as_state_delta_not_by_mutation(stranding_plan_state):
    """AgentTool runs the loop in a fresh child session and forwards ONLY
    event.actions.state_delta to the parent. A mutating implementation would
    still feed the next iteration's {feedback}, so it looks correct -- but the
    second loop call in the same turn would quote the critic's stale 'valid'.
    Asserting on the state dict would pass that bug; this asserts on the delta."""
    stranding_plan_state["feedback"] = "valid"
    events = _run(stranding_plan_state)

    delta = events[0].actions.state_delta
    assert "plot_id" in delta["feedback"]
    assert delta["feedback"].startswith(PLAN_PROBLEM_HEADER)


def test_problems_do_not_escalate_whatever_the_verdict(stranding_plan_state):
    """The stop-check never needs to know whether an iteration remains: if one
    does it runs, and if none does LoopAgent ends anyway with this event last.
    One behaviour serves both."""
    for verdict in ("valid", "retry\n- something else", ""):
        stranding_plan_state["feedback"] = verdict
        events = _run(stranding_plan_state)
        assert events[0].actions.escalate is False
        assert _text(events[0]).startswith("retry")


def test_a_clean_plan_passes_the_verdict_through_untouched(stranding_plan_state):
    """Nothing found must change nothing -- including emitting no delta, so an
    empty verdict still leaves the slot as it was."""
    stranding_plan_state["proposed_construction_plan"]["Plot"]["unique_column_name"] = (
        "plot_id"
    )
    stranding_plan_state["feedback"] = "valid\nWarnings:\n- partial join coverage"
    events = _run(stranding_plan_state)

    assert _text(events[0]) == "valid\nWarnings:\n- partial join coverage"
    assert events[0].actions.escalate is True
    assert not events[0].actions.state_delta


def test_an_absent_plan_passes_through(stranding_plan_state):
    """The empty-plan rule: no plan is not a plan with every column stranded."""
    del stranding_plan_state["proposed_construction_plan"]
    stranding_plan_state["feedback"] = "valid"
    events = _run(stranding_plan_state)

    assert _text(events[0]) == "valid"
    assert not events[0].actions.state_delta


def test_unverified_only_findings_pass_the_verdict_through(stranding_plan_state):
    """AC4's other half. The reachability check fails open: a source it cannot
    read yields a note, never a problem. An implementation that treated notes
    as problems would pass every other test here."""
    stranding_plan_state["approved_file_list"] = ["plots.csv", "absent.csv"]
    stranding_plan_state["feedback"] = "valid"
    events = _run(stranding_plan_state)

    assert _text(events[0]) == "valid"
    assert events[0].actions.escalate is True
    assert not events[0].actions.state_delta


def test_a_stale_composite_is_not_recomposed(stranding_plan_state):
    """Built from the composer's own output, never a copied literal: a
    hand-written header would stay green if the real one were reworded and the
    detector thereby broken."""
    stale = _compose_feedback("valid", ["an earlier problem"])
    stranding_plan_state["feedback"] = stale
    events = _run(stranding_plan_state)

    composite = events[0].actions.state_delta["feedback"]
    assert composite.count(PLAN_PROBLEM_HEADER) == 1
    assert "an earlier problem" not in composite
    assert CRITIC_PREAMBLE not in composite


def test_a_repaired_plan_clears_the_stale_composite(stranding_plan_state):
    """The one pass-through path that emits a delta, and it is not optional:
    round 1's composite already reached the PARENT session via its own delta,
    so emitting nothing here would leave it for prepare_refinement_loop_invocation
    to quote in its 'stopped:' message -- stale mechanical problems for a plan
    that no longer has them."""
    stranding_plan_state["proposed_construction_plan"]["Plot"]["unique_column_name"] = (
        "plot_id"
    )
    stranding_plan_state["feedback"] = _compose_feedback(
        "valid", ["an earlier problem"]
    )
    events = _run(stranding_plan_state)

    assert events[0].actions.state_delta == {"feedback": ""}
    assert _text(events[0]) == EMPTY_VERDICT_SUMMARY
    assert events[0].actions.escalate is True


def test_a_crashing_check_leaves_the_verdict_alone(
    monkeypatch, caplog, stranding_plan_state
):
    """Fail-open in the loop, because approval still refuses the plan. A raise
    here would instead abort the loop mid-turn: a dead turn with no response
    and no spinner, to protect an optimisation approval backstops."""
    # NOT `import ...schema_proposal_agent.agent as module`: sub_agents/__init__.py
    # rebinds the name `schema_proposal_agent` to the LlmAgent, so that form
    # raises ImportError.
    from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent import (
        agent as module,
    )

    def boom(*args, **kwargs):
        raise PermissionError("source unreadable")

    monkeypatch.setattr(module, "find_plan_problems", boom)
    stranding_plan_state["feedback"] = "valid"
    with caplog.at_level(logging.WARNING):
        events = _run(stranding_plan_state)

    assert _text(events[0]) == "valid"
    assert events[0].actions.escalate is True
    assert not events[0].actions.state_delta
    assert "PermissionError" in caplog.text


def test_a_real_adk_state_object_works(stranding_plan_state):
    """The tests above use dict fakes; StateLike exists for ADK's State, which
    is not a Mapping. Exercise it against the type it was written for."""
    from google.adk.sessions.state import State

    state = State(value=dict(stranding_plan_state), delta={})
    state["feedback"] = "valid"
    events = _run(state)

    assert "plot_id" in events[0].actions.state_delta["feedback"]


def test_the_cleared_slot_is_not_quoted_by_the_second_loop_call(stranding_plan_state):
    """End-to-end on the clearing path: apply the delta the stop-check emitted,
    then run the turn-cap callback that short-circuits a second loop call in
    the same turn. Its 'stopped:' message quotes whatever is in the slot, so a
    slot left holding the old composite would hand the coordinator mechanical
    problems for a plan that has since been repaired."""
    from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
        prepare_refinement_loop_invocation,
    )

    stranding_plan_state["proposed_construction_plan"]["Plot"][
        "unique_column_name"
    ] = "plot_id"
    stranding_plan_state["feedback"] = _compose_feedback("valid", ["an earlier problem"])

    events = _run(stranding_plan_state)
    stranding_plan_state.update(events[0].actions.state_delta)

    # calls == 1 already spent this turn, so the next invocation short-circuits
    stranding_plan_state["schema_refinement_calls_this_turn"] = 1
    callback_context = SimpleNamespace(state=stranding_plan_state)
    content = prepare_refinement_loop_invocation(callback_context)

    message = content.parts[0].text
    assert message.startswith("stopped:")
    assert PLAN_PROBLEM_HEADER not in message
    assert "an earlier problem" not in message
