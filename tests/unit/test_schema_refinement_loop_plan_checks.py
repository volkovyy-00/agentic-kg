"""The refinement loop's own verdict when a mechanical check finds problems.

Approval already refuses these plans; catching them here buys back the user
turn that the one-call-per-turn cap would otherwise cost. See
docs/superpowers/specs/2026-09-20-loop-side-plan-checks-design.md.
"""

from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    PLAN_PROBLEM_HEADER,
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
