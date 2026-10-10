"""The rejected-call cap's callbacks, driven the way ADK drives them.

Each reply is simulated in ADK's order: the cap's before-model check, then
each call's hook -- on-tool-error for an unknown name (ADK skips the
after-tool callbacks there), after-tool for anything that ran or was
answered, and mark_stuck from the hidden-transfer refusal. The real ADK
flow is in test_rejected_call_cap_flow.py.
"""

import inspect
from types import SimpleNamespace

import pytest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool

from agentic_kg.common.rejected_call_cap import (
    HIDDEN_TRANSFER_TURN_END,
    MAX_CONSECUTIVE_STUCK_REPLIES,
    MAX_STUCK_REPLIES_PER_TURN,
    STUCK_TURN_END,
    StuckKind,
    _key,
    end_turn_past_cap,
    is_adk_rejection,
    mark_stuck,
    mark_tool_outcome,
    mark_unknown_tool,
    was_stopped,
)
from agentic_kg.common.tool_result import tool_error, tool_success


def ping() -> dict:
    """Answers pong."""
    return tool_success("pong", True)


_REAL_TOOL = FunctionTool(ping)
# Built the way ADK builds it for a name that did not resolve
# (flows/llm_flows/tools/_caller.py, _prepare_single).
_STAND_IN = BaseTool(name="no_such_tool", description="Tool not found")
_REJECTION = dict(
    error="Invoking `ping()` failed as the following mandatory input "
    "parameters are not present:\nvalue"
)
_PLAN_WITH_AN_ERROR_LABEL = {"error": {"construction_type": "node", "label": "error"}}


def _context(state=None, agent="probe_agent"):
    """Stands in for a CallbackContext and a ToolContext: the cap reads only
    agent_name and state."""
    return SimpleNamespace(agent_name=agent, state={} if state is None else state)


# What a real tool answered, for outcomes that reach only the after-tool marker.
_TOOL_RESPONSES = {
    "rejected": _REJECTION,
    "success": tool_success("pong", True),
    "own_error": tool_error("the plan has a problem"),
    # The coordinator's real transfer_to_agent returns None.
    "transfer_none": None,
    "plan_read": _PLAN_WITH_AN_ERROR_LABEL,
}


def _answer(context, outcome):
    if outcome == "unknown":
        mark_unknown_tool(
            tool=_STAND_IN, args={}, tool_context=context, error=ValueError("x")
        )
    elif outcome == "transfer":
        # refuse_transfer_to_agent marks, then answers with a tool_error,
        # which the after-tool marker also sees.
        mark_stuck(context, StuckKind.HIDDEN_TRANSFER)
        mark_tool_outcome(
            tool=_STAND_IN,
            args={},
            tool_context=context,
            tool_response=tool_error("refused"),
        )
    else:
        mark_tool_outcome(
            tool=_REAL_TOOL,
            args={},
            tool_context=context,
            tool_response=_TOOL_RESPONSES[outcome],
        )


def _run(context, replies):
    """Each reply is a tuple of call outcomes; () is a text-only reply.
    Returns the first stop, or what the check after the last reply returned."""
    for reply in replies:
        ended = end_turn_past_cap(callback_context=context, llm_request=None)
        if ended is not None:
            return ended
        for outcome in reply:
            _answer(context, outcome)
    return end_turn_past_cap(callback_context=context, llm_request=None)


def _text_of(response):
    return response.content.parts[0].text


@pytest.mark.parametrize("kind", ["unknown", "rejected", "transfer"])
def test_three_stuck_replies_in_a_row_stop_the_turn(kind):
    context = _context()
    assert _run(context, [(kind,)] * (MAX_CONSECUTIVE_STUCK_REPLIES - 1)) is None
    assert not was_stopped(context.state, context.agent_name)

    ended = _run(context, [(kind,)])

    assert ended is not None
    assert not ended.get_function_calls()
    assert was_stopped(context.state, context.agent_name)


def test_alternating_stuck_and_good_replies_stop_at_the_sixth_stuck_reply():
    context = _context()
    pairs = [("unknown",), ("success",)] * (MAX_STUCK_REPLIES_PER_TURN - 1)
    assert _run(context, pairs) is None
    assert _run(context, [("unknown",)]) is not None


def test_a_mix_of_kinds_counts_toward_one_limit():
    ended = _run(_context(), [("unknown",), ("rejected",), ("transfer",)])
    assert _text_of(ended) == STUCK_TURN_END


def test_hand_over_wording_only_when_every_stuck_reply_was_a_hidden_transfer():
    only_transfers = _run(_context(), [("transfer",)] * MAX_CONSECUTIVE_STUCK_REPLIES)
    assert _text_of(only_transfers) == HIDDEN_TRANSFER_TURN_END

    mixed = _run(_context(), [("transfer",), ("transfer",), ("unknown",)])
    assert _text_of(mixed) == STUCK_TURN_END


def test_a_reply_counts_once_however_many_bad_calls_it_holds():
    context = _context()
    crowded = ("unknown", "unknown", "rejected", "transfer")
    assert _run(context, [crowded] * (MAX_CONSECUTIVE_STUCK_REPLIES - 1)) is None
    assert context.state[_key("total", context.agent_name)] == (
        MAX_CONSECUTIVE_STUCK_REPLIES - 1
    )


@pytest.mark.parametrize(
    "mixed",
    [("unknown", "success"), ("success", "unknown")],
    ids=["bad-first", "good-first"],
)
def test_a_mixed_reply_resets_in_a_row_but_counts_for_the_turn(mixed):
    context = _context()
    replies = [("unknown",), ("unknown",), mixed, ("unknown",), ("unknown",)]
    assert _run(context, replies) is None
    assert context.state[_key("consecutive", context.agent_name)] == 2
    assert context.state[_key("total", context.agent_name)] == 5

    assert _run(_context(), [mixed] * MAX_STUCK_REPLIES_PER_TURN) is not None


def test_one_slip_then_real_work_keeps_the_turn():
    replies = [("unknown",), ("success",)] * 2
    assert _run(_context(), replies) is None


def test_a_tools_own_error_neither_counts_nor_resets():
    """Stuck, own error, stuck, own error, stuck stops at the third stuck
    reply: in a row means no success between them, not adjacent replies."""
    replies = [("unknown",), ("own_error",), ("unknown",), ("own_error",), ("unknown",)]
    assert _run(_context(), replies) is not None


def test_own_errors_alone_never_stop_the_turn():
    assert _run(_context(), [("own_error",)] * 20) is None


def test_a_text_reply_between_retries_does_not_reset_the_count():
    """A streamed reply can arrive as a text-only response followed by the
    call-only one; text is not progress."""
    replies = [(), ("unknown",)] * MAX_CONSECUTIVE_STUCK_REPLIES
    assert _run(_context(), replies) is not None


def test_the_real_transfer_is_progress_not_stuck():
    """The coordinator's transfer_to_agent returns None: progress."""
    context = _context()
    replies = [
        ("unknown",),
        ("unknown",),
        ("transfer_none",),
        ("unknown",),
        ("unknown",),
    ]
    assert _run(context, replies) is None


def test_a_plan_with_an_error_label_is_progress():
    """get_proposed_construction_plan returns the plan keyed by label, and
    'error' is a valid label; its value is a rule dict, never a string."""
    context = _context()
    replies = [("unknown",), ("unknown",), ("plan_read",), ("unknown",), ("unknown",)]
    assert _run(context, replies) is None


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (_REJECTION, True),
        (_PLAN_WITH_AN_ERROR_LABEL, False),
        (dict(error="x", status="error"), False),
        (dict(error="x", detail="y"), False),
        (tool_error("x"), False),
        (tool_success("pong", True), False),
        ({}, False),
        (None, False),
        ("error", False),
    ],
)
def test_adk_rejection_is_a_single_string_valued_error_key(response, expected):
    assert is_adk_rejection(response) is expected


def test_a_real_tool_that_raises_is_not_an_unknown_name():
    """ADK runs the on-tool-error callbacks for a real tool that raised too.
    Only ADK's stand-in is a bare BaseTool."""
    context = _context()
    for _ in range(MAX_CONSECUTIVE_STUCK_REPLIES + 1):
        assert (
            mark_unknown_tool(
                tool=_REAL_TOOL, args={}, tool_context=context, error=ValueError("x")
            )
            is None
        )
        assert end_turn_past_cap(callback_context=context, llm_request=None) is None


def test_the_marks_never_answer_the_call():
    context = _context()
    assert (
        mark_unknown_tool(
            tool=_STAND_IN, args={}, tool_context=context, error=ValueError("x")
        )
        is None
    )
    assert (
        mark_tool_outcome(
            tool=_REAL_TOOL, args={}, tool_context=context, tool_response=_REJECTION
        )
        is None
    )
    assert (
        mark_tool_outcome(
            tool=_REAL_TOOL, args={}, tool_context=context, tool_response=None
        )
        is None
    )


def test_each_agent_keeps_its_own_count():
    state = {}
    first = _context(state, agent="first_agent")
    _run(first, [("unknown",)] * (MAX_CONSECUTIVE_STUCK_REPLIES - 1))

    second = _context(state, agent="second_agent")
    assert _run(second, [("unknown",)]) is None
    assert not was_stopped(state, "second_agent")


def test_everything_lives_in_invocation_scoped_state():
    """temp: keys are dropped from the persisted delta, so each turn (and
    each AgentTool run) starts at zero with no reset callback."""
    context = _context()
    _run(context, [("unknown",), ("success",), ("rejected",)])
    assert context.state
    assert all(key.startswith("temp:") for key in context.state)


def test_callback_parameter_names_are_the_ones_adk_passes():
    """ADK invokes every callback by keyword (flows/llm_flows/tools/_caller.py,
    flows/llm_flows/core/_finalizer.py)."""

    def names(callback):
        return list(inspect.signature(callback).parameters)

    assert names(end_turn_past_cap) == ["callback_context", "llm_request"]
    assert names(mark_tool_outcome) == ["tool", "args", "tool_context", "tool_response"]
    assert names(mark_unknown_tool) == ["tool", "args", "tool_context", "error"]
