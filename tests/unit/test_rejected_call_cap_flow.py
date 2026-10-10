"""The rejected-call cap through ADK's real flow.

Probe agents built here carry the cap the way production agents do
(**agent_guard_callbacks); a scripted model above the LiteLlm adapter makes
the calls. These also pin the ADK 2.10 behaviour the cap relies on -- the
stand-in for an unknown name, the shape of the missing-parameters reply, a
real tool's exception re-raising -- so a google-adk upgrade that changes one
fails here.
"""

import asyncio
from collections import Counter

import pytest
from agent_tree import user_facing_llm_agents
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field

from agentic_kg.common.agent_guards import agent_guard_callbacks
from agentic_kg.common.rejected_call_cap import (
    HIDDEN_TRANSFER_TURN_END,
    MAX_CONSECUTIVE_STUCK_REPLIES,
    MAX_STUCK_REPLIES_PER_TURN,
    STUCK_TURN_END,
    is_adk_rejection,
)
from agentic_kg.common.tool_result import is_error, tool_error, tool_success
from agentic_kg.coordinators.multi_agent.agent import full_workflow_agent
from agentic_kg.coordinators.multi_agent.sub_agents.file_suggestion_agent.agent import (
    file_suggestion_agent,
)


class ScriptedLlm(BaseLlm):
    """Replays a fixed script, one response per call; holds on the last.

    Streaming (SSE), it first yields a partial text chunk, as a streamed reply
    arrives, then the complete reply."""

    responses: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        if stream:
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="…")]),
                partial=True,
            )
        yield self.responses[index]


def _text(text):
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def _calls(*calls):
    """One reply making these calls: each a name, or a (name, args) pair."""
    parts = []
    for call in calls:
        name, args = call if isinstance(call, tuple) else (call, {})
        parts.append(types.Part(function_call=types.FunctionCall(name=name, args=args)))
    return LlmResponse(content=types.Content(role="model", parts=parts))


_UNKNOWN = _calls("no_such_tool")
_MISSING = _calls("needs_value")
_GOOD = _calls("ping")
_REFUSED_BY_TOOL = _calls("refuse")
_HIDDEN = _calls(("transfer_to_agent", {"agent_name": "anyone"}))


def ping() -> dict:
    """Answers pong."""
    return tool_success("pong", True)


def needs_value(value: str) -> dict:
    """Echoes the value."""
    return tool_success("value", value)


def refuse() -> dict:
    """Always reports its own error."""
    return tool_error("not like that")


def explode() -> dict:
    """Always raises."""
    raise ValueError("boom")


def _probe(responses, gated=False):
    return LlmAgent(
        name="cap_probe_agent",
        mode="chat",
        model=ScriptedLlm(model="scripted", responses=responses),
        instruction="You are a probe.",
        tools=[ping, needs_value, refuse, explode],
        **agent_guard_callbacks(gated=gated),
    )


async def _run_turns(agent, app_name, messages=("hello",), run_config=None):
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name, user_id="u1"
    )
    turns = []
    for message in messages:
        turns.append(
            [
                event
                async for event in runner.run_async(
                    user_id="u1",
                    session_id=session.id,
                    new_message=types.Content(
                        role="user", parts=[types.Part(text=message)]
                    ),
                    run_config=run_config or RunConfig(),
                )
            ]
        )
    saved = await runner.session_service.get_session(
        app_name=app_name, user_id="u1", session_id=session.id
    )
    return turns, saved.events


def _final_text(events):
    parts = events[-1].content.parts if events[-1].content else []
    return "".join(part.text or "" for part in parts)


def _responses(events, name):
    return [
        part.function_response.response or {}
        for event in events
        for part in (event.content.parts if event.content else None) or []
        if part.function_response and part.function_response.name == name
    ]


def _unanswered_calls(events):
    """Calls with no matching response in the saved session (per id: a
    reused scripted reply gives every call it makes the same id)."""
    calls, answered = Counter(), Counter()
    for event in events:
        for part in (event.content.parts if event.content else None) or []:
            if part.function_call:
                calls[part.function_call.id] += 1
            if part.function_response:
                answered[part.function_response.id] += 1
    return calls - answered


def _one_turn(agent, app_name, run_config=None):
    (events,), saved = asyncio.run(_run_turns(agent, app_name, run_config=run_config))
    return events, saved


def test_unknown_names_are_capped_and_answered_by_adk():
    agent = _probe(
        [_UNKNOWN] * MAX_CONSECUTIVE_STUCK_REPLIES + [_text("never reached")]
    )
    events, saved = _one_turn(agent, "cap_unknown")

    assert agent.model.call_count == MAX_CONSECUTIVE_STUCK_REPLIES
    assert _final_text(events) == STUCK_TURN_END
    replies = _responses(events, "no_such_tool")
    assert len(replies) == MAX_CONSECUTIVE_STUCK_REPLIES
    assert all("no tool with that name" in reply["error"] for reply in replies)
    assert not _unanswered_calls(saved)


def test_missing_parameters_are_capped_and_have_adks_shape():
    agent = _probe(
        [_MISSING] * MAX_CONSECUTIVE_STUCK_REPLIES + [_text("never reached")]
    )
    events, saved = _one_turn(agent, "cap_missing")

    assert agent.model.call_count == MAX_CONSECUTIVE_STUCK_REPLIES
    assert _final_text(events) == STUCK_TURN_END
    replies = _responses(events, "needs_value")
    assert replies
    assert all(is_adk_rejection(reply) for reply in replies)
    assert not _unanswered_calls(saved)


def test_a_real_tool_that_raises_still_raises():
    agent = _probe([_calls("explode"), _text("never reached")])
    with pytest.raises(ValueError, match="boom"):
        _one_turn(agent, "cap_explode")


def test_alternating_stuck_and_good_replies_stop_at_the_sixth_stuck_reply():
    pairs = [_UNKNOWN, _GOOD] * (MAX_STUCK_REPLIES_PER_TURN - 1)
    stopped = _probe(pairs + [_UNKNOWN, _text("never reached")])
    events, _ = _one_turn(stopped, "cap_alternating")
    assert stopped.model.call_count == 2 * MAX_STUCK_REPLIES_PER_TURN - 1
    assert _final_text(events) == STUCK_TURN_END

    finishes = _probe(pairs + [_text("done")])
    events, _ = _one_turn(finishes, "cap_alternating_done")
    assert _final_text(events) == "done"


def test_a_mix_of_kinds_shares_one_limit_and_uses_the_general_text():
    agent = _probe([_UNKNOWN, _MISSING, _HIDDEN, _text("never reached")], gated=True)
    events, saved = _one_turn(agent, "cap_mixed_kinds")

    assert agent.model.call_count == 3
    assert _final_text(events) == STUCK_TURN_END
    assert not _unanswered_calls(saved)


def test_only_hidden_transfers_get_the_hand_over_text():
    agent = _probe(
        [_HIDDEN] * MAX_CONSECUTIVE_STUCK_REPLIES + [_text("never reached")], gated=True
    )
    events, _ = _one_turn(agent, "cap_only_transfers")
    assert _final_text(events) == HIDDEN_TRANSFER_TURN_END


@pytest.mark.parametrize(
    "mixed",
    [_calls("no_such_tool", "ping"), _calls("ping", "no_such_tool")],
    ids=["bad-first", "good-first"],
)
def test_a_mixed_reply_resets_in_a_row_but_counts_for_the_turn(mixed):
    keeps = _probe([_UNKNOWN, _UNKNOWN, mixed, _UNKNOWN, _UNKNOWN, _text("done")])
    events, _ = _one_turn(keeps, "cap_mixed_reply")
    assert _final_text(events) == "done"

    capped = _probe([mixed] * MAX_STUCK_REPLIES_PER_TURN + [_text("never reached")])
    events, _ = _one_turn(capped, "cap_mixed_reply_ceiling")
    assert capped.model.call_count == MAX_STUCK_REPLIES_PER_TURN
    assert _final_text(events) == STUCK_TURN_END


def test_one_slip_then_real_work_keeps_the_turn():
    agent = _probe([_UNKNOWN, _GOOD, _text("done")])
    events, _ = _one_turn(agent, "cap_one_slip")
    assert _final_text(events) == "done"


def test_text_beside_an_unknown_name_does_not_end_the_turn():
    """Only the hidden-transfer refusal ends a turn on a reply that spoke to
    the user. A model narrating next to a bad call ("Let me check.") keeps its
    turn and can recover (the spec's OQ2)."""
    narrated = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(text="Let me check the files."),
                types.Part(
                    function_call=types.FunctionCall(name="no_such_tool", args={})
                ),
            ],
        )
    )
    agent = _probe([narrated, _GOOD, _text("done")])
    events, _ = _one_turn(agent, "cap_narrated_slip")
    assert agent.model.call_count == 3
    assert _final_text(events) == "done"


def test_a_tools_own_errors_do_not_reset_in_a_row():
    replies = [_UNKNOWN, _REFUSED_BY_TOOL, _UNKNOWN, _REFUSED_BY_TOOL, _UNKNOWN]
    agent = _probe(replies + [_text("never reached")])
    events, _ = _one_turn(agent, "cap_own_errors")
    assert agent.model.call_count == len(replies)
    assert _final_text(events) == STUCK_TURN_END


@pytest.mark.parametrize(
    ("stuck", "gated", "expected"),
    [(_MISSING, False, STUCK_TURN_END), (_HIDDEN, True, HIDDEN_TRANSFER_TURN_END)],
    ids=["ungated", "gated"],
)
def test_the_cap_stops_under_sse_streaming(stuck, gated, expected):
    """adk web streams (the Dev UI's toggle): partial chunks arrive before
    each complete reply, and ADK passes them to the after-model callbacks."""
    agent = _probe(
        [stuck] * MAX_CONSECUTIVE_STUCK_REPLIES + [_text("never reached")], gated=gated
    )
    events, saved = _one_turn(
        agent,
        f"cap_sse_{gated}",
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )
    assert any(event.partial for event in events)  # the stream really had chunks
    assert agent.model.call_count == MAX_CONSECUTIVE_STUCK_REPLIES
    assert _final_text(events) == expected
    assert not _unanswered_calls(saved)


def test_the_next_turn_starts_at_zero():
    agent = _probe(
        [_UNKNOWN] * MAX_CONSECUTIVE_STUCK_REPLIES
        + [_UNKNOWN, _GOOD, _text("turn two done")]
    )
    (first, second), saved = asyncio.run(
        _run_turns(agent, "cap_next_turn", messages=("hello", "and now?"))
    )
    assert _final_text(first) == STUCK_TURN_END
    assert _final_text(second) == "turn two done"
    assert not _unanswered_calls(saved)


@pytest.mark.parametrize(
    "agent", user_facing_llm_agents(), ids=lambda agent: agent.name
)
def test_every_agent_the_user_talks_to_stops_after_three_stuck_replies(
    agent, monkeypatch
):
    """Built from the tree walk, so a new agent in any root is covered without
    anyone adding it here."""
    model = ScriptedLlm(
        model="scripted",
        responses=[_UNKNOWN] * MAX_CONSECUTIVE_STUCK_REPLIES + [_text("never reached")],
    )
    monkeypatch.setattr(agent, "model", model)

    events, saved = _one_turn(agent, f"cap_{agent.name}")

    assert model.call_count == MAX_CONSECUTIVE_STUCK_REPLIES
    assert _final_text(events) == STUCK_TURN_END
    assert not _unanswered_calls(saved)


def test_the_coordinators_real_transfer_is_not_refused(monkeypatch):
    """Not counted is pinned by test_rejected_call_cap.py::test_the_real_transfer_is_progress_not_stuck."""
    coordinator = ScriptedLlm(
        model="scripted",
        responses=[
            _UNKNOWN,
            _UNKNOWN,
            _calls(("transfer_to_agent", {"agent_name": file_suggestion_agent.name})),
        ],
    )
    monkeypatch.setattr(full_workflow_agent, "model", coordinator)
    monkeypatch.setattr(
        file_suggestion_agent,
        "model",
        ScriptedLlm(model="scripted", responses=[_text("file step speaking")]),
    )

    events, saved = _one_turn(full_workflow_agent, "cap_real_transfer")

    assert _final_text(events) == "file step speaking"
    replies = _responses(events, "transfer_to_agent")
    assert replies
    assert not any(is_error(reply) for reply in replies)
    assert any(
        event.actions.transfer_to_agent == file_suggestion_agent.name
        for event in events
    )
    assert not _unanswered_calls(saved)
