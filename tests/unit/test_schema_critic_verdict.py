"""The critic's verdict for a round is the text of its final answer only (KG-25).

On google-adk 2.x, output_key also stores text an agent writes alongside a
tool call, accumulated across its whole run, so the critic's narration ("Let
me look at the goal first.") became part of its verdict. The verdict is now
recorded by record_critic_verdict instead.

Runner-driven, like test_schema_refinement_loop_repair_round.py: what ADK
writes into state for a given reply shape is framework behaviour. Every test
asserts on a single round -- the end state hides the bug, because round 2's
clean answer overwrites round 1's. The plan checks are stubbed to find
nothing, so the loop's route depends on the critic's verdict alone.
"""

import asyncio

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field

from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent import (
    agent as schema_proposal_module,
)
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    EMPTY_VERDICT_SUMMARY,
    root_agent,
    schema_critic_agent,
    schema_proposal_agent,
)

NARRATION = "Let me look at the goal first."


class ScriptedLlm(BaseLlm):
    """One scripted entry per model call; an entry that is a list is yielded
    as several responses from ONE call, the way a streamed reply arrives.
    Holds on the last entry. Records every request."""

    responses: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        self.requests.append(llm_request)
        entry = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        for response in entry if isinstance(entry, list) else [entry]:
            yield response


def _reply(*parts: types.Part, partial: bool = False) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=list(parts)), partial=partial
    )


def _text(text: str) -> types.Part:
    return types.Part(text=text)


def _thought(text: str) -> types.Part:
    return types.Part(text=text, thought=True)


def _call(name: str, args: dict | None = None) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args or {}))


def _narration_and_call() -> LlmResponse:
    """Text sent alongside a tool call, in one unary reply -- the shape that
    reaches the critic in product (AgentTool runs the loop unary)."""
    return _reply(_text(NARRATION), _call("get_approved_user_goal"))


def _silent() -> LlmResponse:
    """A final answer with no text: thought-only, as a reasoning model ends a
    round. Not Part(text=""), which ADK's output_key wrote as "" anyway."""
    return _reply(_thought("thinking it over"))


@pytest.fixture
def no_plan_problems(monkeypatch):
    monkeypatch.setattr(schema_proposal_module, "_plan_problems", lambda state: [])


def _run_loop_once(monkeypatch, critic_script, app_name):
    """One user turn in which the coordinator calls the loop once. Returns
    (the proposal model's requests, the loop's tool result)."""
    monkeypatch.setattr(
        schema_proposal_agent,
        "model",
        ScriptedLlm(model="scripted", responses=[_reply(_text("a proposal"))]),
    )
    monkeypatch.setattr(
        schema_critic_agent,
        "model",
        ScriptedLlm(model="scripted", responses=critic_script),
    )
    monkeypatch.setattr(
        root_agent,
        "model",
        ScriptedLlm(
            model="scripted",
            responses=[
                _reply(_call("schema_refinement_loop", {"request": "propose"})),
                _reply(_text("done")),
            ],
        ),
    )

    async def run():
        runner = InMemoryRunner(agent=root_agent, app_name=app_name)
        session = await runner.session_service.create_session(
            app_name=app_name, user_id="u1"
        )
        return [
            event
            async for event in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[_text("propose")]),
            )
        ]

    events = asyncio.run(run())
    results = [
        str(part.function_response.response.get("result", ""))
        for event in events
        if event.content and event.content.parts
        for part in event.content.parts
        if part.function_response
        and part.function_response.name == "schema_refinement_loop"
    ]
    assert len(results) == 1
    return schema_proposal_agent.model.requests, results[0]


def test_narration_beside_a_tool_call_never_reaches_the_next_round(
    monkeypatch, no_plan_problems
):
    """Round 1's verdict is what the proposal step reads in round 2, where
    its instruction renders {feedback}."""
    requests, _ = _run_loop_once(
        monkeypatch,
        [_narration_and_call(), _reply(_text("retry: add a Supplier node"))],
        "critic_verdict_retry_test",
    )

    assert len(requests) == 2
    round_two = str(requests[1].config.system_instruction)
    assert "retry: add a Supplier node" in round_two
    assert NARRATION not in round_two


def test_narration_beside_several_tool_calls_never_reaches_the_next_round(
    monkeypatch, no_plan_problems
):
    requests, _ = _run_loop_once(
        monkeypatch,
        [
            _reply(
                _text(NARRATION),
                _call("get_approved_user_goal"),
                _call("get_approved_files"),
            ),
            _reply(_text("retry: add a Supplier node")),
        ],
        "critic_verdict_two_calls_test",
    )

    assert len(requests) == 2
    round_two = str(requests[1].config.system_instruction)
    assert "retry: add a Supplier node" in round_two
    assert NARRATION not in round_two


def test_a_silent_final_answer_is_no_verdict_even_after_narration(
    monkeypatch, no_plan_problems
):
    requests, result = _run_loop_once(
        monkeypatch,
        [_narration_and_call(), _silent()],
        "critic_verdict_silent_test",
    )

    assert len(requests) == 1  # no verdict stops the loop after round 1
    assert result == EMPTY_VERDICT_SUMMARY


def test_a_whitespace_only_final_answer_is_no_verdict(monkeypatch, no_plan_problems):
    requests, result = _run_loop_once(
        monkeypatch,
        [_narration_and_call(), _reply(_text("  \n"))],
        "critic_verdict_whitespace_test",
    )

    assert len(requests) == 1
    assert result == EMPTY_VERDICT_SUMMARY


def test_a_valid_final_answer_after_narration_stops_the_loop(
    monkeypatch, no_plan_problems
):
    requests, result = _run_loop_once(
        monkeypatch,
        [_narration_and_call(), _reply(_text("valid"))],
        "critic_verdict_valid_test",
    )

    assert len(requests) == 1
    assert result == "valid"


def test_a_final_answer_in_several_text_parts_is_joined(monkeypatch, no_plan_problems):
    """Joined as ADK joined output_key text; the route still reads the first
    word, and the warnings survive for the user."""
    requests, result = _run_loop_once(
        monkeypatch,
        [_reply(_text("valid"), _text("\nWarnings: 2 suppliers lack quotes"))],
        "critic_verdict_parts_test",
    )

    assert len(requests) == 1
    assert result == "valid\nWarnings: 2 suppliers lack quotes"


def test_thought_parts_never_become_part_of_the_verdict(monkeypatch, no_plan_problems):
    """Regression guard: this one held under output_key too."""
    requests, result = _run_loop_once(
        monkeypatch,
        [_reply(_thought("weighing the joins"), _text("valid"))],
        "critic_verdict_thought_test",
    )

    assert len(requests) == 1
    assert result == "valid"


def test_a_streamed_reply_is_judged_by_its_final_answer_too(
    monkeypatch, no_plan_problems
):
    """A shape the loop does not see today (AgentTool runs it unary): a
    streamed reply flushes its text as a complete, text-only response before
    the tool call arrives, so "the response without a tool call" is not a safe
    test for the final answer."""
    streamed = [
        _reply(_text(NARRATION), partial=True),
        _reply(_text(NARRATION)),
        _reply(_call("get_approved_user_goal")),
    ]
    requests, result = _run_loop_once(
        monkeypatch,
        [streamed, _silent()],
        "critic_verdict_streamed_test",
    )

    assert len(requests) == 1
    assert result == EMPTY_VERDICT_SUMMARY
