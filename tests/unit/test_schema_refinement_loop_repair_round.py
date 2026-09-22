"""The repair round actually happens, and carries the problems to the proposal
step.

Runner-driven rather than a hand-built InvocationContext, for the same reason
test_schema_refinement_loop_turn_cap.py is: whether LoopAgent runs a second
iteration, and what ADK renders into the proposal agent's instruction, are
framework behaviours. The models are scripted so the test is fast and
deterministic.
"""

import asyncio

import fsspec
import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field

from agentic_kg.common.config import reset_settings
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    PLAN_PROBLEM_HEADER,
    root_agent,
    schema_critic_agent,
    schema_proposal_agent,
)


class RecordingLlm(BaseLlm):
    """ScriptedLlm plus a record of every request, so the rendered instruction
    can be inspected."""

    responses: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        self.requests.append(llm_request)
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        yield self.responses[index]


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def _tool_call_response(request_text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="schema_refinement_loop",
                        args={"request": request_text},
                    )
                ),
            ],
        )
    )


# Deliberately a third copy of this fixture (the others are in
# test_construction_plan_tools.py and test_schema_refinement_loop_plan_checks.py).
# Each yields a different shape -- a FakeToolContext, a plain dict, and a
# Runner session state -- and sharing one would couple three test modules to a
# single conftest fixture for the sake of six lines of CSV.
@pytest.fixture
def stranding_sources(monkeypatch):
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


def test_the_proposal_step_is_told_the_problems_in_the_second_round(
    monkeypatch, stranding_sources
):
    """The critic passes the plan in both rounds. The mechanical check must
    still force a second iteration and put the problem in front of the
    proposal step, and the loop's result must still be a retry."""

    async def run():
        monkeypatch.setattr(
            schema_proposal_agent,
            "model",
            RecordingLlm(
                model="recording",
                responses=[_text_response("a minimal schema proposal")],
            ),
        )
        monkeypatch.setattr(
            schema_critic_agent,
            "model",
            RecordingLlm(model="recording", responses=[_text_response("valid")]),
        )
        monkeypatch.setattr(
            root_agent,
            "model",
            RecordingLlm(
                model="recording",
                responses=[
                    _tool_call_response("propose an initial schema"),
                    _text_response("final response"),
                ],
            ),
        )

        runner = InMemoryRunner(agent=root_agent, app_name="repair_round_test")
        session = await runner.session_service.create_session(
            app_name="repair_round_test",
            user_id="u1",
            state=stranding_sources,
        )
        events = [
            event
            async for event in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="please propose a schema")]
                ),
            )
        ]
        return events, schema_proposal_agent.model.requests

    events, requests = asyncio.run(run())

    # AC1: the loop spent a second iteration although the critic said 'valid'.
    assert len(requests) == 2

    # The problem reached the proposal step through the instruction, where
    # {feedback} renders -- NOT merely through conversation history. Asserting
    # on the whole request would pass an implementation that forgot the
    # state_delta entirely.
    instruction = str(requests[1].config.system_instruction)
    assert "plot_id" in instruction
    assert PLAN_PROBLEM_HEADER in instruction

    # KG-30 AC3: the kind reaches the proposal step through its own
    # placeholder -- none in round 1 (the reset), mechanical in round 2.
    assert "Kind of feedback: none" in str(requests[0].config.system_instruction)
    assert "Kind of feedback: mechanical" in instruction

    # AC2: the problem persists (the scripted proposal never changes the plan),
    # so what the coordinator gets back is a retry, not the critic's 'valid'.
    results = [
        str(part.function_response.response.get("result", ""))
        for event in events
        if event.content and event.content.parts
        for part in event.content.parts
        if part.function_response and part.function_response.response
    ]
    assert results
    assert results[0].startswith("retry")


def test_a_second_loop_call_in_the_same_turn_quotes_the_composite(
    monkeypatch, stranding_sources
):
    """The turn cap short-circuits a second schema_refinement_loop call with a
    'stopped:' message quoting the feedback slot. That slot must hold the
    composite, not the critic's 'valid' -- which is exactly what a mutating
    implementation would leave there, since AgentTool forwards only
    state_delta out of the loop's child session."""

    async def run():
        monkeypatch.setattr(
            schema_proposal_agent,
            "model",
            RecordingLlm(
                model="recording",
                responses=[_text_response("a minimal schema proposal")],
            ),
        )
        monkeypatch.setattr(
            schema_critic_agent,
            "model",
            RecordingLlm(model="recording", responses=[_text_response("valid")]),
        )
        monkeypatch.setattr(
            root_agent,
            "model",
            RecordingLlm(
                model="recording",
                responses=[
                    _tool_call_response("propose an initial schema"),
                    _tool_call_response("the user asked for another change"),
                    _text_response("final response"),
                ],
            ),
        )

        runner = InMemoryRunner(agent=root_agent, app_name="second_call_test")
        session = await runner.session_service.create_session(
            app_name="second_call_test",
            user_id="u1",
            state=stranding_sources,
        )
        return [
            event
            async for event in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="please propose a schema")]
                ),
            )
        ]

    events = asyncio.run(run())
    results = [
        str(part.function_response.response.get("result", ""))
        for event in events
        if event.content and event.content.parts
        for part in event.content.parts
        if part.function_response and part.function_response.response
    ]

    assert len(results) == 2
    assert results[1].startswith("stopped:")
    assert "plot_id" in results[1]
    assert "last verdict: valid" not in results[1]

    # KG-30: the kind travelled out of the loop's child session with the delta,
    # so the short-circuit names it.
    assert "mechanical check finding" in results[1]


# KG-29: a critic that ends a round without text. A thought-only part is how a
# reasoning model does that; ADK saves output_key only for a non-thought text
# part, so without the per-round reset the slot would keep last round's verdict.
def _silent_response() -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text="thinking it over", thought=True)],
        )
    )


def _run_turn(monkeypatch, state, *, critic, coordinator, app_name):
    """One user turn through the real Runner; returns (tool results, proposal
    requests, final session state)."""

    async def run():
        monkeypatch.setattr(
            schema_proposal_agent,
            "model",
            RecordingLlm(
                model="recording",
                responses=[_text_response("a minimal schema proposal")],
            ),
        )
        monkeypatch.setattr(
            schema_critic_agent,
            "model",
            RecordingLlm(model="recording", responses=critic),
        )
        monkeypatch.setattr(
            root_agent, "model", RecordingLlm(model="recording", responses=coordinator)
        )
        runner = InMemoryRunner(agent=root_agent, app_name=app_name)
        session = await runner.session_service.create_session(
            app_name=app_name, user_id="u1", state=state
        )
        events = [
            event
            async for event in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="please propose a schema")]
                ),
            )
        ]
        final = await runner.session_service.get_session(
            app_name=app_name, user_id="u1", session_id=session.id
        )
        return events, schema_proposal_agent.model.requests, final.state

    events, requests, final_state = asyncio.run(run())
    results = [
        str(part.function_response.response.get("result", ""))
        for event in events
        if event.content and event.content.parts
        for part in event.content.parts
        if part.function_response and part.function_response.response
    ]
    return results, requests, final_state


def test_a_silent_critic_round_does_not_inherit_the_last_rounds_verdict(
    monkeypatch, stranding_sources
):
    """KG-29 AC1 and AC4. Round 1's critic objects to a plan with no mechanical
    problem; round 2's critic says nothing. Round 2 has no verdict -- yet round
    1's objection still reached round 2's proposal step, which reads {feedback}
    before the critic's reset fires."""
    stranding_sources["proposed_construction_plan"]["Plot"]["unique_column_name"] = (
        "plot_id"
    )
    objection = "retry\n- the relationship names read badly"
    results, requests, final_state = _run_turn(
        monkeypatch,
        stranding_sources,
        critic=[_text_response(objection), _silent_response()],
        coordinator=[
            _tool_call_response("propose an initial schema"),
            _text_response("final response"),
        ],
        app_name="silent_critic_test",
    )

    assert len(requests) == 2
    instruction = str(requests[1].config.system_instruction)
    assert "the relationship names read badly" in instruction
    assert "Kind of feedback: critic" in instruction

    assert results[0].startswith("no verdict:")
    assert "the relationship names read badly" not in results[0]
    assert final_state["feedback"] == ""
    assert final_state["feedback_kind"] == "none"


@pytest.mark.parametrize(
    "header",
    [None, "the loop's own findings, worded differently:"],
    ids=["current-wording", "reworded"],
)
def test_a_repaired_plan_with_a_silent_critic_is_not_reported_as_a_retry(
    monkeypatch, stranding_sources, header
):
    """KG-29 AC2, AC3 and AC5. Round 1 finds a mechanical problem, round 2's
    revision clears it and round 2's critic is silent. The coordinator is told
    plainly there is no verdict -- never 'retry' -- and a second call in the
    same turn says the same. Rewording the loop's own header changes nothing,
    since no behaviour keys on its wording any more."""
    from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent import (
        agent as module,
    )

    if header is not None:
        monkeypatch.setattr(module, "PLAN_PROBLEM_HEADER", header)
    rounds = iter([(["Plot: an earlier problem"], []), ([], [])])
    monkeypatch.setattr(module, "find_plan_problems", lambda state: next(rounds))

    results, requests, _ = _run_turn(
        monkeypatch,
        stranding_sources,
        critic=[_text_response("valid"), _silent_response()],
        coordinator=[
            _tool_call_response("propose an initial schema"),
            _tool_call_response("the user asked for another change"),
            _text_response("final response"),
        ],
        app_name="repaired_plan_test",
    )

    assert len(requests) == 2
    assert "an earlier problem" in str(requests[1].config.system_instruction)

    assert results[0].startswith("no verdict:")
    assert not results[0].startswith("retry")
    assert "an earlier problem" not in results[0]

    assert results[1].startswith("stopped:")
    assert "It recorded no verdict." in results[1]
    assert "an earlier problem" not in results[1]
