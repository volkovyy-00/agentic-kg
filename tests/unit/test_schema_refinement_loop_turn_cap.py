"""Empirical proof that the coordinator's before_agent_callback -- which
resets the schema_refinement_loop turn budget -- fires exactly once per
incoming user message, not once per some other unit of ADK scheduling.

This is deliberately a Runner-driven test rather than a hand-built
InvocationContext: "how many times does ADK call run_async on the
coordinator per user turn" is a framework scheduling behavior, and the only
way to answer it without trusting an assumption is to actually run it
through ADK's own Runner. The three agents' models are replaced with a
scripted fake so the test is fast and deterministic -- it is not testing
whether gpt-5 decides to call the tool, only whether the turn-budget
mechanism holds when it does.
"""
import asyncio

import pytest
from pydantic import Field
from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner

from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    root_agent,
    schema_proposal_agent,
    schema_critic_agent,
)


class ScriptedLlm(BaseLlm):
    """Replays a fixed script of LlmResponses, one per call; holds on the
    last one if more calls arrive than scripted responses."""

    responses: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        yield self.responses[index]


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _tool_call_response(request_text: str) -> LlmResponse:
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(
            name="schema_refinement_loop", args={"request": request_text},
        )),
    ]))


def test_each_user_turn_gets_its_own_one_call_budget():
    async def run():
        # schema_proposal_agent_v1 and schema_critic_agent_v1 only need to
        # end their turn immediately (plain text, no tool calls) so the one
        # allowed schema_refinement_loop invocation per turn completes fast.
        schema_proposal_agent.model = ScriptedLlm(
            model="scripted", responses=[_text_response("a minimal schema proposal")],
        )
        schema_critic_agent.model = ScriptedLlm(
            model="scripted", responses=[_text_response("valid")],
        )
        # Coordinator: turn 1 tries to call schema_refinement_loop TWICE in a
        # row (reproducing the exact live-observed misbehavior this fix
        # guards against), then finalizes. Turn 2 tries once, then finalizes.
        root_agent.model = ScriptedLlm(model="scripted", responses=[
            _tool_call_response("propose an initial schema"),
            _tool_call_response("the user asked for another change"),
            _text_response("turn 1 final response"),
            _tool_call_response("the user asked for yet another change"),
            _text_response("turn 2 final response"),
        ])

        runner = InMemoryRunner(agent=root_agent, app_name="turn_cap_test")
        session = await runner.session_service.create_session(
            app_name="turn_cap_test", user_id="u1",
        )

        turn_1_events = [
            event async for event in runner.run_async(
                user_id="u1", session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="please propose a schema")],
                ),
            )
        ]
        # Captured here, before turn 2 runs a second real round and bumps
        # these counts again -- this is the actual proof (a call-count spy,
        # not a text-content coincidence) that the short-circuited second
        # schema_refinement_loop call in turn 1 never invoked either
        # sub-agent's model at all.
        calls_after_turn_1 = (
            schema_proposal_agent.model.call_count,
            schema_critic_agent.model.call_count,
        )

        turn_2_events = [
            event async for event in runner.run_async(
                user_id="u1", session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="please change it again")],
                ),
            )
        ]
        calls_after_turn_2 = (
            schema_proposal_agent.model.call_count,
            schema_critic_agent.model.call_count,
        )
        return turn_1_events, turn_2_events, calls_after_turn_1, calls_after_turn_2

    turn_1_events, turn_2_events, calls_after_turn_1, calls_after_turn_2 = asyncio.run(run())

    def _function_response_texts(events):
        texts = []
        for event in events:
            if not event.content or not event.content.parts:
                continue
            for part in event.content.parts:
                if part.function_response and part.function_response.response:
                    texts.append(str(part.function_response.response.get("result", "")))
        return texts

    turn_1_results = _function_response_texts(turn_1_events)
    turn_2_results = _function_response_texts(turn_2_events)

    # Turn 1: two schema_refinement_loop calls were made. The first
    # succeeds (real verdict), the second is short-circuited.
    assert len(turn_1_results) == 2
    assert turn_1_results[0] == "valid"
    assert turn_1_results[1].startswith("stopped:")

    # The spy: each sub-agent's model was called exactly once during turn 1,
    # even though schema_refinement_loop was invoked twice -- proving the
    # second invocation's before_agent_callback short-circuit genuinely
    # skipped running schema_proposal_agent_v1 and schema_critic_agent_v1,
    # not just that it happened to return matching text.
    assert calls_after_turn_1 == (1, 1)

    # Turn 2: a fresh message resets the budget, so its one call succeeds
    # again (a second real round) rather than being short-circuited.
    assert len(turn_2_results) == 1
    assert turn_2_results[0] == "valid"
    assert calls_after_turn_2 == (2, 2)
