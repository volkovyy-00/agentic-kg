"""Empirical proof that a gated agent keeps the user across turns.

ADK decides who handles each NEW top-level user message in
Runner._find_agent_to_run (runners.py:474-489): it walks back to the agent
that replied last and returns it only if _is_transferable_across_agent_tree
(492-510) finds disallow_transfer_to_parent unset on that agent and every
ancestor. Setting that flag -- the obvious way to remove ADK's injected
transfer tool, and the one this design rejected -- would therefore send every
in-phase follow-up question back to the coordinator to be re-arbitrated.

That is not a hypothetical. The post-construction window exists precisely so
the user can ask several questions in a row (docs/backlog/
construction-agent-confirm-loop.md), and a coordinator re-decision on each one
is another chance to leave the phase with the confirmation flag still False --
the defect the gates exist to close.

Rooted at the real full_workflow_agent, because that is the only arrangement
in which _find_agent_to_run's interesting branch runs at all: point the Runner
at the gated agent and it becomes root_agent, so the first `event.author ==
root_agent.name` check short-circuits.
"""
import asyncio

from google.genai import types
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from pydantic import Field

from agentic_kg.common.agent_names import MULTI_AGENT_COORDINATOR
from agentic_kg.coordinators.multi_agent.agent import full_workflow_agent
from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent.agent import (
    graph_construction_agent,
)


class ScriptedLlm(BaseLlm):
    """Replays a fixed script of LlmResponses, holding on the last one.

    Copied from test_schema_refinement_loop_turn_cap.py:36 rather than
    imported; tests/unit/fakes.py deliberately does not centralize these.
    """
    responses: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        yield self.responses[index]


def _text(text):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _call(name, args=None):
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part(function_call=types.FunctionCall(name=name, args=args or {})),
    ]))


def test_a_second_question_stays_with_the_construction_agent(monkeypatch):
    """Turn 1 the coordinator delegates; turn 2 must go straight back to the
    construction agent without the coordinator's model being consulted."""
    monkeypatch.setattr(full_workflow_agent, "model", ScriptedLlm(
        model="scripted",
        responses=[_call("transfer_to_agent", {"agent_name": "graph_construction_agent_v1"})],
    ))
    monkeypatch.setattr(graph_construction_agent, "model", ScriptedLlm(
        model="scripted",
        responses=[
            _text("the graph is built -- ask me anything, or move on when ready"),
            _text("here is the answer to your second question"),
        ],
    ))

    async def run():
        runner = InMemoryRunner(agent=full_workflow_agent, app_name="stickiness_test")
        session = await runner.session_service.create_session(
            app_name="stickiness_test", user_id="u1",
        )

        async def turn(text):
            return [
                event async for event in runner.run_async(
                    user_id="u1", session_id=session.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=text)]),
                )
            ]

        turn_1 = await turn("build the graph")
        # Captured before turn 2 runs, so the comparison below is against the
        # coordinator's state at the moment it handed off -- the same
        # call-count-spy technique as
        # test_schema_refinement_loop_turn_cap.py:100.
        coordinator_calls_after_turn_1 = full_workflow_agent.model.call_count
        turn_2 = await turn("one more question about the graph")
        return turn_1, coordinator_calls_after_turn_1, turn_2

    turn_1, coordinator_calls_after_turn_1, turn_2 = asyncio.run(run())

    assert "graph_construction_agent_v1" in [e.author for e in turn_1], (
        "turn 1 never reached the construction agent; the fixture is wrong"
    )

    authors = [event.author for event in turn_2]
    assert "graph_construction_agent_v1" in authors, (
        f"the second question never reached the construction agent; authors were {authors}"
    )
    assert MULTI_AGENT_COORDINATOR not in authors, (
        "the coordinator handled the second question instead of the gated agent; "
        f"authors were {authors}"
    )
    assert full_workflow_agent.model.call_count == coordinator_calls_after_turn_1, (
        "the coordinator re-arbitrated the second question -- something set "
        "disallow_transfer_to_parent on the construction agent or an ancestor"
    )
