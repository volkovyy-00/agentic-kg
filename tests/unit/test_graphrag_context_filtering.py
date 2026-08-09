"""End-to-end proof that graphrag_agent_v2 never sees another agent's output.

Asserts on what reached the model, never on what the model said. The negative
control matters as much as the positive one: without asserting that v1 DOES
receive the sentinel, a test passing because the fixture never produced
foreign context would look identical to a working filter.
"""

import asyncio

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field

from agentic_kg.common.adk_context import FOREIGN_CONTEXT_SENTINEL
from agentic_kg.coordinators.multi_agent.sub_agents.graphrag_agent.variants import (
    variants,
)


class CapturingLlm(BaseLlm):
    """Scripted LLM that also records every request it was handed.

    Extends the pattern in test_schema_refinement_loop_turn_cap.py:36.
    """

    responses: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        self.requests.append(llm_request)
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        yield self.responses[index]


def _text(text):
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def _build_agent(variant_name):
    from google.adk.agents import Agent

    spec = variants[variant_name]
    return Agent(
        name=variant_name,
        model=CapturingLlm(model="scripted", responses=[_text("ok")]),
        description="test",
        instruction=spec["instruction"],
        tools=spec["tools"],
        before_model_callback=spec.get("before_model_callback"),
    )


async def _run_with_foreign_history(agent):
    runner = InMemoryRunner(agent=agent, app_name="ctx_test")
    session = await runner.session_service.create_session(
        app_name="ctx_test", user_id="u1"
    )

    # A message shaped exactly as ADK reshapes another agent's output.
    foreign = types.Content(
        role="user",
        parts=[
            types.Part(text=FOREIGN_CONTEXT_SENTINEL),
            types.Part(text="[schema_critic] said: 4 suppliers have no quote rows"),
        ],
    )
    from google.adk.events.event import Event

    await runner.session_service.append_event(
        session=session, event=Event(author="user", content=foreign)
    )

    async for _ in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part(text="which are orphaned?")]
        ),
    ):
        pass
    return agent.model.requests


def _sentinel_present(requests):
    for req in requests:
        for content in req.contents or []:
            for part in content.parts or []:
                if getattr(part, "text", None) == FOREIGN_CONTEXT_SENTINEL:
                    return True
    return False


def test_v2_never_receives_foreign_context():
    # asyncio.run inside a sync test, matching
    # test_schema_refinement_loop_turn_cap.py:120 -- this repo drives async ADK
    # code without taking a pytest-asyncio dependency.
    requests = asyncio.run(_run_with_foreign_history(_build_agent("graphrag_agent_v2")))
    assert requests, "the model was never called"
    assert not _sentinel_present(requests)


def test_v1_still_receives_it_negative_control():
    """Proves the fixture actually produces foreign context."""
    requests = asyncio.run(_run_with_foreign_history(_build_agent("graphrag_agent_v1")))
    assert requests, "the model was never called"
    assert _sentinel_present(requests)


def test_v1_is_left_intact_for_the_acceptance_ab():
    assert "graphrag_agent_v1" in variants
    assert "before_model_callback" not in variants["graphrag_agent_v1"]


def test_v2_binds_the_profile_wrapper_not_the_bare_schema_tool():
    from agentic_kg.tools.cypher_tools import (
        get_graph_schema_with_profile,
        get_physical_schema,
    )

    tools = variants["graphrag_agent_v2"]["tools"]
    assert get_graph_schema_with_profile in tools
    assert get_physical_schema not in tools
