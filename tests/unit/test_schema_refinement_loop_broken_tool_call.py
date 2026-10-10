"""A plan write cut off at the output cap is retried, not silently dropped.

google-adk 2.10 drops a tool call whose arguments do not parse. Inside the
refinement loop nobody sees the error -- AgentTool returns it only when no
event had content, and the stop-check always emits content -- so a dropped
plan write let the loop return "valid" over an empty plan. The proposal
agent's real model (get_llm's instance, with ToolArgsRepairingClient) is
driven through the real LiteLlm adapter with a stubbed transport; the critic
and the coordinator are scripted above the adapter.
"""

import asyncio
import json

import fsspec
import pytest
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLLMClient
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from litellm import ModelResponse
from pydantic import Field

from agentic_kg.common.config import reset_settings
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    root_agent,
    schema_critic_agent,
    schema_proposal_agent,
)

PLAN_WRITE = "propose_node_constructions"
FULL_ARGUMENTS = json.dumps(
    {
        "node_constructions": [
            {
                "approved_file": "plots.csv",
                "proposed_label": "Plot",
                "unique_column_name": "plot_label",
                "proposed_properties": ["plot_id"],
            }
        ]
    }
)


class ScriptedLlm(BaseLlm):
    responses: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        yield self.responses[index]


def _text(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def _call_the_loop() -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="schema_refinement_loop",
                        args={"request": "propose an initial schema"},
                    )
                )
            ],
        )
    )


def _provider_reply(message: dict, finish_reason: str) -> ModelResponse:
    return ModelResponse(
        choices=[{"index": 0, "finish_reason": finish_reason, "message": message}]
    )


def _plan_write(arguments: str, finish_reason: str) -> ModelResponse:
    return _provider_reply(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": PLAN_WRITE, "arguments": arguments},
                }
            ],
        },
        finish_reason,
    )


@pytest.fixture
def one_approved_file(monkeypatch):
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    with fs.open("/src/plots.csv", "w") as handle:
        handle.write("plot_label,plot_id\nridge,PL-1\nhollow,PL-3\n")
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield {"approved_file_list": ["plots.csv"]}
    fs.store.clear()
    fs.pseudo_dirs.clear()


@pytest.mark.parametrize(
    "streaming_mode", [StreamingMode.NONE, StreamingMode.SSE], ids=["unary", "sse"]
)
def test_a_cut_off_plan_write_is_retried_and_the_plan_is_written(
    monkeypatch, one_approved_file, streaming_mode
):
    """Round 1's plan write is cut off at the output cap. As on 2.9.2, the
    call goes on with empty arguments, the tool reports what is missing, the
    model sends the full call, and the plan exists when the loop returns.

    Also with the Dev UI's streaming toggle on (SSE): AgentTool runs the loop
    unary whatever the caller's mode (google/adk/tools/agent_tool.py), so the
    proposal agent's replies still pass through ToolArgsRepairingClient."""
    replies = [
        _plan_write(FULL_ARGUMENTS[:40], "length"),
        _plan_write(FULL_ARGUMENTS, "tool_calls"),
        _provider_reply({"role": "assistant", "content": "proposal done"}, "stop"),
    ]
    provider_calls = []

    async def acompletion(client, model, messages, tools, **kwargs):
        reply = replies[min(len(provider_calls), len(replies) - 1)]
        provider_calls.append(model)
        return reply

    monkeypatch.setattr(LiteLLMClient, "acompletion", acompletion)
    critic = ScriptedLlm(model="scripted", responses=[_text("valid")])
    monkeypatch.setattr(schema_critic_agent, "model", critic)
    monkeypatch.setattr(
        root_agent,
        "model",
        ScriptedLlm(
            model="scripted", responses=[_call_the_loop(), _text("final response")]
        ),
    )

    async def run():
        runner = InMemoryRunner(agent=root_agent, app_name="broken_plan_write")
        assert runner.plugin_manager.plugins == []
        session = await runner.session_service.create_session(
            app_name="broken_plan_write", user_id="u1", state=one_approved_file
        )
        async for _ in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="propose")]),
            run_config=RunConfig(streaming_mode=streaming_mode),
        ):
            pass
        return await runner.session_service.get_session(
            app_name="broken_plan_write", user_id="u1", session_id=session.id
        )

    final = asyncio.run(run())

    # The proposal agent's own model (get_llm's instance) made all three calls.
    assert provider_calls == [schema_proposal_agent.model.model] * 3
    assert list(final.state["proposed_construction_plan"]) == ["Plot"]
    assert final.state["feedback"] == "valid"
