"""Tool calls whose arguments do not parse behave as on google-adk 2.9.2.

Drives the real LiteLlm adapter: ADK's handling of these calls lives inside
it, below the BaseLlm fakes other tests use. The transport is stubbed by
patching LiteLLMClient.acompletion, which ToolArgsRepairingClient calls through
super(), so each reply is exactly what a provider would send -- text and the
call together, never a patched callback input.
"""

import asyncio
import json

import pytest
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
from google.adk.runners import InMemoryRunner
from google.genai import types
from litellm import ModelResponse

from agentic_kg.common.litellm_client import (
    EMPTY_ARGUMENTS,
    ToolArgsRepairingClient,
    repair_tool_call_arguments,
)

LOOKUP = "lookup"
GET_GOAL = "get_goal"


def _reply(*, arguments=None, tool=LOOKUP, text=None, finish_reason="tool_calls"):
    message: dict = {"role": "assistant", "content": text}
    if arguments is not None:
        message["tool_calls"] = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": tool, "arguments": arguments},
            }
        ]
    return ModelResponse(
        choices=[{"index": 0, "finish_reason": finish_reason, "message": message}]
    )


DONE = _reply(text="done", finish_reason="stop")


class Transport:
    """Stands in for the provider: returns the scripted replies in order, then
    repeats the last one, and counts the calls."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = 0

    async def acompletion(self, client, model, messages, tools, **kwargs):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


@pytest.fixture
def transport(monkeypatch):
    def install(*replies):
        stub = Transport(list(replies))

        async def acompletion(client, model, messages, tools, **kwargs):
            return await stub.acompletion(client, model, messages, tools, **kwargs)

        monkeypatch.setattr(LiteLLMClient, "acompletion", acompletion)
        return stub

    return install


def _agent(client: LiteLLMClient, ran: list) -> LlmAgent:
    def lookup(q: str) -> dict:
        """Looks something up."""
        ran.append((LOOKUP, q))
        return {"status": "success"}

    def get_goal() -> dict:
        """Returns the goal."""
        ran.append((GET_GOAL, None))
        return {"status": "success"}

    return LlmAgent(
        name="probe_agent",
        model=LiteLlm(model="openai/test-model", api_key="test", llm_client=client),
        instruction="Answer.",
        tools=[lookup, get_goal],
        mode="chat",
    )


def _run(agent: LlmAgent) -> list:
    async def run():
        runner = InMemoryRunner(agent=agent, app_name="litellm_client_test")
        # ADK's reflect-and-retry plugin is the only other thing that retries
        # these calls; a plugin added later would change every count below.
        assert runner.plugin_manager.plugins == []
        session = await runner.session_service.create_session(
            app_name="litellm_client_test", user_id="u1"
        )
        return [
            event
            async for event in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text="go")]),
            )
        ]

    return asyncio.run(run())


def _function_calls(events) -> list:
    return [
        part.function_call
        for event in events
        if event.content
        for part in event.content.parts or []
        if part.function_call
    ]


def _first_model_event(events):
    return next(event for event in events if event.author == "probe_agent")


def test_canary_adk_still_drops_a_tool_call_whose_arguments_do_not_parse(transport):
    """Without our client, google-adk 2.10 drops the call and ends the step.
    If this fails, ADK changed its handling (75d84b1): re-check whether
    ToolArgsRepairingClient is still needed, and remove it if not."""
    stub = transport(_reply(arguments='{"q": "ab'), DONE)
    ran: list = []

    events = _run(_agent(LiteLLMClient(), ran))

    assert stub.calls == 1
    assert _function_calls(events) == []
    assert ran == []
    error_event = _first_model_event(events)
    assert error_event.error_code == types.FinishReason.MALFORMED_FUNCTION_CALL
    assert LOOKUP in (error_event.error_message or "")


@pytest.mark.parametrize(
    ("finish_reason", "text", "expected_error_code"),
    [
        ("tool_calls", None, None),
        ("tool_calls", "Let me look that up.", None),
        ("length", None, types.FinishReason.MAX_TOKENS),
        ("length", "Let me look that up.", types.FinishReason.MAX_TOKENS),
    ],
    ids=["broken", "broken-with-text", "cut-off", "cut-off-with-text"],
)
def test_a_call_whose_arguments_do_not_parse_is_sent_on_with_empty_arguments(
    transport, finish_reason, text, expected_error_code
):
    """As on 2.9.2: the call is dispatched with {}, the tool reports its
    missing parameter, and the model gets a second call to retry."""
    stub = transport(
        _reply(arguments='{"q": "ab', text=text, finish_reason=finish_reason), DONE
    )
    ran: list = []

    events = _run(_agent(ToolArgsRepairingClient(), ran))

    assert stub.calls == 2
    calls = _function_calls(events)
    assert [(call.name, call.args) for call in calls] == [(LOOKUP, {})]
    assert ran == []  # the tool's own validation rejected the empty arguments
    first = _first_model_event(events)
    assert first.error_code == expected_error_code
    texts = [part.text for part in first.content.parts if part.text]
    assert texts == ([text] if text else [])


def test_null_arguments_to_a_no_argument_tool_run_it(transport):
    """2.9.2 dispatched `null` as no arguments, so a no-argument tool (the
    get_* family) ran; 2.10 drops it."""
    stub = transport(_reply(arguments="null", tool=GET_GOAL), DONE)
    ran: list = []

    _run(_agent(ToolArgsRepairingClient(), ran))

    assert ran == [(GET_GOAL, None)]
    assert stub.calls == 2


def test_a_list_as_arguments_becomes_a_retry_not_a_crash(transport):
    """2.9.2 raised a ValidationError and the turn crashed; sending {} instead
    lets the model retry. The one deliberate difference this client introduces."""
    stub = transport(_reply(arguments="[]"), DONE)
    ran: list = []

    events = _run(_agent(ToolArgsRepairingClient(), ran))

    assert [(call.name, call.args) for call in _function_calls(events)] == [
        (LOOKUP, {})
    ]
    assert stub.calls == 2


def test_arguments_adk_repairs_itself_are_left_alone(transport):
    """A Python dict literal is one of the shapes ADK's parser repairs; the
    client must not blank it."""
    literal = "{'q': 'abc'}"
    reply = _reply(arguments=literal)
    repair_tool_call_arguments(reply)
    assert reply.choices[0].message.tool_calls[0].function.arguments == literal

    transport(_reply(arguments=literal), DONE)
    ran: list = []
    _run(_agent(ToolArgsRepairingClient(), ran))
    assert ran == [(LOOKUP, "abc")]


def test_valid_arguments_are_left_alone():
    arguments = json.dumps({"q": "abc"})
    reply = _reply(arguments=arguments)
    repair_tool_call_arguments(reply)
    assert reply.choices[0].message.tool_calls[0].function.arguments == arguments


def test_a_broken_call_is_repaired_to_empty_arguments():
    reply = _reply(arguments='{"q": "ab')
    repair_tool_call_arguments(reply)
    assert reply.choices[0].message.tool_calls[0].function.arguments == EMPTY_ARGUMENTS


def test_a_streamed_reply_passes_through_untouched(monkeypatch):
    """Streaming returns a stream wrapper, not a ModelResponse; the client
    hands it back as it came."""
    stream = object()

    async def acompletion(client, model, messages, tools, **kwargs):
        return stream

    monkeypatch.setattr(LiteLLMClient, "acompletion", acompletion)

    result = asyncio.run(
        ToolArgsRepairingClient().acompletion(
            model="openai/test-model", messages=[], tools=None, stream=True
        )
    )

    assert result is stream


def test_only_the_broken_call_of_several_is_repaired():
    """Models send parallel calls; one broken call must not blank the others."""
    good = json.dumps({"q": "abc"})
    reply = ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": LOOKUP, "arguments": good},
                        },
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {"name": LOOKUP, "arguments": '{"q": "ab'},
                        },
                    ],
                },
            }
        ]
    )

    repair_tool_call_arguments(reply)

    calls = reply.choices[0].message.tool_calls
    assert [call.function.arguments for call in calls] == [good, EMPTY_ARGUMENTS]


def test_a_reply_without_tool_calls_is_left_alone():
    reply = _reply(text="just text", finish_reason="stop")
    repair_tool_call_arguments(reply)
    assert reply.choices[0].message.content == "just text"
    assert not reply.choices[0].message.tool_calls


def test_the_log_names_the_tool_but_never_the_arguments(caplog):
    """Arguments can be large and carry the user's data."""
    secret = '{"q": "customer-ledger-2026'
    with caplog.at_level("WARNING", logger="agentic_kg.common.litellm_client"):
        repair_tool_call_arguments(_reply(arguments=secret))

    assert LOOKUP in caplog.text
    assert "customer-ledger-2026" not in caplog.text
