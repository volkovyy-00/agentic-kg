"""The LiteLLM client every agent's model uses: keeps google-adk 2.9's handling
of tool calls whose arguments do not parse.

google-adk 2.10 (75d84b1) drops a tool call whose arguments are not a JSON
object -- broken JSON, JSON cut off at the output cap (`max_tokens` in
llm_catalog.py; the proposal agent's plans are our largest arguments), `null`,
or a list/string/number -- and ends the agent's step with an error event.
2.9.2 dispatched a broken or cut-off call with empty arguments instead, so the
tool reported its missing parameters and the model retried. Inside the schema
refinement loop that difference is silent and harmful: AgentTool shows the
error only when no event had content, the stop-check always emits content, so a
dropped plan write let the loop return "valid" over an empty plan.

This client restores 2.9.2: before ADK parses a non-streaming reply, a call
whose arguments ADK's own parser cannot turn into a dict gets "{}". `null`
then behaves exactly as on 2.9.2 (a no-argument tool runs); a list, string or
number, which crashed the turn on 2.9.2, becomes a missing-parameters error the
model can retry. Every other reply passes through untouched.

Limits, on purpose:
- Non-streaming only. A streamed reply is assembled inside ADK's adapter after
  this client returns. The refinement loop never streams (AgentTool runs it
  unary), so the silent case is covered. Streamed broken or cut-off JSON
  already ended in an error event on 2.9.2. One streamed case does differ and
  is accepted: a tool call whose arguments are a bare `null`, streamed in one
  piece. 2.9.2 sent it on with no arguments; 2.10 drops it with an error event.
  Only agents the user talks to directly stream (AgentTool runs nested agents
  unary), so there the user sees the error and can resend.
- Tool calls ADK reads out of the reply text (a model that writes the call
  as JSON instead of using tool_calls) are not repaired: if such a call's
  arguments do not parse, 2.10 drops it where 2.9.2 sent it on with empty
  arguments. A reply cut off at the output cap never takes this path (ADK
  finds a call in text only when the call's whole JSON object is complete,
  which a cut-off call never is), so this is rare.
- This client does not bound retries: the rejected-call cap
  (rejected_call_cap.py) counts the missing-parameters reply a repaired call
  gets, and stops the agent after 3 in a row or 6 in a turn.

Remove this client when the canary in tests/unit/test_litellm_client.py fails:
it means ADK no longer drops these calls, and the reason for the client is
gone. `_parse_tool_call_arguments` is private; a rename fails this import at
startup, and a change in what it accepts fails the same test file's
pass-through and `null` tests.
"""

import json
import logging
from typing import Any

from google.adk.models.lite_llm import LiteLLMClient, _parse_tool_call_arguments
from litellm import ModelResponse

logger = logging.getLogger(__name__)

EMPTY_ARGUMENTS = "{}"


def _parses_to_object(arguments: Any) -> bool:
    try:
        return isinstance(_parse_tool_call_arguments(arguments), dict)
    except json.JSONDecodeError:
        return False


def repair_tool_call_arguments(response: ModelResponse) -> None:
    """Replace, in place, the arguments of every tool call ADK would drop."""
    for choice in response.choices or []:
        message = getattr(choice, "message", None)
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            if function is None or _parses_to_object(function.arguments):
                continue
            # The arguments themselves are never logged: they can be large and
            # may carry user data.
            arguments = function.arguments
            logger.warning(
                "Tool call %r had arguments that are not a JSON object (%d chars);"
                " sending it on with empty arguments, as google-adk 2.9.2 did.",
                function.name,
                len(arguments) if isinstance(arguments, str) else 0,
            )
            function.arguments = EMPTY_ARGUMENTS


class ToolArgsRepairingClient(LiteLLMClient):
    """LiteLLMClient that applies repair_tool_call_arguments to non-streaming
    replies; see the module docstring."""

    async def acompletion(self, model: Any, messages: Any, tools: Any, **kwargs: Any):
        response = await super().acompletion(
            model=model, messages=messages, tools=tools, **kwargs
        )
        if isinstance(response, ModelResponse):
            repair_tool_call_arguments(response)
        return response
