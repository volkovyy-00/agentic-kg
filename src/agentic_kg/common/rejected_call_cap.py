"""Stop an agent that keeps making tool calls ADK rejects.

google-adk 2.10 answers three kinds of call with an error that invites a
retry, and only RunConfig.max_llm_calls (default 500) would stop a model that
keeps retrying:

- An unknown tool name. ADK puts a bare BaseTool in its place
  (flows/llm_flows/tools/_caller.py, _prepare_single), runs the before-tool
  callbacks, then the on-tool-error callbacks, then answers with
  build_tool_not_found_response. The after-tool callbacks are skipped, so
  mark_unknown_tool, an on-tool-error callback, is the only hook that sees it.
- A call missing a mandatory parameter. FunctionTool.run_async returns a dict
  whose only key is "error", holding a string, and that reaches the
  after-tool callbacks as the tool's result (mark_tool_outcome). Arguments
  that do not parse reach the tool as an empty call and end up here.
  Argument-type validation returns the same shape but is an experimental ADK
  feature, off by default; turned on, it counts the same way. So would ADK's
  tool-confirmation replies, which share the shape; no tool here requires
  confirmation.
- A call to the transfer tool stripped from a gated agent.
  refuse_transfer_to_agent (adk_transfer.py) answers it and calls mark_stuck.

No one hook sees all three, so each marks the reply being answered, and
end_turn_past_cap adds the marks up before the next model call:

- A reply that held any rejected call is one stuck reply, however many such
  calls it held, and adds one to the turn's count.
- A reply in which a tool ran and succeeded sets the in-a-row count back to
  zero; otherwise a stuck reply adds one to it. A tool's own error
  (tool_error) neither counts nor resets: the plan checks rely on the model
  fixing those. So stuck, own error, stuck, own error, stuck stops at the third.
- At MAX_CONSECUTIVE_STUCK_REPLIES in a row, or MAX_STUCK_REPLIES_PER_TURN in
  the turn, the model call is replaced by a reply to the user. The per-turn
  ceiling is for a model that pairs every retry with a tool that always
  succeeds.

Everything lives in temp: state, per agent. It lasts one invocation -- one
user message, or one AgentTool run, which gets its own -- and is never
persisted, so each turn starts at zero with no reset callback. Inside the
schema refinement loop the counts are per loop run, shared by the
proposer's iterations.

Never raises, and every rejected call is answered -- by ADK or by the
refusal -- before the stop, so history never holds an unanswered call.
"""

import logging
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Optional

from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from agentic_kg.common.tool_result import is_error

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_STUCK_REPLIES = 3
MAX_STUCK_REPLIES_PER_TURN = 6


class StuckKind(StrEnum):
    """Which kind of rejected call a reply held. Only HIDDEN_TRANSFER changes
    what the user is told."""

    UNKNOWN_TOOL = "unknown_tool"
    ADK_REJECTED = "adk_rejected"
    HIDDEN_TRANSFER = "hidden_transfer"


# What the user reads when the cap ends a turn: its last word, written for the
# user, not the model. Each must be true of every turn it can end.
#
# When every stuck reply this turn was a hidden transfer, the model was trying
# to hand the user on. Direction-neutral, because a model calls the hidden
# tool to go back as well as forward, and the retrieval agent has no next step.
HIDDEN_TRANSFER_TURN_END = (
    "I couldn't hand you over to another step from here, so I stopped. "
    "Please tell me how you would like to continue."
)
# Any other mix. "couldn't run" is true of all three kinds.
STUCK_TURN_END = (
    "I kept making tool calls that couldn't run, so I stopped. "
    "Please tell me how you would like to continue."
)

_PROGRESS = "progress"  # a tool ran and succeeded in the reply being answered
_CONSECUTIVE = "consecutive"  # stuck replies since a tool last succeeded
_TOTAL = "total"  # stuck replies this turn
_NOT_ONLY_TRANSFERS = "not_only_transfers"  # a stuck reply held another kind
_STOPPED = "stopped"  # the cap ended this agent's run


def _key(item: str, agent_name: str) -> str:
    return f"temp:rejected_call_{item}:{agent_name}"


def _pending(kind: StuckKind) -> str:
    return f"pending_{kind.value}"


def is_adk_rejection(tool_response: Any) -> bool:
    """Whether an after-tool result is ADK's own rejection of the call.

    ADK's shape is exactly one key, "error", holding a string. A tool's own
    result never has it: tools return tool_error (status plus error_message),
    tests/unit/test_no_bare_error_results.py keeps literal "error" keys out of
    the package, and a value the model chose -- a plan keyed by label, where
    "error" is a valid label -- holds a rule dict there, never a string.
    """
    return (
        isinstance(tool_response, Mapping)
        and len(tool_response) == 1
        and isinstance(tool_response.get("error"), str)
    )


def mark_stuck(tool_context: Any, kind: StuckKind) -> None:
    """Mark the reply being answered as holding a rejected call of this kind.

    One flag per kind, only ever set to True, so parallel calls in one reply
    never overwrite each other's mark.
    """
    tool_context.state[_key(_pending(kind), tool_context.agent_name)] = True


def mark_unknown_tool(
    tool: Any, args: dict[str, Any], tool_context: Any, error: Exception
) -> Optional[dict[str, Any]]:
    """on_tool_error_callback: mark a call to a tool name that did not resolve.

    ADK runs these callbacks for an unknown name and for a real tool that
    raised. Only the first is ADK's stand-in, a bare BaseTool (every real tool
    is a subclass), so a real tool's ValueError is not miscounted.

    Always returns None: ADK's own reply, which lists the agent's real tools,
    answers an unknown name, and a real tool's exception re-raises as before.
    The parameter NAMES are load-bearing: ADK passes tool=, args=,
    tool_context= and error= by keyword.
    """
    del args, error  # Part of ADK's keyword contract.
    if type(tool) is BaseTool:
        mark_stuck(tool_context, StuckKind.UNKNOWN_TOOL)
    return None


def mark_tool_outcome(
    tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any
) -> Optional[dict[str, Any]]:
    """after_tool_callback: mark ADK's rejection, or a tool that succeeded.

    A tool's own error (is_error) marks nothing, and nor does the refusal of a
    hidden transfer: it is a tool_error, and marks itself. Anything else ran
    and succeeded -- including None (the coordinator's real
    transfer_to_agent), {} (make_finished) and AgentTool's text.

    Only marks, never answers: returns None, so the response stands. The
    parameter NAMES are load-bearing: ADK passes tool=, args=, tool_context=
    and tool_response= by keyword.
    """
    del tool, args  # Part of ADK's keyword contract.
    if is_adk_rejection(tool_response):
        mark_stuck(tool_context, StuckKind.ADK_REJECTED)
    elif not is_error(tool_response):
        tool_context.state[_key(_PROGRESS, tool_context.agent_name)] = True
    return None


def _take(state: Any, item: str, agent: str) -> bool:
    """Read a mark and clear it if it was set."""
    key = _key(item, agent)
    if not state.get(key):
        return False
    state[key] = False
    return True


def end_turn_past_cap(callback_context: Any, llm_request: Any) -> Optional[LlmResponse]:
    """before_model_callback: add up the last reply's marks; at the cap, reply
    to the user instead of calling the model.

    Must be first among an agent's before-model callbacks (agent_guards.py):
    when it answers, ADK skips the callbacks after it, the model call and
    every after-model callback (flows/llm_flows/core/_model_call.py,
    call_llm_async). That last part keeps a capped critic's stop text away
    from record_critic_verdict.

    A reply with no function calls ends the run, so the user gets an answer,
    and every call in history already has its reply. Records the stop for
    was_stopped.
    """
    del llm_request  # Part of ADK's keyword contract.
    state, agent = callback_context.state, callback_context.agent_name
    kinds = [kind for kind in StuckKind if _take(state, _pending(kind), agent)]
    progress = _take(state, _PROGRESS, agent)
    if kinds:
        total_key = _key(_TOTAL, agent)
        state[total_key] = state.get(total_key, 0) + 1
        if any(kind is not StuckKind.HIDDEN_TRANSFER for kind in kinds):
            state[_key(_NOT_ONLY_TRANSFERS, agent)] = True
    consecutive_key = _key(_CONSECUTIVE, agent)
    if progress:
        if state.get(consecutive_key):
            state[consecutive_key] = 0
    elif kinds:
        state[consecutive_key] = state.get(consecutive_key, 0) + 1
    consecutive = state.get(consecutive_key, 0)
    total = state.get(_key(_TOTAL, agent), 0)
    if (
        consecutive < MAX_CONSECUTIVE_STUCK_REPLIES
        and total < MAX_STUCK_REPLIES_PER_TURN
    ):
        return None
    state[_key(_STOPPED, agent)] = True
    logger.warning(
        "%s made %d stuck replies this turn (%d since a tool last succeeded); "
        "last stuck reply: %s; every stuck reply a hidden transfer: %s; "
        "ending its run",
        agent,
        total,
        consecutive,
        [kind.value for kind in kinds],
        not state.get(_key(_NOT_ONLY_TRANSFERS, agent)),
    )
    text = (
        STUCK_TURN_END
        if state.get(_key(_NOT_ONLY_TRANSFERS, agent))
        else HIDDEN_TRANSFER_TURN_END
    )
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def was_stopped(state: Any, agent_name: str) -> bool:
    """Whether the cap ended this agent's run in the current invocation.

    For a composite that runs agents in steps and must not trust a capped
    step's text: the schema refinement loop's stop-check reads it. temp:
    values are applied to the in-memory session as they are written
    (sessions/base_session_service.py), so a later step of the same
    invocation sees them.
    """
    return bool(state.get(_key(_STOPPED, agent_name)))
