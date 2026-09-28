# src/agentic_kg/common/adk_transfer.py
"""Remove ADK's injected agent-transfer tool from a gated agent's request.

ADK gives every LlmAgent with a parent or peers a `transfer_to_agent` tool and
a system-instruction block advertising it (`agent_transfer.py`). That tool does
not consult the handoff gates, so a gated agent could leave its phase through
it with the confirmation flag still unset -- the exact defect the gates exist
to prevent.

The obvious fix, `disallow_transfer_to_parent`, is deliberately NOT used. That
flag also turns off phase stickiness: `Runner._find_agent_to_run` reads it
through `_is_transferable_across_agent_tree` when choosing who handles each NEW
user message, so setting it sends every in-phase follow-up question back
through the coordinator to be re-arbitrated. A multi-question window is what
the construction phase is for. Stripping the request instead leaves the flag
unset, and `_find_agent_to_run` never inspects request contents. On
google-adk 2.9 either flag also makes a blocked `finished` call raise
ValueError (`_transfer_utils.resolve_and_derive_transfer_context`), so a
`make_finished` target must be the agent's parent or a peer -- one more
reason not to set it.

The cost is coupling: we remove something ADK built, so we depend on the shape
it built it in -- marker phrases in an interpolated instruction block, and the
layout of config.tools. That is why this is tested against real LlmRequest
objects and ADK's own instruction builder, and asserted end-to-end on what
reaches the model, so a google-adk upgrade that changes either shape fails
those tests rather than this module going quietly inert.

Known gap, deliberately not closed: the live (bidi-streaming) path never
applies this. run_live_flow (flows/llm_flows/_live_llm_flow.py) calls
_preprocess_async, so it DOES inject the transfer tool, and opens the
connection with that request. It runs _handle_before_model_callback only in
screen_live_user_content, once per user message sent after the connection is
open, on a copy of the request whose contents are just that message -- so
neither this strip nor drop_foreign_context shapes the request the live
connection was opened with. That is unreachable here because LiteLlm does not
override BaseLlm.connect, which raises NotImplementedError, so no agent in
this tree can run live at all.
"""

import logging
import re
from typing import Any, Optional

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agentic_kg.common.adk_context import drop_foreign_context
from agentic_kg.common.tool_result import ToolResult, tool_error

logger = logging.getLogger(__name__)

TRANSFER_TOOL_NAME = "transfer_to_agent"

# The per-turn count of model replies that called the hidden tool, one key per
# agent. temp: state lives only for the current invocation (one per user
# message) and is never persisted, so each turn starts at zero without a reset
# callback.
HIDDEN_TRANSFER_REPLIES_KEY_PREFIX = "temp:hidden_transfer_replies:"
MAX_HIDDEN_TRANSFER_REPLIES_PER_TURN = 2

HIDDEN_TRANSFER_REFUSAL = (
    "transfer_to_agent is not available to this agent. It hands the user on "
    "only through `finished`, which says what is still missing when it cannot "
    "hand over yet. Do not call transfer_to_agent again."
)

# What the user reads when the cap ends a turn. Written for the user, not the
# model: it is the turn's last word.
HIDDEN_TRANSFER_TURN_END = (
    "I tried to move you to the next step before this one was finished, and "
    "stopped. Please tell me how you would like to continue."
)


def _phrase(text: str) -> "re.Pattern[str]":
    """A marker that tolerates any whitespace between its words.

    ADK builds the block from a hard-wrapped triple-quoted string, so where a
    sentence breaks across lines is an accident of its source formatting --
    1.28 moved "the function call." onto two lines without changing a word.
    Matching word-by-word keeps the markers about wording, not layout.
    """
    return re.compile(r"\s+".join(re.escape(word) for word in text.split()))


# The block _build_transfer_instructions appends. Its body interpolates every
# transfer target's name and description, so there is no fixed literal for the
# whole thing -- this is the invariant opening line.
_TRANSFER_INSTRUCTION_PREFIX = _phrase(
    "You have a list of other agents to transfer to:"
)

# Everything between the opening line and this sentence is interpolated: one
# "Agent name: ... / Agent description: ..." pair per transfer target, and
# those descriptions are author-written prose from each agent's own
# `description=`. So the ending marker below must NOT be searched from the
# block's start -- a peer whose description happened to end "...formats the
# function call." would match first, and the removal would stop mid-block,
# leaving the advertisement in place with no drift warning firing. This
# sentence is the first FIXED text after the interpolated region, so searching
# for the ending from here instead skips every description.
_TRANSFER_INSTRUCTION_BODY_ANCHOR = _phrase(
    "If you are the best to answer the question according to your description"
)

# ...and this closes the fixed body. The end boundary is needed because the
# block is NOT reliably the last thing in the system instruction.
# _preprocess_async runs every request processor first -- agent_transfer last
# among them (AutoFlow) -- and THEN, in a separate loop, each of the agent's
# own tools' process_llm_request. No tool on any gated agent appends to the
# system instruction today, so nothing currently lands after the block. A
# future toolset that did would have its own legitimate instructions silently
# deleted by a truncate-to-end-of-string removal.
_TRANSFER_INSTRUCTION_ENDING = _phrase("the function call.")

# Paragraphs ADK appends after the body, in this order. Each is recognised only
# when it IMMEDIATELY follows what was removed so far (whitespace only in
# between) and it opens with ADK's own words; it then extends the removal to
# its first full stop. Its interpolated tail is agent names, which ADK requires
# to be identifiers, so that full stop is the paragraph's own.
#
# Matching forward and anchored, never searching ahead: a later toolset
# instruction that merely contains similar English ("...hand control back to
# your parent agent.") must not be swallowed along with everything between.
_TRANSFER_TRAILING_PARAGRAPHS = (
    # Always present: names the agents `transfer_to_agent` accepts.
    _phrase(
        f"**NOTE**: the only available agents for `{TRANSFER_TOOL_NAME}` function are"
    ),
    # Present when the agent has a parent and may transfer to it.
    _phrase(
        "If neither you nor the other agents are best for the question, "
        "transfer to your parent agent"
    ),
)


def strip_transfer_to_agent(
    callback_context: Any, llm_request: Any
) -> Optional[LlmResponse]:
    """Remove the injected transfer tool from the request, in place.

    The parameter NAMES are load-bearing: ADK invokes this purely by keyword,
    as callback(callback_context=..., llm_request=...) (_handle_before_model_callback
    in base_llm_flow.py). Renaming either one fails at request time with a
    TypeError, not at import.

    Always returns None so ADK proceeds with the (now stripped) request. The
    Optional[LlmResponse] annotation documents ADK's contract rather than this
    function's behaviour: returning an LlmResponse here would short-circuit the
    model call entirely and send that response back as the turn's output, which
    is never what a strip wants.

    All three surfaces matter. tools_dict is ADK's dispatch table, so removing
    it turns a call the model remembers from an earlier turn into a tool error
    rather than a working exit, and the agent keeps the turn. Wire it with
    transfer_guard_callbacks, which pairs it with refuse_transfer_to_agent:
    that answers such a call before ADK's generic not-found reply and caps it
    per turn.
    config.tools is the schema the provider actually receives, so leaving it
    would keep offering the model the tool. system_instruction is where ADK
    tells the model the tool exists at all.
    """
    was_injected = llm_request.tools_dict.pop(TRANSFER_TOOL_NAME, None) is not None

    config = getattr(llm_request, "config", None)
    if config is None:
        return None

    kept = []
    for tool in config.tools or []:
        declarations = getattr(tool, "function_declarations", None)
        if not declarations:
            # A built-in tool with no function declarations (search, code
            # execution). Nothing to filter; keep it as-is.
            kept.append(tool)
            continue
        remaining = [d for d in declarations if d.name != TRANSFER_TOOL_NAME]
        if not remaining:
            continue
        tool.function_declarations = remaining
        kept.append(tool)
    config.tools = kept

    instruction = config.system_instruction
    if isinstance(instruction, str):
        config.system_instruction = _without_transfer_block(
            instruction, tool_was_injected=was_injected
        )

    return None


def refuse_transfer_to_agent(
    tool: Any, args: dict[str, Any], tool_context: Any
) -> Optional[ToolResult]:
    """Answer a call to the stripped transfer tool before ADK does.

    The strip removes the tool, but a model can still call it from memory.
    ADK 2.9 would answer with build_tool_not_found_response, which invites a
    retry, and nothing but RunConfig.max_llm_calls (default 500) would stop a
    model that keeps retrying. ADK runs before-tool callbacks ahead of that
    reply (_tool_caller.py _execute_single_prepared_call), so this one answers
    first, naming this agent's real exit. Every call gets an answer, so the
    history never holds an unanswered call; the per-turn cap lives in the two
    model callbacks below.

    Returns None for every other tool, so ADK runs it as usual. The parameter
    NAMES are load-bearing: ADK passes tool=, args= and tool_context= by
    keyword.
    """
    del args  # Part of ADK's keyword contract; the refusal does not read it.
    if tool.name != TRANSFER_TOOL_NAME:
        return None
    return tool_error(HIDDEN_TRANSFER_REFUSAL)


def count_hidden_transfer_replies(
    callback_context: Any, llm_response: LlmResponse
) -> None:
    """after_model_callback: count a reply that calls the hidden tool.

    Counts replies, not calls, so parallel calls in one reply are one attempt.
    Skips partial (streaming) chunks, which ADK also passes here and which are
    followed by the full reply. Per agent, since one turn can run two gated
    agents (a confirmed construction handoff runs retrieval inline).
    """
    if llm_response.partial or llm_response.content is None:
        return None
    parts = llm_response.content.parts or []
    if any(
        part.function_call and part.function_call.name == TRANSFER_TOOL_NAME
        for part in parts
    ):
        key = HIDDEN_TRANSFER_REPLIES_KEY_PREFIX + callback_context.agent_name
        callback_context.state[key] = callback_context.state.get(key, 0) + 1
    return None


def end_turn_past_hidden_transfer_cap(
    callback_context: Any, llm_request: Any
) -> Optional[LlmResponse]:
    """before_model_callback: past the cap, reply to the user instead of the model.

    Runs before the model call that would follow the refusals. Returning a
    text response skips that call, and a reply with no function calls ends the
    turn, so the user gets an answer and every call in history already has
    its refusal. The next turn starts at zero.
    """
    del llm_request  # Part of ADK's keyword contract.
    key = HIDDEN_TRANSFER_REPLIES_KEY_PREFIX + callback_context.agent_name
    replies = callback_context.state.get(key, 0)
    if replies <= MAX_HIDDEN_TRANSFER_REPLIES_PER_TURN:
        return None
    logger.warning(
        "%s called %s in %d replies this turn; ending the turn",
        callback_context.agent_name,
        TRANSFER_TOOL_NAME,
        replies,
    )
    return LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part(text=HIDDEN_TRANSFER_TURN_END)]
        )
    )


def transfer_guard_callbacks(gated: bool) -> dict[str, Any]:
    """The callbacks a gated phase agent needs, as Agent(...) keyword args.

    They only work as a set: the strip removes the injected transfer tool,
    drop_foreign_context removes the worked example of it from history,
    refuse_transfer_to_agent answers a call made anyway, and the counter and
    the turn end cap how often a model can make one in a turn. Spread into the
    constructor (`**transfer_guard_callbacks(...)`) so no agent gets one
    without the others. An ungated agent gets none.
    """
    if not gated:
        return {}
    return {
        "before_model_callback": [
            drop_foreign_context,
            strip_transfer_to_agent,
            end_turn_past_hidden_transfer_cap,
        ],
        "after_model_callback": count_hidden_transfer_replies,
        "before_tool_callback": refuse_transfer_to_agent,
    }


def _extend_past_trailing_paragraphs(instruction: str, end: int) -> int:
    for opening in _TRANSFER_TRAILING_PARAGRAPHS:
        next_text = len(instruction) - len(instruction[end:].lstrip())
        match = opening.match(instruction, next_text)
        if match is None:
            continue
        full_stop = instruction.find(".", match.end())
        if full_stop != -1:
            end = full_stop + 1
    return end


def _without_transfer_block(instruction: str, tool_was_injected: bool = True) -> str:
    """Cut out the injected transfer block, and only it.

    Bounded at both ends deliberately, and both bounds are found FORWARD from
    the block's start. Removing from the prefix to the end of the string would
    be correct today and would silently swallow anything a future tool
    appended after it; so, just as silently, would taking the LAST occurrence
    of an ending marker rather than the first.
    """
    prefix = _TRANSFER_INSTRUCTION_PREFIX.search(instruction)
    if prefix is not None and not tool_was_injected:
        # ADK renamed the tool: the block is still being appended (its opening
        # line matched) but nothing named TRANSFER_TOOL_NAME was in tools_dict,
        # so the two tool-level strips above did nothing at all and the model
        # is still being offered a working transfer under its new name. The
        # instruction strip below still fires, which hides this from every
        # assertion phrased on the instruction text -- hence the warning.
        logger.warning(
            "the transfer instruction block was injected but no %s tool was in "
            "tools_dict -- ADK may have renamed the tool, in which case the "
            "renamed tool is NOT being stripped",
            TRANSFER_TOOL_NAME,
        )
    if prefix is None:
        if TRANSFER_TOOL_NAME in instruction:
            # ADK changed the block's opening line. The tool is gone from both
            # the dispatch table and the schema, so the model cannot call it --
            # but it is still being told about something that no longer exists,
            # and the prefix above needs updating.
            logger.warning(
                "system instruction still mentions %s but the expected block "
                "prefix was not found -- ADK's wording may have changed",
                TRANSFER_TOOL_NAME,
            )
        return instruction
    start = prefix.start()

    # Skip the interpolated agent descriptions before looking for the ending.
    # If the anchor is missing, ADK's wording has drifted; searching from the
    # block start is the old, description-sensitive behaviour, which is still
    # better than giving up -- but say so.
    body = _TRANSFER_INSTRUCTION_BODY_ANCHOR.search(instruction, start)
    if body is None:
        logger.warning(
            "found the transfer block's opening line but not its fixed body "
            "sentence -- falling back to searching the end marker from the "
            "block start, which an agent description can shadow; ADK's wording "
            "may have changed",
        )
    ending = _TRANSFER_INSTRUCTION_ENDING.search(
        instruction, start if body is None else body.start()
    )
    if ending is None:
        # Opening line matched but the closing line did not: ADK's wording has
        # drifted. Fall back to removing everything from the prefix, which is
        # what this did before it was bounded -- losing a trailing instruction
        # is worse than leaving the door advertised, so warn loudly.
        logger.warning(
            "found the transfer block's opening line but not its closing "
            "line -- removing to end of instruction, which may discard other "
            "tools' instructions; ADK's wording may have changed",
        )
        return instruction[:start].rstrip()

    end = _extend_past_trailing_paragraphs(instruction, ending.end())

    remainder = instruction[end:].lstrip("\n")
    head = instruction[:start].rstrip()
    if not remainder:
        result = head
    else:
        result = f"{head}\n\n{remainder}" if head else remainder

    if f"`{TRANSFER_TOOL_NAME}`" in result:
        # Every marker matched, yet ADK's quoted tool name survived: a paragraph
        # of the block (the NOTE naming the valid agents, today) was reworded
        # or moved, so its opening no longer matched and it was left behind.
        logger.warning(
            "removed the transfer block but `%s` is still quoted in the system "
            "instruction -- ADK's wording may have changed",
            TRANSFER_TOOL_NAME,
        )
    return result
