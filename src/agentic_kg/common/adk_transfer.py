# src/agentic_kg/common/adk_transfer.py
"""Remove ADK's injected agent-transfer tool from a gated agent's request.

ADK gives every LlmAgent with a parent or peers a `transfer_to_agent` tool and
a system-instruction block advertising it (`agent_transfer.py`). That tool does
not consult the handoff gates, so a gated agent could leave its phase through
it with the confirmation flag still unset -- the exact defect the gates exist
to prevent.

The obvious fix, `disallow_transfer_to_parent`, is deliberately NOT used. That
flag also turns off phase stickiness: `Runner._find_agent_to_run`
(runners.py:474-489) reads it through `_is_transferable_across_agent_tree`
(492-510) when choosing who handles each NEW user message, so setting it sends
every in-phase follow-up question back through the coordinator to be
re-arbitrated. A multi-question window is what the construction phase is for.
Stripping the request instead leaves the flag unset, and `_find_agent_to_run`
never inspects request contents.

The cost is coupling: we remove something ADK built, so we depend on the shape
it built it in -- a marker phrase in an interpolated instruction block, and the
layout of config.tools. That is why this is tested against real LlmRequest
objects and asserted end-to-end on what reaches the model. A google-adk 2.x
upgrade should expect to rewrite this module.

Known gap, deliberately not closed: the live (bidi-streaming) path never runs
this. run_live (base_llm_flow.py:70-79) does call _preprocess_async, so it DOES
inject the transfer tool, but it never reaches _handle_before_model_callback
(base_llm_flow.py:560), which only _call_llm_async on the run_async path calls
-- so neither this strip nor drop_foreign_context applies under run_live. That
is unreachable here because LiteLlm does not override BaseLlm.connect
(base_llm.py:118), which raises NotImplementedError, so no agent in this tree
can run live at all.
"""
import logging
from typing import Any, Optional

from google.adk.models.llm_response import LlmResponse

logger = logging.getLogger(__name__)

TRANSFER_TOOL_NAME = "transfer_to_agent"

# The block agent_transfer.py:85-107 appends. Its body interpolates the parent
# agent's name and every peer's description, so there is no fixed literal for
# the whole thing -- this is the invariant opening line.
_TRANSFER_INSTRUCTION_PREFIX = "You have a list of other agents to transfer to:"

# ...and this is its closing line. The base block (agent_transfer.py:85-99)
# always ends with it. Quoted mid-line because ADK's source hard-wraps the
# surrounding sentence; this fragment sits on a line of its own and so survives
# the wrapping.
#
# The end boundary is needed because the block is NOT reliably the last thing
# in the system instruction. _preprocess_async (base_llm_flow.py:374-405) runs
# every request processor first -- agent_transfer last among them
# (auto_flow.py:44) -- and THEN, in a separate loop, each of the agent's own
# tools' process_llm_request. No tool on either gated agent appends to the
# system instruction today, so nothing currently lands after the block. A
# future toolset that did would have its own legitimate instructions silently
# deleted by a truncate-to-end-of-string removal.
_TRANSFER_INSTRUCTION_ENDING = "the function call."

# Everything between the opening line and this sentence is interpolated: it is
# one "Agent name: ... / Agent description: ..." pair per transfer target
# (agent_transfer.py:88-90), and those descriptions are author-written prose
# from each agent's own `description=`. So the ending marker above must NOT be
# searched from the block's start -- a peer whose description happened to end
# "...formats the function call." would match first, and the removal would stop
# mid-block, leaving the advertisement in place with neither drift warning
# firing. This sentence is the first FIXED text after the interpolated region
# (agent_transfer.py:92-93), so searching for the ending from here instead
# skips every description.
_TRANSFER_INSTRUCTION_BODY_ANCHOR = (
    "If you are the best to answer the question according to your description"
)

# When the agent has a parent, agent_transfer.py:101-106 appends one more
# paragraph immediately after the base block, and that paragraph is what
# actually ends the thing we want removed.
#
# Both markers are matched FORWARD from the block's start, never backward: an
# rfind would jump to the LAST occurrence anywhere later in the instruction, so
# any future toolset instruction happening to contain one of these ordinary
# English fragments would be swallowed whole, silently -- the exact failure the
# two-ended bound exists to prevent.
#
# Extending past the base ending is likewise conditional, not unconditional:
# the addendum is only recognised when the text between the two markers really
# is ADK's addendum, i.e. it opens with the addendum's own first words. A later
# instruction ending "...hand control back to your parent agent." does not
# qualify, and the removal stops at the base ending instead.
_TRANSFER_PARENT_ADDENDUM_OPENING = "Your parent agent is"
_TRANSFER_PARENT_ADDENDUM_ENDING = "to your parent agent."


def strip_transfer_to_agent(callback_context: Any, llm_request: Any) -> Optional[LlmResponse]:
    """Remove the injected transfer tool from the request, in place.

    The parameter NAMES are load-bearing: ADK invokes this purely by keyword,
    as callback(callback_context=..., llm_request=...) (base_llm_flow.py:661).
    Renaming either one fails at request time with a TypeError, not at import.

    Always returns None so ADK proceeds with the (now stripped) request. The
    Optional[LlmResponse] annotation documents ADK's contract rather than this
    function's behaviour: returning an LlmResponse here would short-circuit the
    model call entirely and send that response back as the turn's output, which
    is never what a strip wants.

    All three surfaces matter. tools_dict is ADK's dispatch table, so removing
    it makes a call the model remembers from an earlier turn a hard error
    (functions.py:565-568) rather than a working exit. config.tools is the
    schema the provider actually receives, so leaving it would keep offering
    the model the tool. system_instruction is where ADK tells the model the
    tool exists at all.
    """
    was_injected = llm_request.tools_dict.pop(TRANSFER_TOOL_NAME, None) is not None

    config = getattr(llm_request, "config", None)
    if config is None:
        return None

    kept = []
    for tool in (config.tools or []):
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


def _without_transfer_block(instruction: str, tool_was_injected: bool = True) -> str:
    """Cut out the injected transfer block, and only it.

    Bounded at both ends deliberately, and both bounds are searched FORWARD
    from the block's start. Removing from the prefix to the end of the string
    would be correct today and would silently swallow anything a future tool
    appended after it; so, just as silently, would taking the LAST occurrence
    of an ending marker rather than the first.
    """
    start = instruction.find(_TRANSFER_INSTRUCTION_PREFIX)
    if start != -1 and not tool_was_injected:
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
    if start == -1:
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

    # Skip the interpolated agent descriptions before looking for the ending.
    # If the anchor is missing, ADK's wording has drifted; searching from the
    # block start is the old, description-sensitive behaviour, which is still
    # better than giving up -- but say so.
    body = instruction.find(_TRANSFER_INSTRUCTION_BODY_ANCHOR, start)
    if body == -1:
        logger.warning(
            "found the transfer block's opening line but not its fixed body "
            "sentence -- falling back to searching the end marker from the "
            "block start, which an agent description can shadow; ADK's wording "
            "may have changed",
        )
        body = start
    base_ending = instruction.find(_TRANSFER_INSTRUCTION_ENDING, body)
    if base_ending == -1:
        end = -1
    else:
        end = base_ending + len(_TRANSFER_INSTRUCTION_ENDING)
        addendum_ending = instruction.find(_TRANSFER_PARENT_ADDENDUM_ENDING, end)
        if addendum_ending != -1:
            between = instruction[end:addendum_ending].lstrip()
            if between.startswith(_TRANSFER_PARENT_ADDENDUM_OPENING):
                # Really is agent_transfer.py:101-106's parent paragraph, which
                # follows the base block immediately. Anything else sitting
                # between the two markers means the second marker belongs to
                # someone else's instruction, and the removal stops short.
                end = addendum_ending + len(_TRANSFER_PARENT_ADDENDUM_ENDING)
    if end == -1:
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

    remainder = instruction[end:].lstrip("\n")
    head = instruction[:start].rstrip()
    if not remainder:
        return head
    return f"{head}\n\n{remainder}" if head else remainder
