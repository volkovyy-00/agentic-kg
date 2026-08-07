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
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

TRANSFER_TOOL_NAME = "transfer_to_agent"

# The block agent_transfer.py:85-107 appends. Its body interpolates the parent
# agent's name and every peer's description, so there is no fixed literal for
# the whole thing -- this is the invariant opening line.
_TRANSFER_INSTRUCTION_PREFIX = "You have a list of other agents to transfer to:"

# ...and these are its two possible closing lines: the base block always ends
# with the first, and the parent addendum (agent_transfer.py:101-106), when
# present, ends with the second. Both are quoted mid-line because ADK's source
# hard-wraps the surrounding sentences; these fragments sit on a line of their
# own and so survive the wrapping.
#
# The end boundary is needed because the block is NOT reliably the last thing
# in the system instruction. _preprocess_async (base_llm_flow.py:374-405) runs
# every request processor first -- agent_transfer last among them
# (auto_flow.py:44) -- and THEN, in a separate loop, each of the agent's own
# tools' process_llm_request. No tool on either gated agent appends to the
# system instruction today, so nothing currently lands after the block. A
# future toolset that did would have its own legitimate instructions silently
# deleted by a truncate-to-end-of-string removal.
_TRANSFER_INSTRUCTION_ENDINGS = (
    "the function call.",
    "to your parent agent.",
)


def strip_transfer_to_agent(callback_context: Any, llm_request: Any) -> Optional[None]:
    """Remove the injected transfer tool from the request, in place.

    The parameter NAMES are load-bearing: ADK invokes this purely by keyword,
    as callback(callback_context=..., llm_request=...) (base_llm_flow.py:661).
    Renaming either one fails at request time with a TypeError, not at import.

    Returns None so ADK proceeds with the (now stripped) request; a non-None
    return would short-circuit the model call entirely.

    All three surfaces matter. tools_dict is ADK's dispatch table, so removing
    it makes a call the model remembers from an earlier turn a hard error
    (functions.py:565-568) rather than a working exit. config.tools is the
    schema the provider actually receives, so leaving it would keep offering
    the model the tool. system_instruction is where ADK tells the model the
    tool exists at all.
    """
    llm_request.tools_dict.pop(TRANSFER_TOOL_NAME, None)

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
        config.system_instruction = _without_transfer_block(instruction)

    return None


def _without_transfer_block(instruction: str) -> str:
    """Cut out the injected transfer block, and only it.

    Bounded at both ends deliberately. Removing from the prefix to the end of
    the string would be correct today and would silently swallow anything a
    future tool appended after it -- see _TRANSFER_INSTRUCTION_ENDINGS.
    """
    start = instruction.find(_TRANSFER_INSTRUCTION_PREFIX)
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

    end = -1
    for ending in _TRANSFER_INSTRUCTION_ENDINGS:
        found = instruction.rfind(ending, start)
        if found != -1:
            end = max(end, found + len(ending))
    if end == -1:
        # Opening line matched but neither closing line did: ADK's wording has
        # drifted. Fall back to removing everything from the prefix, which is
        # what this did before it was bounded -- losing a trailing instruction
        # is worse than leaving the door advertised, so warn loudly.
        logger.warning(
            "found the transfer block's opening line but neither closing "
            "line -- removing to end of instruction, which may discard other "
            "tools' instructions; ADK's wording may have changed",
        )
        return instruction[:start].rstrip()

    remainder = instruction[end:].lstrip("\n")
    head = instruction[:start].rstrip()
    if not remainder:
        return head
    return f"{head}\n\n{remainder}" if head else remainder
