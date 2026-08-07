"""Unit tests for the callback that removes ADK's injected transfer tool.

Builds real LlmRequest objects with the real transfer tool rather than fakes,
because the whole point of this callback is coupling to the shape ADK
produces -- a fake shaped the way we assume would hide exactly the drift the
tests exist to catch.

Fixtures assemble requests via BaseTool.process_llm_request (base_tool.py:89-
125), the literal same call agent_transfer.py makes to inject the transfer
tool -- NOT via LlmRequest.append_tools, which has no lazy-init guard for
config.tools and raises AttributeError against a bare LlmRequest() under the
pinned google-adk 1.10.0. process_llm_request also does more than lazy-init:
it calls _find_tool_with_function_declarations and, if a Tool with
declarations already exists, merges the new declaration into that same shared
object instead of creating a new one. So a real production request carries
ONE merged Tool holding every declaration, whereas append_tools would create a
fresh Tool per call. strip_transfer_to_agent is correct under either shape (it
filters per-declaration), but only the merged shape is what ADK actually
produces, and these tests exist specifically to couple to the shape ADK
produces.

Fixture helpers are async (process_llm_request is a coroutine) and are driven
with asyncio.run(...) from inside otherwise-synchronous test functions, this
repo's established pattern for calling async ADK APIs from sync tests.
"""
import asyncio

from google.adk.models.llm_request import LlmRequest
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.transfer_to_agent_tool import transfer_to_agent

from agentic_kg.common.adk_transfer import (
    TRANSFER_TOOL_NAME,
    strip_transfer_to_agent,
)


def keep_me(query: str) -> str:
    """A stand-in for the agent's own tools, which must survive the strip."""
    return query


# The real block ADK appends, with the parent's name interpolated -- there is
# no fixed literal to match against, which is why the strip is prefix-based.
# Verbatim output of agent_transfer._build_target_agents_instructions for one
# transfer target and a parent (confirmed by calling it directly against a
# fake agent tree) -- the brief's original abbreviated version dropped the
# middle "If another agent is better..." paragraph, the only place the literal
# tool name `transfer_to_agent` appears in the instruction text, and the only
# source of the "the function call." ending in _TRANSFER_INSTRUCTION_ENDINGS.
_TRANSFER_INSTRUCTION = """
You have a list of other agents to transfer to:

Agent name: schema_proposal_agent_coordinator
Agent description: proposes a schema

If you are the best to answer the question according to your description, you
can answer it.

If another agent is better for answering the question according to its
description, call `transfer_to_agent` function to transfer the
question to that agent. When transferring, do not generate any text other than
the function call.

Your parent agent is kg_construction_agent_v1. If neither the other agents nor
you are best for answering the question according to the descriptions, transfer
to your parent agent.
"""


def _declaration_names(request):
    names = []
    for tool in (request.config.tools or []):
        for declaration in (getattr(tool, "function_declarations", None) or []):
            names.append(declaration.name)
    return names


async def _request_as_adk_builds_it():
    """Mirrors the real assembly order: the agent's own tools and instruction
    first, then agent_transfer's block last (auto_flow.py:44 appends its
    processor after every SingleFlow processor)."""
    request = LlmRequest()
    request.append_instructions(["You are an expert at knowledge graph construction."])
    await FunctionTool(func=keep_me).process_llm_request(
        tool_context=None, llm_request=request
    )
    request.append_instructions([_TRANSFER_INSTRUCTION])
    await FunctionTool(func=transfer_to_agent).process_llm_request(
        tool_context=None, llm_request=request
    )
    return request


def test_the_fixture_actually_contains_the_tool_negative_control():
    """Without this, every assertion below could pass because the fixture
    never produced the tool in the first place."""
    request = asyncio.run(_request_as_adk_builds_it())
    assert TRANSFER_TOOL_NAME in request.tools_dict
    assert TRANSFER_TOOL_NAME in _declaration_names(request)
    assert TRANSFER_TOOL_NAME in request.config.system_instruction


def test_strip_removes_the_tool_from_the_dispatch_table():
    """tools_dict is what functions.py looks the call up in. Leaving it here
    means ADK will happily run a call the model made from memory."""
    request = asyncio.run(_request_as_adk_builds_it())
    strip_transfer_to_agent(None, request)
    assert TRANSFER_TOOL_NAME not in request.tools_dict


def test_strip_removes_the_declaration_sent_to_the_provider():
    """config.tools is the schema the model actually receives. Popping only
    tools_dict would still offer the model the door -- one layer deeper than
    the one this ticket started with."""
    request = asyncio.run(_request_as_adk_builds_it())
    strip_transfer_to_agent(None, request)
    assert TRANSFER_TOOL_NAME not in _declaration_names(request)


def test_strip_removes_the_instruction_advertising_it():
    request = asyncio.run(_request_as_adk_builds_it())
    strip_transfer_to_agent(None, request)
    assert TRANSFER_TOOL_NAME not in request.config.system_instruction
    assert "other agents to transfer to" not in request.config.system_instruction


def test_strip_leaves_the_agents_own_tool_and_instruction_alone():
    """Catches an over-broad strip that clears config.tools wholesale or
    truncates the system instruction from the wrong place."""
    request = asyncio.run(_request_as_adk_builds_it())
    strip_transfer_to_agent(None, request)
    assert "keep_me" in request.tools_dict
    assert "keep_me" in _declaration_names(request)
    assert "expert at knowledge graph construction" in request.config.system_instruction


def test_strip_keeps_instructions_that_land_after_the_block():
    """Catches a removal that truncates to the end of the string instead of
    bounding itself to the block.

    The block is last today only because no tool on either gated agent appends
    to system_instruction. _preprocess_async (base_llm_flow.py:374-405) runs
    the request processors first -- agent_transfer last among them -- and THEN
    each tool's process_llm_request in a separate loop, so a toolset added
    later could legitimately append after it. An unbounded strip would delete
    that silently: no error, no warning, and the prefix-not-found fallback
    would not fire because the prefix matched fine.
    """
    request = asyncio.run(_request_as_adk_builds_it())
    request.append_instructions(["Some later tool's own instructions."])

    strip_transfer_to_agent(None, request)

    assert TRANSFER_TOOL_NAME not in request.config.system_instruction
    assert "other agents to transfer to" not in request.config.system_instruction
    assert "Some later tool's own instructions." in request.config.system_instruction
    assert "expert at knowledge graph construction" in request.config.system_instruction


async def _request_with_no_transfer_tool():
    request = LlmRequest()
    request.append_instructions(["You are an expert at knowledge graph construction."])
    await FunctionTool(func=keep_me).process_llm_request(
        tool_context=None, llm_request=request
    )
    return request


def test_strip_is_a_no_op_when_the_tool_was_never_injected():
    """The callback runs on every model call, including ones ADK never added
    a transfer tool to. It must not corrupt those."""
    request = asyncio.run(_request_with_no_transfer_tool())
    before = request.config.system_instruction

    strip_transfer_to_agent(None, request)

    assert "keep_me" in request.tools_dict
    assert _declaration_names(request) == ["keep_me"]
    assert request.config.system_instruction == before


def test_the_parameter_names_are_the_ones_adk_passes():
    """ADK invokes before_model_callback purely by keyword
    (base_llm_flow.py:661), so a rename fails at request time with a
    TypeError rather than at import. Same guard adk_context.py carries."""
    import inspect

    parameters = list(inspect.signature(strip_transfer_to_agent).parameters)
    assert parameters == ["callback_context", "llm_request"]
