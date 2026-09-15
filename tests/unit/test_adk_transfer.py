"""Unit tests for the callback that removes ADK's injected transfer tool.

Builds real LlmRequest objects with the real transfer tool and ADK's own
transfer-instruction builder rather than fakes or copied text, because the
whole point of this callback is coupling to the shape ADK produces -- a fixture
shaped the way we assume would hide exactly the drift the tests exist to catch.
An earlier version pinned a hand-copied instruction string; ADK 1.28 reworded
the real block, the strip fell back to its drift path, and every test here kept
passing against the stale copy.

Fixtures assemble requests the way agent_transfer's request processor does:
append the text from `_build_transfer_instructions`, then call the transfer
tool's `process_llm_request`, so the request carries whatever tool and
declaration shape ADK itself produces.

Fixture helpers are async (process_llm_request is a coroutine) and are driven
with asyncio.run(...) from inside otherwise-synchronous test functions, this
repo's established pattern for calling async ADK APIs from sync tests.
"""

import asyncio
import logging
from types import SimpleNamespace

from google.adk.flows.llm_flows.agent_transfer import _build_transfer_instructions
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool

from agentic_kg.common.adk_transfer import (
    TRANSFER_TOOL_NAME,
    _without_transfer_block,
    strip_transfer_to_agent,
)

_PARENT = "kg_construction_agent_v1"


def keep_me(query: str) -> str:
    """A stand-in for the agent's own tools, which must survive the strip."""
    return query


def _transfer_instruction(*, parent=_PARENT, description="proposes a schema"):
    """ADK's real transfer block for one peer, optionally with a parent.

    SimpleNamespace stands in for agents: the builder reads only `.name`,
    `.description`, `.parent_agent` and `.disallow_transfer_to_parent`.
    """
    agent = SimpleNamespace(
        name="graph_construction_agent_v1",
        parent_agent=SimpleNamespace(name=parent) if parent else None,
        disallow_transfer_to_parent=False,
    )
    peer = SimpleNamespace(
        name="schema_proposal_agent_coordinator", description=description
    )
    return _build_transfer_instructions(TRANSFER_TOOL_NAME, agent, [peer])


def _declaration_names(request):
    names = []
    for tool in request.config.tools or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            names.append(declaration.name)
    return names


def _drift_warnings(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "agentic_kg.common.adk_transfer" and r.levelno >= logging.WARNING
    ]


async def _request_as_adk_builds_it(**instruction_kwargs):
    """Mirrors the real assembly order: the agent's own tools and instruction
    first, then agent_transfer's block last (AutoFlow appends its processor
    after every SingleFlow processor)."""
    request = LlmRequest()
    request.append_instructions(["You are an expert at knowledge graph construction."])
    await FunctionTool(func=keep_me).process_llm_request(
        tool_context=None, llm_request=request
    )
    request.append_instructions([_transfer_instruction(**instruction_kwargs)])
    await TransferToAgentTool(
        agent_names=["schema_proposal_agent_coordinator", _PARENT]
    ).process_llm_request(tool_context=None, llm_request=request)
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
    assert "**NOTE**" not in request.config.system_instruction


def test_real_adk_wording_is_stripped_without_any_drift_warning(caplog):
    """The alarm that used to be log-only. Every strip path below falls back
    silently-but-for-a-warning when ADK rewords the block, and the end result
    can still look right (the block is last today). Asserting on the warnings
    themselves, against ADK's own builder, turns a wording change into a
    failing test instead of a log line nobody reads."""
    caplog.set_level(logging.WARNING)
    for kwargs in ({}, {"parent": None}):
        request = asyncio.run(_request_as_adk_builds_it(**kwargs))
        request.append_instructions(["Some later tool's own instructions."])
        strip_transfer_to_agent(None, request)
        assert request.config.system_instruction == (
            "You are an expert at knowledge graph construction.\n\n"
            "Some later tool's own instructions."
        ), kwargs
    assert _drift_warnings(caplog) == []


def test_drift_is_still_reported(caplog):
    """Negative control for the test above: an opening line with no
    recognisable closing line must still warn, or an empty warning list would
    prove nothing."""
    caplog.set_level(logging.WARNING)
    reworded = (
        "Own instruction.\n\n"
        "You have a list of other agents to transfer to:\n\n"
        "Use `transfer_to_agent` when you are done."
    )
    assert _without_transfer_block(reworded) == "Own instruction."
    assert any("wording may have changed" in m for m in _drift_warnings(caplog))


def test_a_reworded_note_paragraph_is_reported(caplog):
    """The trailing paragraphs are optional to the matcher, so a reworded one is
    left behind rather than failing a bound. It still quotes the tool name,
    which is what the final check warns on."""
    caplog.set_level(logging.WARNING)
    reworded = _transfer_instruction().replace(
        "**NOTE**: the only available agents", "**NOTE**: agents available"
    )
    result = _without_transfer_block(reworded)
    assert "**NOTE**: agents available" in result
    assert any("still quoted" in m for m in _drift_warnings(caplog))


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

    The block is last today only because no tool on any gated agent appends to
    system_instruction. _preprocess_async runs the request processors first --
    agent_transfer last among them -- and THEN each tool's process_llm_request
    in a separate loop, so a toolset added later could legitimately append
    after it. An unbounded strip would delete that silently.
    """
    request = asyncio.run(_request_as_adk_builds_it())
    request.append_instructions(["Some later tool's own instructions."])

    strip_transfer_to_agent(None, request)

    assert TRANSFER_TOOL_NAME not in request.config.system_instruction
    assert "other agents to transfer to" not in request.config.system_instruction
    assert "Some later tool's own instructions." in request.config.system_instruction
    assert "expert at knowledge graph construction" in request.config.system_instruction


def test_strip_keeps_a_later_instruction_containing_the_base_ending_marker():
    """The sharper version of the test above.

    That one's trailing text contains no marker, so it passes against a
    removal bounded with a backward search -- which takes the LAST occurrence
    of a marker anywhere after the block rather than the first, and so deletes
    everything from the block's opening line through a later, unrelated
    sentence. "the function call." is ordinary enough English for a future
    toolset to write, and the deletion is silent: no error, no warning.
    """
    request = asyncio.run(_request_as_adk_builds_it())
    request.append_instructions(
        ["Later toolset: when you are done, emit the function call."]
    )

    strip_transfer_to_agent(None, request)

    assert TRANSFER_TOOL_NAME not in request.config.system_instruction
    assert "other agents to transfer to" not in request.config.system_instruction
    assert "Later toolset: when you are done" in request.config.system_instruction
    assert "expert at knowledge graph construction" in request.config.system_instruction


def test_strip_keeps_a_later_instruction_containing_the_parent_marker():
    """Same trap, the parent paragraph. It only extends the removal when it
    immediately follows the block and opens with ADK's own words. A later
    sentence that merely talks about the parent agent must not be swallowed."""
    request = asyncio.run(_request_as_adk_builds_it())
    request.append_instructions(
        ["Later toolset: when stuck, transfer to your parent agent."]
    )

    strip_transfer_to_agent(None, request)

    assert TRANSFER_TOOL_NAME not in request.config.system_instruction
    assert "other agents to transfer to" not in request.config.system_instruction
    assert (
        "Later toolset: when stuck, transfer to your parent agent."
        in request.config.system_instruction
    )
    assert "expert at knowledge graph construction" in request.config.system_instruction


def test_strip_still_removes_the_parent_paragraph_it_is_meant_to():
    """The negative control for the test above: a bound narrowed until it stops
    at the body's ending would pass that test while leaving ADK's real parent
    paragraph in the instruction."""
    request = asyncio.run(_request_as_adk_builds_it())
    assert _PARENT in request.config.system_instruction  # negative control

    strip_transfer_to_agent(None, request)

    assert "transfer to your parent agent" not in request.config.system_instruction
    assert _PARENT not in request.config.system_instruction


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
    (_handle_before_model_callback in base_llm_flow.py), so a rename fails at
    request time with a TypeError rather than at import. Same guard
    adk_context.py carries."""
    import inspect

    parameters = list(inspect.signature(strip_transfer_to_agent).parameters)
    assert parameters == ["callback_context", "llm_request"]


def test_a_peer_description_cannot_shadow_the_blocks_end_marker():
    """A peer whose description ends "...the function call." sits BEFORE the
    block's own closing line, because ADK interpolates every target's
    description into the block ahead of the fixed text. Searching the end
    marker from the block's start would match the description first and stop
    the removal mid-block -- leaving the whole `transfer_to_agent`
    advertisement in the instruction.

    Anchoring the search at the block's first fixed sentence skips every
    description. This is a regression test: search the ending from the prefix
    instead of the body anchor and it fails.
    """
    request = asyncio.run(
        _request_as_adk_builds_it(description="Formats the function call.")
    )
    assert TRANSFER_TOOL_NAME in request.config.system_instruction  # negative control

    strip_transfer_to_agent(None, request)

    instruction = request.config.system_instruction
    assert TRANSFER_TOOL_NAME not in instruction
    assert "You have a list of other agents to transfer to:" not in instruction
    assert "schema_proposal_agent_coordinator" not in instruction
    assert "transfer to your parent agent" not in instruction
    # ...and the agent's own instruction is still there.
    assert instruction == "You are an expert at knowledge graph construction."
