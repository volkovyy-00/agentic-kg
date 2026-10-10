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

from google.adk.flows.llm_flows.agent_transfer import (
    _build_transfer_instructions,
    _get_transfer_targets,
)
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool
from google.genai import types

from agentic_kg.common.adk_context import drop_foreign_context
from agentic_kg.common.adk_transfer import (
    HIDDEN_TRANSFER_TURN_END,
    MAX_CONSECUTIVE_HIDDEN_TRANSFER_REPLIES,
    MAX_HIDDEN_TRANSFER_REPLIES_PER_TURN,
    TRANSFER_TOOL_NAME,
    _state_key,
    _without_transfer_block,
    count_hidden_transfer_replies,
    end_turn_past_hidden_transfer_cap,
    refuse_transfer_to_agent,
    reset_on_real_progress,
    strip_transfer_to_agent,
    transfer_guard_callbacks,
)
from agentic_kg.common.tool_result import is_error, tool_error

_PARENT = "kg_construction_agent_v1"


def keep_me(query: str) -> str:
    """A stand-in for the agent's own tools, which must survive the strip."""
    return query


def _transfer_setup(*, transfer_to_parent=True, description="proposes a schema"):
    """ADK's real transfer block, and the transfer tool's agent names, for a
    gated agent under _PARENT with one peer.

    Both come from one target list, derived by ADK's own _get_transfer_targets
    exactly as its request processor derives them. Hand-built lists let the
    instruction text and the tool's agent_name enum drift apart -- a no-parent
    instruction beside an enum still offering the parent -- which no real
    request can contain.

    transfer_to_parent=False sets disallow_transfer_to_parent, the realistic
    way to get a block with no parent paragraph: without a parent at all ADK
    finds no peers either, and injects no block or tool. SimpleNamespace stands
    in for agents; ADK reads only the attributes set here.
    """
    peer = SimpleNamespace(
        name="schema_proposal_agent_coordinator", description=description
    )
    agent = SimpleNamespace(
        name="graph_construction_agent_v1",
        sub_agents=[],
        mode="chat",
        disallow_transfer_to_parent=not transfer_to_parent,
        disallow_transfer_to_peers=False,
    )
    agent.parent_agent = SimpleNamespace(
        name=_PARENT,
        description="coordinates the construction phases",
        sub_agents=[agent, peer],
        disallow_transfer_to_parent=False,
    )
    targets = _get_transfer_targets(agent)
    instruction = _build_transfer_instructions(TRANSFER_TOOL_NAME, agent, targets)
    return instruction, [target.name for target in targets]


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


async def _request_as_adk_builds_it(**setup_kwargs):
    """Mirrors the real assembly order: the agent's own tools and instruction
    first, then agent_transfer's block last (AutoFlow appends its processor
    after every SingleFlow processor)."""
    request = LlmRequest()
    request.append_instructions(["You are an expert at knowledge graph construction."])
    await FunctionTool(func=keep_me).process_llm_request(
        tool_context=None, llm_request=request
    )
    instruction, agent_names = _transfer_setup(**setup_kwargs)
    request.append_instructions([instruction])
    await TransferToAgentTool(agent_names=agent_names).process_llm_request(
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
    """tools_dict is what ADK looks the call up in (_get_tool in
    flows/llm_flows/tools/_caller.py). Leaving it here means ADK will happily
    run a call the model made from memory."""
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
    for kwargs in ({}, {"transfer_to_parent": False}):
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
    reworded, _ = _transfer_setup()
    reworded = reworded.replace(
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
    (handle_before_model_callback in flows/llm_flows/core/_finalizer.py), so a
    rename fails at request time with a TypeError rather than at import. Same
    guard adk_context.py carries."""
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


_HIDDEN = SimpleNamespace(name=TRANSFER_TOOL_NAME)


def _callback_context(state=None, agent="graph_construction_agent_v1"):
    """Stands in for both a CallbackContext and a ToolContext: the callbacks
    read only agent_name, state and (the refusal) actions."""
    return SimpleNamespace(
        agent_name=agent,
        state={} if state is None else state,
        actions=SimpleNamespace(skip_summarization=None),
    )


def _consecutive(context):
    return context.state.get(_state_key("consecutive", context.agent_name))


def _reply(*names, partial=None, text=None):
    """A model reply calling these tools, or saying something if none; with
    `text`, saying that before the calls."""
    parts = [
        types.Part(function_call=types.FunctionCall(name=name, args={}))
        for name in names
    ] or [types.Part(text="ok")]
    if text is not None and names:
        parts.insert(0, types.Part(text=text))
    return LlmResponse(
        content=types.Content(role="model", parts=parts), partial=partial
    )


def _run_replies(context, replies, failing=()):
    """Each reply as ADK runs it: the turn-end check before it, the counter
    after it, then each call's after-tool callback on its result. The hidden
    tool gets its refusal, a tool named in `failing` an error, any other tool
    a success. Returns what the turn-end check returned last."""
    ended = None
    for reply in replies:
        ended = end_turn_past_hidden_transfer_cap(context, None)
        if ended is not None:
            return ended
        count_hidden_transfer_replies(context, reply)
        for call in reply.get_function_calls():
            tool = SimpleNamespace(name=call.name)
            result = refuse_transfer_to_agent(tool, {}, context)
            if result is None:
                failed = call.name in failing
                result = tool_error("no") if failed else {"status": "success"}
            reset_on_real_progress(tool, {}, context, result)
    return end_turn_past_hidden_transfer_cap(context, None)


def test_refusal_leaves_every_other_tool_alone():
    assert refuse_transfer_to_agent(SimpleNamespace(name="finished"), {}, None) is None


def test_refusal_answers_the_hidden_tool_and_names_the_real_exit():
    """Answered before ADK's generic not-found reply, which invites a retry:
    the model is told which exit this agent really has, and keeps the turn."""
    reply = refuse_transfer_to_agent(
        _HIDDEN, {"agent_name": _PARENT}, _callback_context()
    )

    assert is_error(reply)
    assert "finished" in reply["error_message"]


def test_the_refusal_names_the_agent_and_does_not_forbid_the_tool():
    """The refusal stays in the session, and the coordinator later reads it as
    another agent's output. Its own transfer_to_agent is real, so the refusal
    names whose tool is missing and gives no bare order not to call it."""
    reply = refuse_transfer_to_agent(
        _HIDDEN, {}, _callback_context(agent="user_intent_agent_v2")
    )

    message = reply["error_message"]
    assert message.startswith("user_intent_agent_v2 cannot use transfer_to_agent")
    assert "do not call" not in message.lower()


def test_the_turn_ends_with_a_reply_to_the_user_past_the_cap():
    """The model call after the capped reply is replaced by a text reply, so
    the user gets an answer and the turn ends without another model call."""
    context = _callback_context()
    replies = [_reply(TRANSFER_TOOL_NAME)] * (
        MAX_CONSECUTIVE_HIDDEN_TRANSFER_REPLIES - 1
    )

    assert _run_replies(context, replies) is None

    ended = _run_replies(context, [_reply(TRANSFER_TOOL_NAME)])
    assert ended is not None
    assert not ended.get_function_calls()
    assert ended.content.parts[0].text == HIDDEN_TRANSFER_TURN_END


def test_a_model_that_recovers_after_each_slip_keeps_its_turn():
    """A tool that succeeds resets the count. A model that takes each refusal
    and goes back to work is not stuck, however many times it slips in a long
    turn."""
    context = _callback_context()
    slip_then_work = [_reply(TRANSFER_TOOL_NAME), _reply("read_neo4j_cypher")]

    replies = slip_then_work * MAX_CONSECUTIVE_HIDDEN_TRANSFER_REPLIES
    assert _run_replies(context, replies) is None


def test_a_reply_that_also_calls_a_real_tool_resets_the_count():
    """So the turn never ends on a real result the model has not reported."""
    context = _callback_context()
    replies = [_reply(TRANSFER_TOOL_NAME)] * (
        MAX_CONSECUTIVE_HIDDEN_TRANSFER_REPLIES - 1
    )
    replies.append(_reply(TRANSFER_TOOL_NAME, "confirm_construction_handoff"))

    assert _run_replies(context, replies) is None
    assert _consecutive(context) == 0


def test_the_per_turn_ceiling_holds_when_every_retry_is_paired_with_a_success():
    """A tool that always succeeds (confirm_construction_handoff, a schema
    read) resets the consecutive count on every reply; the per-turn ceiling
    is the one no reset clears."""
    context = _callback_context()
    paired = _reply(TRANSFER_TOOL_NAME, "confirm_construction_handoff")

    early = [paired] * (MAX_HIDDEN_TRANSFER_REPLIES_PER_TURN - 1)
    assert _run_replies(context, early) is None
    assert _run_replies(context, [paired]) is not None


def test_a_reply_that_asks_the_user_something_ends_the_turn_on_its_refusal():
    """A question and the hidden call in one reply: the refusal ends the turn,
    so the question is the user's to answer before the model runs again."""
    context = _callback_context()
    asking = _reply(TRANSFER_TOOL_NAME, text="Which products matter most?")

    count_hidden_transfer_replies(context, asking)
    refuse_transfer_to_agent(_HIDDEN, {}, context)

    assert context.actions.skip_summarization is True


def test_a_reply_that_only_calls_the_hidden_tool_keeps_the_turn():
    context = _callback_context()
    count_hidden_transfer_replies(
        context, _reply(TRANSFER_TOOL_NAME, text="Which products matter most?")
    )
    count_hidden_transfer_replies(context, _reply(TRANSFER_TOOL_NAME))
    refuse_transfer_to_agent(_HIDDEN, {}, context)

    assert not context.actions.skip_summarization


def test_retrying_both_exits_together_is_still_capped():
    """A refused `finished` beside the hidden call is not progress: a model
    that keeps sending both is the same loop, and must not run to ADK's
    500-call limit."""
    context = _callback_context()
    both = _reply(TRANSFER_TOOL_NAME, "finished")

    replies = [both] * MAX_CONSECUTIVE_HIDDEN_TRANSFER_REPLIES
    assert _run_replies(context, replies, failing={"finished"}) is not None


def test_a_text_reply_between_retries_does_not_reset_the_count():
    """A streamed reply can arrive as a text-only response followed by the
    call-only one. Counting that text as progress would let a narrating model
    retry forever."""
    context = _callback_context()
    narrated = [_reply(), _reply(TRANSFER_TOOL_NAME)]

    replies = narrated * MAX_CONSECUTIVE_HIDDEN_TRANSFER_REPLIES
    assert _run_replies(context, replies) is not None


def test_parallel_calls_in_one_reply_count_once():
    context = _callback_context()
    parallel = _reply(TRANSFER_TOOL_NAME, TRANSFER_TOOL_NAME, TRANSFER_TOOL_NAME)

    assert _run_replies(context, [parallel]) is None
    assert _consecutive(context) == 1


def test_replies_without_the_hidden_tool_and_partial_chunks_are_not_counted():
    """ADK passes streaming chunks to after-model callbacks too, and the full
    reply follows them; counting both would count one reply twice."""
    context = _callback_context()
    replies = [
        _reply("finished"),
        _reply(),
        _reply(TRANSFER_TOOL_NAME, partial=True),
        LlmResponse(),
    ]

    assert _run_replies(context, replies) is None
    assert context.state == {}


def test_each_agent_keeps_its_own_count():
    """One invocation can run two gated agents (a confirmed construction
    handoff runs retrieval inline); retrieval must not inherit construction's
    replies."""
    state = {}
    construction = _callback_context(state)
    _run_replies(
        construction,
        [_reply(TRANSFER_TOOL_NAME)] * (MAX_CONSECUTIVE_HIDDEN_TRANSFER_REPLIES - 1),
    )

    retrieval = _callback_context(state, agent="graphrag_agent_v2")
    assert _run_replies(retrieval, [_reply(TRANSFER_TOOL_NAME)]) is None


def test_the_count_lives_in_invocation_scoped_state():
    """temp: keys are dropped from the persisted delta, so each turn starts at
    zero with no reset callback and nothing lands in the session store."""
    context = _callback_context()
    count_hidden_transfer_replies(context, _reply(TRANSFER_TOOL_NAME))

    assert context.state
    assert all(key.startswith("temp:") for key in context.state)


def test_callback_parameter_names_are_the_ones_adk_passes():
    """ADK invokes every callback purely by keyword: tool=, args= and
    tool_context= for before-tool, plus tool_response= for after-tool
    (flows/llm_flows/tools/_caller.py _execute_single_prepared_call),
    callback_context= plus llm_request= or llm_response= for the model callbacks
    (flows/llm_flows/core/_finalizer.py)."""
    import inspect

    def names(callback):
        return list(inspect.signature(callback).parameters)

    assert names(refuse_transfer_to_agent) == ["tool", "args", "tool_context"]
    assert names(end_turn_past_hidden_transfer_cap) == [
        "callback_context",
        "llm_request",
    ]
    assert names(count_hidden_transfer_replies) == [
        "callback_context",
        "llm_response",
    ]
    assert names(reset_on_real_progress) == [
        "tool",
        "args",
        "tool_context",
        "tool_response",
    ]


def test_the_guard_wires_all_its_callbacks_together_or_none():
    """The strip, the context filter, the refusal and its cap only work as a
    set, so a gated agent takes them from one helper."""
    wired = transfer_guard_callbacks(True)

    assert wired == {
        "before_model_callback": [
            end_turn_past_hidden_transfer_cap,
            drop_foreign_context,
            strip_transfer_to_agent,
        ],
        "after_model_callback": count_hidden_transfer_replies,
        "before_tool_callback": refuse_transfer_to_agent,
        "after_tool_callback": reset_on_real_progress,
    }
    assert transfer_guard_callbacks(False) == {}
