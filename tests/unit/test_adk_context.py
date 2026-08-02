# tests/unit/test_adk_context.py
"""Unit tests for the graphrag foreign-context filter.

The canary test deliberately drives ADK's own _convert_foreign_event rather
than asserting on our copy of the sentinel string: asserting our constant
equals our constant proves nothing. google-adk is pinned >=1.10,<2, so a
routine `uv sync` can change that wording; this test is what notices.
"""
from google.adk.events.event import Event
from google.adk.flows.llm_flows.contents import _convert_foreign_event
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from agentic_kg.common.adk_context import (
    FOREIGN_CONTEXT_SENTINEL,
    drop_foreign_context,
)


def _content(role, *parts):
    return types.Content(role=role, parts=list(parts))


def _request(*contents):
    return LlmRequest(contents=list(contents))


def _human_turn():
    """A real user message, which must always survive the filter.

    Several tests need one purely so the request is not ALL foreign: the
    all-foreign guard deliberately leaves such a request untouched, so a
    drop-test built from foreign content alone would assert nothing.
    """
    return _content("user", types.Part(text="which countries dominate sourcing?"))


def test_drops_content_carrying_the_sentinel():
    foreign = _content("user", types.Part(text=FOREIGN_CONTEXT_SENTINEL),
                       types.Part(text="[other_agent] said: 4 suppliers are orphaned"))
    human = _human_turn()
    req = _request(foreign, human)
    drop_foreign_context(None, req)
    assert req.contents == [human]


def test_all_foreign_request_is_left_unfiltered_rather_than_emptied():
    """Would catch: filtering a request down to zero contents.

    Most model backends reject an empty `contents` outright, which surfaces as
    an unhandled exception mid-turn -- the failure mode the rest of this
    codebase works to avoid. Leaving the request unfiltered is the deliberate
    lesser harm. Unreachable through the coordinator, where a real user turn
    always survives; this is a guard, not a live path.
    """
    foreign = _content("user", types.Part(text=FOREIGN_CONTEXT_SENTINEL),
                       types.Part(text="[other_agent] said: anything"))
    req = _request(foreign)
    drop_foreign_context(None, req)
    assert req.contents == [foreign], "an empty contents must never be sent"


def test_keeps_real_user_message():
    human = _human_turn()
    req = _request(human)
    drop_foreign_context(None, req)
    assert req.contents == [human]


def test_keeps_own_model_turn_and_tool_parts():
    said = _content("model", types.Part(text="let me check the schema"))
    call = _content("model", types.Part(
        function_call=types.FunctionCall(name="read_neo4j_cypher", args={"query": "MATCH (n) RETURN n"})))
    resp = _content("user", types.Part(
        function_response=types.FunctionResponse(name="read_neo4j_cypher", response={"status": "success"})))
    req = _request(said, call, resp)
    drop_foreign_context(None, req)
    assert req.contents == [said, call, resp]


def test_drops_only_the_foreign_content_from_a_mixed_history():
    human = _content("user", types.Part(text="hello"))
    foreign = _content("user", types.Part(text=FOREIGN_CONTEXT_SENTINEL),
                       types.Part(text="[x] said: hi"))
    own = _content("model", types.Part(text="hi back"))
    req = _request(human, foreign, own)
    drop_foreign_context(None, req)
    assert req.contents == [human, own]


def test_survives_empty_and_none_content():
    empty = types.Content(role="user", parts=[])
    bare = types.Content(role="user")
    req = _request(empty, bare)
    drop_foreign_context(None, req)
    assert req.contents == [empty, bare]


def test_returns_none_so_the_model_call_proceeds():
    req = _request(_content("user", types.Part(text="hi")))
    assert drop_foreign_context(None, req) is None


def test_canary_adk_still_marks_foreign_events_with_our_sentinel():
    """Fails if a google-adk upgrade changes the foreign-event wording."""
    original = Event(
        author="schema_critic_agent",
        content=_content("model", types.Part(text="4 suppliers have no quote rows")),
    )
    converted = _convert_foreign_event(original)

    assert converted.content.parts[0].text == FOREIGN_CONTEXT_SENTINEL, (
        "ADK's _convert_foreign_event no longer emits our sentinel as part 0. "
        "The graphrag context filter is now a silent no-op. Check the installed "
        "google-adk version against the >=1.10,<2 pin in pyproject.toml."
    )

    # Paired with a surviving human turn: an all-foreign request is
    # deliberately left unfiltered, so the drop needs something to keep.
    human = _human_turn()
    req = _request(converted.content, human)
    drop_foreign_context(None, req)
    assert req.contents == [human]
