"""Both halves of KG-3: what reaches this agent's model, and what its
instruction says about warnings.

Asserts on what reached the model, never on what the model said. A scripted
model can be told to emit any summary at all, so a test resting on its prose
would only be asserting our own script back to us. Whether the summary really
omits the warnings section is verified by hand -- see the plan's Task 6, and
test_construction_handoff_gate.py's docstring for the same split.

The negative control carries as much weight as the positive assertion. Without
proving the sentinel DOES arrive once drop_foreign_context is removed, a test
passing because the fixture never produced foreign context in the first place
would look exactly like a working filter.
"""

import asyncio

from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field

from agentic_kg.common.adk_context import FOREIGN_CONTEXT_SENTINEL
from agentic_kg.common.adk_transfer import strip_transfer_to_agent
from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent.agent import (
    graph_construction_agent,
)

USER_ID = "u1"

# Hand-written, not imported from schema_critic_agent_v1's prompt: importing it
# would make this file fail whenever an unrelated sentence in a long
# instruction is reworded, and a test that breaks for unrelated reasons gets
# deleted. The cost is that this can stop resembling what the critic emits, so
# test_the_critic_still_speaks_of_warnings below pins the one token that
# matters. Vocabulary is deliberately about graph shape, not about the bundled
# dataset -- test_generality.py does not scan this file, so that is the
# project's stated principle honoured by choice, not an enforced rule.
STALE_CRITIC_REPLY = (
    "valid\n"
    "Warnings:\n"
    "- one relationship's join columns overlap on only a minority of the rows read"
)


class CapturingLlm(BaseLlm):
    """Scripted LLM that also records every request it was handed.

    Same shape as the fakes in test_construction_handoff_gate.py and
    test_graphrag_context_filtering.py. Copied rather than imported: no test
    file in this suite imports helpers from another, and fakes.py deliberately
    covers graph-database fakes only.
    """

    responses: list = Field(default_factory=list)
    requests: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
        self.requests.append(llm_request)
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        yield self.responses[index]


def _text(text):
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


def _foreign_event():
    """One event shaped exactly as ADK reshapes another agent's turn.

    _convert_foreign_event (contents.py) sets role and author to 'user' and
    prepends the sentinel as its own part, so role cannot distinguish this from
    a real human turn -- which is why drop_foreign_context keys on the sentinel
    text sitting at parts[0].
    """
    return Event(
        author="user",
        content=types.Content(
            role="user",
            parts=[
                types.Part(text=FOREIGN_CONTEXT_SENTINEL),
                types.Part(text=f"[schema_critic_agent_v1] said: {STALE_CRITIC_REPLY}"),
            ],
        ),
    )


async def _prepare_session(app_name):
    """Create a session and plant the stale critic turn in its history.

    Points InMemoryRunner at the singleton, which is already parented into the
    real tree -- re-parenting it is what base_agent.py:496-505 forbids.
    """
    runner = InMemoryRunner(agent=graph_construction_agent, app_name=app_name)
    session = await runner.session_service.create_session(
        app_name=app_name, user_id=USER_ID
    )
    await runner.session_service.append_event(session=session, event=_foreign_event())
    return runner, session


async def _stored_events(runner, session):
    stored = await runner.session_service.get_session(
        app_name=session.app_name, user_id=USER_ID, session_id=session.id
    )
    return stored.events


async def _drive_one_turn(runner, session, message="summarize the graph"):
    """Run one real user turn and return every request the model received."""
    async for _ in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
    ):
        pass
    return graph_construction_agent.model.requests


def _part_texts(requests):
    """Every part's text across every request, flattened.

    Shared by the two checks below so the request/content/part walk exists
    once: if ADK reshapes any of those three levels, only this breaks, rather
    than one check silently going always-False while the other is fixed.
    """
    return [
        getattr(part, "text", None)
        for request in requests
        for content in (request.contents or [])
        for part in (content.parts or [])
    ]


def _sentinel_present(requests):
    """Exact match, mirroring the filter's own predicate.

    drop_foreign_context decides with `parts[0].text == FOREIGN_CONTEXT_SENTINEL`
    (adk_context.py `_is_foreign`), so this asks the same question rather than a
    looser substring one.
    """
    return FOREIGN_CONTEXT_SENTINEL in _part_texts(requests)


def _text_present(requests, needle):
    """Substring match, for payload text that arrives inside a larger part."""
    return any(needle in (text or "") for text in _part_texts(requests))


def _scripted_model():
    return CapturingLlm(model="scripted", responses=[_text("nothing to do")])


def test_the_fixture_really_lands_in_session_history():
    """Fixture sanity, and nothing more.

    This does NOT stand in for the control. Session storage and
    request-building are separate transformations, so a message sitting in
    storage is no evidence it would have reached the model. Its honest job is
    catching a fixture that silently stopped inserting anything at all.
    """

    async def scenario():
        runner, session = await _prepare_session("ctx_history_test")
        return await _stored_events(runner, session)

    # asyncio.run inside a sync test, matching
    # test_construction_handoff_gate.py -- this repo drives async ADK code
    # without taking a pytest-asyncio dependency.
    events = asyncio.run(scenario())
    texts = [
        part.text
        for event in events
        for part in (event.content.parts if event.content else [])
    ]
    assert FOREIGN_CONTEXT_SENTINEL in texts
    assert any("Warnings:" in (text or "") for text in texts)


def test_without_the_filter_the_stale_warning_does_reach_the_model(monkeypatch):
    """The control, and the load-bearing half of this file.

    Removes drop_foreign_context and NOTHING else -- strip_transfer_to_agent
    stays, so the only difference from the real agent is the one callback under
    test. Monkeypatching the live singleton rather than building a twin Agent
    keeps this control on the same road as the positive case: same object, same
    name, same tools, same instruction, same place in the tree.

    Safe because canonical_before_model_callbacks (llm_agent.py:382-393) is a
    live property re-read on every access, and neither BaseAgent nor LlmAgent
    sets frozen or validate_assignment (base_agent.py:70-74). monkeypatch is
    function-scoped, so both attributes are restored for the next test.
    """
    monkeypatch.setattr(graph_construction_agent, "model", _scripted_model())
    monkeypatch.setattr(
        graph_construction_agent, "before_model_callback", [strip_transfer_to_agent]
    )

    async def scenario():
        runner, session = await _prepare_session("ctx_control_test")
        return await _drive_one_turn(runner, session)

    requests = asyncio.run(scenario())
    assert requests, "the model was never called"
    assert _sentinel_present(requests), (
        "the fixture no longer produces foreign context, so the positive test "
        "in this file would pass for the wrong reason"
    )
    assert _text_present(requests, "Warnings:"), (
        "the sentinel arrived but the critic's own payload did not -- the "
        "sentinel is only the marker that travels with the payload today, and "
        "the payload (not the marker) is what this control is supposed to prove "
        "reaches the model when the filter is absent"
    )


def test_the_stale_critic_warning_never_reaches_the_model(monkeypatch):
    """AC1: the critic's 'Warnings:' turn cannot reach the model request.

    In a traced 0.4.0 session the agent presented a 'Construction warnings'
    section built from exactly this text -- the critic's, from two turns
    earlier -- while the loader had returned no warnings at all. PR #11 wired
    drop_foreign_context onto this agent six days later. This is the proof it
    closes that route.
    """
    monkeypatch.setattr(graph_construction_agent, "model", _scripted_model())

    async def scenario():
        runner, session = await _prepare_session("ctx_filter_test")
        return await _drive_one_turn(runner, session)

    requests = asyncio.run(scenario())
    assert requests, "the model was never called"
    assert not _sentinel_present(requests)
    assert not _text_present(requests, "Warnings:"), (
        "the sentinel was stripped but the critic's own 'Warnings:' payload "
        "still reached the model -- the sentinel is only the marker that "
        "travels with the payload today, and the payload is what KG-3 is "
        "actually about"
    )


def test_the_critic_still_speaks_of_warnings():
    """Pins the one token this file's fixture imitates.

    STALE_CRITIC_REPLY is hand-written, so nothing else notices if the critic's
    vocabulary moves -- the fixture would keep passing while representing a
    word collision that no longer exists.
    """
    from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.variants import (
        variants as schema_variants,
    )

    critic_instruction = schema_variants["schema_critic_agent_v1"]["instruction"]
    assert "Warnings:" in critic_instruction, (
        "schema_critic_agent_v1 no longer emits a line reading 'Warnings:'. "
        "STALE_CRITIC_REPLY in this file imitates the critic's output, and "
        "the word collision between the critic's warnings and the loader's is "
        "the whole scenario these tests reproduce -- so the fixture now stands "
        "for nothing. Update STALE_CRITIC_REPLY to match the critic's new "
        "wording, or delete this file's premise deliberately."
    )


def test_the_instruction_covers_the_no_warnings_case():
    """The other half of the 0.4.0 defect.

    Step 6 told the agent what to do when the build result carries warnings and
    said nothing about when it does not. Silence, plus a word the schema critic
    also uses, is what let a 'Construction warnings' section be assembled from
    the critic's feedback. Wiring alone cannot fix that -- the instruction has
    to name the empty case, and also name the partial-failure case: the loader
    can carry real warnings inside its error message with no top-level
    'warnings' key, and the instruction must not read that key's absence as
    license to suppress them.

    It also has to scope the warnings by which build produced them. This agent's
    own earlier tool results stay in its context -- drop_foreign_context removes
    other agents' turns, not its own -- and a retry after a partial failure is
    the expected flow, so 'that tool result' alone would still license quoting
    build #1's warnings under a build #2 that reported none.

    Whether the model then obeys is not unit-testable and is verified by hand;
    see the plan's Task 6.
    """
    from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent.variants import (
        variants as construction_variants,
    )

    instruction = construction_variants["graph_construction_agent_v1"]["instruction"]
    assert "the most recent 'build_graph_from_construction_rules'" in instruction
    assert "inside its error message on a partial failure" in instruction
    assert "or an earlier build" in instruction
    assert "no warnings section" in instruction


def test_step_six_gloss_covers_over_matching_too():
    """The gloss motivates reporting a warning on an otherwise successful
    build. While it named only under-matching, a model holding an over-match
    warning read a cause that contradicted it -- and rewriting the warning to
    fit is exactly what the rest of step 6 forbids."""
    from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent.variants import (
        variants as construction_variants,
    )

    instruction = construction_variants["graph_construction_agent_v1"]["instruction"]
    assert "far below the rows read, or above it at all" in instruction
    assert "matched far fewer rows than were read" not in instruction
    # The asymmetry is deliberate: under-match is a half-the-rows threshold,
    # over-match has no slack at all.
    assert "far below or far above" not in instruction
