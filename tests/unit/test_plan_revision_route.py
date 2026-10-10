"""After a revision is re-approved, the plan step hands straight back to construction."""

import asyncio

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field

from agentic_kg.common.agent_names import (
    GRAPH_CONSTRUCTION_AGENT,
    MULTI_AGENT_COORDINATOR,
)

# Also parents the sub-agents: without this import a transfer to a peer finds
# no target when this file runs alone.
from agentic_kg.coordinators.multi_agent.agent import full_workflow_agent
from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent.agent import (
    graph_construction_agent,
)
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent import (
    agent as schema_module,
)
from agentic_kg.tools import construction_plan_tools as cpt
from agentic_kg.tools import kg_construction_tools as kg
from agentic_kg.tools import plan_revision_record as record
from agentic_kg.tools import plan_revision_tools
from agentic_kg.tools.kg_construction_tools import APPROVED_CONSTRUCTION_PLAN

_PLAN = {
    "Person": {
        "construction_type": "node",
        "label": "Person",
        "source_file": "p.csv",
        "unique_column_name": "id",
        "properties": [],
    }
}
_FULL = {
    "nodes": 5,
    "relationships": 0,
    "labels": {"Person": 5},
    "relationship_types": {},
    "constraints": [],
    "indexes": [],
}


class FakeActions:
    def __init__(self):
        self.escalate = False
        self.transfer_to_agent = None


class FakeToolContext:
    def __init__(self, state):
        self.state = state
        self.actions = FakeActions()


@pytest.mark.parametrize(
    "has_record, approved, target",
    [
        (False, True, MULTI_AGENT_COORDINATOR),
        (True, False, MULTI_AGENT_COORDINATOR),
        (True, True, GRAPH_CONSTRUCTION_AGENT),
        (False, False, MULTI_AGENT_COORDINATOR),
    ],
)
def test_the_schema_exit_routes_on_record_and_approval(has_record, approved, target):
    state = {APPROVED_CONSTRUCTION_PLAN: _PLAN if approved else None}
    if has_record:
        state[record.PLAN_REVISION_KEY] = {"status": record.PENDING}
    context = FakeToolContext(state)
    assert schema_module.finished(context) == {}
    assert context.actions.transfer_to_agent == target


def test_the_approval_check_says_nothing_without_a_record(monkeypatch):
    monkeypatch.setattr(cpt, "find_plan_problems", lambda state: ([], []))
    result = cpt.get_proposed_construction_plan_with_approval_check(
        FakeToolContext({cpt.PROPOSED_CONSTRUCTION_PLAN: _PLAN})
    )
    assert "revision" not in result
    assert "revision" not in result.get("result", {})


def test_the_approval_check_says_the_user_came_back_while_pending(monkeypatch):
    monkeypatch.setattr(cpt, "find_plan_problems", lambda state: ([], []))
    result = cpt.get_proposed_construction_plan_with_approval_check(
        FakeToolContext(
            {
                cpt.PROPOSED_CONSTRUCTION_PLAN: _PLAN,
                record.PLAN_REVISION_KEY: {"status": record.PENDING},
            }
        )
    )
    assert result["result"]["revision"] == cpt.REVISION_NOTE
    assert "withdrawn" in cpt.REVISION_NOTE


def test_the_approval_check_drops_the_note_once_a_plan_is_approved(monkeypatch):
    monkeypatch.setattr(cpt, "find_plan_problems", lambda state: ([], []))
    result = cpt.get_proposed_construction_plan_with_approval_check(
        FakeToolContext(
            {
                cpt.PROPOSED_CONSTRUCTION_PLAN: _PLAN,
                APPROVED_CONSTRUCTION_PLAN: _PLAN,
                record.PLAN_REVISION_KEY: {"status": record.PENDING},
            }
        )
    )
    assert "revision" not in result.get("result", {})


@pytest.mark.parametrize("status", [record.ASKED, record.ANSWERED])
def test_the_approval_check_has_no_note_while_the_clear_question_is_open(
    monkeypatch, status
):
    monkeypatch.setattr(cpt, "find_plan_problems", lambda state: ([], []))
    result = cpt.get_proposed_construction_plan_with_approval_check(
        FakeToolContext(
            {
                cpt.PROPOSED_CONSTRUCTION_PLAN: _PLAN,
                record.PLAN_REVISION_KEY: {"status": status},
            }
        )
    )
    assert "revision" not in result.get("result", {})


def test_the_approval_check_error_path_carries_the_note_while_pending(monkeypatch):
    monkeypatch.setattr(cpt, "find_plan_problems", lambda state: (["a problem"], []))
    result = cpt.get_proposed_construction_plan_with_approval_check(
        FakeToolContext(
            {
                cpt.PROPOSED_CONSTRUCTION_PLAN: _PLAN,
                record.PLAN_REVISION_KEY: {"status": record.PENDING},
            }
        )
    )
    assert result["status"] == "error"
    assert result["revision"] == cpt.REVISION_NOTE


def test_the_schema_instruction_handles_a_revision():
    text = " ".join(schema_module.root_agent.instruction.split())
    assert "came back from the construction step" in text
    assert "ask what they want to change" in text
    assert "{" not in text
    assert "}" not in text


class CapturingLlm(BaseLlm):
    """Scripted LLM; copied from test_construction_handoff_gate.py, not imported."""

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


def _call(name, args=None):
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(function_call=types.FunctionCall(name=name, args=args or {}))
            ],
        )
    )


def _responses(events, name):
    return [
        part.function_response.response
        for event in events
        for part in (event.content.parts if event.content else []) or []
        if part.function_response and part.function_response.name == name
    ]


def test_re_approval_lands_in_construction_which_asks_and_refuses_a_same_turn_yes(
    monkeypatch,
):
    """The user's 'yes' to the plan cannot also be the yes to the erase."""
    monkeypatch.setattr(cpt, "find_plan_problems", lambda state: ([], []))
    monkeypatch.setattr(
        plan_revision_tools,
        "database_contents",
        lambda: {"status": "success", "contents": _FULL},
    )
    monkeypatch.setattr(
        plan_revision_tools,
        "reset_neo4j_data",
        lambda: pytest.fail("erased in the asking turn"),
    )
    monkeypatch.setattr(
        schema_module.root_agent,
        "model",
        CapturingLlm(
            model="scripted",
            responses=[
                _call("approve_proposed_construction_plan"),
                _call("finished"),
                _text("approved"),
            ],
        ),
    )
    monkeypatch.setattr(
        graph_construction_agent,
        "model",
        CapturingLlm(
            model="scripted",
            responses=[
                _call("check_database_before_rebuild"),
                _call("clear_database_for_rebuild"),
                _text("shall I clear it?"),
            ],
        ),
    )

    async def run():
        runner = InMemoryRunner(agent=schema_module.root_agent, app_name="route_test")
        session = await runner.session_service.create_session(
            app_name="route_test",
            user_id="u1",
            state={
                cpt.PROPOSED_CONSTRUCTION_PLAN: _PLAN,
                APPROVED_CONSTRUCTION_PLAN: None,
                record.PLAN_REVISION_KEY: {"status": record.PENDING},
            },
        )
        events = [
            e
            async for e in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="yes, approve it")]
                ),
            )
        ]
        saved = await runner.session_service.get_session(
            app_name="route_test", user_id="u1", session_id=session.id
        )
        return events, saved.state

    events, state = asyncio.run(run())

    assert GRAPH_CONSTRUCTION_AGENT in [e.author for e in events]
    assert "question" in _responses(events, "check_database_before_rebuild")[0]
    clear = _responses(events, "clear_database_for_rebuild")[0]
    assert clear["error_message"] == plan_revision_tools.SAME_TURN_REFUSAL
    assert state[record.PLAN_REVISION_KEY]["status"] == record.ASKED


def test_the_answer_comes_next_turn_then_the_rebuild_uses_the_record_up(monkeypatch):
    """Two real user messages: the check asks; next turn it says the question was
    asked earlier; clear erases once; the build succeeds; the record is gone; and
    the second message stayed with construction."""
    erased = []
    monkeypatch.setattr(cpt, "find_plan_problems", lambda state: ([], []))
    monkeypatch.setattr(
        plan_revision_tools,
        "database_contents",
        lambda: {"status": "success", "contents": _FULL},
    )
    monkeypatch.setattr(
        plan_revision_tools,
        "reset_neo4j_data",
        lambda: erased.append(True) or {"status": "success", "message": "reset"},
    )
    monkeypatch.setattr(
        kg,
        "construct_domain_graph",
        lambda plan: {"status": "success", "domain_graph_constructed": {}},
    )
    monkeypatch.setattr(
        schema_module.root_agent,
        "model",
        CapturingLlm(
            model="scripted",
            responses=[
                _call("approve_proposed_construction_plan"),
                _call("finished"),
                _text("approved"),
            ],
        ),
    )
    # Rooted at the coordinator so the second message can find construction
    # (ADK searches the runner's own tree); it hands turn 1 to the plan step.
    monkeypatch.setattr(
        full_workflow_agent,
        "model",
        CapturingLlm(
            model="scripted",
            responses=[
                _call(
                    "transfer_to_agent", {"agent_name": schema_module.root_agent.name}
                )
            ],
        ),
    )
    monkeypatch.setattr(
        graph_construction_agent,
        "model",
        CapturingLlm(
            model="scripted",
            responses=[
                _call("check_database_before_rebuild"),
                _text("shall I clear it?"),
                _call("check_database_before_rebuild"),
                _call("clear_database_for_rebuild"),
                _call("build_graph_from_construction_rules"),
                _text("rebuilt"),
            ],
        ),
    )

    async def run():
        runner = InMemoryRunner(agent=full_workflow_agent, app_name="route_test")
        session = await runner.session_service.create_session(
            app_name="route_test",
            user_id="u1",
            state={
                cpt.PROPOSED_CONSTRUCTION_PLAN: _PLAN,
                APPROVED_CONSTRUCTION_PLAN: None,
                record.PLAN_REVISION_KEY: {"status": record.PENDING},
            },
        )
        turns = []
        for message in ("yes, approve it", "yes, clear it"):
            turns.append(
                [
                    e
                    async for e in runner.run_async(
                        user_id="u1",
                        session_id=session.id,
                        new_message=types.Content(
                            role="user", parts=[types.Part(text=message)]
                        ),
                    )
                ]
            )
        saved = await runner.session_service.get_session(
            app_name="route_test", user_id="u1", session_id=session.id
        )
        return turns, saved.state

    (first, second), state = asyncio.run(run())

    assert (
        "earlier turn"
        in _responses(second, "check_database_before_rebuild")[0]["question"]
    )
    assert erased == [True]
    assert state[record.PLAN_REVISION_KEY] is None
    assert {e.author for e in second if e.content} <= {GRAPH_CONSTRUCTION_AGENT, "user"}


@pytest.mark.parametrize(
    "stale",
    [
        {"status": record.ASKED, "asked_in": "inv-old", "contents": _FULL},
        {"status": record.ANSWERED, "cleared": False},
    ],
)
def test_a_fresh_approval_reopens_a_stale_record(monkeypatch, stale):
    """A plan approved again while an old clear question is still open (or after
    a failed rebuild) is a new revision: construction must ask afresh, so the
    approval's own "yes" cannot be read as "yes, clear the database"."""
    monkeypatch.setattr(cpt, "find_plan_problems", lambda state: ([], []))
    state = {cpt.PROPOSED_CONSTRUCTION_PLAN: _PLAN, record.PLAN_REVISION_KEY: stale}
    result = cpt.approve_proposed_construction_plan(FakeToolContext(state))
    assert result["status"] == "success"
    assert state[record.PLAN_REVISION_KEY] == {"status": record.PENDING}


def test_a_first_approval_writes_no_record(monkeypatch):
    monkeypatch.setattr(cpt, "find_plan_problems", lambda state: ([], []))
    state = {cpt.PROPOSED_CONSTRUCTION_PLAN: _PLAN}
    cpt.approve_proposed_construction_plan(FakeToolContext(state))
    assert record.PLAN_REVISION_KEY not in state


def test_the_revision_note_does_not_assume_the_change_is_still_outstanding():
    """The note is shown on every turn until approval, including after the
    requested change was already applied."""
    note = " ".join(cpt.REVISION_NOTE.split())
    assert "apply the change they asked for," not in note
    assert "that it does not include yet" in note
