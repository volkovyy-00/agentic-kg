"""The construction agent's way back to the plan step."""

import ast
import asyncio
import inspect

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
from pydantic import Field

from agentic_kg.common import agent_names
from agentic_kg.common.agent_names import SCHEMA_PROPOSAL_COORDINATOR
from agentic_kg.coordinators.multi_agent.agent import full_workflow_agent
from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent import (
    variants as construction,
)
from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent.agent import (
    graph_construction_agent,
    reset_plan_revision_confirmation,
)
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    root_agent as schema_stage,
)
from agentic_kg.tools import cypher_tools
from agentic_kg.tools import plan_revision_record as record
from agentic_kg.tools.construction_handoff_tools import (
    PLAN_REVISION_CONFIRMED_KEY,
    confirm_construction_handoff,
    confirm_plan_revision,
)
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
_TOOLS = construction.variants["graph_construction_agent_v1"]["tools"]
_EMPTY_CONTENTS = {
    "nodes": 0,
    "relationships": 0,
    "labels": {},
    "relationship_types": {},
    "constraints": [],
    "indexes": [],
}


class FakeActions:
    def __init__(self):
        self.escalate = False
        self.transfer_to_agent = None


class FakeToolContext:
    def __init__(self, state=None):
        self.state = dict(state or {})
        self.actions = FakeActions()


class FakeCallbackContext:
    def __init__(self, state=None):
        self.state = dict(state or {})


def test_agent_names_imports_nothing_from_the_package():
    tree = ast.parse(inspect.getsource(agent_names))
    assert not [
        n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
    ]


def test_both_names_resolve_in_the_tree():
    assert full_workflow_agent.find_agent(SCHEMA_PROPOSAL_COORDINATOR) is not None
    assert (
        full_workflow_agent.find_agent(agent_names.GRAPH_CONSTRUCTION_AGENT) is not None
    )


def test_way_back_refuses_without_confirmation_and_changes_nothing():
    context = FakeToolContext({APPROVED_CONSTRUCTION_PLAN: _PLAN})
    result = construction.return_to_plan(context)
    assert result["status"] == "error"
    assert context.state[APPROVED_CONSTRUCTION_PLAN] == _PLAN
    assert record.PLAN_REVISION_KEY not in context.state
    assert context.actions.transfer_to_agent is None


def test_refusal_with_no_approval_does_not_say_the_plan_is_still_approved():
    context = FakeToolContext({APPROVED_CONSTRUCTION_PLAN: None})
    text = construction.return_to_plan(context)["error_message"]
    assert "still approved" not in text
    assert "still not approved" in text
    assert "nothing was changed" in text


def test_refusal_with_an_approved_plan_still_says_it_is_still_approved():
    context = FakeToolContext({APPROVED_CONSTRUCTION_PLAN: _PLAN})
    text = construction.return_to_plan(context)["error_message"]
    assert "the plan is still approved" in text
    assert "still not approved" not in text


def test_confirmed_way_back_withdraws_records_and_hands_over():
    context = FakeToolContext({APPROVED_CONSTRUCTION_PLAN: _PLAN})
    confirm_plan_revision(context)
    assert construction.return_to_plan(context) == {}
    assert context.state[APPROVED_CONSTRUCTION_PLAN] is None
    assert record.revision(context.state) == {"status": record.PENDING}
    assert context.actions.transfer_to_agent == SCHEMA_PROPOSAL_COORDINATOR


def test_a_second_way_back_resets_the_record_to_pending():
    context = FakeToolContext(
        {
            APPROVED_CONSTRUCTION_PLAN: _PLAN,
            record.PLAN_REVISION_KEY: {
                "status": record.ASKED,
                "asked_in": "x",
                "contents": {},
            },
        }
    )
    confirm_plan_revision(context)
    construction.return_to_plan(context)
    assert record.revision(context.state) == {"status": record.PENDING}


def test_confirming_the_revision_does_not_open_the_retrieval_exit():
    context = FakeToolContext({APPROVED_CONSTRUCTION_PLAN: _PLAN})
    confirm_plan_revision(context)
    assert construction.finished(context)["status"] == "error"
    assert context.actions.transfer_to_agent is None


def test_confirming_the_handoff_does_not_open_the_way_back():
    context = FakeToolContext({APPROVED_CONSTRUCTION_PLAN: _PLAN})
    confirm_construction_handoff(context)
    assert construction.return_to_plan(context)["status"] == "error"
    assert context.state[APPROVED_CONSTRUCTION_PLAN] == _PLAN


def test_the_reset_clears_the_revision_flag():
    context = FakeCallbackContext({PLAN_REVISION_CONFIRMED_KEY: True})
    reset_plan_revision_confirmation(context)
    assert context.state[PLAN_REVISION_CONFIRMED_KEY] is False


def test_both_resets_are_wired_onto_the_construction_agent():
    from agentic_kg.coordinators.multi_agent.sub_agents.graph_construction_agent.agent import (
        reset_construction_handoff_confirmation,
    )

    callbacks = graph_construction_agent.canonical_before_agent_callbacks
    assert reset_construction_handoff_confirmation in callbacks
    assert reset_plan_revision_confirmation in callbacks


def test_the_new_tools_are_in_the_variant():
    from agentic_kg.tools import plan_revision_tools

    for tool in (
        confirm_plan_revision,
        construction.return_to_plan,
        plan_revision_tools.check_database_before_rebuild,
        plan_revision_tools.clear_database_for_rebuild,
        plan_revision_tools.keep_database_for_rebuild,
    ):
        assert tool in _TOOLS
    assert construction.create_uniqueness_constraint in _TOOLS
    assert cypher_tools.create_uniqueness_constraint not in _TOOLS


def test_the_constraint_wrapper_refuses_while_pending_and_sends_no_query(monkeypatch):
    sent = []
    monkeypatch.setattr(
        construction,
        "_shared_create_uniqueness_constraint",
        lambda label, key: sent.append((label, key)),
    )
    for status in (record.PENDING, record.ASKED):
        result = construction.create_uniqueness_constraint(
            "Person",
            "id",
            FakeToolContext({record.PLAN_REVISION_KEY: {"status": status}}),
        )
        assert result == {"status": "error", "error_message": record.REVISION_REFUSAL}
    assert sent == []


def test_the_constraint_wrapper_calls_through_otherwise(monkeypatch):
    monkeypatch.setattr(
        construction,
        "_shared_create_uniqueness_constraint",
        lambda label, key: {"status": "success", "label": label},
    )
    result = construction.create_uniqueness_constraint(
        "Person", "id", FakeToolContext()
    )
    assert result == {"status": "success", "label": "Person"}


def test_the_model_sees_the_same_constraint_tool():
    wrapped = FunctionTool(construction.create_uniqueness_constraint)._get_declaration()
    shared = FunctionTool(cypher_tools.create_uniqueness_constraint)._get_declaration()
    assert wrapped.name == shared.name
    assert wrapped.description == shared.description
    assert wrapped.parameters_json_schema == shared.parameters_json_schema
    assert wrapped.parameters == shared.parameters


def test_the_wrapper_receives_its_context_through_adk(monkeypatch):
    """A functools.wraps wrapper would pass the declaration test above and
    then fail here: ADK would not pass tool_context."""
    monkeypatch.setattr(
        construction,
        "_shared_create_uniqueness_constraint",
        lambda label, key: {"status": "success"},
    )
    tool = FunctionTool(construction.create_uniqueness_constraint)
    context = FakeToolContext({record.PLAN_REVISION_KEY: {"status": record.PENDING}})
    result = asyncio.run(
        tool.run_async(
            args={"label": "Person", "unique_property_key": "id"}, tool_context=context
        )
    )
    assert result == {"status": "error", "error_message": record.REVISION_REFUSAL}


def test_every_tool_the_construction_texts_name_is_wired():
    """Catches a tool named in the instruction or in a tool result that the
    agent cannot call -- the model would be pointed at a tool it lacks."""
    import re

    from agentic_kg.tools import (
        construction_handoff_tools,
        construction_plan_tools,
        cypher_tools,
        kg_construction_tools,
        plan_revision_tools,
    )

    texts = [
        construction.variants["graph_construction_agent_v1"]["instruction"],
        record.REVISION_REFUSAL,
        kg_construction_tools.NOT_APPROVED_MESSAGE,
        plan_revision_tools.NO_REBUILD_PENDING,
        plan_revision_tools.REBUILD_OWED,
        plan_revision_tools.SAME_TURN_REFUSAL,
        plan_revision_tools.NOT_ASKED_REFUSAL,
        plan_revision_tools._question(_EMPTY_CONTENTS, asked_earlier=False),
        plan_revision_tools._question(_EMPTY_CONTENTS, asked_earlier=True),
        construction.return_to_plan(FakeToolContext())["error_message"],
    ]
    known = {
        name
        for module in (
            construction,
            construction_handoff_tools,
            construction_plan_tools,
            cypher_tools,
            kg_construction_tools,
            plan_revision_tools,
        )
        for name, value in vars(module).items()
        if callable(value) and not isinstance(value, type) and not name.startswith("_")
    }
    wired = {tool.__name__ for tool in _TOOLS}
    named = {m for text in texts for m in re.findall(r"'([a-z_][a-z0-9_]*)'", text)}
    assert not (named & known) - wired


def test_the_every_turn_check_comes_before_the_steps():
    text = construction.variants["graph_construction_agent_v1"]["instruction"]
    assert text.index("At the start of every turn") < text.index("Follow these steps")


def test_the_instruction_carries_the_way_back_and_the_every_turn_check():
    text = " ".join(
        construction.variants["graph_construction_agent_v1"]["instruction"].split()
    )
    assert (
        "At the start of every turn, before anything else, call 'check_database_before_rebuild'"
        in text
    )
    assert "any earlier build in this conversation is the previous version" in text
    assert "Never offer the way back while the approval is intact" in text
    assert "'confirm_plan_revision' and then 'return_to_plan'" in text
    assert "{" not in text and "}" not in text


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


async def _turns(agent, state, messages):
    runner = InMemoryRunner(agent=agent, app_name="way_back_test")
    session = await runner.session_service.create_session(
        app_name="way_back_test", user_id="u1", state=state
    )
    turns = []
    for message in messages:
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
        app_name="way_back_test", user_id="u1", session_id=session.id
    )
    return turns, saved.state


def test_the_way_back_lands_in_the_schema_stage_and_stays_there(monkeypatch):
    monkeypatch.setattr(
        graph_construction_agent,
        "model",
        CapturingLlm(
            model="scripted",
            responses=[
                _call("confirm_plan_revision"),
                _call("return_to_plan"),
                _text("back to the plan"),
            ],
        ),
    )
    monkeypatch.setattr(
        schema_stage,
        "model",
        CapturingLlm(model="scripted", responses=[_text("here is the current plan")]),
    )
    # Start at the coordinator, as test_phase_stickiness.py does: ADK picks the
    # agent for each new message by searching the runner's own tree, so a runner
    # rooted at the construction agent could never find the schema stage.
    monkeypatch.setattr(
        full_workflow_agent,
        "model",
        CapturingLlm(
            model="scripted",
            responses=[
                _call(
                    "transfer_to_agent",
                    {"agent_name": agent_names.GRAPH_CONSTRUCTION_AGENT},
                )
            ],
        ),
    )

    (first, second), state = asyncio.run(
        _turns(
            full_workflow_agent,
            {APPROVED_CONSTRUCTION_PLAN: _PLAN},
            ["rename Person to Customer in the plan", "and make the key customer_id"],
        )
    )

    assert SCHEMA_PROPOSAL_COORDINATOR in [e.author for e in first]
    assert {e.author for e in second if e.content} <= {
        SCHEMA_PROPOSAL_COORDINATOR,
        "user",
    }
    assert state[APPROVED_CONSTRUCTION_PLAN] is None
    assert state[record.PLAN_REVISION_KEY] == {"status": record.PENDING}


def test_moving_on_with_a_changed_plan_unbuilt_says_the_graph_is_the_earlier_build():
    """A user may leave for retrieval without answering the clear question; they
    are let go, but told their questions will be answered from the earlier build."""
    text = " ".join(
        construction.variants["graph_construction_agent_v1"]["instruction"].split()
    )
    step_9 = text[
        text.index("9. only when the user says") : text.index("Changing the plan:")
    ]
    assert "approved and not yet built" in step_9
    assert "still holds the earlier build" in step_9
