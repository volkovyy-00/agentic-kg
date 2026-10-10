"""A refinement loop whose proposal or review step the rejected-call cap
stopped reports 'stuck:', never a clean review.

Runs the real loop under the real coordinator with scripted models above the
LiteLlm adapter. The cap's temp: values live in the AgentTool run's own
invocation, which is how the stop-check sees them and why they never reach
the coordinator.
"""

import asyncio

import fsspec
import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import Field

from agentic_kg.common.config import reset_settings
from agentic_kg.common.rejected_call_cap import STUCK_TURN_END
from agentic_kg.coordinators.multi_agent.sub_agents.schema_proposal_agent.agent import (
    FEEDBACK_KIND_KEY,
    STUCK_LOOP_RESULT,
    VerdictKind,
    record_critic_verdict,
    root_agent,
    schema_critic_agent,
    schema_proposal_agent,
)

PLAN_WRITE = "propose_node_constructions"
NODE_CONSTRUCTIONS = [
    {
        "approved_file": "plots.csv",
        "proposed_label": "Plot",
        "unique_column_name": "plot_label",
        "proposed_properties": ["plot_id"],
    }
]


class ScriptedLlm(BaseLlm):
    responses: list = Field(default_factory=list)
    call_count: int = 0

    async def generate_content_async(self, llm_request, stream: bool = False):
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


_UNKNOWN = _call("no_such_tool")
_LOOP = _call("schema_refinement_loop", {"request": "propose an initial schema"})


@pytest.fixture
def one_approved_file(monkeypatch):
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    fs.pseudo_dirs.clear()
    with fs.open("/src/plots.csv", "w") as handle:
        handle.write("plot_label,plot_id\nridge,PL-1\nhollow,PL-3\n")
    monkeypatch.setenv("SOURCE_URI", "memory://src")
    reset_settings()
    yield {"approved_file_list": ["plots.csv"]}
    fs.store.clear()
    fs.pseudo_dirs.clear()


def _script(monkeypatch, *, proposer, critic, coordinator):
    models = {
        "proposer": ScriptedLlm(model="scripted", responses=proposer),
        "critic": ScriptedLlm(model="scripted", responses=critic),
        "coordinator": ScriptedLlm(model="scripted", responses=coordinator),
    }
    monkeypatch.setattr(schema_proposal_agent, "model", models["proposer"])
    monkeypatch.setattr(schema_critic_agent, "model", models["critic"])
    monkeypatch.setattr(root_agent, "model", models["coordinator"])
    return models


def _run(app_name, state):
    async def run():
        runner = InMemoryRunner(agent=root_agent, app_name=app_name)
        session = await runner.session_service.create_session(
            app_name=app_name, user_id="u1", state=state
        )
        events = [
            event
            async for event in runner.run_async(
                user_id="u1",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text="propose")]
                ),
            )
        ]
        final = await runner.session_service.get_session(
            app_name=app_name, user_id="u1", session_id=session.id
        )
        return events, final

    return asyncio.run(run())


def _loop_results(events):
    return [
        str((part.function_response.response or {}).get("result", ""))
        for event in events
        for part in (event.content.parts if event.content else None) or []
        if part.function_response
        and part.function_response.name == "schema_refinement_loop"
    ]


def _final_text(events):
    parts = events[-1].content.parts if events[-1].content else []
    return "".join(part.text or "" for part in parts)


def test_the_stuck_result_cannot_be_read_as_another_verdict():
    for prefix in ("retry", "valid", "stopped:", "no verdict:"):
        assert not STUCK_LOOP_RESULT.lower().startswith(prefix)
    assert STUCK_LOOP_RESULT.startswith("stuck:")


def test_a_stuck_proposer_makes_the_loop_report_stuck(monkeypatch, one_approved_file):
    """The critic still runs and says 'valid'; the loop reports stuck, ends
    after one iteration, and the coordinator's own count is untouched."""
    models = _script(
        monkeypatch,
        proposer=[_UNKNOWN] * 3 + [_text("never reached")],
        critic=[_text("valid")],
        coordinator=[_LOOP, _UNKNOWN, _UNKNOWN, _text("coordinator speaking")],
    )

    events, final = _run("stuck_proposer", one_approved_file)

    assert _loop_results(events) == [STUCK_LOOP_RESULT]
    assert models["critic"].call_count == 1  # one iteration, not two
    assert final.state["feedback"] == ""
    assert final.state[FEEDBACK_KIND_KEY] == VerdictKind.NONE.value
    assert _final_text(events) == "coordinator speaking"
    assert not any(key.startswith("temp:") for key in final.state)


def test_a_stuck_critic_makes_the_loop_report_stuck_and_never_records_the_stop(
    monkeypatch, one_approved_file
):
    seen = []

    def spy(callback_context, llm_response):
        seen.append(llm_response)
        return record_critic_verdict(callback_context, llm_response)

    monkeypatch.setattr(schema_critic_agent, "after_model_callback", spy)
    _script(
        monkeypatch,
        proposer=[_text("a proposal")],
        critic=[_UNKNOWN] * 3 + [_text("never reached")],
        coordinator=[_LOOP, _text("coordinator speaking")],
    )

    events, final = _run("stuck_critic", one_approved_file)

    assert _loop_results(events) == [STUCK_LOOP_RESULT]
    assert final.state["feedback"] == ""
    assert seen
    texts = [
        part.text or ""
        for response in seen
        for part in (response.content.parts if response.content else None) or []
    ]
    assert STUCK_TURN_END not in texts


def test_a_proposer_recovers_from_two_rejected_plan_writes(
    monkeypatch, one_approved_file
):
    _script(
        monkeypatch,
        proposer=[
            _call(PLAN_WRITE),
            _call(PLAN_WRITE),
            _call(PLAN_WRITE, {"node_constructions": NODE_CONSTRUCTIONS}),
            _text("proposal done"),
        ],
        critic=[_text("valid")],
        coordinator=[_LOOP, _text("coordinator speaking")],
    )

    events, final = _run("proposer_recovers", one_approved_file)

    assert _loop_results(events) == ["valid"]
    assert list(final.state["proposed_construction_plan"]) == ["Plot"]


def test_the_coordinators_count_does_not_seed_a_loop_run(
    monkeypatch, one_approved_file
):
    """Two stuck coordinator replies, then the loop, whose proposer slips twice
    and recovers: neither agent is stopped."""
    _script(
        monkeypatch,
        proposer=[_UNKNOWN, _UNKNOWN, _text("a proposal")],
        critic=[_text("valid")],
        coordinator=[_UNKNOWN, _UNKNOWN, _LOOP, _text("coordinator speaking")],
    )

    events, _ = _run("count_scope", one_approved_file)

    (result,) = _loop_results(events)
    assert not result.startswith("stuck:")
    assert _final_text(events) == "coordinator speaking"


def test_the_coordinator_is_told_what_stuck_means():
    instruction = root_agent.instruction
    assert "'stuck:' when its proposal or review step got stuck" in instruction
    assert "If the verdict the loop returns begins with 'stuck:'" in instruction
