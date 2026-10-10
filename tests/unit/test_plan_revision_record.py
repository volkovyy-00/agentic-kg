"""The record of a plan revision, and the build's gate on it."""

import ast
import inspect

from agentic_kg.tools import kg_construction_tools as kg
from agentic_kg.tools import plan_revision_record as record

_PLAN = {
    "Person": {
        "construction_type": "node",
        "label": "Person",
        "source_file": "p.csv",
        "unique_column_name": "id",
        "properties": [],
    }
}


class FakeToolContext:
    def __init__(self, state):
        self.state = state


def test_no_record_reads_as_none():
    assert record.revision({}) is None
    assert record.revision({record.PLAN_REVISION_KEY: None}) is None


def test_the_record_moves_through_its_states():
    state = {}
    record.mark_pending(state)
    assert record.revision(state) == {"status": record.PENDING}
    record.mark_asked(state, "inv-1", {"nodes": 3})
    assert record.revision(state) == {
        "status": record.ASKED,
        "asked_in": "inv-1",
        "contents": {"nodes": 3},
    }
    record.mark_answered(state, cleared=True)
    assert record.revision(state) == {"status": record.ANSWERED, "cleared": True}
    record.clear_revision(state)
    assert state[record.PLAN_REVISION_KEY] is None


def test_pending_overwrites_any_state():
    for prior in (
        {"status": record.ASKED, "asked_in": "x", "contents": {}},
        {"status": record.ANSWERED, "cleared": False},
    ):
        state = {record.PLAN_REVISION_KEY: prior}
        record.mark_pending(state)
        assert record.revision(state) == {"status": record.PENDING}


def test_refusal_only_while_pending_or_asked():
    assert record.revision_refusal({}) is None
    assert (
        record.revision_refusal(
            {record.PLAN_REVISION_KEY: {"status": record.ANSWERED, "cleared": None}}
        )
        is None
    )
    for status in (record.PENDING, record.ASKED):
        text = record.revision_refusal({record.PLAN_REVISION_KEY: {"status": status}})
        assert text == record.REVISION_REFUSAL
    assert "check_database_before_rebuild" in record.REVISION_REFUSAL


def test_the_record_module_imports_nothing_from_the_tools_package():
    """It is imported by the build, the plan tools and the revision tools; an
    import from any of them would be a cycle."""
    tree = ast.parse(inspect.getsource(record))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("agentic_kg.tools")
            assert node.level == 0


def test_the_build_refuses_while_the_question_is_unsettled(monkeypatch):
    called = []
    monkeypatch.setattr(kg, "construct_domain_graph", lambda plan: called.append(plan))
    for status in (record.PENDING, record.ASKED):
        result = kg.build_graph_from_construction_rules(
            FakeToolContext(
                {
                    kg.APPROVED_CONSTRUCTION_PLAN: _PLAN,
                    record.PLAN_REVISION_KEY: {"status": status},
                }
            )
        )
        assert result == {"status": "error", "error_message": record.REVISION_REFUSAL}
    assert called == []


def test_approval_is_checked_before_the_record(monkeypatch):
    monkeypatch.setattr(
        kg, "construct_domain_graph", lambda plan: {"status": "success"}
    )
    result = kg.build_graph_from_construction_rules(
        FakeToolContext(
            {
                kg.APPROVED_CONSTRUCTION_PLAN: None,
                record.PLAN_REVISION_KEY: {"status": record.ANSWERED, "cleared": True},
            }
        )
    )
    assert result["error_message"] == kg.NOT_APPROVED_MESSAGE


def test_a_successful_rebuild_uses_the_record_up(monkeypatch):
    monkeypatch.setattr(
        kg,
        "construct_domain_graph",
        lambda plan: {"status": "success", "domain_graph_constructed": {}},
    )
    state = {
        kg.APPROVED_CONSTRUCTION_PLAN: _PLAN,
        record.PLAN_REVISION_KEY: {"status": record.ANSWERED, "cleared": False},
    }
    kg.build_graph_from_construction_rules(FakeToolContext(state))
    assert record.revision(state) is None


def test_a_failed_rebuild_keeps_the_answer(monkeypatch):
    monkeypatch.setattr(
        kg,
        "construct_domain_graph",
        lambda plan: {"status": "error", "error_message": "boom"},
    )
    answered = {"status": record.ANSWERED, "cleared": True}
    state = {kg.APPROVED_CONSTRUCTION_PLAN: _PLAN, record.PLAN_REVISION_KEY: answered}
    kg.build_graph_from_construction_rules(FakeToolContext(state))
    assert record.revision(state) == answered


def test_a_first_build_writes_no_record(monkeypatch):
    monkeypatch.setattr(
        kg, "construct_domain_graph", lambda plan: {"status": "success"}
    )
    state = {kg.APPROVED_CONSTRUCTION_PLAN: _PLAN}
    kg.build_graph_from_construction_rules(FakeToolContext(state))
    assert record.PLAN_REVISION_KEY not in state
