"""The approval can be withdrawn, and every reader agrees on what that means.

ADK's State cannot delete a key, so withdrawal writes None, and a missing key,
None, or an empty value all read as "not approved".
"""

import pytest

from agentic_kg.tools import construction_plan_tools as cpt
from agentic_kg.tools import kg_construction_tools as kg

_PLAN = {
    "Person": {
        "construction_type": "node",
        "source_file": "p.csv",
        "label": "Person",
        "unique_column_name": "id",
        "properties": [],
    }
}


class FakeToolContext:
    def __init__(self, state):
        self.state = state


def test_the_key_is_spelled_once():
    assert cpt.APPROVED_CONSTRUCTION_PLAN is kg.APPROVED_CONSTRUCTION_PLAN


@pytest.mark.parametrize(
    "state",
    [
        {},
        {kg.APPROVED_CONSTRUCTION_PLAN: None},
        {kg.APPROVED_CONSTRUCTION_PLAN: {}},
        {kg.APPROVED_CONSTRUCTION_PLAN: []},
    ],
)
def test_missing_none_and_empty_all_read_as_not_approved(state):
    assert kg.approved_plan(state) is None


def test_an_approved_plan_reads_back():
    assert kg.approved_plan({kg.APPROVED_CONSTRUCTION_PLAN: _PLAN}) == _PLAN


def test_withdraw_writes_none_and_reads_as_not_approved():
    state = {kg.APPROVED_CONSTRUCTION_PLAN: _PLAN}
    kg.withdraw_approval(state)
    assert state[kg.APPROVED_CONSTRUCTION_PLAN] is None
    assert kg.approved_plan(state) is None


@pytest.mark.parametrize("value", [None, {}])
def test_the_build_refuses_a_withdrawn_plan_and_sends_no_query(value, monkeypatch):
    called = []
    monkeypatch.setattr(kg, "construct_domain_graph", lambda plan: called.append(plan))
    result = kg.build_graph_from_construction_rules(
        FakeToolContext({kg.APPROVED_CONSTRUCTION_PLAN: value})
    )
    assert result["status"] == "error"
    assert result["error_message"] == kg.NOT_APPROVED_MESSAGE
    assert "needs approval at the plan step" in kg.NOT_APPROVED_MESSAGE
    assert called == []


def test_the_getter_returns_an_error_when_not_approved():
    result = cpt.get_approved_construction_plan(
        FakeToolContext({kg.APPROVED_CONSTRUCTION_PLAN: None})
    )
    assert result == {"status": "error", "error_message": kg.NOT_APPROVED_MESSAGE}


def test_the_getter_returns_the_plan_itself_when_approved():
    result = cpt.get_approved_construction_plan(
        FakeToolContext({kg.APPROVED_CONSTRUCTION_PLAN: _PLAN})
    )
    assert result == _PLAN
