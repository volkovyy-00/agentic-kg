"""Unit tests for the proposed/approved construction plan state tools.

Regression coverage for a live session in which a targeted revision ("remove
INCLUDED_IN") caused the schema proposal agent to re-derive the whole plan and
overwrite an earlier, user-requested fix: the Assembly node reverted from being
keyed by `assembly_name` to being keyed by the per-row `assembly_id`, while the
ASSEMBLY_OF relationship kept joining on `assembly_name`. That plan was approved
and built, producing 64 wrong Assembly nodes and zero ASSEMBLY_OF edges.

Two layers are covered here:
  * the state layer behaves correctly in isolation (overwrite-by-key semantics),
    so the drift was not a state-management bug; and
  * approval now mechanically refuses the inconsistent plan the drift produced.
"""
import pytest

from agentic_kg.tools import construction_plan_tools as cpt
from agentic_kg.tools.construction_plan_tools import (
    propose_node_construction,
    propose_relationship_construction,
    remove_node_construction,
    get_proposed_construction_plan,
    approve_proposed_construction_plan,
    check_construction_plan_consistency,
    PROPOSED_CONSTRUCTION_PLAN,
    APPROVED_CONSTRUCTION_PLAN,
)


class FakeToolContext:
    def __init__(self, state=None):
        self.state = state or {}


@pytest.fixture
def ctx():
    return FakeToolContext()


@pytest.fixture
def any_column_exists(monkeypatch):
    """Make the propose tools' search_file sanity check always pass."""
    def fake_search_file(file_path, pattern):
        return {"status": "success",
                "search_results": {"metadata": {"lines_found": 1}}}
    monkeypatch.setattr(cpt, "search_file", fake_search_file)


# --- state layer: overwrite semantics ---------------------------------------

def test_propose_node_twice_same_label_overwrites_unique_column(ctx, any_column_exists):
    propose_node_construction("assemblies.csv", "Assembly", "assembly_id",
                              ["component_name", "quantity", "product_id"], ctx)
    propose_node_construction("assemblies.csv", "Assembly", "assembly_name",
                              ["assembly_id", "product_id"], ctx)

    plan = get_proposed_construction_plan(ctx)
    assert list(plan) == ["Assembly"], "same label must replace, not accumulate"
    assert plan["Assembly"]["unique_column_name"] == "assembly_name"
    assert plan["Assembly"]["properties"] == ["assembly_id", "product_id"]


def test_propose_node_overwrite_is_total_not_a_merge(ctx, any_column_exists):
    """A re-propose drops fields the previous entry had; nothing is carried over.

    This is why a from-scratch re-derivation silently undoes a targeted fix.
    """
    propose_node_construction("assemblies.csv", "Assembly", "assembly_name",
                              ["assembly_id", "product_id"], ctx)
    propose_node_construction("assemblies.csv", "Assembly", "assembly_id",
                              ["component_name"], ctx)

    plan = get_proposed_construction_plan(ctx)
    assert plan["Assembly"]["unique_column_name"] == "assembly_id"
    assert plan["Assembly"]["properties"] == ["component_name"]


def test_remove_then_repropose_leaves_only_the_new_entry(ctx, any_column_exists):
    propose_node_construction("assemblies.csv", "Assembly", "assembly_id", [], ctx)
    remove_node_construction("Assembly", ctx)
    assert get_proposed_construction_plan(ctx) == {}

    propose_node_construction("assemblies.csv", "Assembly", "assembly_name", [], ctx)
    plan = get_proposed_construction_plan(ctx)
    assert plan["Assembly"]["unique_column_name"] == "assembly_name"


def test_get_proposed_plan_reflects_state_not_a_snapshot(ctx, any_column_exists):
    propose_node_construction("suppliers.csv", "Supplier", "supplier_id", ["name"], ctx)
    first = get_proposed_construction_plan(ctx)
    propose_node_construction("products.csv", "Product", "product_id", [], ctx)
    assert set(get_proposed_construction_plan(ctx)) == {"Supplier", "Product"}
    assert first is ctx.state[PROPOSED_CONSTRUCTION_PLAN]


# --- consistency check ------------------------------------------------------

def _drifted_plan():
    """The exact shape that was approved and built in the failing session."""
    return {
        "Product": {"construction_type": "node", "source_file": "products.csv",
                    "label": "Product", "unique_column_name": "product_id",
                    "properties": ["product_name"]},
        "Assembly": {"construction_type": "node", "source_file": "assemblies.csv",
                     "label": "Assembly", "unique_column_name": "assembly_id",
                     "properties": ["component_name", "quantity", "product_id"]},
        "ASSEMBLY_OF": {"construction_type": "relationship", "source_file": "assemblies.csv",
                        "relationship_type": "ASSEMBLY_OF",
                        "from_node_label": "Assembly", "from_node_column": "assembly_name",
                        "to_node_label": "Product", "to_node_column": "product_id",
                        "properties": ["quantity"]},
    }


def _consistent_plan():
    plan = _drifted_plan()
    plan["Assembly"]["unique_column_name"] = "assembly_name"
    plan["Assembly"]["properties"] = ["assembly_id", "product_id"]
    return plan


def test_consistency_check_flags_join_column_the_node_does_not_carry():
    problems = check_construction_plan_consistency(_drifted_plan())
    assert len(problems) == 1
    assert "ASSEMBLY_OF" in problems[0]
    assert "assembly_name" in problems[0]


def test_consistency_check_accepts_the_plan_the_user_asked_for():
    assert check_construction_plan_consistency(_consistent_plan()) == []


def test_consistency_check_allows_join_on_a_declared_property():
    plan = _consistent_plan()
    # product_id is a property of Assembly, not its key, but it does exist on the node
    plan["ASSEMBLY_OF"]["from_node_column"] = "assembly_name"
    plan["ASSEMBLY_OF"]["to_node_column"] = "product_id"
    assert check_construction_plan_consistency(plan) == []


def test_consistency_check_flags_relationship_to_an_undefined_label():
    plan = _consistent_plan()
    del plan["Assembly"]
    problems = check_construction_plan_consistency(plan)
    assert any("no node construction" in p for p in problems)


def test_consistency_check_ignores_non_dict_and_empty_plans():
    assert check_construction_plan_consistency({}) == []
    assert check_construction_plan_consistency([]) == []


# --- approval gate ----------------------------------------------------------

def test_approve_refuses_the_drifted_plan_and_does_not_record_it(ctx):
    ctx.state[PROPOSED_CONSTRUCTION_PLAN] = _drifted_plan()
    result = approve_proposed_construction_plan(ctx)

    assert result["status"] == "error"
    assert "ASSEMBLY_OF" in result["error_message"]
    assert APPROVED_CONSTRUCTION_PLAN not in ctx.state


def test_approve_records_a_consistent_plan(ctx):
    ctx.state[PROPOSED_CONSTRUCTION_PLAN] = _consistent_plan()
    result = approve_proposed_construction_plan(ctx)

    assert result["status"] == "success"
    assert ctx.state[APPROVED_CONSTRUCTION_PLAN] == _consistent_plan()


def test_approve_refuses_when_there_is_no_proposed_plan(ctx):
    result = approve_proposed_construction_plan(ctx)
    assert result["status"] == "error"
    assert APPROVED_CONSTRUCTION_PLAN not in ctx.state
